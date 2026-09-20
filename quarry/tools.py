"""Hand-written OpenAI-style tool schemas and their handlers.

No framework. Each tool is a `Tool` with an explicit JSON schema, a docstring the
description is taken from, and an argument validator that rejects bad calls with a
message the model can act on. A rejected call is an observation, not a crash: the
agent feeds the validation error back and lets the model try again.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .dataset import Dataset
from .sandbox import DEFAULT_TIMEOUT_S, ExecutionResult, run_python

MAX_SQL_ROWS = 200
SQL_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex)\b",
    re.IGNORECASE,
)


class ToolArgumentError(ValueError):
    """Raised when a tool call's arguments fail validation."""


@dataclass
class ToolResult:
    ok: bool
    observation: str
    payload: dict = field(default_factory=dict)
    execution: ExecutionResult | None = None
    terminal: bool = False


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., ToolResult]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate(self, arguments: Any) -> dict:
        if not isinstance(arguments, dict):
            raise ToolArgumentError(
                f"{self.name}: arguments must be a JSON object, got {type(arguments).__name__}."
            )
        props = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])
        missing = [k for k in required if k not in arguments or arguments[k] is None]
        if missing:
            raise ToolArgumentError(
                f"{self.name}: missing required argument(s): {', '.join(missing)}."
            )
        unknown = [k for k in arguments if k not in props]
        if unknown:
            raise ToolArgumentError(
                f"{self.name}: unknown argument(s): {', '.join(sorted(unknown))}. "
                f"Accepted: {', '.join(sorted(props)) or '(none)'}."
            )
        cleaned = {}
        for key, value in arguments.items():
            expected = props[key].get("type")
            cleaned[key] = _coerce(self.name, key, value, expected)
        return cleaned


_TYPE_NAMES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _coerce(tool: str, key: str, value: Any, expected: str | None) -> Any:
    if expected is None:
        return value
    if expected == "array" and isinstance(value, str):
        # Models sometimes hand back a JSON-encoded array in a string field.
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            raise ToolArgumentError(f"{tool}: `{key}` must be an array, got a plain string.")
        if not isinstance(parsed, list):
            raise ToolArgumentError(f"{tool}: `{key}` must be an array.")
        return parsed
    py_type = _TYPE_NAMES.get(expected)
    if py_type and not isinstance(value, py_type):
        raise ToolArgumentError(
            f"{tool}: `{key}` must be of type {expected}, got {type(value).__name__}."
        )
    if expected == "string" and not value.strip():
        raise ToolArgumentError(f"{tool}: `{key}` must not be empty.")
    return value


class AnalysisSession:
    """Holds the dataset, the run directory and everything the tools produced.

    `observed_outputs` is the evidence pool the verify step checks answers against:
    if a number is not in here, the model did not compute it.
    """

    def __init__(self, dataset: Dataset, run_dir: str | Path,
                 timeout_s: float = DEFAULT_TIMEOUT_S):
        self.dataset = dataset
        self.run_dir = Path(run_dir).resolve()
        self.workdir = self.run_dir / "workdir"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.df_path = self.workdir / "df.pkl"
        dataset.df.to_pickle(self.df_path)
        self.executions: list[ExecutionResult] = []
        self.observed_outputs: list[str] = []
        self.chart_path: str | None = None
        self.sandbox_violations = 0
        self.final_answer: dict | None = None
        self._conn: sqlite3.Connection | None = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = self.dataset.sqlite_connection()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def record_output(self, text: str) -> None:
        if text and text.strip():
            self.observed_outputs.append(text)

    def evidence(self) -> str:
        return "\n".join(self.observed_outputs)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _inspect_schema(session: AnalysisSession) -> ToolResult:
    """Return the table's columns, dtypes, null counts, cardinality and sample rows.

    Call this before writing code if anything about the data is uncertain. It is
    cheap and it is the only way to learn how a column is actually stored.
    """
    text = session.dataset.schema_text()
    session.record_output(text)
    return ToolResult(True, text, payload=session.dataset.schema())


def _run_python(session: AnalysisSession, code: str) -> ToolResult:
    """Execute pandas code in the sandbox. `df` is bound to the table; assign your
    answer to `result`. Imports are limited to pandas, numpy, matplotlib, json, math,
    datetime, re and statistics. No filesystem, no network, 30s limit.
    """
    ex = run_python(code, session.workdir, session.df_path, timeout_s=session.timeout_s)
    session.executions.append(ex)
    if ex.guard is not None and ex.failure_kind == "guard_violation":
        session.sandbox_violations += 1
    obs = ex.observation()
    if ex.ok:
        session.record_output(obs)
    return ToolResult(ex.ok, obs, payload={"code": code}, execution=ex)


def _run_sql(session: AnalysisSession, query: str) -> ToolResult:
    """Run a read-only SQL query against the table in SQLite and return the rows.

    Only SELECT and WITH statements are accepted, one statement at a time. Results
    are capped at 200 rows.
    """
    stripped = query.strip().rstrip(";").strip()
    if not stripped:
        return ToolResult(False, "run_sql: the query was empty.")
    head = stripped.split(None, 1)[0].lower()
    if head not in {"select", "with"}:
        session.sandbox_violations += 1
        return ToolResult(False, f"run_sql: only SELECT/WITH queries are allowed, got `{head}`.")
    if ";" in stripped:
        return ToolResult(False, "run_sql: send exactly one statement, without a trailing `;`.")
    if SQL_FORBIDDEN.search(stripped):
        session.sandbox_violations += 1
        return ToolResult(False, "run_sql: this connection is read-only; data-modifying "
                                 "statements are rejected.")
    try:
        frame = pd.read_sql_query(stripped, session.connection)
    except Exception as exc:  # noqa: BLE001 - sqlite raises several unrelated types
        return ToolResult(
            False,
            f"run_sql failed with {type(exc).__name__}: {exc}\n"
            f"The table is named `{session.dataset.table}`.",
            payload={"query": stripped},
        )
    truncated = len(frame) > MAX_SQL_ROWS
    shown = frame.head(MAX_SQL_ROWS)
    obs = f"{len(frame)} row(s)"
    obs += " (showing first 200)" if truncated else ""
    obs += ":\n" + shown.to_string(index=False)
    session.record_output(obs)
    return ToolResult(True, obs, payload={
        "query": stripped, "row_count": int(len(frame)),
        "rows": json.loads(shown.to_json(orient="records")),
    })


def _make_chart(session: AnalysisSession, code: str) -> ToolResult:
    """Execute matplotlib code in the sandbox and save the figure to chart.png.

    Same rules as run_python. `df` is bound; the Agg backend is already selected;
    you must call plt.savefig('chart.png').
    """
    ex = run_python(code, session.workdir, session.df_path,
                    timeout_s=session.timeout_s, expect_chart=True)
    session.executions.append(ex)
    if ex.failure_kind == "guard_violation":
        session.sandbox_violations += 1
    if ex.ok and ex.chart_path:
        session.chart_path = ex.chart_path
    obs = ex.observation()
    if ex.ok:
        session.record_output(obs)
    return ToolResult(ex.ok, obs, payload={"code": code, "chart_path": ex.chart_path},
                      execution=ex)


def _final_answer(session: AnalysisSession, answer: str,
                  supporting_values: list) -> ToolResult:
    """Finish the run. `answer` is prose for the user. `supporting_values` lists the
    computed values the answer rests on, each exactly as it appeared in execution
    output, so the claim can be checked against what actually ran.
    """
    values = [str(v) for v in supporting_values]
    if not values:
        raise ToolArgumentError(
            "final_answer: supporting_values must contain at least one value that appeared "
            "in execution output. If you have not computed anything yet, run code first."
        )
    payload = {"answer": answer, "supporting_values": values}
    session.final_answer = payload
    return ToolResult(True, "Answer recorded.", payload=payload, terminal=True)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _describe(fn: Callable) -> str:
    return " ".join((fn.__doc__ or "").split())


def build_tools(session: AnalysisSession) -> "ToolRegistry":
    tools = [
        Tool("inspect_schema", _describe(_inspect_schema),
             {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
             lambda: _inspect_schema(session)),
        Tool("run_python", _describe(_run_python),
             {"type": "object",
              "properties": {"code": {
                  "type": "string",
                  "description": "Python source. `df` is bound to the table. Assign the answer "
                                 "to `result`. print() output is returned to you.",
              }},
              "required": ["code"], "additionalProperties": False},
             lambda code: _run_python(session, code)),
        Tool("run_sql", _describe(_run_sql),
             {"type": "object",
              "properties": {"query": {
                  "type": "string",
                  "description": f"A single read-only SQL statement against table "
                                 f"`{session.dataset.table}`.",
              }},
              "required": ["query"], "additionalProperties": False},
             lambda query: _run_sql(session, query)),
        Tool("make_chart", _describe(_make_chart),
             {"type": "object",
              "properties": {"code": {
                  "type": "string",
                  "description": "matplotlib code that ends with plt.savefig('chart.png').",
              }},
              "required": ["code"], "additionalProperties": False},
             lambda code: _make_chart(session, code)),
        Tool("final_answer", _describe(_final_answer),
             {"type": "object",
              "properties": {
                  "answer": {"type": "string",
                             "description": "The answer to the user's question, in plain prose."},
                  "supporting_values": {
                      "type": "array", "items": {"type": "string"},
                      "description": "Computed values backing the answer, written exactly as they "
                                     "appeared in execution output (e.g. \"0.4471\", \"North\").",
                  },
              },
              "required": ["answer", "supporting_values"], "additionalProperties": False},
             lambda answer, supporting_values: _final_answer(session, answer, supporting_values)),
    ]
    return ToolRegistry(tools)


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def call(self, name: str, arguments: Any) -> ToolResult:
        if name not in self._tools:
            raise ToolArgumentError(
                f"Unknown tool {name!r}. Available tools: {', '.join(self._tools)}."
            )
        tool = self._tools[name]
        return tool.handler(**tool.validate(arguments))
