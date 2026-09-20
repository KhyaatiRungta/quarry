"""The agent loop. Hand-written ReAct-style, no framework.

The whole loop is one function you can read top to bottom: ask the model, execute
the tool calls it emits, append the real observation, repeat until final_answer,
max_steps or the cost budget. The two parts worth reading closely are the repair
path (`_repair_feedback` and the `repair_*` counters) and the verify gate before an
answer is allowed out.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .dataset import Dataset
from .llm import LLMResponse, Usage
from .tools import AnalysisSession, ToolArgumentError, ToolResult, build_tools
from .verify import VerifyReport, verify_answer

PROMPT_DIR = Path(__file__).parent / "prompts"
TRACE_SCHEMA_VERSION = 1

DEFAULT_MAX_STEPS = 8
DEFAULT_COST_BUDGET_USD = 0.50
DEFAULT_MAX_REPAIRS = 4
EXECUTION_TOOLS = frozenset({"run_python", "run_sql", "make_chart"})


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text()


@dataclass
class RunResult:
    run_id: str
    question: str
    dataset: str
    ok: bool
    stop_reason: str
    answer: str | None = None
    supporting_values: list[str] = field(default_factory=list)
    verify: VerifyReport | None = None
    steps: int = 0
    tool_calls: int = 0
    repair_attempts: int = 0
    repair_successes: int = 0
    failure_kinds: list[str] = field(default_factory=list)
    sandbox_violations: int = 0
    usage: Usage = field(default_factory=Usage)
    wall_s: float = 0.0
    chart_path: str | None = None
    trace_path: str | None = None
    run_dir: str | None = None

    @property
    def grounded(self) -> bool:
        return bool(self.verify and self.verify.grounded)

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "question": self.question,
            "dataset": self.dataset,
            "ok": self.ok,
            "stop_reason": self.stop_reason,
            "answer": self.answer,
            "supporting_values": self.supporting_values,
            "verify": self.verify.as_dict() if self.verify else None,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "repair_attempts": self.repair_attempts,
            "repair_successes": self.repair_successes,
            "failure_kinds": self.failure_kinds,
            "sandbox_violations": self.sandbox_violations,
            "usage": self.usage.as_dict(),
            "wall_s": round(self.wall_s, 3),
            "chart_path": self.chart_path,
        }


class QuarryAgent:
    """One agent instance answers one question about one dataset.

    self_correction=False is the ablation used by the benchmark: the repair loop is
    removed, so the first failed execution ends the run. Everything else -- the
    prompt, the tools, the sandbox, the verify gate -- is identical, which is what
    makes the on/off comparison mean something.
    """

    def __init__(self, client, dataset: Dataset, runs_dir: str | Path = "runs",
                 max_steps: int = DEFAULT_MAX_STEPS,
                 cost_budget_usd: float = DEFAULT_COST_BUDGET_USD,
                 self_correction: bool = True,
                 max_repairs: int = DEFAULT_MAX_REPAIRS,
                 verify_retries: int = 1,
                 timeout_s: float = 30.0,
                 run_id: str | None = None,
                 log=None):
        self.client = client
        self.dataset = dataset
        self.runs_dir = Path(runs_dir)
        self.max_steps = max_steps
        self.cost_budget_usd = cost_budget_usd
        self.self_correction = self_correction
        self.max_repairs = max_repairs
        self.verify_retries = verify_retries
        self.timeout_s = timeout_s
        self.run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        self._log = log or (lambda msg: None)

    # -- public ----------------------------------------------------------
    def ask(self, question: str) -> RunResult:
        run_dir = self.runs_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        session = AnalysisSession(self.dataset, run_dir, timeout_s=self.timeout_s)
        tools = build_tools(session)
        system = load_prompt("system").format(
            schema=self.dataset.schema_text(),
            max_steps=self.max_steps,
            timeout=int(self.timeout_s),
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

        trace_steps: list[dict] = []
        started = time.time()
        result = RunResult(
            run_id=self.run_id, question=question, dataset=self.dataset.name,
            ok=False, stop_reason="max_steps", run_dir=str(run_dir),
        )
        pending_failure: str | None = None
        verify_retries_left = self.verify_retries
        no_tool_nudges = 0

        try:
            for step_no in range(1, self.max_steps + 1):
                result.steps = step_no
                response = self.client.complete(messages, tools=tools.schemas())
                result.usage = result.usage + response.usage
                step = {
                    "index": step_no,
                    "type": "model_turn",
                    "model": response.model,
                    "text": response.text,
                    "usage": response.usage.as_dict(),
                    "cumulative_cost_usd": round(result.usage.cost_usd, 6),
                    "tool_calls": [],
                }
                trace_steps.append(step)

                if not response.tool_calls:
                    no_tool_nudges += 1
                    messages.append({"role": "assistant", "content": response.text})
                    if no_tool_nudges > 1:
                        result.stop_reason = "no_tool_call"
                        break
                    messages.append({
                        "role": "user",
                        "content": "You must answer by calling a tool. Use run_python or run_sql "
                                   "to compute the answer, then final_answer.",
                    })
                    step["nudged"] = True
                    continue

                messages.append(_assistant_message(response))
                terminal = False
                # All tool messages for this turn must be appended before any user
                # message, so repair feedback is deferred until the turn is drained.
                repair_feedback: str | None = None

                for call in response.tool_calls:
                    result.tool_calls += 1
                    # A repair is any tool call issued while an unresolved failure is
                    # outstanding, including the schema lookup a model often does
                    # before rewriting the code. The failure stays outstanding until an
                    # execution tool actually succeeds.
                    is_repair = pending_failure is not None and call["name"] != "final_answer"
                    if is_repair:
                        result.repair_attempts += 1
                    outcome, tool_result = self._dispatch(tools, call)
                    step["tool_calls"].append({
                        "id": call["id"],
                        "name": call["name"],
                        "arguments": call["arguments"],
                        "is_repair": is_repair,
                        "ok": outcome["ok"],
                        "observation": outcome["observation"],
                        "duration_s": outcome.get("duration_s", 0.0),
                        "failure_kind": outcome.get("failure_kind"),
                        "execution": outcome.get("execution"),
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": outcome["observation"],
                    })

                    if not outcome["ok"]:
                        kind = outcome.get("failure_kind") or "tool_error"
                        result.failure_kinds.append(kind)
                        self._log(f"step {step_no}: {call['name']} failed ({kind})")
                        if not self.self_correction:
                            result.stop_reason = "failed_without_repair"
                            terminal = True
                            continue
                        if result.repair_attempts >= self.max_repairs:
                            result.stop_reason = "repair_budget_exhausted"
                            terminal = True
                            continue
                        pending_failure = kind
                        repair_feedback = load_prompt("repair").format(
                            failure_kind=kind,
                            observation=outcome["observation"],
                            attempt=result.repair_attempts + 1,
                        )
                        continue

                    if call["name"] in EXECUTION_TOOLS:
                        if is_repair:
                            result.repair_successes += 1
                        pending_failure = None
                        repair_feedback = None

                    if tool_result is None or not tool_result.terminal:
                        continue

                    report = verify_answer(tool_result.payload["supporting_values"],
                                           session.evidence())
                    trace_steps.append({
                        "index": step_no,
                        "type": "verify",
                        "supporting_values": tool_result.payload["supporting_values"],
                        "report": report.as_dict(),
                    })
                    if report.grounded or verify_retries_left <= 0 or not self.self_correction:
                        result.verify = report
                        result.answer = tool_result.payload["answer"]
                        result.supporting_values = tool_result.payload["supporting_values"]
                        result.ok = True
                        result.stop_reason = ("final_answer" if report.grounded
                                              else "final_answer_ungrounded")
                        terminal = True
                        break
                    verify_retries_left -= 1
                    session.final_answer = None
                    repair_feedback = load_prompt("verify").format(
                        unsupported="\n".join(f"  - {v}" for v in report.unsupported))
                    self._log(f"step {step_no}: verify rejected {report.unsupported}")

                if repair_feedback and not terminal:
                    messages.append({"role": "user", "content": repair_feedback})

                if terminal:
                    break
                if result.usage.cost_usd > self.cost_budget_usd:
                    result.stop_reason = "cost_budget_exceeded"
                    break

            result.wall_s = time.time() - started
            result.sandbox_violations = session.sandbox_violations
            result.chart_path = session.chart_path
            trace_path = run_dir / "trace.json"
            trace_path.write_text(json.dumps(
                self._trace(result, question, messages, trace_steps), indent=2, default=str))
            result.trace_path = str(trace_path)
            return result
        finally:
            session.close()

    # -- internals -------------------------------------------------------
    def _dispatch(self, tools, call: dict) -> tuple[dict, ToolResult | None]:
        """Run one tool call. Argument errors are observations, not exceptions."""
        if call.get("malformed_json"):
            return ({"ok": False, "observation":
                     f"Your arguments for {call['name']} were not valid JSON: "
                     f"{call['raw_arguments'][:400]}",
                     "failure_kind": "bad_arguments"}, None)
        try:
            tool_result = tools.call(call["name"], call["arguments"])
        except ToolArgumentError as exc:
            return ({"ok": False, "observation": str(exc), "failure_kind": "bad_arguments"}, None)
        except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the run
            return ({"ok": False,
                     "observation": f"The tool raised {type(exc).__name__}: {exc}",
                     "failure_kind": "tool_error"}, None)

        outcome = {
            "ok": tool_result.ok,
            "observation": tool_result.observation,
            "failure_kind": None,
        }
        ex = tool_result.execution
        if ex is not None:
            outcome["duration_s"] = round(ex.duration_s, 3)
            outcome["failure_kind"] = ex.failure_kind
            outcome["execution"] = {
                "ok": ex.ok,
                "code": tool_result.payload.get("code"),
                "stdout": ex.stdout,
                "stderr": ex.stderr,
                "result_repr": ex.result_repr,
                "result_type": ex.result_type,
                "exception_type": ex.exception_type,
                "traceback": ex.traceback,
                "chart_path": ex.chart_path,
                "duration_s": round(ex.duration_s, 3),
                "guard": ex.guard.as_dict() if ex.guard else None,
                "failure_kind": ex.failure_kind,
            }
        elif not tool_result.ok:
            outcome["failure_kind"] = "tool_error"
        if call["name"] == "run_sql" and tool_result.payload:
            outcome["execution"] = {
                "ok": tool_result.ok,
                "code": tool_result.payload.get("query"),
                "stdout": tool_result.observation,
                "stderr": "",
                "result_repr": None, "result_type": "sql_rows",
                "exception_type": None, "traceback": None, "chart_path": None,
                "duration_s": 0.0, "guard": None, "failure_kind": None,
            }
        return outcome, tool_result

    def _trace(self, result: RunResult, question: str, messages: list[dict],
               steps: list[dict]) -> dict:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "run_id": result.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "dataset": {
                "name": self.dataset.name,
                "table": self.dataset.table,
                "source": self.dataset.source_path,
                "rows": int(len(self.dataset.df)),
                "columns": [str(c) for c in self.dataset.df.columns],
            },
            "config": {
                "model": getattr(self.client, "model", "unknown"),
                "offline": bool(getattr(self.client, "offline", False)),
                "max_steps": self.max_steps,
                "cost_budget_usd": self.cost_budget_usd,
                "self_correction": self.self_correction,
                "max_repairs": self.max_repairs,
                "verify_retries": self.verify_retries,
                "timeout_s": self.timeout_s,
            },
            "steps": steps,
            "summary": result.summary(),
            "messages": messages,
        }


def _assistant_message(response: LLMResponse) -> dict:
    return {
        "role": "assistant",
        "content": response.text or None,
        "tool_calls": [
            {
                "id": c["id"],
                "type": "function",
                "function": {"name": c["name"], "arguments": c["raw_arguments"]},
            }
            for c in response.tool_calls
        ],
    }
