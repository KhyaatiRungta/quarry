"""Static AST guard plus subprocess execution for model-written Python.

READ THIS BEFORE TRUSTING IT
----------------------------
This is defence-in-depth against a CONFUSED model, not a security boundary
against an ADVERSARIAL one. The generated code runs as your user, on your
filesystem, with your network. A determined attacker who controls the model
output can get past a static allowlist; the README lists the specific holes
we know about (pandas `eval`/`query`, C-extension reach-through,
resource exhaustion below the caps). If you need a real boundary, run this
inside a container, a VM, or a seccomp/gVisor sandbox and treat this module as
the first of several layers rather than the only one.

What it does buy you:
  - a model that absent-mindedly writes `import os; os.system(...)` is stopped
    before execution, with a specific message it can learn from;
  - runaway loops die on a wall-clock timeout;
  - every run is confined to its own working directory;
  - a crash cannot take the agent process down, because it is a subprocess.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

# Modules the generated code may import. Root package only: `matplotlib.pyplot`
# is allowed because its root is `matplotlib`.
ALLOWED_IMPORTS = frozenset({
    "pandas", "numpy", "matplotlib", "json", "math", "datetime", "re", "statistics",
})

# Names that are never allowed to be referenced, whatever the context.
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "exit", "quit", "help", "memoryview",
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
    "ctypes", "pickle", "requests", "urllib", "http", "builtins", "__builtins__",
})

# Import targets that get a louder message because they are the classic escapes.
BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
    "ctypes", "pickle", "requests", "urllib", "http", "builtins", "multiprocessing",
    "threading", "tempfile", "glob", "io", "codecs", "platform", "signal", "resource",
})

READ_MODES = frozenset({"r", "rb", "rt", "br", "tr"})


@dataclass
class Violation:
    code: str
    message: str
    lineno: int = 0
    col: int = 0

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "lineno": self.lineno, "col": self.col}

    def __str__(self) -> str:
        return f"line {self.lineno}: [{self.code}] {self.message}"


@dataclass
class GuardReport:
    ok: bool
    violations: list[Violation] = field(default_factory=list)
    syntax_error: str | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "violations": [v.as_dict() for v in self.violations],
            "syntax_error": self.syntax_error,
        }

    def feedback(self) -> str:
        """The message handed back to the model. Specific, so it can fix it."""
        if self.syntax_error:
            return f"Your code did not parse: {self.syntax_error}"
        lines = ["The sandbox rejected your code before running it:"]
        lines += [f"  {v}" for v in self.violations]
        lines.append(
            "Allowed imports: " + ", ".join(sorted(ALLOWED_IMPORTS)) +
            ". The dataframe is already bound as `df`; you never need to read files or "
            "touch the OS."
        )
        return "\n".join(lines)


class _Guard(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[Violation] = []

    def _add(self, code: str, message: str, node: ast.AST) -> None:
        self.violations.append(
            Violation(code, message, getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
        )

    # -- imports ---------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module(alias.name, node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            self._add("relative-import", "Relative imports are not allowed.", node)
        elif node.module:
            self._check_module(node.module, node)
        self.generic_visit(node)

    def _check_module(self, dotted: str, node: ast.AST) -> None:
        root = dotted.split(".")[0]
        if root in BLOCKED_MODULES:
            self._add("blocked-import",
                      f"`{dotted}` is blocked. The sandbox has no OS, filesystem or network access.",
                      node)
        elif root not in ALLOWED_IMPORTS:
            self._add("import-not-allowed",
                      f"`{dotted}` is not on the import allowlist.", node)

    # -- names and attributes -------------------------------------------
    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self._add("forbidden-name", f"`{node.id}` may not be used.", node)
        elif node.id.startswith("__") and node.id.endswith("__"):
            self._add("dunder-name", f"Dunder name `{node.id}` may not be used.", node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self._add("dunder-attribute",
                      f"Dunder attribute access `.{node.attr}` may not be used.", node)
        self.generic_visit(node)

    # -- calls -----------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            self._check_open(node)
        elif isinstance(func, ast.Attribute) and func.attr in {"system", "popen", "spawn", "fork"}:
            self._add("forbidden-call", f"`.{func.attr}(...)` may not be called.", node)
        self.generic_visit(node)

    def _check_open(self, node: ast.Call) -> None:
        """`open` is permitted only to read a plain filename inside the run directory."""
        if not node.args:
            self._add("open-unsafe", "open() needs a literal filename argument.", node)
            return
        target = node.args[0]
        if not (isinstance(target, ast.Constant) and isinstance(target.value, str)):
            self._add("open-unsafe",
                      "open() is only allowed with a literal filename inside the run directory.",
                      node)
            return
        name = target.value
        if name.startswith("/") or name.startswith("~") or ".." in name.split("/"):
            self._add("open-escape",
                      f"open({name!r}) points outside the run directory.", node)
        mode = "r"
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
            mode = str(node.args[1].value)
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = str(kw.value.value)
        if mode not in READ_MODES:
            self._add("open-write",
                      f"open(..., {mode!r}) would write. Writing is limited to chart.png, "
                      "which the sandbox saves for you.", node)


def guard_code(code: str) -> GuardReport:
    """Walk the AST and report every rule the code breaks.

    AST walking rather than regex: `import os` and `import  os` and
    `import os.path as p` and `from os import system` are one rule here, and a
    string containing the text "import os" is correctly not a violation.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return GuardReport(ok=False, syntax_error=f"{exc.msg} (line {exc.lineno})")
    guard = _Guard()
    guard.visit(tree)
    return GuardReport(ok=not guard.violations, violations=guard.violations)


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------

import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MEMORY_MB = 2048
MAX_STREAM_CHARS = 8000
MAX_REPR_CHARS = 4000

RUNNER_FILENAME = "_quarry_runner.py"
CODE_FILENAME = "generated.py"
OUTCOME_FILENAME = "outcome.json"
DATAFRAME_FILENAME = "df.pkl"
CHART_FILENAME = "chart.png"

# Runner harness. Written into the run directory and executed as the subprocess
# entry point. It is NOT subject to the AST guard; only the model's code is.
RUNNER_SOURCE = '''\
import json, sys, traceback, io, contextlib, os

OUTCOME = sys.argv[1]
CODE = sys.argv[2]
DF_PATH = sys.argv[3] if len(sys.argv) > 3 else ""

outcome = {
    "ok": False, "stdout": "", "stderr": "", "result_repr": None, "result_type": None,
    "exception_type": None, "traceback": None, "result_json": None,
}
buf_out, buf_err = io.StringIO(), io.StringIO()
try:
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    import numpy as np
    ns = {"pd": pd, "np": np}
    if DF_PATH:
        if not os.path.exists(DF_PATH):
            raise RuntimeError("sandbox harness: dataframe not found at " + DF_PATH)
        ns["df"] = pd.read_pickle(DF_PATH)
    source = open(CODE).read()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        exec(compile(source, "generated.py", "exec"), ns, ns)
    result = ns.get("result", None)
    outcome["ok"] = True
    if result is not None:
        outcome["result_type"] = type(result).__name__
        try:
            outcome["result_repr"] = repr(result)
        except Exception:
            outcome["result_repr"] = "<unreprable object>"
        try:
            if isinstance(result, (int, float, str, bool)):
                outcome["result_json"] = result
            elif isinstance(result, (np.integer,)):
                outcome["result_json"] = int(result)
            elif isinstance(result, (np.floating,)):
                outcome["result_json"] = float(result)
            elif isinstance(result, pd.Series):
                outcome["result_json"] = json.loads(result.head(50).to_json())
            elif isinstance(result, pd.DataFrame):
                outcome["result_json"] = json.loads(result.head(50).to_json(orient="records"))
            else:
                outcome["result_json"] = json.loads(json.dumps(result, default=str))
        except Exception:
            outcome["result_json"] = None
except BaseException as exc:
    outcome["ok"] = False
    outcome["exception_type"] = type(exc).__name__
    outcome["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
finally:
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass
    outcome["stdout"] = buf_out.getvalue()
    outcome["stderr"] = buf_err.getvalue()
    with open(OUTCOME, "w") as fh:
        json.dump(outcome, fh, default=str)
'''


@dataclass
class ExecutionResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    result_repr: str | None = None
    result_type: str | None = None
    chart_path: str | None = None
    duration_s: float = 0.0
    exception_type: str | None = None
    traceback: str | None = None
    # Extras the agent needs; not part of the minimum contract.
    result_json: object = None
    guard: GuardReport | None = None
    failure_kind: str | None = None  # guard_violation|syntax_error|runtime_error|timeout|empty_result

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "result_repr": self.result_repr,
            "result_type": self.result_type,
            "chart_path": self.chart_path,
            "duration_s": round(self.duration_s, 3),
            "exception_type": self.exception_type,
            "traceback": self.traceback,
            "result_json": self.result_json,
            "failure_kind": self.failure_kind,
            "guard": self.guard.as_dict() if self.guard else None,
        }

    def observation(self, max_chars: int = MAX_STREAM_CHARS) -> str:
        """The text handed back to the model as the tool result."""
        parts = []
        if self.guard is not None and not self.guard.ok:
            return self.guard.feedback()
        if self.stdout.strip():
            parts.append("stdout:\n" + _truncate(self.stdout, max_chars))
        if self.stderr.strip():
            parts.append("stderr:\n" + _truncate(self.stderr, max_chars))
        if self.ok:
            if self.result_repr is None:
                parts.append(
                    "Execution succeeded but `result` was never assigned. Assign your answer "
                    "to a variable called `result`."
                )
            else:
                parts.append(f"result ({self.result_type}):\n{_truncate(self.result_repr, MAX_REPR_CHARS)}")
            if self.chart_path:
                parts.append(f"chart saved: {os.path.basename(self.chart_path)}")
        else:
            if self.failure_kind == "timeout":
                parts.append(f"Execution timed out after {self.duration_s:.1f}s and was killed.")
            else:
                parts.append(
                    f"Execution failed with {self.exception_type}. Traceback (most recent call last):\n"
                    + _truncate_traceback(self.traceback or "")
                )
        return "\n\n".join(p for p in parts if p) or "(no output)"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2:]
    return f"{head}\n... [{len(text) - limit} characters elided] ...\n{tail}"


def _truncate_traceback(tb: str, keep_lines: int = 24) -> str:
    """Keep the tail of a traceback: the frames nearest the error are the useful ones."""
    lines = tb.rstrip().splitlines()
    if len(lines) <= keep_lines:
        return "\n".join(lines)
    return "... [%d earlier frames elided] ...\n" % (len(lines) - keep_lines) + "\n".join(lines[-keep_lines:])


def memory_limit_supported() -> bool:
    """RLIMIT_AS is honoured on Linux. On macOS it is accepted but not enforced for
    the mappings numpy/pandas make, so we do not pretend it is a memory cap there."""
    return platform.system() == "Linux"


def _preexec(memory_mb: int | None):  # pragma: no cover - runs in the forked child
    def _apply():
        os.setsid()
        if memory_mb:
            try:
                limit = memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except (ValueError, OSError):
                pass
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except (ValueError, OSError, AttributeError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
        except (ValueError, OSError):
            pass
    return _apply


def run_python(code: str, workdir: str | Path, dataframe_path: str | Path | None = None,
               timeout_s: float = DEFAULT_TIMEOUT_S, memory_mb: int | None = DEFAULT_MEMORY_MB,
               expect_chart: bool = False, python_executable: str | None = None) -> ExecutionResult:
    """Guard, then execute `code` in a fresh subprocess rooted at `workdir`."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    report = guard_code(code)
    (workdir / CODE_FILENAME).write_text(code)
    if not report.ok:
        return ExecutionResult(
            ok=False,
            guard=report,
            exception_type="SyntaxError" if report.syntax_error else "SandboxViolation",
            failure_kind="syntax_error" if report.syntax_error else "guard_violation",
            traceback=report.feedback(),
        )

    chart_file = workdir / CHART_FILENAME
    if chart_file.exists():
        chart_file.unlink()
    outcome_file = workdir / OUTCOME_FILENAME
    if outcome_file.exists():
        outcome_file.unlink()
    (workdir / RUNNER_FILENAME).write_text(RUNNER_SOURCE)

    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(workdir),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        # Deliberately NOT inheriting the parent environment: no API keys reach the child.
    }

    # -I is isolated mode: no user site-packages, no PYTHON* env influence, cwd not
    # prepended to sys.path. We invoke the current interpreter so the venv's pandas
    # and matplotlib are still importable.
    argv = [
        python_executable or sys.executable, "-I",
        RUNNER_FILENAME, OUTCOME_FILENAME, CODE_FILENAME,
        # Absolute: the child runs with cwd set to the run directory, so a relative
        # path from the caller would silently resolve to nothing.
        str(Path(dataframe_path).resolve()) if dataframe_path else "",
    ]
    started = time.time()
    timed_out = False
    proc = subprocess.Popen(
        argv, cwd=str(workdir), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        preexec_fn=_preexec(memory_mb if memory_limit_supported() else None),
    )
    try:
        proc_out, proc_err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):  # pragma: no cover
            proc.kill()
        proc_out, proc_err = proc.communicate()
    duration = time.time() - started

    if timed_out:
        return ExecutionResult(
            ok=False, stdout=proc_out or "", stderr=proc_err or "",
            duration_s=duration, exception_type="Timeout", failure_kind="timeout",
            traceback=f"Killed after exceeding the {timeout_s:.0f}s execution limit.",
            guard=report,
        )

    if not outcome_file.exists():
        return ExecutionResult(
            ok=False, stdout=proc_out or "", stderr=proc_err or "", duration_s=duration,
            exception_type="SandboxCrash", failure_kind="runtime_error",
            traceback=(proc_err or "The sandbox subprocess died without reporting an outcome. "
                       "This usually means it was killed by the OS (memory)."),
            guard=report,
        )

    data = json.loads(outcome_file.read_text())
    chart_path = str(chart_file) if chart_file.exists() else None
    ok = bool(data["ok"])
    failure_kind = None
    if not ok:
        failure_kind = "runtime_error"
    elif expect_chart and not chart_path:
        ok = False
        failure_kind = "empty_result"
        data["exception_type"] = "NoChartProduced"
        data["traceback"] = ("The code ran but no chart.png was written. Save the figure with "
                             "plt.savefig('chart.png').")
    elif data.get("result_repr") is None and not expect_chart:
        failure_kind = "empty_result"

    return ExecutionResult(
        ok=ok,
        stdout=data.get("stdout", "") + (proc_out or ""),
        stderr=data.get("stderr", "") + (proc_err or ""),
        result_repr=data.get("result_repr"),
        result_type=data.get("result_type"),
        result_json=data.get("result_json"),
        chart_path=chart_path,
        duration_s=duration,
        exception_type=data.get("exception_type"),
        traceback=data.get("traceback"),
        guard=report,
        failure_kind=failure_kind,
    )
