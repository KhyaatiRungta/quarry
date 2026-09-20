import pandas as pd
import pytest

from quarry.sandbox import CHART_FILENAME, run_python


def test_executes_and_returns_result(workdir, df_path):
    ex = run_python("result = int(df['units'].sum())", workdir, df_path)
    assert ex.ok
    assert ex.result_repr == "10"
    assert ex.result_type == "int"
    assert ex.result_json == 10


def test_dataframe_is_injected_as_df(workdir, df_path):
    ex = run_python("result = list(df.columns)", workdir, df_path)
    assert ex.ok and "revenue" in ex.result_repr


def test_captures_stdout(workdir, df_path):
    ex = run_python("print('hello from the sandbox')\nresult = 1", workdir, df_path)
    assert "hello from the sandbox" in ex.stdout


def test_captures_traceback_of_a_runtime_error(workdir, df_path):
    ex = run_python("result = df['missing_column'].sum()", workdir, df_path)
    assert not ex.ok
    assert ex.exception_type == "KeyError"
    assert ex.failure_kind == "runtime_error"
    assert "KeyError" in ex.traceback
    assert "missing_column" in ex.traceback
    assert "KeyError" in ex.observation()


def test_timeout_kills_the_process(workdir, df_path):
    ex = run_python("while True:\n    pass", workdir, df_path, timeout_s=2)
    assert not ex.ok
    assert ex.failure_kind == "timeout"
    assert ex.duration_s < 10
    assert "timed out" in ex.observation()


def test_guard_violation_never_reaches_execution(workdir, df_path, tmp_path):
    marker = tmp_path / "written.txt"
    ex = run_python(f"import os\nopen({str(marker)!r}, 'w').write('x')", workdir, df_path)
    assert not ex.ok
    assert ex.failure_kind == "guard_violation"
    assert not marker.exists()


def test_syntax_error_is_reported_as_syntax_error(workdir, df_path):
    ex = run_python("result = df[", workdir, df_path)
    assert not ex.ok and ex.failure_kind == "syntax_error"


def test_missing_result_is_flagged_as_empty_result(workdir, df_path):
    ex = run_python("total = df['units'].sum()", workdir, df_path)
    assert ex.ok
    assert ex.failure_kind == "empty_result"
    assert "`result` was never assigned" in ex.observation()


def test_chart_is_written_and_detected(workdir, df_path):
    code = ("import matplotlib.pyplot as plt\n"
            "df.plot(kind='bar', x='region', y='revenue')\n"
            "plt.savefig('chart.png')\nresult = 'done'")
    ex = run_python(code, workdir, df_path, expect_chart=True)
    assert ex.ok
    assert ex.chart_path and ex.chart_path.endswith(CHART_FILENAME)
    assert (workdir / CHART_FILENAME).stat().st_size > 0


def test_expecting_a_chart_that_was_not_saved_fails(workdir, df_path):
    ex = run_python("result = 1", workdir, df_path, expect_chart=True)
    assert not ex.ok
    assert ex.exception_type == "NoChartProduced"


def test_run_is_confined_to_its_own_workdir(workdir, df_path, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    ex = run_python("result = 1", workdir, df_path)
    assert ex.ok
    assert not list(other.iterdir())


def test_subprocess_env_does_not_carry_the_api_key(workdir, df_path, monkeypatch):
    """The child gets a hand-built environment, so credentials never reach model code."""
    from quarry import sandbox

    captured = {}
    real_popen = sandbox.subprocess.Popen

    def spy(argv, **kwargs):
        captured.update(kwargs.get("env") or {})
        return real_popen(argv, **kwargs)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-should-not-leak")
    monkeypatch.setattr(sandbox.subprocess, "Popen", spy)
    ex = sandbox.run_python("result = 'ran'", workdir, df_path)
    assert ex.ok
    assert captured, "the sandbox did not pass an explicit environment"
    assert "OPENROUTER_API_KEY" not in captured
    assert captured["MPLBACKEND"] == "Agg"
