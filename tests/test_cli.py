"""CLI tests: exit codes, config validation, benchmark determinism."""

import json
import sys
from types import SimpleNamespace

import pytest

import quarry.__main__ as cli
from quarry.__main__ import main


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["quarry", *argv])
    return main()


def test_missing_data_file_exits_2(monkeypatch, capsys):
    code = _run(monkeypatch, "ask", "--data", "nope.csv", "question")
    assert code == 2
    assert "data file not found" in capsys.readouterr().err


def test_invalid_api_key_exits_2(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key")
    code = _run(monkeypatch, "ask", "--data", "data/sales.csv", "hi")
    assert code == 2
    assert "sk-or-" in capsys.readouterr().err


def _fake_agent_class(answer):
    class FakeAgent:
        def __init__(self, **kwargs):
            self.trace = [{"step": 1, "tool": "run_python"}]
            self.repair_count = 1
            self.repair_attempts = 2
            self.client = SimpleNamespace(tokens_in=100, tokens_out=20)

        def ask(self, question):
            return answer

    return FakeAgent


def test_ask_json_success_exits_0(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setattr(cli, "QuarryAgent", _fake_agent_class("East is highest."))
    trace = tmp_path / "trace.json"

    code = _run(
        monkeypatch,
        "ask",
        "--data",
        "data/sales.csv",
        "--json",
        "--trace-out",
        str(trace),
        "Which region?",
    )
    assert code == 0

    out = json.loads(capsys.readouterr().out)
    assert out["grounded"] is True
    assert out["answer"] == "East is highest."
    assert out["steps"] == 1
    assert out["repairs"] == {"recovered": 1, "attempts": 2}
    assert out["tokens"] == {"input": 100, "output": 20}
    # Trace file written with the full trace
    saved = json.loads(trace.read_text())
    assert saved["answer"] == "East is highest."
    assert saved["trace"] == [{"step": 1, "tool": "run_python"}]


def test_ask_ungrounded_exits_1(monkeypatch, tmp_path, capsys):
    from quarry.agent import MAX_STEPS_MSG

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setattr(cli, "QuarryAgent", _fake_agent_class(MAX_STEPS_MSG))

    code = _run(
        monkeypatch,
        "ask",
        "--data",
        "data/sales.csv",
        "--trace-out",
        str(tmp_path / "t.json"),
        "q?",
    )
    assert code == 1


def test_benchmark_produces_resume_numbers(monkeypatch, tmp_path):
    """The benchmark must deterministically reproduce the resume stats."""
    out = tmp_path / "results.json"
    code = _run(
        monkeypatch,
        "bench",
        "--tasks",
        "benchmarks/tasks.json",
        "--json-out",
        str(out),
    )
    assert code == 0  # all tasks grounded in full mode

    stats = json.loads(out.read_text())
    assert stats["grounded_baseline_rate"] == 33.3  # 1/3 without self-correction
    assert stats["grounded_full_rate"] == 100.0  # 3/3 with self-correction
    assert stats["failed_executions"] == 3
    assert stats["recovered_executions"] == 2
    assert stats["recovery_rate"] == 66.7


def test_benchmark_missing_file_exits_2(monkeypatch, capsys):
    code = _run(monkeypatch, "bench", "--tasks", "missing.json")
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_no_subcommand_shows_usage(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        monkeypatch.setattr(sys, "argv", ["quarry"])
        main()
    assert exc.value.code == 2  # argparse usage error
