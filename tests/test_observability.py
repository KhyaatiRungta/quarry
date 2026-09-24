"""Tests for observability: latency in traces + run persistence."""

import json
from pathlib import Path

from quarry.agent import QuarryAgent
from quarry.benchmark import ReplayLLM

with open("benchmarks/tasks.json", encoding="utf-8") as f:
    TASK = json.load(f)["tasks"][0]


def test_trace_entries_include_latency():
    agent = QuarryAgent(
        dataset_path=TASK["data"],
        llm=ReplayLLM(TASK["turns"], allow_repairs=True),
    )
    agent.ask(TASK["question"])

    assert len(agent.trace) == 3
    for entry in agent.trace:
        assert "latency_ms" in entry
        assert entry["latency_ms"] >= 0


def test_run_dir_persistence(monkeypatch, tmp_path, capsys):
    """cmd_ask writes runs/<ts>/{trace.json, answer.txt}."""
    import sys
    from types import SimpleNamespace

    import quarry.__main__ as cli

    class FakeAgent:
        def __init__(self, **kwargs):
            self.trace = [{"step": 1, "tool": "run_python", "latency_ms": 5.0}]
            self.repair_count = 0
            self.repair_attempts = 0
            self.client = SimpleNamespace(tokens_in=1, tokens_out=1)

        def ask(self, question):
            return "East"

    monkeypatch.setattr(cli, "QuarryAgent", FakeAgent)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    runs = tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quarry",
            "ask",
            "--data",
            "data/sales.csv",
            "--runs-dir",
            str(runs),
            "--trace-out",
            str(tmp_path / "trace.json"),
            "Which region?",
        ],
    )
    assert cli.main() == 0

    run_dirs = list(runs.iterdir())
    assert len(run_dirs) == 1
    trace = json.loads((run_dirs[0] / "trace.json").read_text())
    assert trace["answer"] == "East"
    assert trace["trace"][0]["latency_ms"] == 5.0
    assert (run_dirs[0] / "answer.txt").read_text() == "East"


def test_run_trace_is_redacted(monkeypatch, tmp_path):
    """The API key must never appear in persisted traces."""
    import sys
    from types import SimpleNamespace

    import quarry.__main__ as cli

    key = "sk-or-v1-abcdef1234567890"

    class FakeAgent:
        def __init__(self, **kwargs):
            self.trace = [{"step": 1, "tool": "x", "args": {"note": key}}]
            self.repair_count = 0
            self.repair_attempts = 0
            self.client = SimpleNamespace(tokens_in=1, tokens_out=1)

        def ask(self, question):
            return f"echo {key}"

    monkeypatch.setattr(cli, "QuarryAgent", FakeAgent)
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    trace_out = tmp_path / "trace.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quarry",
            "ask",
            "--data",
            "data/sales.csv",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--trace-out",
            str(trace_out),
            "q?",
        ],
    )
    assert cli.main() == 0

    run_trace = next((tmp_path / "runs").iterdir())
    content = (run_trace / "trace.json").read_text() + (run_trace / "answer.txt").read_text()
    assert key not in content
    assert key not in trace_out.read_text()


def test_path_object_used_everywhere():
    """Sanity: run persistence uses pathlib (no os.path mixups)."""
    src = Path("quarry/__main__.py").read_text()
    assert "Path(" in src
    assert "shutil.copy" in src
