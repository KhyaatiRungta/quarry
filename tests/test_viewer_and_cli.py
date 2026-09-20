import json

import pytest

from quarry.cli import main
from quarry.viewer import render_trace, render_trace_file


def test_shipped_example_trace_renders(repo_root, tmp_path):
    out = render_trace_file(repo_root / "runs/example/trace.json", tmp_path / "trace.html")
    html = open(out).read()
    assert html.startswith("<!doctype html>")
    assert "Quarry trace" in html
    assert "repair" in html                     # the repaired step is labelled
    assert "total_revenue" in html              # the code that failed is shown
    assert "KeyError" in html                   # the traceback it was given
    assert "data:image/png;base64," in html     # the chart is inlined
    assert "<script" not in html                # standalone, no javascript


def test_renderer_escapes_html_in_model_output():
    trace = {
        "run_id": "x", "question": "<script>alert(1)</script>", "created_at": "now",
        "dataset": {"name": "d", "rows": 1, "columns": ["a"]},
        "config": {"model": "m"}, "summary": {"usage": {}}, "steps": [],
    }
    html = render_trace(trace)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_ask_offline_runs_end_to_end(tmp_path, capsys, monkeypatch, repo_root):
    monkeypatch.chdir(repo_root)
    code = main([
        "ask", "--offline", "--data", "data/sales.csv",
        "--runs-dir", str(tmp_path / "runs"),
        "--render-trace", str(tmp_path / "trace.html"),
        "Which region has the highest total revenue, and by how much does it beat the runner-up?",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "repair run_python -> ok" in out
    assert "grounded=yes" in out
    assert (tmp_path / "trace.html").exists()


def test_ask_without_a_key_and_without_offline_fails_with_a_clear_message(
        tmp_path, capsys, monkeypatch, repo_root):
    monkeypatch.chdir(tmp_path)          # away from any .env in the repo
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    code = main(["ask", "--data", str(repo_root / "data/sales.csv"),
                 "--runs-dir", str(tmp_path / "runs"), "anything?"])
    assert code == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_view_renders_a_trace(tmp_path, capsys, repo_root):
    code = main(["view", str(repo_root / "runs/example/trace.json"),
                 "--out", str(tmp_path / "t.html")])
    assert code == 0
    assert (tmp_path / "t.html").exists()


def test_bench_offline_writes_results(tmp_path, repo_root, monkeypatch):
    monkeypatch.chdir(repo_root)
    code = main(["bench", "--offline", "--limit", "3", "--arms", "on",
                 "--out", str(tmp_path / "results"),
                 "--runs-dir", str(tmp_path / "runs")])
    assert code == 0
    payload = json.loads((tmp_path / "results/bench_results.json").read_text())
    assert payload["mode"] == "offline"
    assert payload["tasks_total"] >= 30
    assert payload["tasks_run"] == 3
    assert "on" in payload["arms"]
    assert (tmp_path / "results/RESULTS.md").exists()


def test_unknown_dataset_is_a_clean_error(tmp_path, capsys):
    code = main(["ask", "--offline", "--data", str(tmp_path / "nope.csv"), "q?"])
    assert code == 2
    assert "No such dataset" in capsys.readouterr().err
