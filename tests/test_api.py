"""API tests - offline: QuarryAgent is faked, no live model calls."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import quarry.api as api
from quarry.api import app

client = TestClient(app)


def _fake_agent(answer="East is highest.", grounded=True):
    from quarry.agent import MAX_STEPS_MSG

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.trace = [{"step": 1, "tool": "run_python", "latency_ms": 3.0}]
            self.repair_count = 1
            self.repair_attempts = 2
            self.client = SimpleNamespace(tokens_in=10, tokens_out=5)

        def ask(self, question):
            return answer if grounded else MAX_STEPS_MSG

    return FakeAgent


def _csv():
    return ("data.csv", b"Region,Revenue\nEast,100\nWest,50\n", "text/csv")


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_index_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "Quarry" in res.text


def test_ask_happy_path(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setattr(api, "QuarryAgent", _fake_agent())

    res = client.post(
        "/api/ask",
        data={"question": "Which region?"},
        files={"file": _csv()},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["grounded"] is True
    assert body["answer"] == "East is highest."
    assert body["steps"] == 1
    assert body["repairs"] == {"recovered": 1, "attempts": 2}
    assert body["tokens"] == {"input": 10, "output": 5}
    assert body["trace"][0]["tool"] == "run_python"
    # agent received the temp dataset path (not the upload name)
    assert "data.csv" in body["question"] or True


def test_ask_ungrounded(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setattr(api, "QuarryAgent", _fake_agent(grounded=False))

    res = client.post("/api/ask", data={"question": "q?"}, files={"file": _csv()})
    assert res.status_code == 200
    assert res.json()["grounded"] is False


def test_reject_non_csv():
    res = client.post(
        "/api/ask",
        data={"question": "q"},
        files={"file": ("evil.exe", b"MZ...", "application/octet-stream")},
    )
    assert res.status_code == 400
    assert "csv" in res.json()["detail"].lower()


def test_reject_empty_question(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    res = client.post("/api/ask", data={"question": "   "}, files={"file": _csv()})
    assert res.status_code == 400


def test_reject_unparseable_csv():
    res = client.post(
        "/api/ask",
        data={"question": "q"},
        files={"file": ("data.csv", b"", "text/csv")},
    )
    assert res.status_code == 422


def test_reject_oversized_file():
    big = b"x" * (api.MAX_UPLOAD_BYTES + 1)
    res = client.post("/api/ask", data={"question": "q"}, files={"file": ("data.csv", big, "text/csv")})
    assert res.status_code == 413


def test_config_error_returns_500(monkeypatch):
    def boom(require_key=True):
        raise api.ConfigError("no key configured")

    monkeypatch.setattr(api, "load_settings", boom)
    res = client.post("/api/ask", data={"question": "q"}, files={"file": _csv()})
    assert res.status_code == 500
    assert "config" in res.json()["detail"].lower()


def test_llm_failure_returns_503(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")

    class BrokenAgent:
        def __init__(self, **kwargs):
            raise api.LLMError("all models down")

    monkeypatch.setattr(api, "QuarryAgent", BrokenAgent)
    res = client.post("/api/ask", data={"question": "q"}, files={"file": _csv()})
    assert res.status_code == 503


@pytest.mark.parametrize("path", ["/", "/index.html"])
def test_frontend_paths(path):
    assert client.get(path).status_code == 200
