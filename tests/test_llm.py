import json

import pytest

from quarry.llm import (DEFAULT_MODEL, LLMClient, LLMError, OfflineLLMClient,
                        OfflineReplayExhausted, Usage, estimate_cost, load_env_file,
                        resolve_api_key)


def test_usage_adds_up():
    total = Usage(10, 2, 0.001, 0.5) + Usage(5, 3, 0.002, 0.25)
    assert total.prompt_tokens == 15
    assert total.completion_tokens == 5
    assert total.cost_usd == pytest.approx(0.003)
    assert total.latency_s == pytest.approx(0.75)


def test_cost_estimate_uses_the_model_price():
    assert estimate_cost(DEFAULT_MODEL, 1_000_000, 0) == pytest.approx(3.0)
    assert estimate_cost(DEFAULT_MODEL, 0, 1_000_000) == pytest.approx(15.0)


def test_no_key_is_needed_until_a_call_is_made(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = LLMClient()                      # constructing is fine
    with pytest.raises(LLMError) as exc:
        client.complete([{"role": "user", "content": "hi"}])
    assert "OPENROUTER_API_KEY" in str(exc.value)
    assert "--offline" in str(exc.value)


def test_env_file_loader_does_not_override_the_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('OPENROUTER_API_KEY="from-file"\n# comment\nQUARRY_MODEL=some/model\n')
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("QUARRY_MODEL", "already/set")
    load_env_file(env)
    import os
    assert os.environ["OPENROUTER_API_KEY"] == "from-file"
    assert os.environ["QUARRY_MODEL"] == "already/set"


def test_model_id_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("QUARRY_MODEL", "vendor/model-x")
    assert LLMClient().model == "vendor/model-x"


def test_offline_client_replays_in_order():
    client = OfflineLLMClient([
        {"text": "one", "tool_calls": [{"name": "run_python", "arguments": {"code": "result = 1"}}]},
        {"text": "two"},
    ])
    first = client.complete([])
    assert first.tool_calls[0]["name"] == "run_python"
    assert json.loads(first.tool_calls[0]["raw_arguments"]) == {"code": "result = 1"}
    assert client.complete([]).text == "two"
    with pytest.raises(OfflineReplayExhausted):
        client.complete([])


def test_offline_client_tracks_usage():
    client = OfflineLLMClient([{"text": "a"}, {"text": "b"}])
    client.complete([])
    client.complete([])
    assert client.call_count == 2
    assert client.total_usage.cost_usd > 0


def test_offline_client_matches_a_fixture_by_question(repo_root):
    client = OfflineLLMClient.from_dir(
        repo_root / "fixtures/offline",
        question="What is the mean temperature across all readings?")
    assert client.scenario == "sensors_messy"


def test_offline_client_reports_an_unmatched_question(repo_root):
    with pytest.raises(LLMError) as exc:
        OfflineLLMClient.from_dir(repo_root / "fixtures/offline",
                                  question="what is the airspeed velocity of a swallow?")
    assert "Available scenarios" in str(exc.value)


def test_both_clients_expose_the_same_surface():
    for attribute in ("complete", "model", "total_usage", "offline"):
        assert hasattr(LLMClient(), attribute)
        assert hasattr(OfflineLLMClient([]), attribute)
