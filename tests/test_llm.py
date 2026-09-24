"""Failover / retry tests for LLMClient - fully offline (fake client)."""

from types import SimpleNamespace

import httpx2
import pytest
from openai import APIStatusError, RateLimitError

from quarry.llm import LLMClient, LLMError

OK_MSG = SimpleNamespace(content="hello", tool_calls=None)
OK_RESPONSE = SimpleNamespace(choices=[SimpleNamespace(message=OK_MSG)], error=None, usage=None)
EMBEDDED_ERROR = SimpleNamespace(choices=None, error={"message": "Upstream overloaded"}, usage=None)


def _rate_limit_error():
    req = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    resp = httpx2.Response(429, request=req, text="{}")
    return RateLimitError("rate limited", response=resp, body=None)


def _not_found_error():
    req = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    resp = httpx2.Response(404, request=req, text="{}")
    return APIStatusError("model not found", response=resp, body=None)


class FakeClient:
    """Records which models were called and replays scripted outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs["model"])
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_fails_over_on_embedded_error():
    fake = FakeClient([EMBEDDED_ERROR, OK_RESPONSE])
    client = LLMClient(client=fake, models=("model-a", "model-b"))
    msg = client.chat([{"role": "user", "content": "hi"}])
    assert msg is OK_MSG
    assert fake.calls == ["model-a", "model-b"]


def test_retries_after_rate_limit():
    fake = FakeClient([_rate_limit_error(), OK_RESPONSE])
    client = LLMClient(client=fake, models=("model-a",))
    msg = client.chat([{"role": "user", "content": "hi"}])
    assert msg is OK_MSG
    assert fake.calls == ["model-a", "model-a"]


def test_all_models_fail_raises_llm_error():
    fake = FakeClient([EMBEDDED_ERROR])
    client = LLMClient(client=fake, models=("model-a", "model-b"))
    with pytest.raises(LLMError, match="All 2 models failed"):
        client.chat([{"role": "user", "content": "hi"}])
    # 2 attempts per model = 4 calls
    assert len(fake.calls) == 4


def test_non_retriable_error_fails_fast():
    fake = FakeClient([_not_found_error()])
    client = LLMClient(client=fake, models=("model-a", "model-b"))
    with pytest.raises(LLMError, match="API error 404"):
        client.chat([{"role": "user", "content": "hi"}])
    # No rotation on non-retriable errors
    assert fake.calls == ["model-a"]


def test_rotates_wrapping_around_model_list():
    # outcomes: a fails, b fails, a fails, b succeeds
    fake = FakeClient([EMBEDDED_ERROR, EMBEDDED_ERROR, EMBEDDED_ERROR, OK_RESPONSE])
    client = LLMClient(client=fake, models=("model-a", "model-b"))
    msg = client.chat([{"role": "user", "content": "hi"}])
    assert msg is OK_MSG
    assert fake.calls == ["model-a", "model-b", "model-a", "model-b"]
