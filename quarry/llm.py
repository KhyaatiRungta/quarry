"""OpenRouter-backed LLM client.

Deliberately a single file with no framework. The agent loop in `agent.py` only
depends on `LLMClient.complete()` returning a normalised `LLMResponse`, so the
offline replay client below is a drop-in substitute for tests and demos.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
FAST_MODEL = "anthropic/claude-3.5-haiku"
JUDGE_MODEL = "anthropic/claude-3.5-sonnet"

# Fallback per-million-token prices used only when OpenRouter does not return
# usage accounting. Estimated costs are flagged as such in the trace.
_FALLBACK_PRICES = {
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "anthropic/claude-3.5-haiku": (0.80, 4.00),
}
_DEFAULT_PRICE = (3.00, 15.00)

MAX_ATTEMPTS = 4


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cost_usd + other.cost_usd,
            self.latency_s + other.latency_s,
        )

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_s": round(self.latency_s, 3),
        }


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    raw: dict = field(default_factory=dict)


def load_env_file(path: str | Path = ".env") -> None:
    """Tiny .env loader. Never overwrites an already-set environment variable."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_api_key() -> str | None:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"):
        load_env_file(candidate)
        if os.environ.get("OPENROUTER_API_KEY"):
            return os.environ["OPENROUTER_API_KEY"]
    return None


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = _FALLBACK_PRICES.get(model, _DEFAULT_PRICE)
    return (prompt_tokens / 1e6) * pin + (completion_tokens / 1e6) * pout


def normalise_tool_calls(message: Any) -> list[dict]:
    """Normalise SDK tool calls to plain dicts: {id, name, arguments(dict), raw_arguments}."""
    calls = []
    for tc in getattr(message, "tool_calls", None) or []:
        fn = getattr(tc, "function", None)
        raw_args = getattr(fn, "arguments", "") or ""
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {}
        calls.append({
            "id": getattr(tc, "id", "") or f"call_{len(calls)}",
            "name": getattr(fn, "name", "") or "",
            "arguments": args,
            "raw_arguments": raw_args,
            "malformed_json": bool(raw_args) and not isinstance(args, dict),
        })
    return calls


class LLMClient:
    """Thin OpenRouter wrapper. The API key is only required at call time."""

    def __init__(self, model: str | None = None, temperature: float = 0.0,
                 log=None, app_title: str = "Quarry"):
        self.model = model or os.environ.get("QUARRY_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self.total_usage = Usage()
        self.call_count = 0
        self._client = None
        self._log = log or (lambda msg: None)
        self._app_title = app_title

    @property
    def offline(self) -> bool:
        return False

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        key = resolve_api_key()
        if not key:
            raise LLMError(
                "No OPENROUTER_API_KEY found. Export it, or put it in a .env file at the "
                "repo root (see .env.example). To run without a key use --offline."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
            raise LLMError("The `openai` package is required: pip install -r requirements.txt") from exc
        self._client = OpenAI(
            base_url=BASE_URL,
            api_key=key,
            # Optional OpenRouter ranking headers.
            default_headers={"X-Title": self._app_title},
        )
        return self._client

    def complete(self, messages: list[dict], tools: list[dict] | None = None,
                 max_tokens: int = 2048, response_format: dict | None = None) -> LLMResponse:
        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "extra_body": {"usage": {"include": True}},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format

        started = time.time()
        last_exc: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                completion = client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - provider errors are not a stable hierarchy
                last_exc = exc
                if not _is_retryable(exc) or attempt == MAX_ATTEMPTS:
                    raise LLMError(f"OpenRouter call failed after {attempt} attempt(s): {exc}") from exc
                delay = min(2 ** attempt, 16) * (0.6 + random.random() * 0.8)
                self._log(f"llm retry {attempt}/{MAX_ATTEMPTS - 1} in {delay:.1f}s: {exc}")
                time.sleep(delay)
        else:  # pragma: no cover - loop always breaks or raises
            raise LLMError(str(last_exc))

        latency = time.time() - started
        raw = completion.model_dump() if hasattr(completion, "model_dump") else dict(completion)
        choice = completion.choices[0]
        message = choice.message
        usage_obj = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        cost = getattr(usage_obj, "cost", None)
        if cost is None:
            cost = (raw.get("usage") or {}).get("cost")
        if cost is None:
            cost = estimate_cost(self.model, prompt_tokens, completion_tokens)
        usage = Usage(prompt_tokens, completion_tokens, float(cost), latency)
        self.total_usage = self.total_usage + usage
        self.call_count += 1
        return LLMResponse(
            text=(message.content or ""),
            tool_calls=normalise_tool_calls(message),
            usage=usage,
            model=getattr(completion, "model", self.model),
            raw=raw,
        )


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if isinstance(status, int):
        return status == 429 or 500 <= status < 600
    text = str(exc).lower()
    return any(t in text for t in ("429", "rate limit", "timeout", "temporarily", "502", "503", "504"))


class OfflineReplayExhausted(LLMError):
    pass


class OfflineLLMClient:
    """Replays canned model turns from a fixture file. Same interface as LLMClient.

    The fixtures only stand in for the model. Tool execution, the sandbox, timings
    and the trace are all real, which is what makes an offline run worth showing.
    """

    def __init__(self, responses: list[dict], model: str = "offline/replay",
                 scenario: str = "inline", temperature: float = 0.0):
        self.responses = list(responses)
        self.model = model
        self.scenario = scenario
        self.temperature = temperature
        self.total_usage = Usage()
        self.call_count = 0
        self._cursor = 0

    @property
    def offline(self) -> bool:
        return True

    @classmethod
    def from_file(cls, path: str | Path) -> "OfflineLLMClient":
        data = json.loads(Path(path).read_text())
        return cls(data["responses"], model=data.get("model", "offline/replay"),
                   scenario=data.get("scenario", Path(path).stem))

    @classmethod
    def from_dir(cls, directory: str | Path, question: str | None = None,
                 scenario: str | None = None) -> "OfflineLLMClient":
        directory = Path(directory)
        files = sorted(directory.glob("*.json"))
        if not files:
            raise LLMError(f"No offline fixtures found in {directory}")
        if scenario:
            path = directory / f"{scenario}.json"
            if not path.exists():
                raise LLMError(f"Unknown offline scenario {scenario!r} in {directory}")
            return cls.from_file(path)
        if question:
            q = question.lower().strip()
            for path in files:
                data = json.loads(path.read_text())
                for needle in data.get("match_questions", []):
                    if needle.lower().strip() == q:
                        return cls.from_file(path)
            for path in files:
                data = json.loads(path.read_text())
                for needle in data.get("match_questions", []):
                    if needle.lower().strip() in q or q in needle.lower().strip():
                        return cls.from_file(path)
        raise LLMError(
            f"No offline fixture matches that question. Available scenarios: "
            f"{', '.join(p.stem for p in files)}"
        )

    def complete(self, messages: list[dict], tools: list[dict] | None = None,
                 max_tokens: int = 2048, response_format: dict | None = None) -> LLMResponse:
        if self._cursor >= len(self.responses):
            raise OfflineReplayExhausted(
                f"Offline fixture {self.scenario!r} ran out of responses after "
                f"{self._cursor} turns. The agent asked for more than the fixture scripts."
            )
        canned = self.responses[self._cursor]
        self._cursor += 1
        usage = Usage(
            int(canned.get("prompt_tokens", 900)),
            int(canned.get("completion_tokens", 130)),
            float(canned.get("cost_usd", 0.0)),
            float(canned.get("latency_s", 0.0)),
        )
        if not usage.cost_usd:
            usage.cost_usd = estimate_cost(DEFAULT_MODEL, usage.prompt_tokens, usage.completion_tokens)
        self.total_usage = self.total_usage + usage
        self.call_count += 1
        calls = []
        for i, tc in enumerate(canned.get("tool_calls", [])):
            args = tc.get("arguments", {})
            calls.append({
                "id": tc.get("id", f"call_{self._cursor}_{i}"),
                "name": tc["name"],
                "arguments": args,
                "raw_arguments": tc.get("raw_arguments", json.dumps(args)),
                "malformed_json": tc.get("malformed_json", False),
            })
        return LLMResponse(
            text=canned.get("text", ""),
            tool_calls=calls,
            usage=usage,
            model=self.model,
            raw={"offline_scenario": self.scenario, "turn": self._cursor},
        )
