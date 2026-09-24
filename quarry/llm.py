"""LLM client - talks to OpenRouter with automatic model failover."""

import logging
import time

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from quarry.config import load_settings

LOG = logging.getLogger("quarry.llm")

# Transient errors worth retrying (429 rate limit, 5xx overload, network)
RETRIABLE = (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)


class LLMError(Exception):
    """Raised when no model could produce a response after all retries."""


class LLMClient:
    def __init__(self, client=None, models=None, retry_delay=None):
        """client/models/retry_delay can be injected for testing (offline)."""
        if client is None:
            settings = load_settings(require_key=True)
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.api_key,
            )
            models = settings.models if models is None else models
            retry_delay = settings.retry_delay if retry_delay is None else retry_delay
        self.client = client
        self.models = tuple(models or ())
        # 2 attempts per model before rotating to the next
        self.max_attempts = len(self.models) * 2
        self.retry_delay = retry_delay or 0.0
        self.tokens_in = 0
        self.tokens_out = 0

    def chat(self, messages, tools=None):
        """Send messages to a model, get a response back.

        On rate limits / overload, fails over to the next model in the chain:
        attempt 0 -> models[0], attempt 1 -> models[1], ... wrapping around
        until attempts are exhausted.
        """
        last_error = "unknown error"

        for attempt in range(self.max_attempts):
            model = self.models[attempt % len(self.models)]
            kwargs = {"model": model, "messages": messages}
            if tools:
                kwargs["tools"] = tools

            try:
                response = self.client.chat.completions.create(**kwargs)
            except RETRIABLE as exc:
                last_error = f"{model}: {type(exc).__name__} ({str(exc)[:120]})"
                LOG.warning("transient error, retrying: %s", last_error)
                time.sleep(self.retry_delay)
                continue
            except APIStatusError as exc:
                # Non-retriable (401 bad key, 404 bad model) - fail fast
                raise LLMError(f"API error {exc.status_code} for model '{model}': {str(exc)[:300]}") from exc

            # OpenRouter can return HTTP 200 with an embedded error - check body
            if getattr(response, "error", None):
                err = response.error
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                last_error = f"{model}: {msg}"
                LOG.warning("embedded API error, failing over: %s", last_error)
                time.sleep(self.retry_delay)
                continue

            if response.choices:
                if response.usage:
                    self.tokens_in += response.usage.prompt_tokens
                    self.tokens_out += response.usage.completion_tokens
                LOG.debug("response from %s (tokens in=%s out=%s)", model, self.tokens_in, self.tokens_out)
                return response.choices[0].message

            last_error = f"{model}: empty response (no choices)"
            time.sleep(self.retry_delay)

        raise LLMError(
            f"All {len(self.models)} models failed after {self.max_attempts} attempts. "
            f"Last error: {last_error}. "
            f"Free models are shared/rate-limited - wait a minute and retry, "
            f"or edit QUARRY_MODELS in .env."
        )
