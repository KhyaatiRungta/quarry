"""Typed configuration with validation and clear error messages."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Fallback chain if .env does not define QUARRY_MODELS / QUARRY_MODEL
DEFAULT_MODELS = (
    "openrouter/free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nex-agi/nex-n2.5-pro:free",
    "cohere/north-mini-code:free",
)

PLACEHOLDER_KEY = "your-free-key-here"
KEYS_URL = "https://openrouter.ai/keys"


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    models: tuple[str, ...]
    max_steps: int
    max_repairs: int
    retry_delay: float


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def load_settings(require_key: bool = False) -> Settings:
    """Load settings from .env / environment.

    require_key=True validates the API key (use for live API calls);
    offline modes (bench, tests) skip validation.
    """
    load_dotenv()

    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if require_key:
        if not key or key == PLACEHOLDER_KEY:
            raise ConfigError(
                f"OPENROUTER_API_KEY is missing or still the placeholder. "
                f"Get a free key at {KEYS_URL} and put it in .env"
            )
        if not key.startswith("sk-or-"):
            raise ConfigError(
                f"OPENROUTER_API_KEY looks invalid (must start with 'sk-or-'). Copy it again from {KEYS_URL}"
            )

    models = tuple(m.strip() for m in os.getenv("QUARRY_MODELS", "").split(",") if m.strip())
    if not models:
        single = os.getenv("QUARRY_MODEL", "").strip()
        if single:
            models = (single,) + tuple(m for m in DEFAULT_MODELS if m != single)
        else:
            models = DEFAULT_MODELS

    return Settings(
        api_key=key,
        models=models,
        max_steps=_int_env("QUARRY_MAX_STEPS", 10),
        max_repairs=_int_env("QUARRY_MAX_REPAIRS", 3),
        retry_delay=float(os.getenv("QUARRY_RETRY_DELAY", "3") or 3),
    )
