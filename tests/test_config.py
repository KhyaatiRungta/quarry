"""Config validation tests - isolated from the real .env file."""

import pytest

import quarry.config as config_mod
from quarry.config import DEFAULT_MODELS, ConfigError, load_settings

ENV_VARS = (
    "OPENROUTER_API_KEY",
    "QUARRY_MODELS",
    "QUARRY_MODEL",
    "QUARRY_MAX_STEPS",
    "QUARRY_MAX_REPAIRS",
)


@pytest.fixture(autouse=True)
def bare_env(monkeypatch):
    """Ignore the real .env; start from an empty environment."""
    monkeypatch.setattr(config_mod, "load_dotenv", lambda: None)
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_missing_key_raises():
    with pytest.raises(ConfigError, match="missing or still the placeholder"):
        load_settings(require_key=True)


def test_placeholder_key_raises(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "your-free-key-here")
    with pytest.raises(ConfigError, match="placeholder"):
        load_settings(require_key=True)


def test_bad_key_prefix_raises(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-valid")
    with pytest.raises(ConfigError, match="sk-or-"):
        load_settings(require_key=True)


def test_offline_mode_skips_key_validation():
    settings = load_settings(require_key=False)
    assert settings.api_key == ""


def test_invalid_max_steps_raises(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setenv("QUARRY_MAX_STEPS", "abc")
    with pytest.raises(ConfigError, match="QUARRY_MAX_STEPS must be an integer"):
        load_settings(require_key=True)


def test_default_models_used_when_unset():
    settings = load_settings()
    assert settings.models == DEFAULT_MODELS


def test_single_model_prepends_defaults(monkeypatch):
    monkeypatch.setenv("QUARRY_MODEL", "my/model:free")
    settings = load_settings()
    assert settings.models[0] == "my/model:free"
    assert settings.models[1:] == DEFAULT_MODELS


def test_models_list_parsed_from_csv(monkeypatch):
    monkeypatch.setenv("QUARRY_MODELS", " model-a , model-b ,, ")
    settings = load_settings()
    assert settings.models == ("model-a", "model-b")


def test_defaults_for_steps_and_repairs():
    settings = load_settings()
    assert settings.max_steps == 10
    assert settings.max_repairs == 3
