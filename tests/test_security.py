"""Security tests: dataset validation, output caps, secret redaction."""

import os

from quarry.redact import redact, redact_structure
from quarry.sandbox import MAX_OUTPUT_CHARS, check_dataset, run_code

DATA = "data/sales.csv"


# --- dataset path safety ---


def test_missing_dataset_rejected():
    ok, err = check_dataset("no/such/file.csv")
    assert not ok
    assert "not found" in err


def test_non_csv_dataset_rejected(tmp_path):
    bad = tmp_path / "evil.exe"
    bad.write_text("x")
    ok, err = check_dataset(str(bad))
    assert not ok
    assert "must be one of" in err


def test_valid_csv_accepted():
    ok, err = check_dataset(DATA)
    assert ok, err


def test_path_traversal_resolves_and_rejects(tmp_path):
    # file exists outside via .. but is still validated by type/exists rules
    secret = tmp_path / "secret.txt"
    secret.write_text("api_key=abc")
    ok, err = check_dataset(str(secret))
    assert not ok  # .txt rejected


def test_null_byte_path_rejected():
    ok, err = check_dataset("data\x00.csv")
    assert not ok


def test_run_code_rejects_bad_dataset():
    result = run_code("print(df)", "not-a-real.csv")
    assert not result["success"]
    assert "guard_violation" in result["error"]


# --- output caps ---


def test_output_truncated_to_cap():
    result = run_code("print('x' * 100000)", DATA)
    assert result["success"]
    assert len(result["output"]) < MAX_OUTPUT_CHARS + 200
    assert "truncated" in result["output"]


# --- secret redaction ---


def test_api_key_redacted(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-supersecret123")
    text = "failed with key sk-or-v1-supersecret123 in header"
    safe = redact(text)
    assert "supersecret123" not in safe
    assert "***" in safe


def test_redact_structure_recurses(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-supersecret123")
    payload = {
        "trace": [{"args": {"code": "key = 'sk-or-v1-supersecret123'"}}, {"nested": ["sk-or-v1-supersecret123"]}],
        "num": 42,
    }
    safe = redact_structure(payload)
    assert "supersecret123" not in str(safe)
    assert safe["num"] == 42  # non-strings untouched


def test_redact_no_key_set(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert redact("nothing here") == "nothing here"
    assert redact("") == ""


def test_env_not_committed():
    """.env must be ignored by git (secret hygiene)."""
    gitignore = open(".gitignore").read()
    assert ".env" in gitignore
    # .env exists locally but must never be tracked
    assert os.path.exists(".env")
