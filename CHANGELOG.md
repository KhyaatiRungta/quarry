# Changelog

All notable changes to Quarry are documented here.

## [1.0.0] - 2026-09-24

### Added
- **ReAct agent loop** (`agent.py`) — think → act → observe, under 400 lines, no agent framework.
- **Sandbox** (`sandbox.py`) — AST import allowlisting, banned-function guard, dunder blocking,
  subprocess isolation (`python -I`, minimal env), timeout, output caps, dataset path validation.
- **Verify gate** (`verify.py`) — every number in a final answer must appear in real execution
  output (exact match, 1% numeric tolerance, normalized text match).
- **Offline benchmark harness** (`benchmark.py`) — replays recorded model turns with *real*
  sandbox execution; reproduces the headline numbers deterministically:
  - grounded-answer rate: **33.3% → 100%** with self-correction
  - failed executions recovered: **66.7%** (2/3)
- **Model failover chain** (`llm.py`) — rotates through a list of free OpenRouter models on
  rate limits / overload, including HTTP-200-with-embedded-error responses.
- **Typed config** (`config.py`) — validated env vars with actionable error messages.
- **Secret redaction** (`redact.py`) — API keys never land in traces, runs, or output.
- **Run persistence** — every `ask` saves `runs/<timestamp>/{trace.json, answer.txt, *.png}`.
- **Latency per step** recorded in every trace entry.
- **FastAPI web API** (`api.py`) + **static frontend** (`static/index.html`):
  upload CSV → ask → grounded answer, trace, and charts in the browser.
- **CLI** — `quarry ask` / `quarry bench`, `--json`, `--verbose`, exit codes
  (0 grounded, 1 failed, 2 config/usage error).
- **Tests** — 64 tests, 90% coverage: sandbox guard, verify gate, failover, CLI, config,
  security (path traversal, output caps, redaction), API.
- **CI** — GitHub Actions: ruff lint/format + pytest with `--cov-fail-under=80` + offline
  benchmark gate (no API key required).
- **Packaging** — `pyproject.toml`, `pip install -e .`, `quarry` console command.

### Security
- `.env` git-ignored; only `sk-or-` keys validated; all persisted traces redacted.
- Datasets restricted to existing `.csv` files (realpath-resolved, null-byte rejected).
- Generated code cannot import outside the allowlist, call `eval`/`exec`, or touch dunders.
