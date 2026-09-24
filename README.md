# ⛏ Quarry

[![CI](https://github.com/KhyaatiRungta/quarry/actions/workflows/ci.yml/badge.svg)](https://github.com/KhyaatiRungta/quarry/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-64%20passed-brightgreen)](#-testing)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)](#-testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**A sandboxed, self-correcting data-analyst agent.** Ask a question about a CSV — Quarry
writes pandas code, executes it in an isolated subprocess, feeds the real traceback back to
itself on failure via a hand-written ReAct loop (< 400 lines, no agent framework), and
refuses to answer unless every number is verified against actual execution output.

```
$ quarry ask --data data/sales.csv "Which region has the highest total revenue?"

============================================================
East
============================================================
Steps used: 3 | Repairs: 0 recovered / 0 failures
```

---

## Architecture

```
                 ┌──────────────────────────────────────────┐
  question ────▶ │  agent.py — ReAct loop (<400 lines)      │
                 │   THINK  → model picks a tool            │
                 │   ACT    → tool executes for real        │
                 │   OBSERVE→ result/traceback fed back     │
                 │   REPEAT → until verified answer         │
                 └─────┬───────────┬────────────┬───────────┘
                       │           │            │
                 ┌─────▼─────┐ ┌───▼──────┐ ┌───▼───────┐
                 │ llm.py    │ │sandbox.py│ │ verify.py │
                 │ failover  │ │ AST guard│ │ grounding │
                 │ chain of  │ │ subproc  │ │ gate: all │
                 │ free      │ │ timeout  │ │ numbers   │
                 │ models    │ │ out caps │ │ must be   │
                 │ + retries │ │ path     │ │ REAL      │
                 └───────────┘ │ checks   │ └───────────┘
                               └──────────┘
```

| Module | Responsibility |
|--------|----------------|
| `quarry/agent.py` | ReAct loop, repair budget, latency-tracked trace |
| `quarry/sandbox.py` | AST allowlist, `python -I` subprocess, timeout, output caps, dataset validation |
| `quarry/verify.py` | Mechanical grounding check (exact / 1% numeric / normalized) |
| `quarry/llm.py` | OpenRouter client with model failover + retry on 429/5xx/embedded errors |
| `quarry/benchmark.py` | Offline harness: recorded turns + real execution |
| `quarry/config.py` | Typed, validated environment configuration |
| `quarry/redact.py` | API-key redaction for traces/runs/output |
| `quarry/api.py` | FastAPI: `POST /api/ask` (upload CSV + question) |

## Benchmark (offline, deterministic)

The benchmark replays *recorded model turns* while executing the generated code **for real**
— no API key needed. It is asserted in CI:

| Metric | Result |
|--------|--------|
| Grounded answers — baseline (no self-correction) | **1/3 = 33.3%** |
| Grounded answers — full (self-correction) | **3/3 = 100%** |
| Failed executions recovered by self-correction | **2/3 = 66.7%** |

```
$ quarry bench
Grounded answers, baseline (no self-correction): 1/3 = 33.3%
Grounded answers, full (self-correction):        3/3 = 100.0%
Failed executions: 3 | recovered by self-correction: 2 = 66.7%
```

## Quickstart

```bash
git clone https://github.com/KhyaatiRungta/quarry.git
cd quarry
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -e ".[dev,api]"

# free key: https://openrouter.ai/keys
echo OPENROUTER_API_KEY=sk-or-... > .env

quarry ask --data data/sales.csv "Which region has the highest total revenue?"
```

### Model failover

Free models are rate-limited; Quarry rotates automatically. Configure in `.env`:

```
QUARRY_MODELS=openrouter/free,nvidia/nemotron-3-super-120b-a12b:free,nex-agi/nex-n2.5-pro:free
```

### Web UI + API

**Windows:** double-click `Start-UI.bat` — it starts the server and opens
http://127.0.0.1:8477 in your browser (if the server is already running it just opens it).

```bash
uvicorn quarry.api:app --port 8477
# open http://127.0.0.1:8477 — upload a CSV, ask, see trace + charts
```

```bash
curl -X POST http://127.0.0.1:8477/api/ask \
  -F "question=Which region has the highest total revenue?" \
  -F "file=@data/sales.csv"
```

## Security model

- **AST allowlist** — only `pandas`, `numpy`, `matplotlib`, `json`, `math`, `datetime`,
  `re`, `statistics` can be imported; `eval`, `exec`, `__import__`, dunder access blocked
  *structurally* (not by text matching).
- **Subprocess isolation** — `python -I`, temp working dir, minimal environment
  (no API keys reach generated code), 30s timeout, 20K-char output cap.
- **Dataset validation** — existing `.csv` only, realpath-resolved (kills `../` traversal),
  null bytes rejected.
- **Grounding gate** — a number that never appeared in real output cannot be an answer.
- **Secret hygiene** — `.env` ignored, all traces/runs pass through `redact()`.

## Testing

```bash
pytest --cov=quarry --cov-fail-under=80   # 64 tests, 90% coverage
ruff check quarry tests && ruff format --check quarry tests
```

CI runs lint + tests + the offline benchmark on every push (Python 3.10 & 3.12).

## CLI exit codes

| Code | Meaning |
|------|---------|
| 0 | Grounded answer produced |
| 1 | Agent failed to produce a verified answer |
| 2 | Config / usage error (bad key, missing file) |

## License

[MIT](LICENSE) © Khyaati Rungta
