# Quarry

A data analyst agent that writes code, **executes it in a sandbox**, reads its own errors,
and corrects itself. Point it at a CSV or a SQLite table, ask in English, get an answer,
a chart and a replayable trace with tokens, cost and latency per step.

No agent framework. The ReAct loop, the tool schemas, the sandbox and the trace format are
all hand-written, because the loop is the part worth reading.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# no API key: recorded model turns, real sandboxed execution
python -m quarry ask --offline --data data/sales.csv \
  "Which region has the highest total revenue, and by how much does it beat the runner-up?"

pytest                                   # passes with no key
python -m quarry view runs/example/trace.json --out /tmp/trace.html
open site/index.html                     # the product page, opens from file://
```

Live:

```bash
export OPENROUTER_API_KEY=sk-or-...      # or put it in .env (see .env.example)
export QUARRY_MODEL=anthropic/claude-3.5-sonnet   # optional
python -m quarry ask --data data/sensors.csv "What is the mean temperature?"
python -m quarry bench                   # both arms, writes results/
```

## What a run looks like

```
$ python -m quarry ask --offline --data data/sales.csv "Which region has the highest total revenue, ..."
step 1: run_python -> FAILED runtime_error        # KeyError: 'Column not found: total_revenue'
step 2: repair inspect_schema -> ok               # stops guessing, looks at the real dtypes
step 3: repair run_python -> ok                   # correct column, plus the runner-up gap
step 4: make_chart -> ok
step 5: final_answer -> ok
------------------------------------------------------------------------
North is the highest-revenue region ... with 588920.0 in total revenue. It beats the
runner-up, East, by 128565.0, which is about 27.9 percent more than East's total.
------------------------------------------------------------------------
steps=5  tool_calls=5  repairs=1/2  grounded=yes
tokens=10487+599  cost=$0.0404  wall=1.8s
trace: runs/20260920T185119-b36898/trace.json
```

That run is committed at `runs/example/trace.json` and rendered at `site/trace.html`.

## Architecture

```
question
   |
   v
inspect_schema ---> system prompt carries real dtypes, null counts, cardinality, samples
   |
   v
model --(tool call)--> AST guard --(passes)--> subprocess sandbox --> observation --+
   |                       |                        |                              |
   |                  rejected: which rule      failed: real traceback              |
   +<----------------------+------------------------+------------------------------+
   |                                REPAIR
   v
final_answer(answer, supporting_values) --> verify: every value must appear in real
                                            execution output, else send it back once
   |
   v
answer + chart.png + trace.json
```

| file | what it is |
| --- | --- |
| `quarry/sandbox.py` | AST guard + subprocess execution. The security-relevant file. |
| `quarry/tools.py` | Five hand-written tool schemas with argument validation. |
| `quarry/agent.py` | The loop: dispatch, repair accounting, verify gate, trace writer. |
| `quarry/verify.py` | Groundedness check of claimed values against real output. |
| `quarry/llm.py` | OpenRouter client with retries, usage accounting, offline replay. |
| `quarry/viewer.py` | `trace.json` to a standalone HTML page. |
| `quarry/bench/` | 43 tasks, three graders, the ON/OFF ablation runner. |
| `quarry/prompts/` | Prompts as files, not string literals in code. |

## The sandbox, honestly

**It is defence in depth against a confused model, not a security boundary against an
adversarial one.** Code that passes the guard runs as your user with your filesystem and your
network. If you do not trust the model output, run this in a container.

What it does do:

- **Subprocess, never `exec` in-process.** `Popen` in its own session, its own working
  directory under `runs/<id>/workdir/`, a hand-built environment that does not contain
  `OPENROUTER_API_KEY`, `-I` isolated mode, killed by process group on timeout (default 30s).
- **AST allowlist, not regex.** Imports outside `{pandas, numpy, matplotlib, json, math,
  datetime, re, statistics}` are rejected at the node level, so `import os`, `import os.path
  as p` and `from os import system` are one rule and the string `"import os"` in a comment is
  not a violation. Also rejected: `eval`, `exec`, `compile`, `__import__`, `getattr`,
  `globals`, `locals`, any dunder attribute access, and `open` unless it is a literal filename
  inside the run directory in a read mode.
- **Rejections are specific.** The model is told which rule it broke and what the allowlist is,
  so a guard violation is a repairable failure rather than a dead end.
- **`rlimit`s where they work.** `RLIMIT_AS` is applied on Linux only; on macOS it is not
  honoured for the mappings numpy and pandas make, and the code says so instead of pretending.
  `RLIMIT_FSIZE` (64 MB) and `RLIMIT_NPROC` are applied where available.
- **`run_sql` is read-only.** SELECT/WITH only, one statement, no stacked statements, 200 row cap.

Known holes: `df.query()` and `pd.eval()` evaluate expression strings inside pandas after the
guard has finished; C extensions are reachable through allowed modules; resource exhaustion
below the caps is possible; network is not actually blocked, the import allowlist just makes it
awkward.

## Self-correction

Failures are classified, not lumped together: `syntax_error`, `guard_violation`,
`runtime_error`, `timeout`, `empty_result`, `bad_arguments`. The real traceback goes back to
the model truncated from the front, because the frames nearest the error are the useful ones.

A repair is any tool call issued while an unresolved failure is outstanding, including the
`inspect_schema` lookup a model usually does before rewriting the code. The failure stays
outstanding until an execution tool actually succeeds, so `repair_successes / repair_attempts`
means what it says.

The ablation is one flag:

```python
QuarryAgent(client, dataset, self_correction=False)   # first failed execution ends the run
```

Everything else is identical between arms, which is what makes the comparison worth printing.

## The verify gate

`final_answer(answer, supporting_values)` requires the values the answer rests on. Each is
matched against the text the sandbox actually produced during the run:

- verbatim match after normalisation, or
- a numeric match within a small relative tolerance, or
- a correct rounding at the precision the model wrote (`0.4471` for a computed `0.44705882`), or
- a percent/proportion conversion.

Anything else is unsupported: the model is sent back once to compute it, and if it insists the
answer is returned with `stop_reason="final_answer_ungrounded"` and the offending values listed.
This is a groundedness check, not a correctness check. A number computed by wrong code passes.

## Benchmark

43 tasks over three bundled datasets, graded three ways and labelled with which:

| grader | count | how |
| --- | ---: | --- |
| `numeric` | 33 | expected value computed by the reference implementation in `scripts/build_tasks.py`, matched within tolerance |
| `predicate` | 7 | the answer must contain given strings (a region, a month, a date) |
| `judge` | 3 | an LLM grades an open-ended answer against a written rubric |

Difficulty spread: 15 easy, 17 medium, 11 hard. Tags cover aggregation, groupby, filters,
time-series, correlation, nulls, type coercion and deliberately messy columns. Ground truth is
never typed by hand and never produced by the agent under test.

```bash
python -m quarry bench                 # full suite, both arms -> results/bench_results.json + RESULTS.md
python -m quarry bench --limit 5 -v
python -m quarry bench --offline       # harness smoke test, no key, unfixtured tasks are skipped
python scripts/render_site_results.py  # inject the numbers into site/index.html
```

### Results

The full live suite has **not** been run here; it needs an API key and real tokens. Those rows
are marked `pending` on the site and in `results/RESULTS.md`, and no number anywhere in this
repo is invented.

What has been measured is the offline harness run: recorded model turns, real sandboxed
execution, real grading, real timings, over the 3 tasks that have a fixture.

| metric | self-correction ON | self-correction OFF |
| --- | ---: | ---: |
| task success rate | 100.0% | 33.3% |
| tasks graded | 3 | 3 |
| median steps | 3 | 1 |
| repair rate | 66.7% | 0.0% |
| repair success rate | 66.7% | n/a |
| grounded answer rate | 100.0% | 33.3% |
| mean cost / task | $0.0231 | $0.0063 |
| p95 latency | 2.19s | 0.66s |
| sandbox violations | 0 | 0 |

Three tasks is a demonstration that the harness and the ablation work end to end, not evidence
about model quality. Read it as such.

## Datasets

Seeded generators in `scripts/make_data.py`, regenerable and committed:

- `data/passengers.csv` — 340 rows, a titanic-shaped table with missing ages and cabins.
- `data/sales.csv` — 2065 orders over 2024-2025, with trend, seasonality and a weekend dip.
- `data/sensors.csv` — 600 readings, deliberately filthy: numbers stored as text with unit
  suffixes, `N/A` and `-999` sentinels, station and status codes differing only by case,
  timestamps in three formats, missing placements.

## Tests

```bash
pytest          # 160 tests, no API key required
```

The largest file is `tests/test_ast_guard.py`, which asserts each forbidden construct
individually: a regression that quietly allows one through is the failure mode that matters.
The agent tests drive the loop with `OfflineLLMClient` and cover the repair path, the ablation,
the verify gate rejecting an unsupported claim, and every bound (`max_steps`, repair budget,
cost budget).

## Limitations

- The sandbox is not a security boundary (see above).
- One table at a time. No cross-table joins, no schema discovery across files.
- No memory between questions; every `ask` starts from an empty message list.
- The LLM judge grades 3 of 43 tasks with a single sampled verdict and no inter-rater check.
- Charting is matplotlib only and is never graded: a valid, useless figure passes.
- The datasets are synthetic, so the success rate is a measurement on this suite, not a general
  capability number.
- One sample per task per arm, so the gap between arms carries unquantified sampling noise.

## Layout

```
quarry/          package: sandbox, tools, agent, verify, llm, viewer, cli, prompts/, bench/
data/            three bundled CSVs
fixtures/offline/ recorded model turns for --offline
runs/example/    a committed real trace
results/         bench_results.json + RESULTS.md
scripts/         data generator, task builder, site results injector
site/            hand-written static page + the rendered trace
tests/           pytest, no key needed
```

## Links

- [github.com/Manavarya09/quarry](https://github.com/Manavarya09/quarry)
- Sibling projects: [strata](https://github.com/Manavarya09/strata),
  [caliper](https://github.com/Manavarya09/caliper)
