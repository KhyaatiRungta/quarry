# Quarry benchmark results

Generated 2026-09-20T18:54:33+00:00.

- mode: **offline**
- model: `offline/replay`
- tasks in suite: 43
- status: **offline_smoke**

Offline mode. Only tasks with a recorded fixture run; every other task is reported as skipped, not as a failure. These numbers describe the harness working end to end, not model quality. The full suite needs a live key.

| metric | self-correction ON | self-correction OFF |
| --- | ---: | ---: |
| task success rate | 100.0% | 33.3% |
| tasks graded | 3 | 3 |
| tasks skipped | 40 | 40 |
| median steps | 3 | 1 |
| repair rate (runs with >=1 repair) | 66.7% | 0.0% |
| repair success rate | 66.7% | n/a |
| grounded answer rate | 100.0% | 33.3% |
| mean cost / task (USD) | $0.0231 | $0.0063 |
| median latency (s) | 1.102 | 0.614 |
| p95 latency (s) | 2.189 | 0.662 |
| sandbox violations | 0 | 0 |

## Per task

| task | difficulty | grader | ON | OFF |
| --- | --- | --- | --- | --- |
| pass-01 | easy | numeric | skipped | skipped |
| pass-02 | easy | numeric | pass | pass |
| pass-03 | easy | numeric | skipped | skipped |
| pass-04 | easy | numeric | skipped | skipped |
| pass-05 | easy | numeric | skipped | skipped |
| pass-06 | easy | predicate | skipped | skipped |
| pass-07 | medium | numeric | skipped | skipped |
| pass-08 | medium | numeric | skipped | skipped |
| pass-09 | hard | numeric | skipped | skipped |
| pass-10 | medium | numeric | skipped | skipped |
| pass-11 | medium | numeric | skipped | skipped |
| pass-12 | medium | numeric | skipped | skipped |
| pass-13 | hard | judge | skipped | skipped |
| sales-01 | easy | numeric | skipped | skipped |
| sales-02 | easy | predicate | pass | fail |
| sales-03 | easy | numeric | skipped | skipped |
| sales-04 | easy | predicate | skipped | skipped |
| sales-05 | easy | numeric | skipped | skipped |
| sales-06 | medium | numeric | skipped | skipped |
| sales-07 | medium | predicate | skipped | skipped |
| sales-08 | medium | numeric | skipped | skipped |
| sales-09 | medium | numeric | skipped | skipped |
| sales-10 | medium | numeric | skipped | skipped |
| sales-11 | hard | predicate | skipped | skipped |
| sales-12 | hard | numeric | skipped | skipped |
| sales-13 | easy | numeric | skipped | skipped |
| sales-14 | medium | numeric | skipped | skipped |
| sales-15 | hard | judge | skipped | skipped |
| sens-01 | easy | numeric | skipped | skipped |
| sens-02 | medium | numeric | pass | fail |
| sens-03 | medium | numeric | skipped | skipped |
| sens-04 | medium | numeric | skipped | skipped |
| sens-05 | medium | numeric | skipped | skipped |
| sens-06 | hard | numeric | skipped | skipped |
| sens-07 | easy | numeric | skipped | skipped |
| sens-08 | easy | numeric | skipped | skipped |
| sens-09 | medium | numeric | skipped | skipped |
| sens-10 | hard | predicate | skipped | skipped |
| sens-11 | hard | numeric | skipped | skipped |
| sens-12 | medium | numeric | skipped | skipped |
| sens-13 | hard | predicate | skipped | skipped |
| sens-14 | hard | numeric | skipped | skipped |
| sens-15 | hard | judge | skipped | skipped |
