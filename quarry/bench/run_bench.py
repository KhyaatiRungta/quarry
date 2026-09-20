"""Run the benchmark with self-correction ON and OFF and write the results files.

    python -m quarry bench                 # live, needs OPENROUTER_API_KEY
    python -m quarry bench --offline       # replays fixtures; only fixtured tasks run
    python -m quarry bench --limit 5 -v

The two arms differ in exactly one flag, `QuarryAgent(self_correction=...)`. Tasks
without an offline fixture are recorded as skipped rather than as failures, and the
summary says so, because a skip is not evidence about the agent.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..agent import QuarryAgent, load_prompt
from ..dataset import Dataset
from ..llm import JUDGE_MODEL, LLMClient, LLMError, OfflineLLMClient
from .grading import grade

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS = Path(__file__).parent / "tasks.jsonl"
ARMS = {"on": True, "off": False}


def load_tasks(path: str | Path | None = None, limit: int | None = None) -> list[dict]:
    path = Path(path or DEFAULT_TASKS)
    tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return tasks[:limit] if limit else tasks


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct * (len(ordered) - 1)))))
    return round(ordered[idx], 3)


def summarise(records: list[dict]) -> dict:
    ran = [r for r in records if not r.get("skipped")]
    graded = [r for r in ran if r["grade"].get("passed") is not None]
    passed = [r for r in graded if r["grade"]["passed"]]
    repair_runs = [r for r in ran if r["repair_attempts"] > 0]
    attempts = sum(r["repair_attempts"] for r in ran)
    successes = sum(r["repair_successes"] for r in ran)
    costs = [r["cost_usd"] for r in ran]
    latencies = [r["wall_s"] for r in ran]
    return {
        "tasks_attempted": len(ran),
        "tasks_graded": len(graded),
        "tasks_skipped": len(records) - len(ran),
        "judge_skipped": sum(1 for r in ran if r["grade"].get("skipped")),
        "task_success_rate": round(len(passed) / len(graded), 4) if graded else None,
        "median_steps": statistics.median([r["steps"] for r in ran]) if ran else None,
        "median_tool_calls": statistics.median([r["tool_calls"] for r in ran]) if ran else None,
        "repair_rate": round(len(repair_runs) / len(ran), 4) if ran else None,
        "repair_attempts": attempts,
        "repair_success_rate": round(successes / attempts, 4) if attempts else None,
        "grounded_rate": round(sum(1 for r in ran if r["grounded"]) / len(ran), 4) if ran else None,
        "mean_cost_usd_per_task": round(sum(costs) / len(costs), 5) if costs else None,
        "total_cost_usd": round(sum(costs), 4) if costs else 0.0,
        "p95_latency_s": _percentile(latencies, 0.95),
        "median_latency_s": _percentile(latencies, 0.5),
        "sandbox_violations": sum(r["sandbox_violations"] for r in ran),
    }


def run_arm(arm: str, tasks: list[dict], args, judge_client, log) -> list[dict]:
    self_correction = ARMS[arm]
    records: list[dict] = []
    judge_prompt = load_prompt("judge")
    for i, task in enumerate(tasks, 1):
        dataset_path = ROOT / task["dataset"]
        base = {"id": task["id"], "arm": arm, "dataset": task["dataset"],
                "difficulty": task["difficulty"], "tags": task["tags"],
                "graded_by": task["graded_by"], "question": task["question"]}
        try:
            if args.offline:
                client = OfflineLLMClient.from_dir(
                    args.fixtures or ROOT / "fixtures" / "offline", question=task["question"])
            else:
                client = LLMClient(model=args.model)
        except LLMError as exc:
            log(f"[{arm}] {i}/{len(tasks)} {task['id']}: skipped ({exc})")
            records.append({**base, "skipped": True, "skip_reason": "no offline fixture"})
            continue

        agent = QuarryAgent(
            client, Dataset.load(dataset_path),
            runs_dir=Path(args.runs_dir) / arm,
            max_steps=args.max_steps,
            self_correction=self_correction,
            run_id=f"{task['id']}-{arm}",
        )
        started = time.time()
        try:
            result = agent.ask(task["question"])
        except LLMError as exc:
            log(f"[{arm}] {i}/{len(tasks)} {task['id']}: run error ({exc})")
            records.append({**base, "skipped": True, "skip_reason": f"run error: {exc}"})
            continue
        verdict = grade(task, result.answer, result.supporting_values,
                        judge_client=judge_client, judge_prompt=judge_prompt)
        records.append({
            **base,
            "skipped": False,
            "passed": verdict.get("passed"),
            "grade": verdict,
            "answer": result.answer,
            "supporting_values": result.supporting_values,
            "stop_reason": result.stop_reason,
            "steps": result.steps,
            "tool_calls": result.tool_calls,
            "repair_attempts": result.repair_attempts,
            "repair_successes": result.repair_successes,
            "failure_kinds": result.failure_kinds,
            "sandbox_violations": result.sandbox_violations,
            "grounded": result.grounded,
            "cost_usd": round(result.usage.cost_usd, 6),
            "wall_s": round(time.time() - started, 3),
            "trace_path": result.trace_path,
        })
        mark = {True: "pass", False: "FAIL", None: "skip"}[verdict.get("passed")]
        log(f"[{arm}] {i}/{len(tasks)} {task['id']}: {mark} "
            f"steps={result.steps} repairs={result.repair_successes}/{result.repair_attempts}")
    return records


def write_results_md(payload: dict, path: Path) -> None:
    arms = payload["arms"]
    rows = [
        ("task success rate", "task_success_rate", "pct"),
        ("tasks graded", "tasks_graded", "int"),
        ("tasks skipped", "tasks_skipped", "int"),
        ("median steps", "median_steps", "num"),
        ("repair rate (runs with >=1 repair)", "repair_rate", "pct"),
        ("repair success rate", "repair_success_rate", "pct"),
        ("grounded answer rate", "grounded_rate", "pct"),
        ("mean cost / task (USD)", "mean_cost_usd_per_task", "usd"),
        ("median latency (s)", "median_latency_s", "num"),
        ("p95 latency (s)", "p95_latency_s", "num"),
        ("sandbox violations", "sandbox_violations", "int"),
    ]

    def fmt(value, kind):
        if value is None:
            return "n/a"
        if kind == "pct":
            return f"{100 * value:.1f}%"
        if kind == "usd":
            return f"${value:.4f}"
        if kind == "int":
            return str(int(value))
        return f"{value:g}"

    lines = [
        "# Quarry benchmark results",
        "",
        f"Generated {payload['generated_at']}.",
        "",
        f"- mode: **{payload['mode']}**",
        f"- model: `{payload['model']}`",
        f"- tasks in suite: {payload['tasks_total']}",
        f"- status: **{payload['status']}**",
        "",
        payload["note"],
        "",
        "| metric | self-correction ON | self-correction OFF |",
        "| --- | ---: | ---: |",
    ]
    for label, key, kind in rows:
        on = fmt(arms.get("on", {}).get(key), kind)
        off = fmt(arms.get("off", {}).get(key), kind)
        lines.append(f"| {label} | {on} | {off} |")
    lines += ["", "## Per task", "",
              "| task | difficulty | grader | ON | OFF |", "| --- | --- | --- | --- | --- |"]
    by_task: dict[str, dict] = {}
    for record in payload["per_task"]:
        entry = by_task.setdefault(record["id"], {"difficulty": record["difficulty"],
                                                  "graded_by": record["graded_by"]})
        if record.get("skipped"):
            entry[record["arm"]] = "skipped"
        else:
            entry[record["arm"]] = {True: "pass", False: "fail", None: "unjudged"}[
                record["grade"].get("passed")]
    for tid, entry in by_task.items():
        lines.append(f"| {tid} | {entry['difficulty']} | {entry['graded_by']} | "
                     f"{entry.get('on', '-')} | {entry.get('off', '-')} |")
    lines.append("")
    path.write_text("\n".join(lines))


def main(args) -> int:
    def log(msg: str) -> None:
        print(msg, file=sys.stderr)

    tasks = load_tasks(args.tasks, args.limit)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; use on and/or off.")

    judge_client = None
    if not args.offline and not args.no_judge:
        judge_client = LLMClient(model=JUDGE_MODEL)

    per_task: list[dict] = []
    summaries: dict[str, dict] = {}
    for arm in arms:
        records = run_arm(arm, tasks, args, judge_client, log)
        per_task.extend(records)
        summaries[arm] = summarise(records)

    ran_any = any(s["tasks_attempted"] for s in summaries.values())
    if args.offline:
        status = "offline_smoke" if ran_any else "pending"
        note = ("Offline mode. Only tasks with a recorded fixture run; every other task is "
                "reported as skipped, not as a failure. These numbers describe the harness "
                "working end to end, not model quality. The full suite needs a live key.")
    elif len(tasks) < payload_total(args):
        status = "partial"
        note = f"Partial run: {len(tasks)} of {payload_total(args)} tasks (--limit)."
    else:
        status = "complete"
        note = ("Live run over the full task set. Numeric tasks are graded against values "
                "computed by the reference implementation in scripts/build_tasks.py; the three "
                "open-ended tasks are graded by an LLM judge and are marked as such.")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "offline" if args.offline else "live",
        "model": "offline/replay" if args.offline else (args.model or LLMClient().model),
        "judge_model": None if args.offline or args.no_judge else JUDGE_MODEL,
        "tasks_total": payload_total(args),
        "tasks_run": len(tasks),
        "status": status,
        "note": note,
        "arms": summaries,
        "per_task": per_task,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench_results.json").write_text(json.dumps(payload, indent=2, default=str))
    write_results_md(payload, out_dir / "RESULTS.md")
    log(f"wrote {out_dir / 'bench_results.json'} and {out_dir / 'RESULTS.md'}")
    for arm, summary in summaries.items():
        log(f"  {arm}: graded={summary['tasks_graded']} "
            f"success={summary['task_success_rate']} repairs={summary['repair_attempts']}")
    return 0


def payload_total(args) -> int:
    return len(load_tasks(args.tasks, None))
