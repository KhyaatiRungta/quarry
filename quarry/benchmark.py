"""Offline benchmark harness.

Replays RECORDED model turns (no API calls) while executing the generated
code for REAL in the sandbox. Each task runs in two modes:

  baseline - self-correction disabled (turns marked "repair" are skipped)
  full     - self-correction enabled  (all recorded turns replayed)

Metrics produced (matching the resume claims):
  * grounded-answer rate: baseline vs full
  * self-correction recovery rate over failed executions
"""

import argparse
import json
from types import SimpleNamespace

from quarry.agent import MAX_STEPS_MSG, QuarryAgent


class ReplayMessage:
    """Mimics an OpenAI assistant message carrying one tool call."""

    def __init__(self, turn, idx):
        self.tool_calls = [
            SimpleNamespace(
                id=f"call_replay_{idx}",
                function=SimpleNamespace(
                    name=turn["tool"],
                    arguments=json.dumps(turn.get("args", {})),
                ),
            )
        ]
        self.content = None

    def model_dump(self):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ],
        }


class ExhaustedMessage:
    """Text-only reply once recorded turns run out; agent spins to max_steps."""

    def __init__(self):
        self.tool_calls = None
        self.content = "replay: no more recorded turns"

    def model_dump(self):
        return {"role": "assistant", "content": self.content}


class ReplayLLM:
    """Drop-in replacement for LLMClient that replays recorded turns."""

    def __init__(self, turns, allow_repairs=True):
        self.turns = [t for t in turns if allow_repairs or not t.get("repair")]
        self.i = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def chat(self, messages, tools=None):
        if self.i >= len(self.turns):
            return ExhaustedMessage()
        turn = self.turns[self.i]
        self.i += 1
        return ReplayMessage(turn, self.i)


def _exec_stats(exec_log):
    """Failed executions, and how many were followed by a success (recovered)."""
    failed = [i for i, e in enumerate(exec_log) if not e["success"]]
    recovered = sum(1 for i in failed if any(e["success"] for e in exec_log[i + 1 :]))
    return len(failed), recovered


def run_benchmark(tasks_path, json_out=None, quiet=False):
    with open(tasks_path, encoding="utf-8") as f:
        tasks = json.load(f)["tasks"]

    results = []
    base_grounded = 0
    full_grounded = 0
    total_failed = 0
    total_recovered = 0

    for i, task in enumerate(tasks, 1):
        question = task["question"]
        data = task["data"]
        turns = task["turns"]

        # --- baseline: self-correction disabled ---
        agent_b = QuarryAgent(dataset_path=data, llm=ReplayLLM(turns, allow_repairs=False))
        ans_b = agent_b.ask(question)
        grounded_b = ans_b != MAX_STEPS_MSG
        if grounded_b:
            base_grounded += 1

        # --- full: self-correction enabled ---
        agent_f = QuarryAgent(dataset_path=data, llm=ReplayLLM(turns, allow_repairs=True))
        ans_f = agent_f.ask(question)
        grounded_f = ans_f != MAX_STEPS_MSG
        if grounded_f:
            full_grounded += 1

        failed, recovered = _exec_stats(agent_f.exec_log)
        total_failed += failed
        total_recovered += recovered

        results.append(
            {
                "name": task["name"],
                "question": question,
                "baseline": {"answer": ans_b, "grounded": grounded_b},
                "full": {"answer": ans_f, "grounded": grounded_f},
                "executions": agent_f.exec_log,
                "failed_executions": failed,
                "recovered": recovered,
            }
        )

        if not quiet:
            b = "GROUNDED" if grounded_b else "stuck"
            fu = "GROUNDED" if grounded_f else "stuck"
            print(f"[{i}/{len(tasks)}] {task['name']}")
            print(f"    baseline (no self-correction): {b}")
            print(f"    full     (self-correction):    {fu}")
            print(f"    answer: {ans_f}")

    n = len(tasks)
    stats = {
        "tasks": n,
        "grounded_baseline_count": base_grounded,
        "grounded_baseline_rate": round(base_grounded / n * 100, 1),
        "grounded_full_count": full_grounded,
        "grounded_full_rate": round(full_grounded / n * 100, 1),
        "failed_executions": total_failed,
        "recovered_executions": total_recovered,
        "recovery_rate": round((total_recovered / total_failed * 100) if total_failed else 100.0, 1),
        "results": results,
    }

    if not quiet:
        print()
        print("=" * 60)
        print("QUARRY OFFLINE BENCHMARK (recorded turns, real sandbox)")
        print("=" * 60)
        print(
            f"Grounded answers, baseline (no self-correction): {base_grounded}/{n} = {stats['grounded_baseline_rate']}%"
        )
        print(f"Grounded answers, full (self-correction):        {full_grounded}/{n} = {stats['grounded_full_rate']}%")
        print(
            f"Failed executions: {total_failed} | recovered by "
            f"self-correction: {total_recovered} = {stats['recovery_rate']}%"
        )

    if json_out:
        with open(json_out, "w") as f:
            json.dump(stats, f, indent=2)
        if not quiet:
            print(f"Results written to {json_out}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quarry offline benchmark")
    parser.add_argument("--tasks", default="benchmarks/tasks.json")
    parser.add_argument("--json-out", default="bench_results.json")
    args = parser.parse_args()
    run_benchmark(args.tasks, json_out=args.json_out)
