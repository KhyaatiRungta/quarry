"""Command-line interface for Quarry."""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

from quarry.agent import MAX_STEPS_MSG, QuarryAgent
from quarry.benchmark import run_benchmark
from quarry.config import ConfigError, load_settings
from quarry.redact import redact, redact_structure

LOG = logging.getLogger("quarry.cli")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def cmd_ask(args: argparse.Namespace) -> int:
    try:
        settings = load_settings(require_key=True)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    agent = QuarryAgent(
        dataset_path=args.data,
        max_steps=settings.max_steps,
        max_repairs=settings.max_repairs,
    )
    question = " ".join(args.question)
    answer = agent.ask(question)
    grounded = answer != MAX_STEPS_MSG

    result = {
        "question": question,
        "answer": answer,
        "grounded": grounded,
        "steps": len(agent.trace),
        "repairs": {
            "recovered": agent.repair_count,
            "attempts": agent.repair_attempts,
        },
        "tokens": {
            "input": agent.client.tokens_in,
            "output": agent.client.tokens_out,
        },
    }

    # Redact secrets, then write trace (for debugging / replay)
    trace_payload = redact_structure({**result, "trace": agent.trace})
    with open(args.trace_out, "w") as f:
        json.dump(trace_payload, f, indent=2)
    LOG.info("trace written to %s", args.trace_out)

    # Persist the run: runs/<timestamp>/{trace.json, answer.txt, *.png}
    run_dir = Path(args.runs_dir) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trace.json").write_text(json.dumps(trace_payload, indent=2))
    (run_dir / "answer.txt").write_text(redact(answer))
    for png in Path(".").glob("*.png"):
        shutil.copy(png, run_dir / png.name)
    LOG.info("run saved to %s", run_dir)

    answer = redact(answer)
    if args.json:
        print(json.dumps(redact_structure(result), indent=2))
    else:
        print()
        print("=" * 60)
        print(answer)
        print("=" * 60)
        print(f"Steps used: {result['steps']}")
        print(f"Repairs: {agent.repair_count} recovered / {agent.repair_attempts} failures")
        print(f"Tokens: {result['tokens']['input']} in / {result['tokens']['output']} out")
        print(f"Trace written to {args.trace_out}")
        print(f"Run saved to {run_dir}")
        if Path("chart.png").exists():
            print("Chart saved: chart.png")

    return 0 if grounded else 1


def cmd_bench(args: argparse.Namespace) -> int:
    try:
        stats = run_benchmark(args.tasks, json_out=args.json_out)
    except FileNotFoundError:
        print(f"Benchmark file not found: {args.tasks}", file=sys.stderr)
        return 2
    # Exit 0 only if the full agent grounded every task
    return 0 if stats["grounded_full_count"] == stats["tasks"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="quarry", description="Quarry: sandboxed self-correcting data analyst agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="Ask a question about a dataset")
    ask.add_argument("--data", required=True, help="Path to CSV file")
    ask.add_argument("--trace-out", default="trace.json", help="Where to write the trace file")
    ask.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    ask.add_argument("--runs-dir", default="runs", help="Directory for persisted runs")
    ask.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    ask.add_argument("question", nargs="+", help="Your question")

    bench = sub.add_parser("bench", help="Run the offline benchmark (no API calls)")
    bench.add_argument("--tasks", default="benchmarks/tasks.json", help="Benchmark task file")
    bench.add_argument("--json-out", default="bench_results.json", help="Results file")
    bench.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", False))

    if args.cmd == "ask":
        if not Path(args.data).exists():
            print(f"Error: data file not found: {args.data}", file=sys.stderr)
            return 2
        return cmd_ask(args)

    return cmd_bench(args)


if __name__ == "__main__":
    sys.exit(main())
