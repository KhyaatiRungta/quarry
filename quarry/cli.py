"""Command line entry point.

    python -m quarry ask --data data/sales.csv "which region earns most?"
    python -m quarry ask --offline --data data/sales.csv "..."
    python -m quarry bench --limit 10
    python -m quarry view runs/<id>/trace.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dataset import Dataset
from .llm import LLMClient, LLMError, OfflineLLMClient

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "offline"

RULE = "-" * 72


def _log(msg: str) -> None:
    print(f"  {msg}", file=sys.stderr)


def build_client(args, question: str | None = None):
    if args.offline:
        return OfflineLLMClient.from_dir(
            args.fixtures or FIXTURE_DIR, question=question,
            scenario=getattr(args, "scenario", None),
        )
    return LLMClient(model=args.model, log=_log)


def cmd_ask(args) -> int:
    from .agent import QuarryAgent

    dataset = Dataset.load(args.data, table=args.table)
    client = build_client(args, question=args.question)
    agent = QuarryAgent(
        client, dataset,
        runs_dir=args.runs_dir,
        max_steps=args.max_steps,
        cost_budget_usd=args.budget,
        self_correction=not args.no_self_correction,
        timeout_s=args.timeout,
        log=_log if args.verbose else None,
    )
    print(f"question: {args.question}")
    print(f"dataset:  {dataset.source_path}  ({len(dataset.df)} rows, {len(dataset.df.columns)} cols)")
    print(f"model:    {getattr(client, 'model', 'unknown')}"
          f"{'  [offline replay]' if args.offline else ''}")
    print(RULE)
    result = agent.ask(args.question)

    for step in _iter_trace_steps(result):
        print(step)
    print(RULE)
    if result.answer:
        print(result.answer)
    else:
        print(f"No answer produced. stop_reason={result.stop_reason}")
    print(RULE)
    v = result.verify
    print(f"steps={result.steps}  tool_calls={result.tool_calls}  "
          f"repairs={result.repair_successes}/{result.repair_attempts}  "
          f"grounded={'yes' if result.grounded else 'no'}"
          f"{' (' + ', '.join(v.unsupported) + ' unsupported)' if v and v.unsupported else ''}")
    print(f"tokens={result.usage.prompt_tokens}+{result.usage.completion_tokens}  "
          f"cost=${result.usage.cost_usd:.4f}  wall={result.wall_s:.1f}s")
    if result.chart_path:
        print(f"chart: {result.chart_path}")
    print(f"trace: {result.trace_path}")
    if args.render_trace:
        from .viewer import render_trace_file
        out = render_trace_file(result.trace_path, args.render_trace)
        print(f"viewer: {out}")
    return 0 if result.ok else 1


def _iter_trace_steps(result):
    trace = json.loads(Path(result.trace_path).read_text())
    for step in trace["steps"]:
        if step["type"] != "model_turn":
            continue
        for call in step["tool_calls"]:
            flag = "repair " if call["is_repair"] else ""
            status = "ok" if call["ok"] else f"FAILED {call.get('failure_kind')}"
            yield f"step {step['index']}: {flag}{call['name']} -> {status}"


def cmd_bench(args) -> int:
    from .bench.run_bench import main as bench_main
    return bench_main(args)


def cmd_view(args) -> int:
    from .viewer import render_trace_file
    out = render_trace_file(args.trace, args.out)
    print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quarry",
                                     description="A data analyst agent with a sandbox and a repair loop.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--offline", action="store_true",
                       help="replay canned model turns from fixtures; no API key needed")
        p.add_argument("--fixtures", default=None, help="offline fixture directory")
        p.add_argument("--model", default=None, help="OpenRouter model id")

    ask = sub.add_parser("ask", help="answer one question about one table")
    ask.add_argument("question")
    ask.add_argument("--data", required=True, help="path to a .csv or .sqlite file")
    ask.add_argument("--table", default=None, help="table name, for sqlite sources")
    ask.add_argument("--runs-dir", default="runs")
    ask.add_argument("--max-steps", type=int, default=8)
    ask.add_argument("--budget", type=float, default=0.50, help="cost ceiling in USD")
    ask.add_argument("--timeout", type=float, default=30.0, help="per-execution timeout in seconds")
    ask.add_argument("--no-self-correction", action="store_true",
                     help="ablation: stop at the first failed execution")
    ask.add_argument("--scenario", default=None, help="force a named offline fixture")
    ask.add_argument("--render-trace", default=None, metavar="PATH",
                     help="also render the trace to a standalone HTML file")
    ask.add_argument("-v", "--verbose", action="store_true")
    common(ask)
    ask.set_defaults(func=cmd_ask)

    bench = sub.add_parser("bench", help="run the benchmark, self-correction on and off")
    bench.add_argument("--limit", type=int, default=None, help="run only the first N tasks")
    bench.add_argument("--tasks", default=None, help="path to tasks.jsonl")
    bench.add_argument("--out", default="results", help="output directory")
    bench.add_argument("--runs-dir", default="runs/bench")
    bench.add_argument("--arms", default="on,off", help="comma-separated: on, off")
    bench.add_argument("--no-judge", action="store_true", help="skip LLM-judged tasks")
    bench.add_argument("--max-steps", type=int, default=8)
    bench.add_argument("-v", "--verbose", action="store_true")
    common(bench)
    bench.set_defaults(func=cmd_bench)

    view = sub.add_parser("view", help="render a trace.json to standalone HTML")
    view.add_argument("trace")
    view.add_argument("--out", default=None, help="output .html path (default: next to the trace)")
    view.set_defaults(func=cmd_view)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (LLMError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
