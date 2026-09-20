"""Render a trace.json into a standalone static HTML page.

No framework, no JavaScript, no external assets. The chart, if the run produced
one, is inlined as a data URI so the file can be emailed or opened from disk.
Same design language as site/: warm off-white, serif prose, monospace for
everything a machine produced.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

CSS = """
:root { --bg:#faf8f3; --fg:#16150f; --muted:#6b6557; --rule:#d8d3c7; --ink:#1f3a5f;
        --fail:#8a2f22; --panel:#f3efe6; --nav-bg:rgba(250,248,243,0.82); }
@media (prefers-color-scheme: dark) {
  :root { --bg:#12110e; --fg:#e8e4da; --muted:#9a9384; --rule:#302d26; --ink:#9db8d8;
          --fail:#d98c7a; --panel:#1a1814; --nav-bg:rgba(18,17,14,0.82); }
}
* { box-sizing: border-box; }
body { margin:0; padding:0 0 96px; background:var(--bg); color:var(--fg);
       font-family: ui-serif, Georgia, 'Times New Roman', serif; line-height:1.55; }
.nav { position:sticky; top:0; z-index:10; background:var(--nav-bg);
       backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
       border-bottom:1px solid var(--rule); }
.nav-inner { max-width:1080px; margin:0 auto; padding:12px 28px; display:flex;
             align-items:baseline; gap:24px; }
.wordmark { font-family: ui-monospace,'SF Mono',Menlo,monospace; font-size:0.95rem;
            font-weight:600; color:var(--fg); text-decoration:none; }
.nav-links { margin-left:auto; display:flex; gap:22px; }
.nav-links a { font-family: ui-monospace,'SF Mono',Menlo,monospace; font-size:0.78rem;
               color:var(--muted); text-decoration:none; }
.nav-links a:hover { color:var(--fg); }
.wrap { max-width:1080px; margin:0 auto; padding:48px 28px 0; }
.prose { max-width:68ch; }
h1 { font-size:1.6rem; font-weight:600; margin:0 0 4px; letter-spacing:-0.01em; }
h2 { font-size:1.05rem; font-weight:600; margin:0 0 12px; }
a { color:var(--ink); }
.mono, code, pre, table, .meta, .label { font-family: ui-monospace,'SF Mono',Menlo,monospace; }
.meta { font-size:0.78rem; color:var(--muted); }
.kv { font-size:0.8rem; border-collapse:collapse; margin:16px 0 0; }
.kv td { padding:2px 18px 2px 0; vertical-align:top; }
.kv td:first-child { color:var(--muted); }
.step { border-top:1px solid var(--rule); padding:26px 0 4px; }
.step-head { display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; }
.num { font-family: ui-monospace,'SF Mono',Menlo,monospace; font-size:0.78rem;
       color:var(--muted); min-width:5.5em; }
.tag { font-family: ui-monospace,'SF Mono',Menlo,monospace; font-size:0.7rem;
       letter-spacing:0.06em; text-transform:uppercase; border:1px solid var(--rule);
       padding:1px 6px; color:var(--muted); }
.tag.repair { border-color:var(--fail); color:var(--fail); }
.tag.fail { border-color:var(--fail); color:var(--fail); }
.say { margin:10px 0 0; }
pre { background:var(--panel); border:1px solid var(--rule); padding:12px 14px;
      overflow-x:auto; font-size:0.78rem; line-height:1.45; margin:10px 0 0; white-space:pre; }
pre.out { background:transparent; border:0; border-left:2px solid var(--rule);
          padding:2px 0 2px 14px; white-space:pre-wrap; }
pre.err { border-left-color:var(--fail); color:var(--fail); }
.label { font-size:0.72rem; letter-spacing:0.06em; text-transform:uppercase;
         color:var(--muted); margin:14px 0 0; }
.cost { margin-left:auto; font-family: ui-monospace,'SF Mono',Menlo,monospace;
        font-size:0.75rem; color:var(--muted); }
img.chart { max-width:560px; width:100%; border:1px solid var(--rule); margin-top:12px;
            background:#fff; }
table.sum { border-collapse:collapse; font-size:0.8rem; margin-top:8px; }
table.sum td, table.sum th { border-bottom:1px solid var(--rule); padding:5px 20px 5px 0;
                             text-align:left; }
table.sum td.n { text-align:right; font-variant-numeric:tabular-nums; }
footer { border-top:1px solid var(--rule); margin-top:48px; padding-top:18px;
         font-family: ui-monospace,'SF Mono',Menlo,monospace; font-size:0.76rem;
         color:var(--muted); }
footer .links { display:flex; gap:20px; flex-wrap:wrap; margin-top:8px; }
footer a { color:var(--muted); text-decoration:none; }
footer a:hover { color:var(--fg); text-decoration:underline; }
@media (max-width:620px) { .wrap, .nav-inner { padding-left:18px; padding-right:18px; }
  .nav-links { gap:14px; } }
"""

NAV = (
    '<nav class="nav"><div class="nav-inner">'
    '<a class="wordmark" href="index.html">quarry</a>'
    '<div class="nav-links">'
    '<a href="index.html#overview">Overview</a>'
    '<a href="index.html#how">How it works</a>'
    '<a href="index.html#benchmarks">Benchmarks</a>'
    '<a href="https://github.com/Manavarya09/quarry">GitHub</a>'
    "</div></div></nav>"
)

FOOTER = (
    '<footer><div>quarry &middot; 2026 &middot; trace rendered by '
    '<code>python -m quarry view</code></div>'
    '<div class="links">'
    '<a href="https://github.com/Manavarya09/quarry">github.com/Manavarya09/quarry</a>'
    '<a href="https://github.com/Manavarya09/strata">strata</a>'
    '<a href="https://github.com/Manavarya09/caliper">caliper</a>'
    "</div></footer>"
)


def _esc(text) -> str:
    return html.escape("" if text is None else str(text))


def _pre(text: str, cls: str = "") -> str:
    return f'<pre class="{cls}">{_esc(text)}</pre>' if text and text.strip() else ""


def _chart_img(chart_path: str | None) -> str:
    if not chart_path:
        return ""
    p = Path(chart_path)
    if not p.exists():
        return f'<p class="meta">chart written to {_esc(chart_path)} (not embedded: file missing)</p>'
    data = base64.b64encode(p.read_bytes()).decode()
    return f'<img class="chart" alt="chart produced by the run" src="data:image/png;base64,{data}">'


def render_trace(trace: dict) -> str:
    cfg = trace.get("config", {})
    summary = trace.get("summary", {})
    ds = trace.get("dataset", {})
    usage = summary.get("usage", {})

    parts = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>Quarry trace {_esc(trace.get('run_id'))}</title>",
        f"<style>{CSS}</style></head><body>",
        NAV,
        '<div class="wrap">',
        "<div class=\"prose\">",
        "<p class=\"meta\">QUARRY &middot; REPLAYABLE RUN TRACE</p>",
        f"<h1>{_esc(trace.get('question'))}</h1>",
        f"<p class=\"meta\">run {_esc(trace.get('run_id'))} &middot; "
        f"{_esc(trace.get('created_at'))}</p>",
        "<table class=\"kv\">",
        f"<tr><td>dataset</td><td>{_esc(ds.get('name'))} &middot; {_esc(ds.get('rows'))} rows "
        f"&middot; {_esc(len(ds.get('columns', [])))} columns</td></tr>",
        f"<tr><td>model</td><td>{_esc(cfg.get('model'))}"
        f"{' &middot; offline replay' if cfg.get('offline') else ''}</td></tr>",
        f"<tr><td>limits</td><td>max_steps={_esc(cfg.get('max_steps'))} &middot; "
        f"budget=${_esc(cfg.get('cost_budget_usd'))} &middot; "
        f"timeout={_esc(cfg.get('timeout_s'))}s &middot; "
        f"self_correction={'on' if cfg.get('self_correction') else 'off'}</td></tr>",
        "</table>",
    ]
    if cfg.get("offline"):
        parts.append(
            '<p class="meta" style="margin-top:14px">Offline replay: the model turns below were '
            'recorded, every tool execution, traceback, duration and result on this page was '
            'produced by really running the code.</p>'
        )
    parts.append("</div>")

    for step in trace.get("steps", []):
        if step.get("type") == "verify":
            parts.append(_render_verify(step))
            continue
        parts.append(_render_model_turn(step))

    parts.append(_render_summary(summary, usage))
    parts.append(FOOTER)
    parts.append("</div></body></html>")
    return "\n".join(parts)


def _render_model_turn(step: dict) -> str:
    out = ['<section class="step">',
           '<div class="step-head">',
           f'<span class="num">step {_esc(step.get("index"))}</span>',
           '<span class="tag">model</span>']
    usage = step.get("usage", {})
    out.append(
        f'<span class="cost">{_esc(usage.get("prompt_tokens"))}+'
        f'{_esc(usage.get("completion_tokens"))} tok &middot; '
        f'{_esc(usage.get("latency_s"))}s &middot; running '
        f'${_esc(step.get("cumulative_cost_usd"))}</span></div>')
    if step.get("text"):
        out.append(f'<p class="say prose">{_esc(step["text"])}</p>')
    if step.get("nudged"):
        out.append('<p class="meta">No tool call in this turn; the loop nudged the model.</p>')

    for call in step.get("tool_calls", []):
        out.append(_render_tool_call(call))
    out.append("</section>")
    return "\n".join(out)


def _render_tool_call(call: dict) -> str:
    tags = [f'<span class="tag">{_esc(call["name"])}</span>']
    if call.get("is_repair"):
        tags.append('<span class="tag repair">repair</span>')
    if not call.get("ok"):
        tags.append(f'<span class="tag fail">{_esc(call.get("failure_kind") or "failed")}</span>')
    head = ('<div class="step-head" style="margin-top:18px">'
            '<span class="num"></span>' + "".join(tags)
            + f'<span class="cost">{_esc(call.get("duration_s"))}s</span></div>')
    body = [head]
    execution = call.get("execution") or {}
    code = execution.get("code") or call.get("arguments", {}).get("code") \
        or call.get("arguments", {}).get("query")
    if code:
        body.append('<p class="label">code it wrote</p>')
        body.append(_pre(code))
    if call["name"] == "final_answer":
        args = call.get("arguments", {})
        body.append('<p class="label">answer</p>')
        body.append(f'<p class="say prose">{_esc(args.get("answer"))}</p>')
        body.append('<p class="label">supporting values</p>')
        body.append(_pre(", ".join(str(v) for v in args.get("supporting_values", []))))
        return "\n".join(body)
    if execution.get("stdout"):
        body.append('<p class="label">stdout</p>')
        body.append(_pre(execution["stdout"], "out"))
    if execution.get("stderr"):
        body.append('<p class="label">stderr</p>')
        body.append(_pre(execution["stderr"], "out err"))
    if execution.get("traceback") and not execution.get("ok"):
        body.append('<p class="label">traceback fed back to the model</p>')
        body.append(_pre(execution["traceback"], "out err"))
    elif not call.get("ok"):
        body.append('<p class="label">observation fed back to the model</p>')
        body.append(_pre(call.get("observation"), "out err"))
    if execution.get("result_repr"):
        body.append('<p class="label">result</p>')
        body.append(_pre(f'({execution.get("result_type")}) {execution["result_repr"]}', "out"))
    if call["name"] == "inspect_schema" and call.get("ok"):
        body.append('<p class="label">schema returned</p>')
        body.append(_pre(call.get("observation"), "out"))
    if execution.get("chart_path"):
        body.append(_chart_img(execution["chart_path"]))
    return "\n".join(body)


def _render_verify(step: dict) -> str:
    report = step.get("report", {})
    ok = report.get("grounded")
    tag = '<span class="tag">grounded</span>' if ok else '<span class="tag fail">not grounded</span>'
    rows = [
        '<section class="step">',
        '<div class="step-head"><span class="num">verify</span>', tag, '</div>',
        '<p class="say prose">Each supporting value is matched against the text that actually '
        'came out of the sandbox during this run.</p>',
        '<table class="sum"><tr><th>value</th><th>found in execution output</th></tr>',
    ]
    for value in report.get("supported", []):
        rows.append(f'<tr><td class="mono">{_esc(value)}</td><td class="mono">yes</td></tr>')
    for value in report.get("unsupported", []):
        rows.append(f'<tr><td class="mono">{_esc(value)}</td>'
                    f'<td class="mono" style="color:var(--fail)">no</td></tr>')
    rows.append("</table></section>")
    return "\n".join(rows)


def _render_summary(summary: dict, usage: dict) -> str:
    verify = summary.get("verify") or {}
    rows = [
        ("outcome", summary.get("stop_reason")),
        ("steps", summary.get("steps")),
        ("tool calls", summary.get("tool_calls")),
        ("repair attempts", summary.get("repair_attempts")),
        ("repairs that succeeded", summary.get("repair_successes")),
        ("failure kinds", ", ".join(summary.get("failure_kinds") or []) or "none"),
        ("sandbox violations", summary.get("sandbox_violations")),
        ("grounded", "yes" if verify.get("grounded") else "no"),
        ("tokens", f'{usage.get("prompt_tokens")} in / {usage.get("completion_tokens")} out'),
        ("cost", f'${usage.get("cost_usd")}'),
        ("wall clock", f'{summary.get("wall_s")}s'),
    ]
    body = ['<section class="step"><div class="step-head">'
            '<span class="num">summary</span></div><table class="sum">']
    for key, value in rows:
        body.append(f'<tr><td>{_esc(key)}</td><td class="mono n">{_esc(value)}</td></tr>')
    body.append("</table>")
    if summary.get("chart_path"):
        body.append(_chart_img(summary["chart_path"]))
    body.append("</section>")
    return "\n".join(body)


def render_trace_file(trace_path: str | Path, out_path: str | Path | None = None) -> str:
    trace_path = Path(trace_path)
    trace = json.loads(trace_path.read_text())
    out = Path(out_path) if out_path else trace_path.with_suffix(".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_trace(trace))
    return str(out)
