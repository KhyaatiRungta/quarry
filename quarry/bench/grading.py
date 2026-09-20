"""Grading for benchmark tasks.

Three graders, and every result says which one produced it:

  numeric    the expected value was computed by the reference implementation in
             scripts/build_tasks.py; a run passes if any number it reports lands
             inside the tolerance. This covers 33 of the 43 tasks.
  predicate  the answer must contain given strings (a region name, a month).
  judge      an LLM reads a rubric and returns pass/fail. Used only for the three
             deliberately open-ended tasks, and marked as such everywhere.

An answer is graded on its prose plus the values it declared as supporting, so a
correct number buried in prose still counts.
"""
from __future__ import annotations

import json
import re

NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    for token in NUMBER_RE.findall(text or ""):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return out


def _candidates(answer: str, supporting_values: list[str]) -> list[float]:
    values = _numbers(" ".join(str(v) for v in supporting_values))
    values += _numbers(answer or "")
    extra = []
    for v in values:
        # A model may report a proportion as a percentage or the other way round.
        extra += [v / 100.0, v * 100.0]
    return values + extra


def grade_numeric(task: dict, answer: str, supporting_values: list[str]) -> dict:
    expected = float(task["expected_value"])
    tol = float(task.get("tolerance", 0.005))
    relative = bool(task.get("relative_tolerance", True))
    window = abs(expected) * tol if relative else tol
    window = max(window, 1e-9)
    best = None
    for candidate in _candidates(answer, supporting_values):
        delta = abs(candidate - expected)
        if best is None or delta < best:
            best = delta
        if delta <= window:
            return {"passed": True, "method": "numeric", "expected": expected,
                    "matched": candidate, "window": window}
    return {"passed": False, "method": "numeric", "expected": expected,
            "closest_delta": best, "window": window,
            "detail": "no reported number was within tolerance"}


def grade_predicate(task: dict, answer: str, supporting_values: list[str]) -> dict:
    predicate = task["expected_predicate"]
    kind = predicate.get("type")
    haystack = (str(answer or "") + " " + " ".join(str(v) for v in supporting_values)).lower()
    if kind == "contains_all":
        missing = [v for v in predicate["values"] if str(v).lower() not in haystack]
        return {"passed": not missing, "method": "predicate", "type": kind,
                "missing": missing}
    raise ValueError(f"Unknown predicate type {kind!r} in task {task['id']}")


def grade_with_judge(task: dict, answer: str, supporting_values: list[str],
                     judge_client, prompt_template: str) -> dict:
    prompt = prompt_template.format(
        question=task["question"],
        rubric=task["rubric"],
        answer=answer or "(no answer)",
        supporting_values=", ".join(str(v) for v in supporting_values) or "(none)",
    )
    response = judge_client.complete(
        [{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=300,
    )
    verdict, reason = "fail", "judge returned unparseable output"
    text = response.text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            verdict = str(parsed.get("verdict", "fail")).lower()
            reason = str(parsed.get("reason", ""))[:300]
        except json.JSONDecodeError:
            pass
    return {"passed": verdict == "pass", "method": "judge", "reason": reason,
            "judge_cost_usd": round(response.usage.cost_usd, 6)}


def grade(task: dict, answer: str | None, supporting_values: list[str] | None,
          judge_client=None, judge_prompt: str | None = None) -> dict:
    supporting_values = supporting_values or []
    if not answer:
        return {"passed": False, "method": task.get("graded_by", "numeric"),
                "detail": "the run produced no answer"}
    graded_by = task.get("graded_by", "numeric")
    if graded_by == "numeric":
        return grade_numeric(task, answer, supporting_values)
    if graded_by == "predicate":
        return grade_predicate(task, answer, supporting_values)
    if graded_by == "judge":
        if judge_client is None:
            return {"passed": None, "method": "judge", "skipped": True,
                    "detail": "no judge available (offline or --no-judge)"}
        return grade_with_judge(task, answer, supporting_values, judge_client, judge_prompt)
    raise ValueError(f"Unknown grader {graded_by!r} for task {task['id']}")
