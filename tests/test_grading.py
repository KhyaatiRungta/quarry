import pytest

from quarry.bench.grading import grade, grade_numeric
from quarry.bench.run_bench import load_tasks, summarise


def numeric_task(value, tolerance=0.005, relative=True):
    return {"id": "t", "graded_by": "numeric", "expected_value": value,
            "tolerance": tolerance, "relative_tolerance": relative}


@pytest.mark.parametrize("reported,passes", [
    ("75.61", True),
    ("75.6", True),           # inside 0.5% relative tolerance
    ("75.0", False),          # outside it
    ("100.0", False),
])
def test_relative_tolerance(reported, passes):
    result = grade_numeric(numeric_task(75.61), f"The average is {reported}.", [reported])
    assert result["passed"] is passes


def test_exact_tolerance_admits_nothing_else():
    task = numeric_task(340, tolerance=0, relative=False)
    assert grade_numeric(task, "340 passengers", ["340"])["passed"]
    assert not grade_numeric(task, "341 passengers", ["341"])["passed"]


def test_absolute_tolerance():
    task = numeric_task(22.0, tolerance=1, relative=False)
    assert grade_numeric(task, "about 23 days", ["23"])["passed"]
    assert not grade_numeric(task, "about 25 days", ["25"])["passed"]


def test_percentage_and_proportion_are_interchangeable():
    task = numeric_task(0.447059)
    assert grade_numeric(task, "44.7% survived", ["44.7"])["passed"]


def test_a_number_only_in_the_prose_still_counts():
    task = numeric_task(1788730.0)
    assert grade_numeric(task, "Total revenue is 1788730.0 across the two years.", [])["passed"]


def test_thousands_separators_are_handled():
    task = numeric_task(1788730.0)
    assert grade_numeric(task, "Total revenue is 1,788,730.00.", [])["passed"]


def test_predicate_grading_is_case_insensitive():
    task = {"id": "t", "graded_by": "predicate",
            "expected_predicate": {"type": "contains_all", "values": ["North"]}}
    assert grade(task, "the north region leads", [])["passed"]
    assert not grade(task, "the west region leads", [])["passed"]


def test_missing_answer_fails_without_raising():
    result = grade(numeric_task(1.0), None, [])
    assert result["passed"] is False
    assert "no answer" in result["detail"]


def test_judge_task_without_a_judge_is_skipped_not_failed():
    task = {"id": "t", "graded_by": "judge", "question": "q", "rubric": "r"}
    result = grade(task, "some answer", [])
    assert result["passed"] is None
    assert result["skipped"] is True


def test_judge_parses_a_verdict(monkeypatch):
    class FakeUsage:
        cost_usd = 0.001

    class FakeResponse:
        text = 'Sure: {"verdict": "pass", "reason": "covers the rubric"}'
        usage = FakeUsage()

    class FakeJudge:
        def complete(self, messages, **kwargs):
            return FakeResponse()

    task = {"id": "t", "graded_by": "judge", "question": "q", "rubric": "r"}
    result = grade(task, "answer", ["1"], judge_client=FakeJudge(),
                   judge_prompt="{question}{rubric}{answer}{supporting_values}")
    assert result["passed"] is True
    assert result["reason"] == "covers the rubric"


def test_task_file_is_wellformed():
    tasks = load_tasks()
    assert len(tasks) >= 30
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids))
    datasets = {t["dataset"] for t in tasks}
    assert len(datasets) >= 3
    difficulties = {t["difficulty"] for t in tasks}
    assert difficulties == {"easy", "medium", "hard"}
    for task in tasks:
        assert task["graded_by"] in {"numeric", "predicate", "judge"}
        assert task["question"].strip()
        assert task["tags"]
        if task["graded_by"] == "numeric":
            assert isinstance(task["expected_value"], float)
        elif task["graded_by"] == "predicate":
            assert task["expected_predicate"]["values"]
        else:
            assert task["rubric"].strip()


def test_messy_tasks_exist_to_exercise_the_repair_loop():
    tasks = load_tasks()
    messy = [t for t in tasks if "messy" in t["tags"] or "type-coercion" in t["tags"]]
    assert len(messy) >= 6


def test_summarise_ignores_skipped_runs():
    records = [
        {"skipped": True},
        {"skipped": False, "grade": {"passed": True}, "steps": 3, "tool_calls": 3,
         "repair_attempts": 1, "repair_successes": 1, "sandbox_violations": 0,
         "grounded": True, "cost_usd": 0.02, "wall_s": 2.0},
        {"skipped": False, "grade": {"passed": False}, "steps": 5, "tool_calls": 5,
         "repair_attempts": 0, "repair_successes": 0, "sandbox_violations": 1,
         "grounded": False, "cost_usd": 0.04, "wall_s": 4.0},
    ]
    summary = summarise(records)
    assert summary["tasks_attempted"] == 2
    assert summary["tasks_skipped"] == 1
    assert summary["task_success_rate"] == 0.5
    assert summary["repair_rate"] == 0.5
    assert summary["repair_success_rate"] == 1.0
    assert summary["sandbox_violations"] == 1
