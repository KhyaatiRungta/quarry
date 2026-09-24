"""Agent loop tests driven by replayed (recorded) model turns - no API needed."""

import json

from quarry.agent import MAX_STEPS_MSG, QuarryAgent
from quarry.benchmark import ReplayLLM

with open("benchmarks/tasks.json", encoding="utf-8") as f:
    TASKS = {t["name"]: t for t in json.load(f)["tasks"]}


def test_full_mode_recovers_and_answers():
    task = TASKS["highest-revenue-region"]
    agent = QuarryAgent(
        dataset_path=task["data"],
        llm=ReplayLLM(task["turns"], allow_repairs=True),
    )
    answer = agent.ask(task["question"])

    assert answer == "East has the highest total revenue at 675,000."
    # First exec crashes (KeyError: region), repair succeeds
    assert agent.repair_attempts == 1
    assert agent.repair_count == 1
    assert len(agent.trace) == 3
    assert [e["success"] for e in agent.exec_log] == [False, True]


def test_baseline_mode_gets_stuck():
    task = TASKS["highest-revenue-region"]
    agent = QuarryAgent(
        dataset_path=task["data"],
        llm=ReplayLLM(task["turns"], allow_repairs=False),
    )
    answer = agent.ask(task["question"])

    # Repair turn skipped -> verify gate has no real output -> ungrounded
    assert answer == MAX_STEPS_MSG


def test_unrecovered_failure_still_grounded():
    task = TASKS["units-by-region"]
    agent = QuarryAgent(
        dataset_path=task["data"],
        llm=ReplayLLM(task["turns"], allow_repairs=True),
    )
    answer = agent.ask(task["question"])

    assert "995" in answer
    # 1 success then 1 unrecovered failure (KeyError: Units)
    assert [e["success"] for e in agent.exec_log] == [True, False]
