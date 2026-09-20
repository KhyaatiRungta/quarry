"""Agent loop behaviour, driven entirely by OfflineLLMClient. No API key needed."""
import json

import pytest

from quarry.agent import QuarryAgent
from quarry.dataset import Dataset
from quarry.llm import OfflineLLMClient


def turn(name=None, text="", **arguments):
    call = [{"name": name, "arguments": arguments}] if name else []
    return {"text": text, "tool_calls": call, "prompt_tokens": 800, "completion_tokens": 100}


@pytest.fixture
def sales(repo_root):
    return Dataset.load(repo_root / "data/sales.csv")


def make_agent(responses, dataset, tmp_path, **kwargs):
    return QuarryAgent(OfflineLLMClient(responses), dataset,
                       runs_dir=tmp_path / "runs", **kwargs)


GOOD_CODE = ("by_region = df.groupby('region')['revenue'].sum().sort_values(ascending=False)\n"
             "print(by_region)\nresult = by_region.idxmax()")
BAD_CODE = "result = df.groupby('region')['total_revenue'].sum()"


def test_happy_path_answers_and_writes_a_trace(sales, tmp_path):
    agent = make_agent([
        turn("run_python", code=GOOD_CODE),
        turn("final_answer", answer="North leads.", supporting_values=["North", "588920.0"]),
    ], sales, tmp_path)
    result = agent.ask("Which region earns most?")
    assert result.ok
    assert result.stop_reason == "final_answer"
    assert result.repair_attempts == 0
    assert result.grounded
    trace = json.loads(open(result.trace_path).read())
    assert trace["schema_version"] == 1
    assert trace["summary"]["ok"] is True
    assert [s["type"] for s in trace["steps"]] == ["model_turn", "model_turn", "verify"]
    assert trace["steps"][0]["tool_calls"][0]["execution"]["code"] == GOOD_CODE


def test_repair_loop_triggers_on_a_failed_execution(sales, tmp_path):
    agent = make_agent([
        turn("run_python", code=BAD_CODE),
        turn("run_python", text="KeyError, wrong column", code=GOOD_CODE),
        turn("final_answer", answer="North leads.", supporting_values=["North"]),
    ], sales, tmp_path)
    result = agent.ask("Which region earns most?")
    assert result.ok
    assert result.repair_attempts == 1
    assert result.repair_successes == 1
    assert result.failure_kinds == ["runtime_error"]
    trace = json.loads(open(result.trace_path).read())
    assert trace["steps"][1]["tool_calls"][0]["is_repair"] is True


def test_the_real_traceback_is_fed_back_to_the_model(sales, tmp_path):
    agent = make_agent([
        turn("run_python", code=BAD_CODE),
        turn("run_python", code=GOOD_CODE),
        turn("final_answer", answer="North leads.", supporting_values=["North"]),
    ], sales, tmp_path)
    result = agent.ask("Which region earns most?")
    trace = json.loads(open(result.trace_path).read())
    feedback = [m for m in trace["messages"] if m["role"] == "tool"][0]["content"]
    assert "KeyError" in feedback
    assert "total_revenue" in feedback
    repair_prompt = [m for m in trace["messages"]
                     if m["role"] == "user" and "repair attempt" in (m.get("content") or "")]
    assert repair_prompt, "the repair prompt was not appended"


def test_a_repair_after_a_schema_lookup_still_counts_as_a_repair(sales, tmp_path):
    agent = make_agent([
        turn("run_python", code=BAD_CODE),
        turn("inspect_schema"),
        turn("run_python", code=GOOD_CODE),
        turn("final_answer", answer="North leads.", supporting_values=["North"]),
    ], sales, tmp_path)
    result = agent.ask("Which region earns most?")
    assert result.repair_attempts == 2
    assert result.repair_successes == 1
    assert result.ok


def test_self_correction_off_stops_at_the_first_failure(sales, tmp_path):
    agent = make_agent([
        turn("run_python", code=BAD_CODE),
        turn("run_python", code=GOOD_CODE),
        turn("final_answer", answer="North leads.", supporting_values=["North"]),
    ], sales, tmp_path, self_correction=False)
    result = agent.ask("Which region earns most?")
    assert not result.ok
    assert result.stop_reason == "failed_without_repair"
    assert result.steps == 1
    assert result.repair_attempts == 0


def test_guard_violation_counts_and_is_repairable(sales, tmp_path):
    agent = make_agent([
        turn("run_python", code="import os\nresult = os.listdir('.')"),
        turn("run_python", code=GOOD_CODE),
        turn("final_answer", answer="North leads.", supporting_values=["North"]),
    ], sales, tmp_path)
    result = agent.ask("Which region earns most?")
    assert result.sandbox_violations == 1
    assert result.failure_kinds == ["guard_violation"]
    assert result.ok


def test_verify_rejects_an_unsupported_claim_and_asks_again(sales, tmp_path):
    agent = make_agent([
        turn("run_python", code=GOOD_CODE),
        turn("final_answer", answer="Atlantis leads with 999999.",
             supporting_values=["Atlantis", "999999"]),
        turn("final_answer", answer="North leads.", supporting_values=["North"]),
    ], sales, tmp_path)
    result = agent.ask("Which region earns most?")
    assert result.ok
    assert result.grounded
    assert result.answer == "North leads."
    trace = json.loads(open(result.trace_path).read())
    verifies = [s for s in trace["steps"] if s["type"] == "verify"]
    assert len(verifies) == 2
    assert verifies[0]["report"]["unsupported"] == ["Atlantis", "999999"]


def test_a_persistently_unsupported_claim_is_returned_but_flagged(sales, tmp_path):
    agent = make_agent([
        turn("run_python", code=GOOD_CODE),
        turn("final_answer", answer="South leads.", supporting_values=["999999"]),
        turn("final_answer", answer="South leads.", supporting_values=["999999"]),
    ], sales, tmp_path)
    result = agent.ask("Which region earns most?")
    assert result.stop_reason == "final_answer_ungrounded"
    assert not result.grounded
    assert result.verify.unsupported == ["999999"]


def test_max_steps_is_bounded(sales, tmp_path):
    agent = make_agent([turn("run_python", code="result = 1")] * 10, sales, tmp_path,
                       max_steps=3)
    result = agent.ask("loop forever?")
    assert result.steps == 3
    assert result.stop_reason == "max_steps"
    assert not result.ok


def test_cost_budget_stops_the_run(sales, tmp_path):
    agent = make_agent([turn("run_python", code="result = 1")] * 10, sales, tmp_path,
                       cost_budget_usd=0.0001)
    result = agent.ask("expensive?")
    assert result.stop_reason == "cost_budget_exceeded"
    assert result.steps == 1


def test_bad_tool_arguments_become_an_observation_not_a_crash(sales, tmp_path):
    agent = make_agent([
        turn("run_python", nonsense="x"),
        turn("run_python", code=GOOD_CODE),
        turn("final_answer", answer="North leads.", supporting_values=["North"]),
    ], sales, tmp_path)
    result = agent.ask("Which region earns most?")
    assert result.ok
    assert "bad_arguments" in result.failure_kinds


def test_a_turn_with_no_tool_call_is_nudged_once_then_ends(sales, tmp_path):
    agent = make_agent([turn(text="I think it is North."), turn(text="Still North.")],
                       sales, tmp_path)
    result = agent.ask("Which region earns most?")
    assert result.stop_reason == "no_tool_call"
    assert not result.ok


def test_repair_budget_is_bounded(sales, tmp_path):
    agent = make_agent([turn("run_python", code=BAD_CODE)] * 6, sales, tmp_path, max_repairs=2)
    result = agent.ask("Which region earns most?")
    assert result.stop_reason == "repair_budget_exhausted"
    assert result.repair_attempts <= 2
