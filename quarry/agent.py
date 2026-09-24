"""Agent - the ReAct loop that ties everything together."""

import json
import time

import pandas as pd

from quarry.llm import LLMClient
from quarry.sandbox import run_code
from quarry.tools import TOOLS
from quarry.verify import verify_answer

# Returned when the loop runs out of steps without a verified answer
MAX_STEPS_MSG = "Max steps reached without a final answer."

SYSTEM_PROMPT = """You are a data analyst agent.

RULES:
- Use run_python to analyze data. pandas is pre-loaded as 'df'.
- Call inspect_schema first if you need to check columns.
- If code fails, read the error and fix it.
- When done, call final_answer with your answer and supporting values.
- Only use values that appear in actual execution output. Never make up numbers.
- For charts, use make_chart and always call plt.savefig('chart.png')."""


class QuarryAgent:
    def __init__(self, dataset_path, max_steps=10, max_repairs=3, llm=None, artifacts_dir="."):
        self.client = llm or LLMClient()
        self.dataset_path = dataset_path
        self.max_steps = max_steps
        self.max_repairs = max_repairs
        self.artifacts_dir = artifacts_dir  # where chart PNGs are copied
        self.messages = []
        self.trace = []
        self.exec_log = []  # Every sandbox execution: {"tool": ..., "success": ...}
        self.all_output = ""
        self.repair_count = 0
        self.repair_attempts = 0

    def ask(self, question: str) -> str:
        """Main method: ask a question, get an answer."""
        # Build initial messages
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        outstanding_failure = None

        # THE REACT LOOP
        for step in range(1, self.max_steps + 1):
            # THINK: Ask model what to do
            response = self.client.chat(self.messages, tools=TOOLS)

            # Check if model called any tools
            if not response.tool_calls:
                # Model responded with plain text
                self.messages.append(response.model_dump())
                continue

            # Add model's response to history
            self.messages.append(response.model_dump())

            # ACT: Execute each tool call
            for tool_call in response.tool_calls:
                fn_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                # Record in trace (with latency for observability)
                trace_entry = {"step": step, "tool": fn_name, "args": args}
                self.trace.append(trace_entry)

                # Execute the tool
                started = time.perf_counter()
                observation = self._execute_tool(fn_name, args, outstanding_failure)
                trace_entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)

                # Track failures
                if "FAILED" in str(observation):
                    outstanding_failure = observation
                    self.repair_attempts += 1
                elif fn_name in ("run_python", "make_chart"):
                    # Success clears the outstanding failure
                    if outstanding_failure:
                        self.repair_count += 1
                        outstanding_failure = None

                # Check if final answer was returned
                if fn_name == "final_answer" and "VERIFY OK" in observation:
                    return args["answer"]

                # Feed observation back to model
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": observation,
                    }
                )

                # Check repair budget
                if self.repair_attempts >= self.max_repairs:
                    self.messages.append(
                        {
                            "role": "system",
                            "content": "Repair budget exceeded. Give your best final_answer now.",
                        }
                    )

        return MAX_STEPS_MSG

    def _execute_tool(self, fn_name, args, outstanding_failure) -> str:
        """Execute a single tool call."""

        if fn_name == "inspect_schema":
            return self._inspect_schema()

        elif fn_name == "run_python":
            result = run_code(args.get("code", ""), self.dataset_path, artifacts_dir=self.artifacts_dir)
            self.exec_log.append({"tool": "run_python", "success": result["success"]})
            if result["success"]:
                self.all_output += result["output"] + "\n"
                # Update trace with output
                self.trace[-1]["output"] = result["output"]
                return f"OUTPUT:\n{result['output']}"
            else:
                # Failure - include repair budget info
                budget_left = self.max_repairs - self.repair_attempts
                if budget_left <= 0:
                    return f"FAILED: {result['error']}\nBudget exceeded."
                return f"FAILED: {result['error']}\nFix the code and try again. ({budget_left} repairs left)"

        elif fn_name == "make_chart":
            result = run_code(args.get("code", ""), self.dataset_path, artifacts_dir=self.artifacts_dir)
            self.exec_log.append({"tool": "make_chart", "success": result["success"]})
            if result["success"]:
                self.all_output += result["output"] + "\n"
                return f"Chart saved.\n{result['output']}"
            else:
                return f"FAILED: {result['error']}"

        elif fn_name == "final_answer":
            # VERIFY GATE
            is_valid, message = verify_answer(
                args.get("answer", ""),
                args.get("supporting_values", []),
                self.all_output,
            )
            if is_valid:
                return f"VERIFY OK: {message}"
            else:
                return f"VERIFY FAILED: {message}\nRecompute the value and try again."

        return f"Unknown tool: {fn_name}"

    def _inspect_schema(self) -> str:
        """Get dataset schema."""
        df = pd.read_csv(self.dataset_path)
        info = f"Columns: {list(df.columns)}\n"
        info += f"Dtypes:\n{df.dtypes.to_string()}\n"
        info += f"Shape: {df.shape}\n"
        info += f"Sample:\n{df.head().to_string()}"
        return info
