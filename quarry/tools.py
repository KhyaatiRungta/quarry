"""Tool definitions - tells the model what it can do."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_schema",
            "description": "Get column names, dtypes, and sample rows of the dataset",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code in a sandbox. pandas is pre-loaded as 'df'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_chart",
            "description": "Create a matplotlib chart. Save it to chart.png.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code that creates a chart with plt.savefig('chart.png')",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Provide the final answer with supporting values from execution output",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete answer to the question",
                    },
                    "supporting_values": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exact values from execution output that support the answer",
                    },
                },
                "required": ["answer", "supporting_values"],
            },
        },
    },
]
