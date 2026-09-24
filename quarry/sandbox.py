"""Sandbox - runs code safely with AST guard + subprocess isolation."""

import ast
import os
import shutil
import subprocess
import sys
import tempfile

# Only these imports are allowed
ALLOWED_IMPORTS = {
    "pandas",
    "numpy",
    "matplotlib",
    "json",
    "math",
    "datetime",
    "re",
    "statistics",
}

# These functions are always banned
BANNED_NAMES = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "getattr",
    "globals",
    "locals",
}

# Cap captured output so a print('x'*1e9) loop cannot exhaust memory
MAX_OUTPUT_CHARS = 20_000

# Only plain CSV files may be loaded as datasets
ALLOWED_DATASET_SUFFIXES = {".csv"}


def check_code(code: str) -> tuple:
    """Check code using AST. Returns (is_safe, error_message)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax_error: {e}"

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return False, (
                        f"guard_violation: import '{alias.name}' not allowed. Allowed: {sorted(ALLOWED_IMPORTS)}"
                    )

        if isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return False, (
                        f"guard_violation: from '{node.module}' import not allowed. Allowed: {sorted(ALLOWED_IMPORTS)}"
                    )

        # Check banned function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BANNED_NAMES:
                    return False, f"guard_violation: '{node.func.id}' is not allowed"

        # Check dunder attribute access (like __class__, __globals__)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return False, (f"guard_violation: dunder attribute '{node.attr}' is not allowed")

    return True, ""


def check_dataset(dataset_path: str) -> tuple:
    """Validate the dataset path. Returns (is_valid, error_message)."""
    if not dataset_path or "\x00" in dataset_path:
        return False, "guard_violation: invalid dataset path"

    # resolve() collapses '..' - reject anything that escaped via traversal
    resolved = os.path.realpath(dataset_path)
    if not os.path.isfile(resolved):
        return False, f"guard_violation: dataset not found: {dataset_path}"

    suffix = os.path.splitext(resolved)[1].lower()
    if suffix not in ALLOWED_DATASET_SUFFIXES:
        return False, (f"guard_violation: dataset must be one of {sorted(ALLOWED_DATASET_SUFFIXES)}, got '{suffix}'")

    return True, ""


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n... [truncated at {MAX_OUTPUT_CHARS} chars]"


def run_code(code: str, dataset_path: str, timeout: int = 30, artifacts_dir: str = ".") -> dict:
    """
    Run code in an isolated subprocess.
    Returns: {"success": bool, "output": str, "error": str, "artifacts": list}
    """
    # Step 1: AST security check
    is_safe, error = check_code(code)
    if not is_safe:
        return {"success": False, "output": "", "error": error, "artifacts": []}

    # Step 1b: Dataset path validation
    ds_ok, ds_error = check_dataset(dataset_path)
    if not ds_ok:
        return {"success": False, "output": "", "error": ds_error, "artifacts": []}

    # Step 2: Create isolated working directory
    with tempfile.TemporaryDirectory() as workdir:
        # Copy dataset into workdir so code can read it
        shutil.copy(os.path.realpath(dataset_path), os.path.join(workdir, "data.csv"))

        # Step 3: Build full script (auto-load pandas as 'df')
        full_script = f"""
import pandas as pd
import numpy as np
import os

# Auto-load dataset
if os.path.exists('data.csv'):
    df = pd.read_csv('data.csv')

{code}
"""

        # Step 4: Run in isolated subprocess
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", full_script],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                # No API keys leak; MPLBACKEND=Agg lets charts run headless
                env={"MPLBACKEND": "Agg"},
            )

            # Extract chart artifacts (*.png) before temp dir is deleted
            artifacts = []
            for fname in os.listdir(workdir):
                if fname.endswith(".png"):
                    shutil.copy(
                        os.path.join(workdir, fname),
                        os.path.join(artifacts_dir, fname),
                    )
                    artifacts.append(fname)

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": _truncate(result.stdout),
                    "error": "",
                    "artifacts": artifacts,
                }
            else:
                # Get last 5 lines of traceback (most useful)
                lines = result.stderr.strip().split("\n")
                short_error = "\n".join(lines[-5:])
                return {
                    "success": False,
                    "output": _truncate(result.stdout),
                    "error": _truncate(short_error),
                    "artifacts": artifacts,
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"timeout: Code took too long ({timeout}s limit)",
                "artifacts": [],
            }
