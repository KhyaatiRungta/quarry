"""Verify gate - checks answers against real execution output."""

import re


def verify_answer(answer: str, supporting_values: list, execution_output: str) -> tuple:
    """
    Check that every supporting value actually appears in real output.
    Returns (is_valid, message).
    """
    if not supporting_values:
        return True, "No supporting values to check"

    for value in supporting_values:
        # Method 1: Exact string match (normalized)
        if _normalize(value) in _normalize(execution_output):
            continue

        # Method 2: Numeric match with tolerance
        if _numeric_match(value, execution_output):
            continue

        # FAILED: value not found in real output
        return False, (f"Value '{value}' not found in execution output. The model may be making it up.")

    return True, "All values verified"


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _numeric_match(value_str: str, text: str) -> bool:
    """Check if a numeric value appears within 1% tolerance."""
    try:
        cleaned = re.sub(r"[^\d.\-]", "", str(value_str))
        target = float(cleaned)
    except (ValueError, TypeError):
        return False

    # Find all numbers in the text
    numbers = re.findall(r"-?\d+\.?\d*", text)

    for num_str in numbers:
        try:
            num = float(num_str)
            # Within 1% relative tolerance
            if abs(num - target) / max(abs(target), 1) < 0.01:
                return True
        except ValueError:
            continue

    return False
