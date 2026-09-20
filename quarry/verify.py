"""Groundedness check for a final answer.

The model is required to list the computed values its answer rests on. This module
checks each one against the text that actually came out of the sandbox during the
run. It is deliberately mechanical: no second model opinion, just string and
numeric matching against real execution output.

A value counts as supported if either
  - it appears verbatim in the observed output (after light normalisation), or
  - it parses as a number and some number in the output either equals it within a
    small relative tolerance, or rounds to it at the precision the model wrote.

The second rule is what makes this usable: reporting 0.4471 for a computed
0.44705882 is honest rounding, not a fabricated number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
REL_TOL = 1e-6


@dataclass
class VerifyReport:
    grounded: bool
    checked: int = 0
    supported: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "grounded": self.grounded,
            "checked": self.checked,
            "supported": self.supported,
            "unsupported": self.unsupported,
        }


def _parse_number(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def _decimals(token: str) -> int:
    token = token.replace(",", "")
    if "." not in token or "e" in token.lower():
        return 0
    return len(token.split(".", 1)[1])


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def value_is_supported(value: str, evidence: str, evidence_numbers: list[float] | None = None) -> bool:
    value = str(value).strip()
    if not value:
        return False
    norm_value = _normalise(value)
    norm_evidence = _normalise(evidence)
    if norm_value and norm_value in norm_evidence:
        return True

    tokens = NUMBER_RE.findall(value)
    if len(tokens) != 1 or _normalise(NUMBER_RE.sub("", value).replace("%", "")).strip(" .,:;()"):
        # Not a bare number (e.g. "North" or "3 of 4 regions"): verbatim match only.
        return False
    claimed = _parse_number(tokens[0])
    if claimed is None:
        return False
    if evidence_numbers is None:
        evidence_numbers = extract_numbers(evidence)
    places = _decimals(tokens[0])
    half_ulp = 0.5 * (10 ** -places) * 1.000001 if places else 0.5000001
    for actual in evidence_numbers:
        if abs(actual - claimed) <= max(abs(claimed) * REL_TOL, 1e-12):
            return True
        if abs(actual - claimed) <= half_ulp:
            return True
        # Percentages written against a proportion, and vice versa.
        for scaled in (actual * 100.0, actual / 100.0):
            if abs(scaled - claimed) <= max(abs(claimed) * REL_TOL, half_ulp):
                return True
    return False


def extract_numbers(evidence: str) -> list[float]:
    out = []
    for token in NUMBER_RE.findall(evidence):
        parsed = _parse_number(token)
        if parsed is not None:
            out.append(parsed)
    return out


def verify_answer(supporting_values: list[str], evidence: str) -> VerifyReport:
    numbers = extract_numbers(evidence)
    supported, unsupported = [], []
    for value in supporting_values:
        (supported if value_is_supported(value, evidence, numbers) else unsupported).append(str(value))
    return VerifyReport(
        grounded=not unsupported,
        checked=len(supporting_values),
        supported=supported,
        unsupported=unsupported,
    )
