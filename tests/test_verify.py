import pytest

from quarry.verify import verify_answer

EVIDENCE = """
result (float64):
np.float64(0.44705882352941179)
region      revenue
North      588920.0
East       460355.0
rows 600
usable 487
"""


@pytest.mark.parametrize("value", [
    "0.44705882352941179",   # verbatim
    "0.4471",                # honest rounding
    "0.447",
    "44.7",                  # proportion reported as a percentage
    "North",
    "588920.0",
    "588,920.0",             # thousands separator
    "487",
])
def test_supported_values_pass(value):
    assert verify_answer([value], EVIDENCE).grounded


@pytest.mark.parametrize("value", [
    "0.51",          # plausible but never computed
    "0.4482",        # close, still not a rounding of anything present
    "700000",
    "West",          # a real category that this run did not compute
    "Southampton",
])
def test_unsupported_values_are_caught(value):
    report = verify_answer([value], EVIDENCE)
    assert not report.grounded
    assert report.unsupported == [value]


def test_report_separates_supported_from_unsupported():
    report = verify_answer(["North", "0.4471", "1234567"], EVIDENCE)
    assert not report.grounded
    assert report.supported == ["North", "0.4471"]
    assert report.unsupported == ["1234567"]
    assert report.checked == 3


def test_empty_evidence_supports_nothing():
    assert not verify_answer(["1"], "").grounded


def test_a_claim_with_no_values_is_vacuously_grounded():
    assert verify_answer([], EVIDENCE).grounded
