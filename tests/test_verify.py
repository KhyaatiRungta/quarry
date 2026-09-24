"""Tests for the verify gate."""

from quarry.verify import verify_answer


def test_value_present_passes():
    ok, msg = verify_answer("East wins", ["675000"], "East 675000\nNorth 530000")
    assert ok, msg


def test_value_absent_fails():
    ok, msg = verify_answer("Answer", ["999999"], "East 675000")
    assert not ok
    assert "999999" in msg


def test_numeric_tolerance():
    # within 1% relative tolerance
    ok, _ = verify_answer("a", ["675000"], "value: 675500")
    assert ok


def test_empty_supporting_values_pass():
    ok, _ = verify_answer("free-form answer", [], "no output")
    assert ok


def test_text_match_normalized():
    ok, _ = verify_answer("a", ["East"], "region EAST has top sales")
    assert ok
