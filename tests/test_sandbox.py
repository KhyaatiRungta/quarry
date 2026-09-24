"""Tests for the sandbox: AST guard + real subprocess execution."""

from quarry.sandbox import check_code, run_code

DATA = "data/sales.csv"


def test_blocks_import_os():
    ok, err = check_code("import os")
    assert not ok
    assert "guard_violation" in err


def test_blocks_from_import():
    ok, err = check_code("from os import system")
    assert not ok
    assert "guard_violation" in err


def test_blocks_eval():
    ok, err = check_code("x = eval('1')")
    assert not ok
    assert "eval" in err


def test_blocks_dunder_attribute():
    ok, err = check_code("x = ().__class__")
    assert not ok
    assert "dunder" in err


def test_allows_pandas():
    ok, err = check_code("import pandas as pd\nprint(df.head())")
    assert ok, err


def test_syntax_error_rejected():
    ok, err = check_code("def broken(:")
    assert not ok
    assert "syntax_error" in err


def test_real_execution_succeeds():
    result = run_code("print(int(df['Revenue'].sum()))", DATA)
    assert result["success"], result["error"]
    assert "2010000" in result["output"]


def test_crash_reports_error():
    result = run_code("df.groupby('region')", DATA)
    assert not result["success"]
    assert "KeyError" in result["error"]
