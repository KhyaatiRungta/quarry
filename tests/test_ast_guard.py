"""The AST guard is the security-relevant part, so it gets the widest test.

Each forbidden construct is asserted individually: a regression that quietly
allows one of these through is the failure mode that matters.
"""
import pytest

from quarry.sandbox import guard_code

FORBIDDEN = [
    ("import os", "blocked-import"),
    ("import sys", "blocked-import"),
    ("import subprocess", "blocked-import"),
    ("import socket", "blocked-import"),
    ("import shutil", "blocked-import"),
    ("import os.path", "blocked-import"),
    ("import os as o", "blocked-import"),
    ("from os import system", "blocked-import"),
    ("from subprocess import run", "blocked-import"),
    ("import requests", "blocked-import"),
    ("import pickle", "blocked-import"),
    ("import ctypes", "blocked-import"),
    ("from . import thing", "relative-import"),
    ("import scipy", "import-not-allowed"),
    ("import sklearn", "import-not-allowed"),
    ("result = eval('1+1')", "forbidden-name"),
    ("exec('x = 1')", "forbidden-name"),
    ("__import__('os')", "forbidden-name"),
    ("result = compile('1', '<s>', 'eval')", "forbidden-name"),
    ("result = globals()", "forbidden-name"),
    ("result = locals()", "forbidden-name"),
    ("result = getattr(df, 'shape')", "forbidden-name"),
    ("result = input('give me')", "forbidden-name"),
    ("result = df.__class__", "dunder-attribute"),
    ("result = df.__class__.__bases__", "dunder-attribute"),
    ("result = ().__class__.__mro__[1]", "dunder-attribute"),
    ("result = __builtins__", "forbidden-name"),
    ("open('/etc/passwd')", "open-escape"),
    ("open('../secrets.txt')", "open-escape"),
    ("open('~/notes.txt')", "open-escape"),
    ("open('out.csv', 'w')", "open-write"),
    ("open('out.csv', mode='a')", "open-write"),
    ("f = open(some_variable)", "open-unsafe"),
    ("df.to_csv(open('x.csv','w'))", "open-write"),
]


@pytest.mark.parametrize("code,expected_code", FORBIDDEN)
def test_guard_rejects(code, expected_code):
    report = guard_code(code)
    assert not report.ok, f"guard allowed: {code!r}"
    assert expected_code in [v.code for v in report.violations], \
        f"{code!r} gave {[v.code for v in report.violations]}, expected {expected_code}"


ALLOWED = [
    "result = df['revenue'].sum()",
    "import pandas as pd\nresult = pd.Series([1, 2]).mean()",
    "import numpy as np\nresult = np.mean(df['units'])",
    "import matplotlib.pyplot as plt\nplt.plot([1,2])\nplt.savefig('chart.png')\nresult = 1",
    "import json, math, re, statistics\nresult = math.sqrt(4)",
    "from datetime import datetime\nresult = str(datetime(2024,1,1))",
    "notes = 'import os is not allowed'\nresult = len(notes)",
    "result = open('notes.txt').read()",
    "result = df.query('revenue > 5').shape[0]",
    "for i in range(3):\n    print(i)\nresult = i",
]


@pytest.mark.parametrize("code", ALLOWED)
def test_guard_allows(code):
    report = guard_code(code)
    assert report.ok, f"guard rejected legitimate code {code!r}: {report.violations}"


def test_guard_reports_syntax_error_without_crashing():
    report = guard_code("result = df[")
    assert not report.ok
    assert report.syntax_error
    assert "did not parse" in report.feedback()


def test_guard_feedback_names_the_rule_and_the_allowlist():
    feedback = guard_code("import os").feedback()
    assert "os" in feedback
    assert "pandas" in feedback


def test_guard_collects_every_violation_not_just_the_first():
    report = guard_code("import os\nimport socket\nresult = eval('1')")
    assert len(report.violations) == 3
