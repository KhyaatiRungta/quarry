import pytest

from quarry.dataset import Dataset
from quarry.tools import AnalysisSession, ToolArgumentError, build_tools


@pytest.fixture
def session(tmp_path, repo_root):
    s = AnalysisSession(Dataset.load(repo_root / "data/sales.csv"), tmp_path / "run")
    yield s
    s.close()


@pytest.fixture
def tools(session):
    return build_tools(session)


def test_every_tool_has_a_schema_with_a_description(tools):
    for schema in tools.schemas():
        fn = schema["function"]
        assert schema["type"] == "function"
        assert fn["description"].strip()
        assert fn["parameters"]["type"] == "object"
        assert fn["parameters"]["additionalProperties"] is False


def test_expected_tools_exist(tools):
    assert set(tools.names) == {
        "inspect_schema", "run_python", "run_sql", "make_chart", "final_answer"}


def test_inspect_schema_reports_columns_nulls_and_samples(tools):
    result = tools.call("inspect_schema", {})
    assert result.ok
    assert "order_date" in result.observation
    assert "first rows:" in result.observation
    assert result.payload["row_count"] == 2065


@pytest.mark.parametrize("name,args,needle", [
    ("run_python", {}, "missing required argument"),
    ("run_python", {"code": "result = 1", "cod": "x"}, "unknown argument"),
    ("run_python", {"code": 42}, "must be of type string"),
    ("run_python", {"code": "   "}, "must not be empty"),
    ("run_sql", {"query": "SELECT 1", "limit": 5}, "unknown argument"),
    ("final_answer", {"answer": "a"}, "missing required argument"),
    ("final_answer", {"answer": "a", "supporting_values": []}, "at least one value"),
])
def test_bad_arguments_are_rejected(tools, name, args, needle):
    with pytest.raises(ToolArgumentError) as exc:
        tools.call(name, args)
    assert needle in str(exc.value)


def test_unknown_tool_is_rejected(tools):
    with pytest.raises(ToolArgumentError) as exc:
        tools.call("delete_everything", {})
    assert "Unknown tool" in str(exc.value)


def test_non_object_arguments_are_rejected(tools):
    with pytest.raises(ToolArgumentError):
        tools.call("run_python", ["code"])


def test_supporting_values_encoded_as_a_json_string_are_accepted(tools):
    result = tools.call("final_answer",
                        {"answer": "North", "supporting_values": '["North", "1.0"]'})
    assert result.terminal
    assert result.payload["supporting_values"] == ["North", "1.0"]


def test_run_python_executes_and_records_evidence(tools, session):
    result = tools.call("run_python", {"code": "result = round(float(df.revenue.sum()), 2)"})
    assert result.ok
    assert "1788730.0" in result.observation
    assert "1788730.0" in session.evidence()


def test_failed_execution_is_not_recorded_as_evidence(tools, session):
    result = tools.call("run_python", {"code": "result = df['nope'].sum()"})
    assert not result.ok
    assert "KeyError" in result.observation
    assert "nope" not in session.evidence()


def test_run_sql_returns_rows(tools):
    result = tools.call("run_sql", {"query": "SELECT region, SUM(revenue) AS r FROM sales "
                                            "GROUP BY region ORDER BY r DESC"})
    assert result.ok
    assert result.payload["row_count"] == 4
    assert "North" in result.observation


@pytest.mark.parametrize("query", [
    "DROP TABLE sales",
    "DELETE FROM sales",
    "UPDATE sales SET revenue = 0",
    "INSERT INTO sales VALUES (1)",
    "PRAGMA table_info(sales)",
])
def test_run_sql_rejects_writes(tools, session, query):
    result = tools.call("run_sql", {"query": query})
    assert not result.ok
    assert session.sandbox_violations >= 1


def test_run_sql_rejects_stacked_statements(tools):
    result = tools.call("run_sql", {"query": "SELECT 1; DROP TABLE sales"})
    assert not result.ok
    assert "one statement" in result.observation


def test_run_sql_error_message_names_the_table(tools):
    result = tools.call("run_sql", {"query": "SELECT * FROM nope"})
    assert not result.ok
    assert "sales" in result.observation


def test_make_chart_requires_a_saved_figure(tools):
    result = tools.call("make_chart", {"code": "result = 1"})
    assert not result.ok
    assert "chart.png" in result.observation


def test_make_chart_writes_a_chart(tools, session):
    code = ("import matplotlib.pyplot as plt\n"
            "df.groupby('region').revenue.sum().plot(kind='bar')\n"
            "plt.savefig('chart.png')\nresult = 'ok'")
    result = tools.call("make_chart", {"code": code})
    assert result.ok
    assert session.chart_path and session.chart_path.endswith("chart.png")
