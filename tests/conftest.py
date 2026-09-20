import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def small_df() -> pd.DataFrame:
    return pd.DataFrame({
        "region": ["North", "South", "North", "East"],
        "revenue": [10.0, 4.0, 6.0, 8.0],
        "units": [1, 2, 3, 4],
    })


@pytest.fixture
def workdir(tmp_path) -> Path:
    d = tmp_path / "workdir"
    d.mkdir()
    return d


@pytest.fixture
def df_path(workdir, small_df) -> Path:
    path = workdir / "df.pkl"
    small_df.to_pickle(path)
    return path
