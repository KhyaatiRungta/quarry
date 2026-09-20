"""Dataset loading and schema inspection.

One table at a time, by design (see Limitations). A CSV is loaded into pandas and
also mirrored into an in-memory SQLite table so both the pandas and the SQL tool
paths exist against identical data. A .sqlite/.db file is read the other way round.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MAX_SAMPLE_ROWS = 3
MAX_CATEGORY_EXAMPLES = 6
CARDINALITY_EXAMPLE_LIMIT = 25


@dataclass
class Dataset:
    name: str
    table: str
    df: pd.DataFrame
    source_path: str
    source_kind: str  # csv | sqlite

    @classmethod
    def load(cls, path: str | Path, table: str | None = None) -> "Dataset":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such dataset: {path}")
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            # Everything is read as-is. Messy columns stay messy on purpose: coercing
            # them here would hide exactly the failures the repair loop exists for.
            df = pd.read_csv(path, sep=sep)
            return cls(path.stem, table or path.stem, df, str(path), "csv")
        if suffix in {".sqlite", ".db", ".sqlite3"}:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                tables = [r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
                if not tables:
                    raise ValueError(f"{path} contains no tables.")
                chosen = table or tables[0]
                if chosen not in tables:
                    raise ValueError(f"Table {chosen!r} not in {path}. Available: {', '.join(tables)}")
                df = pd.read_sql_query(f'SELECT * FROM "{chosen}"', conn)
            return cls(path.stem, chosen, df, str(path), "sqlite")
        raise ValueError(f"Unsupported dataset type {suffix!r}. Use .csv or .sqlite.")

    def sqlite_connection(self) -> sqlite3.Connection:
        """A fresh in-memory connection holding this dataset as a single table."""
        conn = sqlite3.connect(":memory:")
        self.df.to_sql(self.table, conn, index=False, if_exists="replace")
        return conn

    def schema(self) -> dict:
        df = self.df
        columns = []
        for col in df.columns:
            series = df[col]
            nulls = int(series.isna().sum())
            nunique = int(series.nunique(dropna=True))
            entry = {
                "name": str(col),
                "dtype": str(series.dtype),
                "null_count": nulls,
                "null_pct": round(100.0 * nulls / max(len(df), 1), 1),
                "distinct": nunique,
            }
            if nunique <= CARDINALITY_EXAMPLE_LIMIT:
                values = series.dropna().unique()[:MAX_CATEGORY_EXAMPLES]
                entry["values"] = [str(v) for v in values]
            else:
                entry["examples"] = [str(v) for v in series.dropna().head(3).tolist()]
            if pd.api.types.is_numeric_dtype(series) and nunique > 1:
                entry["min"] = _clean(series.min())
                entry["max"] = _clean(series.max())
            columns.append(entry)
        return {
            "dataset": self.name,
            "table": self.table,
            "source": self.source_path,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": columns,
            "sample_rows": df.head(MAX_SAMPLE_ROWS).astype(str).to_dict(orient="records"),
        }

    def schema_text(self) -> str:
        """Compact plain-text rendering of the schema for the system prompt.

        Text, not JSON: it is shorter for the same information, and column notes
        such as `stored as text` read as instructions rather than as data.
        """
        s = self.schema()
        lines = [
            f"TABLE {s['table']}  ({s['row_count']} rows, {s['column_count']} columns)",
            f"source: {Path(s['source']).name}",
            "",
            "columns:",
        ]
        for c in s["columns"]:
            bits = [f"  {c['name']}  {c['dtype']}"]
            if c["null_count"]:
                bits.append(f"nulls={c['null_count']} ({c['null_pct']}%)")
            bits.append(f"distinct={c['distinct']}")
            if "values" in c:
                bits.append("values=[" + ", ".join(repr(v) for v in c["values"]) + "]")
            elif "examples" in c:
                bits.append("e.g. " + ", ".join(repr(v) for v in c["examples"]))
            if "min" in c:
                bits.append(f"range={c['min']}..{c['max']}")
            lines.append("  ".join(bits))
        lines.append("")
        lines.append("first rows:")
        for row in s["sample_rows"]:
            lines.append("  " + " | ".join(f"{k}={v!r}" for k, v in row.items()))
        return "\n".join(lines)


def _clean(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return int(f) if f == int(f) and abs(f) < 1e15 else round(f, 4)
