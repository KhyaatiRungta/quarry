"""Build quarry/bench/tasks.jsonl.

Every expected value below is COMPUTED here from the shipped CSVs by a
hand-written reference implementation, never typed in by hand and never produced
by the agent under test. Re-run this if the datasets change.

    python scripts/build_tasks.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "quarry" / "bench" / "tasks.jsonl"

passengers = pd.read_csv(ROOT / "data/passengers.csv")
sales = pd.read_csv(ROOT / "data/sales.csv")
sales["order_date"] = pd.to_datetime(sales["order_date"])
sensors = pd.read_csv(ROOT / "data/sensors.csv")


def clean_temp(frame: pd.DataFrame) -> pd.Series:
    raw = frame["temp_c"].astype("string").str.strip().str.replace(" C", "", regex=False)
    return pd.to_numeric(raw.replace({"N/A": None, "": None}), errors="coerce")


def clean_humidity(frame: pd.DataFrame) -> pd.Series:
    raw = frame["humidity"].astype("string").str.strip().str.rstrip("%")
    values = pd.to_numeric(raw, errors="coerce")
    return values.where(values > -900)


def station_norm(frame: pd.DataFrame) -> pd.Series:
    return frame["station"].str.upper().str.strip()


def parse_ts(frame: pd.DataFrame) -> pd.Series:
    iso = pd.to_datetime(frame["recorded_at"], format="ISO8601", errors="coerce")
    dmy = pd.to_datetime(frame["recorded_at"], format="%d/%m/%Y %H:%M", errors="coerce")
    return iso.fillna(dmy)


TASKS: list[dict] = []


def task(tid, dataset, question, difficulty, tags, *, value=None, tolerance=None,
         relative=True, predicate=None, rubric=None):
    entry = {
        "id": tid, "dataset": dataset, "question": question,
        "difficulty": difficulty, "tags": tags,
    }
    if rubric is not None:
        entry["graded_by"] = "judge"
        entry["rubric"] = rubric
    elif predicate is not None:
        entry["graded_by"] = "predicate"
        entry["expected_predicate"] = predicate
    else:
        entry["graded_by"] = "numeric"
        entry["expected_value"] = round(float(value), 6)
        entry["tolerance"] = tolerance if tolerance is not None else 0.005
        entry["relative_tolerance"] = relative
    TASKS.append(entry)


# --- passengers -----------------------------------------------------------
P = "data/passengers.csv"
task("pass-01", P, "How many passengers are in the table?", "easy", ["count"],
     value=len(passengers), tolerance=0, relative=False)
task("pass-02", P, "What fraction of passengers survived?", "easy", ["aggregation"],
     value=passengers.survived.mean())
task("pass-03", P, "What was the average fare paid by first class passengers?", "easy",
     ["groupby", "filter"], value=passengers.loc[passengers.pclass == 1, "fare"].mean())
task("pass-04", P, "How many passengers have no recorded age?", "easy", ["nulls"],
     value=int(passengers.age.isna().sum()), tolerance=0, relative=False)
task("pass-05", P, "What is the survival rate among female passengers?", "easy",
     ["groupby", "filter"],
     value=passengers.loc[passengers.sex == "female", "survived"].mean())
task("pass-06", P, "Which embarkation port did the most passengers leave from?", "easy",
     ["groupby", "categorical"],
     predicate={"type": "contains_all",
                "values": [str(passengers.embark_port.value_counts().idxmax())]})
task("pass-07", P, "What is the median age of passengers, ignoring missing ages?", "medium",
     ["nulls", "aggregation"], value=passengers.age.median())
task("pass-08", P, "How many passengers travelled with at least one sibling or spouse aboard?",
     "medium", ["filter", "count"], value=int((passengers.sibsp >= 1).sum()),
     tolerance=0, relative=False)
task("pass-09", P, "What is the survival rate of third class passengers under 18? "
     "Passengers with a missing age should be excluded.", "hard", ["nulls", "filter", "groupby"],
     value=passengers[(passengers.pclass == 3) & (passengers.age < 18)].survived.mean())
task("pass-10", P, "How much higher is the average fare of survivors than of non-survivors?",
     "medium", ["groupby", "arithmetic"],
     value=(passengers[passengers.survived == 1].fare.mean()
            - passengers[passengers.survived == 0].fare.mean()))
task("pass-11", P, "What is the Pearson correlation between fare and survival?", "medium",
     ["correlation"], value=passengers.fare.corr(passengers.survived), tolerance=0.02)
task("pass-12", P, "What share of first class passengers have a recorded cabin number?",
     "medium", ["nulls", "filter"],
     value=passengers.loc[passengers.pclass == 1, "cabin"].notna().mean())
task("pass-13", P, "Summarise how survival varied with passenger class and sex, and say which "
     "single factor separates survivors most sharply.", "hard", ["open-ended", "groupby"],
     rubric="Must report that first class survived at a much higher rate than classes 2 and 3 "
            "(roughly 0.65 vs roughly 0.38), must report that women survived at a far higher "
            "rate than men, and must conclude that sex is the sharper separator of the two.")

# --- sales ----------------------------------------------------------------
S = "data/sales.csv"
task("sales-01", S, "What is the total revenue across all orders?", "easy", ["aggregation"],
     value=sales.revenue.sum())
task("sales-02", S, "Which region has the highest total revenue?", "easy",
     ["groupby", "categorical"],
     predicate={"type": "contains_all",
                "values": [str(sales.groupby("region").revenue.sum().idxmax())]})
task("sales-03", S, "How many orders are in the table?", "easy", ["count"],
     value=len(sales), tolerance=0, relative=False)
task("sales-04", S, "Which product sold the most units in total?", "easy",
     ["groupby", "categorical"],
     predicate={"type": "contains_all",
                "values": [str(sales.groupby("product").units.sum().idxmax())]})
task("sales-05", S, "What is the average order value, in revenue per order?", "easy",
     ["aggregation"], value=sales.revenue.mean())
_2024 = sales[sales.order_date.dt.year == 2024].revenue.sum()
_2025 = sales[sales.order_date.dt.year == 2025].revenue.sum()
task("sales-06", S, "By what percentage did total revenue grow from 2024 to 2025?", "medium",
     ["time-series", "arithmetic"], value=100 * (_2025 - _2024) / _2024, tolerance=0.01)
_monthly = sales.set_index("order_date").revenue.resample("MS").sum()
task("sales-07", S, "Which calendar month had the highest total revenue? Give it as YYYY-MM.",
     "medium", ["time-series", "groupby"],
     predicate={"type": "contains_all", "values": [_monthly.idxmax().strftime("%Y-%m")]})
task("sales-08", S, "How many orders were placed in the fourth quarter of 2025?", "medium",
     ["time-series", "filter"],
     value=int(((sales.order_date >= "2025-10-01") & (sales.order_date <= "2025-12-31")).sum()),
     tolerance=0, relative=False)
task("sales-09", S, "What is the average order value on the partner channel?", "medium",
     ["groupby", "filter"], value=sales[sales.channel == "partner"].revenue.mean())
task("sales-10", S, "What percentage of total revenue comes from the Quarry Core product?",
     "medium", ["groupby", "arithmetic"],
     value=100 * sales[sales["product"] == "Quarry Core"].revenue.sum() / sales.revenue.sum())
_dow = sales.assign(dow=sales.order_date.dt.day_name()).groupby("dow").revenue.sum()
task("sales-11", S, "Which day of the week has the lowest total revenue?", "hard",
     ["time-series", "groupby"],
     predicate={"type": "contains_all", "values": [str(_dow.idxmin())]})
_direct = sales[sales.channel == "direct"].groupby("product").unit_price.mean()
_partner = sales[sales.channel == "partner"].groupby("product").unit_price.mean()
task("sales-12", S, "On average across products, what percentage discount does the partner "
     "channel get on unit price compared with the direct channel?", "hard",
     ["groupby", "join", "arithmetic"],
     value=float((100 * (1 - _partner / _direct)).mean()), tolerance=0.02)
task("sales-13", S, "How many distinct products are sold?", "easy", ["count", "categorical"],
     value=sales["product"].nunique(), tolerance=0, relative=False)
task("sales-14", S, "What is the total revenue in the first half of 2024?", "medium",
     ["time-series", "filter"],
     value=sales[(sales.order_date >= "2024-01-01") & (sales.order_date <= "2024-06-30")].revenue.sum())
task("sales-15", S, "Describe the shape of revenue over the two years: the trend, any "
     "seasonality, and the weekday effect.", "hard", ["open-ended", "time-series"],
     rubric="Must state that revenue grows over the two years, must identify a seasonal "
            "within-year cycle (a mid-year peak and a turn-of-year trough), and must note that "
            "weekend days take substantially less revenue than weekdays.")

# --- sensors --------------------------------------------------------------
N = "data/sensors.csv"
_temp = clean_temp(sensors)
_hum = clean_humidity(sensors)
_stn = station_norm(sensors)
_status = sensors.status.str.lower().str.strip()
_ts = parse_ts(sensors)
task("sens-01", N, "How many readings are in the table?", "easy", ["count"],
     value=len(sensors), tolerance=0, relative=False)
task("sens-02", N, "What is the mean temperature in degrees C across all readings?", "medium",
     ["messy", "type-coercion"], value=_temp.mean())
task("sens-03", N, "How many readings have a usable numeric temperature?", "medium",
     ["messy", "nulls"], value=int(_temp.notna().sum()), tolerance=0, relative=False)
task("sens-04", N, "How many distinct physical stations are there? Station codes are recorded "
     "inconsistently.", "medium", ["messy", "categorical"],
     value=int(_stn.nunique()), tolerance=0, relative=False)
task("sens-05", N, "How many readings have status OK? Status is recorded with inconsistent "
     "capitalisation.", "medium", ["messy", "categorical"],
     value=int((_status == "ok").sum()), tolerance=0, relative=False)
task("sens-06", N, "What is the mean humidity? Some values carry a percent sign and some rows "
     "use -999 as a missing marker.", "hard", ["messy", "type-coercion", "nulls"],
     value=_hum.mean())
task("sens-07", N, "What is the highest battery percentage recorded?", "easy", ["aggregation"],
     value=sensors.battery_pct.max())
task("sens-08", N, "How many readings have no recorded placement?", "easy", ["nulls"],
     value=int(sensors.placement.isna().sum()), tolerance=0, relative=False)
task("sens-09", N, "What proportion of readings are in a degraded state? Status casing is "
     "inconsistent.", "medium", ["messy", "categorical"],
     value=float((_status == "degraded").mean()))
task("sens-10", N, "Which station recorded the most readings, once station codes are normalised "
     "for capitalisation?", "hard", ["messy", "groupby"],
     predicate={"type": "contains_all", "values": [str(_stn.value_counts().idxmax()).split("-")[0]]})
task("sens-11", N, "What is the mean temperature for outdoor readings? Placement casing is "
     "inconsistent and temperature is stored as text.", "hard",
     ["messy", "type-coercion", "filter"],
     value=float(_temp[sensors.placement.str.lower().str.strip() == "outdoor"].mean()))
task("sens-12", N, "How many humidity readings use the -999 missing marker?", "medium",
     ["messy", "nulls"],
     value=int((pd.to_numeric(sensors.humidity.astype("string").str.rstrip("%"),
                              errors="coerce") <= -900).sum()),
     tolerance=0, relative=False)
task("sens-13", N, "What is the date of the earliest reading? Timestamps are stored in more than "
     "one format. Answer as YYYY-MM-DD.", "hard", ["messy", "time-series", "type-coercion"],
     predicate={"type": "contains_all", "values": [_ts.min().strftime("%Y-%m-%d")]})
task("sens-14", N, "Over how many whole days does the data span, from the earliest to the latest "
     "reading?", "hard", ["messy", "time-series"],
     value=float((_ts.max() - _ts.min()).days), tolerance=1, relative=False)
task("sens-15", N, "What are the main data quality problems in this table, and which columns are "
     "affected?", "hard", ["open-ended", "messy"],
     rubric="Must identify at least three of: temperature stored as text with a ' C' unit "
            "suffix and 'N/A' values, humidity carrying percent signs plus a -999 sentinel, "
            "station codes differing only by capitalisation, status values differing only by "
            "capitalisation, missing placement values, and timestamps in more than one format. "
            "Must name the affected columns.")

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as fh:
    for entry in TASKS:
        fh.write(json.dumps(entry) + "\n")

by_diff = {}
for t in TASKS:
    by_diff[t["difficulty"]] = by_diff.get(t["difficulty"], 0) + 1
print(f"{len(TASKS)} tasks -> {OUT}")
print("difficulty:", by_diff)
print("graded_by:", {k: sum(1 for t in TASKS if t["graded_by"] == k)
                     for k in ("numeric", "predicate", "judge")})
