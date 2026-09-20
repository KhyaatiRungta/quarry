"""Generate the three bundled demo datasets. Deterministic: seeded, no network.

Run: python scripts/make_data.py
The generated CSVs are committed, so this only needs re-running if the shape changes.
"""
import csv
import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def write(name, header, rows):
    DATA.mkdir(exist_ok=True)
    with open(DATA / name, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def make_passengers(n=340):
    rng = random.Random(11)
    first = ["Ada", "Owen", "Mira", "Jonas", "Lena", "Piet", "Rosa", "Caleb", "Nina", "Hugo",
             "Elsie", "Tomas", "Clara", "Bram", "Iris", "Otto", "Maeve", "Gus", "Sofia", "Emil"]
    last = ["Halloran", "Vestergaard", "Nakamura", "Okonkwo", "Beaumont", "Kowalski", "Ferreira",
            "Lindqvist", "Ashworth", "Dvorak", "Mancini", "Bergstrom", "Toussaint", "Rowan"]
    ports = ["Southampton", "Cherbourg", "Queenstown"]
    rows = []
    for i in range(1, n + 1):
        pclass = rng.choices([1, 2, 3], weights=[0.22, 0.24, 0.54])[0]
        sex = rng.choice(["female", "male"])
        age = None if rng.random() < 0.18 else round(max(0.5, rng.gauss(38 - 6 * pclass, 14)), 1)
        fare = round(abs(rng.gauss({1: 82, 2: 22, 3: 13}[pclass], {1: 40, 2: 10, 3: 6}[pclass])) + 3, 2)
        sibsp = rng.choices([0, 1, 2, 3], weights=[0.66, 0.24, 0.07, 0.03])[0]
        parch = rng.choices([0, 1, 2], weights=[0.74, 0.18, 0.08])[0]
        base = 0.14 + (0.34 if sex == "female" else 0.0) + {1: 0.32, 2: 0.16, 3: 0.0}[pclass]
        if age is not None and age < 12:
            base += 0.18
        survived = 1 if rng.random() < min(base, 0.95) else 0
        port = None if rng.random() < 0.02 else rng.choices(ports, weights=[0.7, 0.2, 0.1])[0]
        cabin = f"{rng.choice('ABCDE')}{rng.randint(1, 120)}" if pclass == 1 and rng.random() < 0.7 else None
        rows.append([
            i,
            f"{rng.choice(first)} {rng.choice(last)}",
            pclass, sex,
            "" if age is None else age,
            sibsp, parch, fare,
            "" if port is None else port,
            "" if cabin is None else cabin,
            survived,
        ])
    write("passengers.csv",
          ["passenger_id", "name", "pclass", "sex", "age", "sibsp", "parch", "fare",
           "embark_port", "cabin", "survived"], rows)


def make_sales():
    rng = random.Random(23)
    regions = ["North", "South", "East", "West"]
    products = [("Quarry Core", 480.0), ("Quarry Lite", 120.0), ("Bedrock Add-on", 260.0),
                ("Seam Analytics", 750.0)]
    channels = ["direct", "partner", "self-serve"]
    rows = []
    start = date(2024, 1, 1)
    oid = 100000
    for day_ix in range((date(2025, 12, 31) - start).days + 1):
        d = start + timedelta(days=day_ix)
        seasonal = 1.0 + 0.28 * math.sin((day_ix / 365.0) * 2 * math.pi)
        weekend = 0.45 if d.weekday() >= 5 else 1.0
        trend = 1.0 + day_ix / 1400.0
        n_orders = max(0, int(rng.gauss(3.1 * seasonal * weekend * trend, 1.2)))
        for _ in range(n_orders):
            oid += 1
            prod, price = rng.choice(products)
            region = rng.choices(regions, weights=[0.32, 0.20, 0.27, 0.21])[0]
            channel = rng.choices(channels, weights=[0.44, 0.26, 0.30])[0]
            units = rng.choices([1, 2, 3, 5, 10], weights=[0.5, 0.22, 0.14, 0.09, 0.05])[0]
            disc = {"direct": 0.0, "partner": 0.15, "self-serve": 0.05}[channel]
            unit_price = round(price * (1 - disc), 2)
            rows.append([oid, d.isoformat(), region, prod, channel, units, unit_price,
                         round(units * unit_price, 2)])
    write("sales.csv",
          ["order_id", "order_date", "region", "product", "channel", "units", "unit_price", "revenue"],
          rows)


def make_sensors(n=600):
    """Deliberately messy: mixed timestamp formats, numbers stored as strings with units,
    sentinel nulls, inconsistent categorical casing, and an ambiguous column pair."""
    rng = random.Random(37)
    stations = ["ALPHA-1", "alpha-1", "BRAVO-2", "bravo-2", "CHARLIE-3", "DELTA-4"]
    statuses = ["OK", "ok", "Degraded", "DEGRADED", "offline", "Offline"]
    rows = []
    t = datetime(2025, 3, 1, 0, 0)
    for i in range(1, n + 1):
        t = t + timedelta(minutes=rng.randint(20, 90))
        if rng.random() < 0.3:
            ts = t.strftime("%d/%m/%Y %H:%M")
        elif rng.random() < 0.5:
            ts = t.isoformat(sep=" ")
        else:
            ts = t.strftime("%Y-%m-%dT%H:%M:%S")
        temp = rng.gauss(17.5, 6.5)
        if rng.random() < 0.10:
            temp_c = ""
        elif rng.random() < 0.10:
            temp_c = "N/A"
        elif rng.random() < 0.18:
            temp_c = f"{temp:.1f} C"
        else:
            temp_c = f"{temp:.1f}"
        hum = rng.uniform(21, 96)
        humidity = f"{hum:.0f}%" if rng.random() < 0.45 else f"{hum:.1f}"
        if rng.random() < 0.06:
            humidity = "-999"
        batt = rng.uniform(4, 100)
        rows.append([
            i,
            rng.choice(stations),
            ts,
            temp_c,
            humidity,
            rng.choice(statuses),
            f"{batt:.1f}",
            rng.choice(["indoor", "outdoor", "Outdoor", ""]),
        ])
    write("sensors.csv",
          ["reading_id", "station", "recorded_at", "temp_c", "humidity", "status",
           "battery_pct", "placement"], rows)


if __name__ == "__main__":
    make_passengers()
    make_sales()
    make_sensors()
