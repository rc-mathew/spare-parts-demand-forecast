"""Synthetic stand-in with the same 7-column schema as project_service_data.service_data.
For pipeline smoke tests only. Never report results from it.
"""
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
# part: (base weight, peak months)
PARTS = {
    "ENGINE OIL 15W40": (10, []), "OIL FILTER": (9, []), "AIR FILTER ELEMENT": (6, [3, 4, 5]),
    "FUEL FILTER": (5, []), "BRAKE PAD SET FRONT": (5, [7, 8, 9]), "BRAKE SHOE REAR": (3, [7, 8, 9]),
    "CLUTCH PLATE": (2, []), "WIPER BLADE": (3, [6, 7, 8, 9]), "BATTERY 12V": (2, [12, 1, 2]),
    "AC GAS REFILL": (3, [3, 4, 5, 6]), "COOLANT 5L": (3, [3, 4, 5]), "TYRE RETREAD": (2, []),
    "HEADLAMP BULB": (2, [11, 12, 1]), "FAN BELT": (2, []), "LABOUR GENERAL SERVICE": (8, []),
}
MODELS = ["Tata Ace", "Ashok Leyland Dost", "Mahindra Bolero Pickup", "Eicher Pro 2049", "Tata 407"]
PARTNERS = [f"Fleet Operator {i}" for i in range(1, 9)]


def make(n_weeks=156, lines_per_week=182, start="2022-01-03"):
    names = list(PARTS)
    veh = pd.DataFrame({
        "no": [f"KA{rng.integers(1, 60):02d}AB{i:04d}" for i in range(1200)],
        "model": rng.choice(MODELS, 1200), "partner": rng.choice(PARTNERS, 1200),
        "km0": rng.integers(5_000, 150_000, 1200),
    })
    rows = []
    for w in range(n_weeks):
        monday = pd.Timestamp(start) + pd.Timedelta(weeks=w)
        wt = np.array([PARTS[n][0] * (1.8 if monday.month in PARTS[n][1] else 1) for n in names], float)
        n = rng.poisson(lines_per_week)
        picks = rng.choice(len(names), n, p=wt / wt.sum())
        for k in picks:
            v = veh.iloc[rng.integers(len(veh))]
            inv = monday + pd.Timedelta(days=int(rng.integers(0, 7)))
            rows.append({
                "Invoice Date": inv,
                "Job Card Date": inv - pd.Timedelta(days=int(rng.integers(0, 5))),
                "Business Partner Name": v["partner"], "Vehicle No.": v["no"], "Vehicle Model": v["model"],
                "Current KM Reading": float(v["km0"] + w * 90 + rng.integers(0, 400)),
                "INVOICE LINE TEXT": f"{names[k]} - {rng.integers(1000, 9999)}",
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/sample_service_data.csv"
    df = make()
    df.to_csv(out, index=False)
    print(out, df.shape)
