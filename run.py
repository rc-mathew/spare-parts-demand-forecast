"""python run.py [--csv data/x.csv] [--centre partner]  -> models/, outputs/"""
import argparse

import pandas as pd

from src import pipeline as P

ap = argparse.ArgumentParser()
ap.add_argument("--csv", help="CSV instead of MySQL (used when DB_HOST is unset)")
ap.add_argument("--centre", choices=["partner"], help="use Business Partner Name as the service-centre proxy")
a = ap.parse_args()

m = P.run(P.clean(P.load_raw(a.csv)), centre_col=a.centre)
print(pd.DataFrame(m["table"]).T.round(3).to_string())
print("improvement vs naive:", {k: round(v, 3) for k, v in m["improvement_vs_naive"].items()})
print("target (>=20% fewer overstock AND stockout events) met:", m["target_met"])
