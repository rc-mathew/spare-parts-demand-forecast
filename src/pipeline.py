"""Spare-part demand forecasting: load -> clean -> weekly panel -> features -> LightGBM -> 4-week forecast.

Direct multi-step design: every feature uses data at least H weeks old, so one model
forecasts all H weeks ahead without recursive prediction and without look-ahead leakage.
"""
import json
import os
import re
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

H = 4  # forecast horizon (weeks) and minimum lag
TEST_WEEKS = 12  # last N observed weeks held out for evaluation
RAW_COLS = {
    "Invoice Date": "invoice_date",
    "Job Card Date": "job_card_date",
    "Business Partner Name": "partner",
    "Vehicle No.": "vehicle_no",
    "Vehicle Model": "vehicle_model",
    "Current KM Reading": "km",
    "INVOICE LINE TEXT": "line_text",
}
FEATURES = [
    "lag_4", "lag_5", "lag_6", "lag_8", "lag_12",
    "roll_mean_4", "roll_mean_8", "roll_mean_13", "roll_std_8",
    "km_lag", "km_bin", "jobs_lag",
    "week_of_year", "month", "woy_sin", "woy_cos",
    "item_c", "centre_c",
]


# ---------- load / clean ----------
def load_raw(csv=None, cache="data/raw_service_data.csv"):
    """MySQL if DB_HOST is set (credentials from env, never in code), else a CSV."""
    if os.getenv("DB_HOST"):
        from sqlalchemy import create_engine
        from sqlalchemy.engine import URL

        url = URL.create(
            "mysql+pymysql",
            username=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            host=os.environ["DB_HOST"],
            port=int(os.getenv("DB_PORT", 3306)),
            database=os.environ["DB_NAME"],
        )
        df = pd.read_sql(f"SELECT * FROM {os.getenv('DB_TABLE', 'service_data')}", create_engine(url))
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        return df
    path = csv or cache
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Set DB_* env vars (see .env.example) or pass a CSV.")
    return pd.read_csv(path)


def norm_item(text):
    """'OIL FILTER - 4521' -> 'oil filter'. Digits/punctuation dropped, first 4 words kept."""
    return " ".join(re.sub(r"[^a-z ]", " ", str(text).lower()).split()[:4])


def clean(df):
    df = df.rename(columns=lambda c: c.strip()).rename(columns=RAW_COLS)
    for c in ("invoice_date", "job_card_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["invoice_date"] = df["invoice_date"].fillna(df["job_card_date"])
    df["km"] = pd.to_numeric(df["km"], errors="coerce")
    df = df.dropna(subset=["invoice_date", "line_text"]).copy()
    df["item"] = df["line_text"].map(norm_item)
    # repeated identical lines are kept: there is no quantity column, a repeat can mean qty > 1
    return df[df["item"] != ""].reset_index(drop=True)


# ---------- weekly panel ----------
def weekly_panel(df, centre_col=None, top_items=25, top_centres=5):
    """Full item x centre x week grid. demand = invoice lines per week (no qty column exists).

    centre_col: column used as 'service centre' ('partner' is the only proxy in the data).
    Returns (panel, last_observed_week). The panel includes H future weeks with demand = NaN.
    """
    df = df.copy()
    df["week"] = df["invoice_date"].dt.normalize() - pd.to_timedelta(df["invoice_date"].dt.weekday, unit="D")
    d0, d1 = df["invoice_date"].min().normalize(), df["invoice_date"].max().normalize()
    first = d0 - pd.Timedelta(days=d0.weekday())
    if d0 != first:
        first += pd.Timedelta(weeks=1)  # drop partial first week
    last = d1 - pd.Timedelta(days=d1.weekday())
    if d1 != last + pd.Timedelta(days=6):
        last -= pd.Timedelta(weeks=1)  # drop partial last week
    df = df[(df["week"] >= first) & (df["week"] <= last)]
    jobs = df.groupby("week").size()  # shop-wide volume, all items

    df["centre"] = df[centre_col].astype(str).str.strip() if centre_col else "ALL"
    df = df[df["item"].isin(df["item"].value_counts().head(top_items).index)]
    df = df[df["centre"].isin(df["centre"].value_counts().head(top_centres).index)]

    g = df.groupby(["item", "centre", "week"]).agg(demand=("item", "size"), km_mean=("km", "mean")).reset_index()
    weeks = pd.DataFrame({"week": pd.date_range(first, last + pd.Timedelta(weeks=H), freq="7D")})
    p = g[["item", "centre"]].drop_duplicates().merge(weeks, how="cross").merge(g, on=["item", "centre", "week"], how="left")
    p["demand"] = np.where(p["week"] <= last, p["demand"].fillna(0), np.nan)
    p["jobs"] = p["week"].map(jobs)
    return p.sort_values(["item", "centre", "week"]).reset_index(drop=True), last


# ---------- features ----------
def add_features(p):
    p = p.copy()
    g = p.groupby(["item", "centre"])
    for k in (H, H + 1, H + 2, H + 4, H + 8):
        p[f"lag_{k}"] = g["demand"].shift(k)
    for w in (4, 8, 13):
        p[f"roll_mean_{w}"] = g["demand"].transform(lambda s, w=w: s.shift(H).rolling(w, min_periods=1).mean())
    p["roll_std_8"] = g["demand"].transform(lambda s: s.shift(H).rolling(8, min_periods=2).std())
    p["km_lag"] = g["km_mean"].transform(lambda s: s.shift(H).ffill())
    p["km_bin"] = (p["km_lag"] // 10000).clip(upper=15)  # 10k-km service-interval bins
    wk_jobs = p.groupby("week")["jobs"].first().sort_index().shift(H)
    p["jobs_lag"] = p["week"].map(wk_jobs)
    p["week_of_year"] = p["week"].dt.isocalendar().week.astype(int)
    p["month"] = p["week"].dt.month
    p["woy_sin"] = np.sin(2 * np.pi * p["week_of_year"] / 52)
    p["woy_cos"] = np.cos(2 * np.pi * p["week_of_year"] / 52)
    p["item_c"] = p["item"].astype("category")
    p["centre_c"] = p["centre"].astype("category")
    return p


def split(p, last):
    """Time-based split. Returns (train, test, future)."""
    obs = p[p["demand"].notna()].dropna(subset=["lag_4"])
    cut = last - pd.Timedelta(weeks=TEST_WEEKS - 1)
    return obs[obs["week"] < cut], obs[obs["week"] >= cut], p[p["week"] > last]


# ---------- model / evaluation ----------
def fit(train):
    m = lgb.LGBMRegressor(
        objective="poisson", n_estimators=400, learning_rate=0.03, num_leaves=15,
        min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=0, verbose=-1,
    )
    return m.fit(train[FEATURES], train["demand"])


def evaluate(y, pred):
    """Order quantity = rounded forecast. overstock: ordered > needed; stockout: ordered < needed."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    order = np.rint(pred)
    over, under = np.maximum(order - y, 0), np.maximum(y - order, 0)
    nz = y > 0
    return {
        "MAE": float(np.abs(y - pred).mean()),
        "RMSE": float(np.sqrt(((y - pred) ** 2).mean())),
        "WAPE": float(np.abs(y - pred).sum() / max(y.sum(), 1e-9)),
        "MAPE_nonzero": float((np.abs(y - pred)[nz] / y[nz]).mean()) if nz.any() else float("nan"),
        "overstock_events": int((over > 0).sum()),
        "stockout_events": int((under > 0).sum()),
        "overstock_units": float(over.sum()),
        "stockout_units": float(under.sum()),
    }


def compare(test, pred):
    rows = {
        "naive_lag4": evaluate(test["demand"], test["lag_4"]),  # same info set as the model
        "moving_avg_4": evaluate(test["demand"], test["roll_mean_4"]),
        "lightgbm": evaluate(test["demand"], pred),
    }
    tab = pd.DataFrame(rows).T
    n, m = tab.loc["naive_lag4"], tab.loc["lightgbm"]
    gain = {k: float(1 - m[k] / n[k]) if n[k] else float("nan") for k in
            ("overstock_events", "stockout_events", "overstock_units", "stockout_units")}
    return tab, gain


# ---------- end to end ----------
def run(df, centre_col=None, out="outputs", models="models", top_items=25, top_centres=5):
    Path(out).mkdir(exist_ok=True, parents=True)
    Path(models).mkdir(exist_ok=True, parents=True)
    panel, last = weekly_panel(df, centre_col, top_items, top_centres)
    p = add_features(panel)
    train, test, future = split(p, last)

    tab, gain = compare(test, fit(train).predict(test[FEATURES]))
    final = fit(pd.concat([train, test]))  # refit on all observed data for the real forecast
    joblib.dump({"model": final, "features": FEATURES, "horizon": H}, f"{models}/demand_model.joblib")

    fc = future[["item", "centre", "week"]].copy()
    fc["predicted_demand"] = np.clip(final.predict(future[FEATURES]), 0, None).round(2)
    fc.rename(columns={"item": "part_name", "centre": "service_centre"}).to_csv(f"{out}/forecast.csv", index=False)

    tp = test[["item", "centre", "week", "demand", "lag_4"]].copy()
    tp["pred"] = fit(train).predict(test[FEATURES])
    tp.to_csv(f"{out}/test_predictions.csv", index=False)
    p[p["demand"].notna()][["item", "centre", "week", "demand"]].to_csv(f"{out}/weekly_history.csv", index=False)
    pd.DataFrame({"feature": FEATURES, "gain": final.booster_.feature_importance("gain")}) \
        .sort_values("gain", ascending=False).to_csv(f"{out}/feature_importance.csv", index=False)

    metrics = {
        "horizon_weeks": H, "test_weeks": TEST_WEEKS, "last_observed_week": str(last.date()),
        "n_items": int(p["item"].nunique()), "n_centres": int(p["centre"].nunique()),
        "n_train_rows": len(train), "n_test_rows": len(test),
        "table": tab.to_dict("index"), "improvement_vs_naive": gain,
        "target_met": bool(all(v >= 0.20 for k, v in gain.items() if k.endswith("events"))),
    }
    Path(f"{out}/metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
