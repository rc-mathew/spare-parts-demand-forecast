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
# services, labour and supplies: not spare parts. Edit to taste; clean() flags them, weekly_panel() drops them.
NON_PART = r"service|consumable|polish|labou?r|wash|lubrication|charge|inspection|alignment|balancing|fitting|discount|round off|check ?up|cleaning"
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
            username=os.environ["DB_USER"].strip(),
            password=os.environ["DB_PASSWORD"].strip(),
            host=os.environ["DB_HOST"].strip(),
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


def _dates(s):
    """dd-mm-yy text (the DB format) first; ISO / real datetimes otherwise. dayfirst=True would corrupt ISO dates."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    d = pd.to_datetime(s, format="%d-%m-%y", errors="coerce")
    return d if d.isna().mean() <= 0.5 else pd.to_datetime(s, errors="coerce")


def clean(df):
    # match columns ignoring case, spaces, dots and underscores ("Invoice Date" == "invoice_date")
    key = lambda c: re.sub(r"[^a-z0-9]", "", str(c).lower())
    want = {key(k): v for k, v in RAW_COLS.items()}
    df = df.rename(columns={c: want[key(c)] for c in df.columns if key(c) in want})
    missing = set(RAW_COLS.values()) - set(df.columns)
    if missing:
        raise KeyError(f"columns not found: {sorted(missing)}. Columns present: {list(df.columns)}")
    for c in ("invoice_date", "job_card_date"):
        df[c] = _dates(df[c])
    df["invoice_date"] = df["invoice_date"].fillna(df["job_card_date"])
    df["km"] = pd.to_numeric(df["km"].astype(str).str.replace(",", ""), errors="coerce")
    df = df.dropna(subset=["invoice_date", "line_text"]).copy()
    df["item"] = df["line_text"].map(norm_item)
    # repeated identical lines are kept: there is no quantity column, a repeat can mean qty > 1
    df = df[df["item"] != ""].reset_index(drop=True)
    df["is_part"] = ~df["item"].str.contains(NON_PART, regex=True)
    return df


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
    jobs = df.groupby("week").size()  # shop-wide volume, all lines
    if "is_part" in df:
        df = df[df["is_part"]]

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


# ---------- time-series cross-validation ----------
def cv(obs, n_splits=5, test_weeks=8, min_train=26):
    """Rolling-origin CV by week (sklearn TimeSeriesSplit on the sorted weeks). obs = observed rows with lags."""
    from sklearn.model_selection import TimeSeriesSplit

    weeks = np.sort(obs["week"].unique())
    n = min(n_splits, (len(weeks) - min_train) // test_weeks)
    if n < 2:
        return pd.DataFrame()
    rows = []
    for tr, te in TimeSeriesSplit(n_splits=n, test_size=test_weeks).split(weeks):
        a, b = obs[obs["week"].isin(weeks[tr])], obs[obs["week"].isin(weeks[te])]
        t, g = compare(b, fit(a).predict(b[FEATURES]))
        rows.append({"test_from": str(weeks[te][0])[:10], "WAPE_lightgbm": t.loc["lightgbm", "WAPE"],
                     "WAPE_naive": t.loc["naive_lag4", "WAPE"], "MAE_lightgbm": t.loc["lightgbm", "MAE"],
                     **{f"gain_{k}": v for k, v in g.items()}})
    return pd.DataFrame(rows)


# ---------- classical models (ARIMA with Fourier seasonality, Prophet) on the top series ----------
def _arima(y, fut_weeks):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    def fourier(w):
        woy = pd.Series(w).dt.isocalendar().week.astype(int).to_numpy()
        return np.column_stack([f(2 * np.pi * k * woy / 52) for k in (1, 2) for f in (np.sin, np.cos)])

    ex = len(y) >= 60  # annual terms need enough history
    try:
        m = SARIMAX(y.to_numpy(float), exog=fourier(y.index) if ex else None, order=(1, 0, 1), trend="c").fit(disp=False)
        return np.clip(m.forecast(len(fut_weeks), exog=fourier(fut_weeks) if ex else None), 0, None)
    except Exception:
        return np.repeat(y.tail(4).mean(), len(fut_weeks))


def _prophet(y, fut_weeks):
    import logging
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    m = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=len(y) >= 104)
    m.fit(pd.DataFrame({"ds": y.index, "y": y.to_numpy(float)}))
    return np.clip(m.predict(pd.DataFrame({"ds": fut_weeks}))["yhat"].to_numpy(), 0, None)


def classical_compare(p, last, test_pred, top=8):
    """4-week-ahead ARIMA/Prophet vs LightGBM and naive on the top series (centres summed), rolling origins over the test window.
    test_pred: DataFrame item, centre, week, demand, lag_4, pred."""
    hist = p[p["demand"].notna()].groupby(["item", "week"])["demand"].sum().unstack(0).fillna(0)
    items = hist.sum().sort_values(ascending=False).head(top).index
    start = last - pd.Timedelta(weeks=TEST_WEEKS - 1)
    models = {"arima_fourier": _arima}
    try:
        import prophet  # noqa: F401
        models["prophet"] = _prophet
    except Exception:
        pass
    preds = {m: [] for m in models}
    for item in items:
        for o in pd.date_range(start, last, freq=f"{H * 7}D"):
            fw = pd.date_range(o, periods=H, freq="7D")
            y = hist.loc[hist.index < o, item]
            for name, f in models.items():
                preds[name].append(pd.DataFrame({"item": item, "week": fw, name: f(y, fw)}))
    tp = test_pred[test_pred["item"].isin(items)].groupby(["item", "week"])[["demand", "lag_4", "pred"]].sum().reset_index()
    for name, parts in preds.items():
        tp = tp.merge(pd.concat(parts), on=["item", "week"], how="left")
    tp = tp[tp["week"] <= last].dropna()
    cols = {"naive_lag4": "lag_4", "lightgbm": "pred", **{m: m for m in models}}
    rows = {k: evaluate(tp["demand"], tp[c]) for k, c in cols.items()}
    return pd.DataFrame(rows).T[["MAE", "RMSE", "WAPE"]], tp


# ---------- plain-language recommendations from the real results ----------
def recommendations(out="outputs"):
    m = json.loads(Path(f"{out}/metrics.json").read_text())
    acc = pd.read_csv(f"{out}/item_accuracy.csv")
    fc = pd.read_csv(f"{out}/forecast.csv")
    hist = pd.read_csv(f"{out}/weekly_history.csv", parse_dates=["week"])
    g, t = m["improvement_vs_naive"], m["table"]["lightgbm"]
    r = []
    top = fc.groupby("part_name")["predicted_demand"].sum().sort_values(ascending=False).head(5)
    r.append("Order first: " + ", ".join(f"{k} (~{v:.0f} lines in the next {m['horizon_weeks']} weeks)" for k, v in top.items()) + ".")
    if m["target_met"]:
        r.append(f"The model cut overstock events by {g['overstock_events']:.0%} and stockout events by {g['stockout_events']:.0%} versus the naive baseline: the 20% target is met on the hold-out weeks.")
    else:
        r.append(f"Target not met: events fell {g['overstock_events']:.0%} (overstock) and {g['stockout_events']:.0%} (stockout) versus naive, below 20% on at least one. "
                 f"Units fell {g['overstock_units']:.0%} and {g['stockout_units']:.0%}, so use the model for sizing orders and judge it on units, not event counts.")
    if t["stockout_units"] > t["overstock_units"]:
        r.append(f"The model under-orders more than it over-orders ({t['stockout_units']:.0f} short vs {t['overstock_units']:.0f} excess units on hold-out): add a safety-stock buffer on fast-moving parts.")
    else:
        r.append(f"The model over-orders more than it under-orders ({t['overstock_units']:.0f} excess vs {t['stockout_units']:.0f} short units on hold-out): trim order quantities on slow parts before adding buffer.")
    bad = acc[(acc["actual"] >= 20) & (acc["WAPE_model"] > 0.6)].sort_values("actual", ascending=False).head(5)["item"].tolist()
    if bad:
        r.append("Keep manual review for hard-to-forecast parts (hold-out WAPE above 60%): " + ", ".join(bad) + ".")
    if m["n_weeks"] >= 52:
        mo = hist.assign(month=hist["week"].dt.month).groupby(["item", "month"])["demand"].mean().unstack()
        ratio = (mo.max(axis=1) / mo.mean(axis=1)).sort_values(ascending=False)
        seas = [f"{i} (peak month {int(mo.loc[i].idxmax())})" for i in ratio[ratio >= 1.5].head(4).index]
        if seas:
            r.append("Seasonal parts: stock up the month before the peak for " + ", ".join(seas) + ".")
    else:
        r.append(f"Only {m['n_weeks']} weeks of history: annual seasonality cannot be learned yet. Re-train when a full year or more is available.")
    r.append("Close the data gaps: record quantity, part number and service-centre ID on every invoice line. Demand is currently a count of lines, and services and parts share one text field.")
    return r


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
    tp.groupby("item").apply(lambda d: pd.Series({
        "actual": d["demand"].sum(),
        "WAPE_model": (d["demand"] - d["pred"]).abs().sum() / max(d["demand"].sum(), 1),
        "WAPE_naive": (d["demand"] - d["lag_4"]).abs().sum() / max(d["demand"].sum(), 1)}), include_groups=False) \
        .reset_index().to_csv(f"{out}/item_accuracy.csv", index=False)
    cvt = cv(pd.concat([train, test]))
    cmp_tab, _ = classical_compare(p, last, tp)
    cmp_tab.to_csv(f"{out}/model_comparison.csv")
    cvt.to_csv(f"{out}/cv_folds.csv", index=False)
    p[p["demand"].notna()][["item", "centre", "week", "demand"]].to_csv(f"{out}/weekly_history.csv", index=False)
    pd.DataFrame({"feature": FEATURES, "gain": final.booster_.feature_importance("gain")}) \
        .sort_values("gain", ascending=False).to_csv(f"{out}/feature_importance.csv", index=False)

    metrics = {
        "horizon_weeks": H, "test_weeks": TEST_WEEKS, "last_observed_week": str(last.date()),
        "n_items": int(p["item"].nunique()), "n_centres": int(p["centre"].nunique()),
        "n_train_rows": len(train), "n_test_rows": len(test),
        "table": tab.to_dict("index"), "improvement_vs_naive": gain,
        "target_met": bool(all(v >= 0.20 for k, v in gain.items() if k.endswith("events"))),
        "n_weeks": int(p.loc[p["demand"].notna(), "week"].nunique()),
        "cv_mean": cvt.drop(columns="test_from").mean().to_dict() if len(cvt) else {},
        "cv_std": cvt.drop(columns="test_from").std().to_dict() if len(cvt) else {},
        "cv_folds": len(cvt),
        "model_comparison": cmp_tab.to_dict("index"),
    }
    Path(f"{out}/metrics.json").write_text(json.dumps(metrics, indent=2))
    metrics["recommendations"] = recommendations(out)
    Path(f"{out}/metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
