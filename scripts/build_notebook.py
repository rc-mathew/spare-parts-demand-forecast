"""Builds notebooks/spare_parts_demand_forecasting.ipynb (unexecuted). Run the notebook to fill outputs."""
import nbformat as nbf

M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SRC = open("src/pipeline.py").read()
cells = [
M("""# Spare-Part Demand Forecasting - NewX Services (Ref PM-PR-0027)
**Goal:** forecast spare-part demand per part x service centre x week, 4 weeks ahead, and beat a naive baseline by >= 20% on overstock and stockout events.

**Flow:** load -> clean -> EDA -> weekly panel -> features -> model vs baselines -> importance -> forecast -> recommendations.
Logic lives in `src/pipeline.py` (also used by `run.py` and the Streamlit app). Run this notebook from the repo root or `notebooks/`.

**Standalone:** the next two cells install packages and write `src/pipeline.py` (the full pipeline) so this notebook runs alone in Colab or Jupyter.
**Data access:** if `data/raw_service_data.csv` exists it is used. Otherwise the notebook asks for the MySQL host/user/password. No credentials are stored in this notebook."""),
C("!pip install -q lightgbm sqlalchemy pymysql\nimport os; os.makedirs(\"src\", exist_ok=True)"),
C("%%writefile src/pipeline.py\n" + SRC),
C("""import os, getpass
os.makedirs("src", exist_ok=True); open("src/__init__.py", "a").close()
if not os.path.exists("data/raw_service_data.csv") and not os.getenv("DB_HOST"):
    os.environ.update(DB_HOST=input("DB host: "), DB_PORT="3306", DB_NAME="project_service_data", DB_TABLE="service_data",
                      DB_USER=input("DB user: "), DB_PASSWORD=getpass.getpass("DB password: "))"""),
C("""import os, sys, warnings
warnings.filterwarnings("ignore")
if os.path.basename(os.getcwd()) == "notebooks": os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from src import pipeline as P
sns.set_theme(style="whitegrid")

CENTRE_COL = None      # the table has no service-centre column. Set "partner" to use Business Partner Name as a proxy.
TOP_ITEMS, TOP_CENTRES = 25, 5"""),
M("## 1. Load and clean"),
C("""raw = P.load_raw()
print(raw.shape); raw.head()"""),
C("""df = P.clean(raw)
print(f"{len(raw)} raw rows -> {len(df)} after cleaning")
print(df.isna().mean().round(3).to_string())
df[["invoice_date","vehicle_model","km","line_text","item"]].head(10)"""),
M("## 2. EDA"),
C("""print("date range:", df.invoice_date.min().date(), "->", df.invoice_date.max().date())
print("vehicles:", df.vehicle_no.nunique(), "| models:", df.vehicle_model.nunique(), "| partners:", df.partner.nunique(), "| distinct items:", df.item.nunique())
df.km.describe().round(0)"""),
C("""fig, ax = plt.subplots(1, 2, figsize=(14, 4))
df.set_index("invoice_date").resample("MS").size().plot(ax=ax[0], title="Invoice lines per month")
df.item.value_counts().head(20).sort_values().plot.barh(ax=ax[1], title="Top 20 items by line count")
plt.tight_layout()"""),
M("**Seasonality** - share of each top item's lines by calendar month (a flat row = no seasonality)."),
C("""top = df.item.value_counts().head(12).index
s = df[df.item.isin(top)].assign(month=lambda d: d.invoice_date.dt.month)
sm = pd.crosstab(s.item, s.month, normalize="index")
plt.figure(figsize=(10, 5)); sns.heatmap(sm, cmap="YlOrRd", annot=True, fmt=".2f"); plt.title("Monthly share of demand by item")"""),
M("**Demand by vehicle model and odometer band** (10,000-km service-interval bins)."),
C("""fig, ax = plt.subplots(1, 2, figsize=(14, 4))
df.vehicle_model.value_counts().head(10).sort_values().plot.barh(ax=ax[0], title="Lines by vehicle model")
df.assign(km_bin=(df.km // 10000).clip(upper=15) * 10).groupby("km_bin").size().plot.bar(ax=ax[1], title="Lines by odometer band (k km)")
plt.tight_layout()"""),
C("""topm = df.vehicle_model.value_counts().head(6).index
ct = pd.crosstab(df[df.vehicle_model.isin(topm)].vehicle_model, df[df.vehicle_model.isin(topm)].item.where(lambda s: s.isin(top), "other"), normalize="index")
ct.plot.bar(stacked=True, figsize=(11, 4), colormap="tab20", title="Item mix by vehicle model").legend(bbox_to_anchor=(1, 1))"""),
M("## 3. Weekly panel (item x centre x week)\nDemand = invoice lines per week: there is no quantity column. Partial first/last weeks are dropped; missing weeks are zeros."),
C("""panel, last = P.weekly_panel(df, CENTRE_COL, TOP_ITEMS, TOP_CENTRES)
print("last observed full week:", last.date(), "| series:", panel[["item","centre"]].drop_duplicates().shape[0])
obs = panel[panel.demand.notna()]
print("zero-demand share (intermittency):", round((obs.demand == 0).mean(), 3))
obs.groupby("week").demand.sum().plot(figsize=(11, 3), title="Total weekly demand, modelled items")"""),
M("## 4. Features\nEvery feature uses data at least **H = 4 weeks old** (lags 4/5/6/8/12, rolling mean/std shifted by 4, last-known odometer + 10k-km bin, shop-wide job volume, week-of-year sin/cos). One direct model therefore forecasts weeks T+1..T+4 with no recursion and no leakage (verified in `tests/`)."),
C("""feat = P.add_features(panel)
train, test, future = P.split(feat, last)
print(len(train), "train rows |", len(test), "test rows (last", P.TEST_WEEKS, "weeks) |", len(future), "future rows")
feat[P.FEATURES].describe().T.round(2)"""),
M("## 5. Model vs baselines\nBaselines: **naive** = demand 4 weeks ago (same information as the model), **moving average** of 4 weeks. Model: LightGBM, Poisson loss (counts, many zeros). Order quantity = rounded forecast; *overstock* = ordered above need, *stockout* = ordered below need. MAPE is undefined on zero weeks so it is computed on non-zero weeks only."),
C("""model = P.fit(train)
pred = model.predict(test[P.FEATURES])
tab, gain = P.compare(test, pred)
display(tab.round(3))
print({k: f"{v:.1%}" for k, v in gain.items()})
print("Target (>=20% fewer overstock AND stockout events vs naive):", all(gain[k] >= .2 for k in ("overstock_events","stockout_events")))"""),
C("""t = test.assign(pred=pred).groupby("week")[["demand", "lag_4", "pred"]].sum()
t.rename(columns={"demand":"actual","lag_4":"naive","pred":"LightGBM"}).plot(figsize=(11, 3.5), title="Hold-out: total weekly demand")"""),
C("""sel = test.assign(pred=pred).groupby("item").apply(lambda d: pd.Series({
    "actual": d.demand.sum(), "WAPE_model": (d.demand - d.pred).abs().sum() / max(d.demand.sum(), 1),
    "WAPE_naive": (d.demand - d.lag_4).abs().sum() / max(d.demand.sum(), 1)})).sort_values("actual", ascending=False)
sel.round(3).head(15)"""),
M("## 6. Feature importance"),
C("""imp = pd.Series(model.booster_.feature_importance("gain"), index=P.FEATURES).sort_values()
imp.plot.barh(figsize=(8, 5), title="LightGBM gain importance")"""),
M("## 7. Final model, forecast and artifacts\nRefits on all observed weeks, saves `models/demand_model.joblib`, `outputs/forecast.csv` (part_name x service_centre x week -> predicted_demand) and the other outputs."),
C("""m = P.run(df, CENTRE_COL, top_items=TOP_ITEMS, top_centres=TOP_CENTRES)
fc = pd.read_csv("outputs/forecast.csv", parse_dates=["week"])
display(fc.head(12))
fc.groupby("week").predicted_demand.sum().round(1)"""),
C("""pn = fc.groupby("part_name").predicted_demand.sum().sort_values(ascending=False).head(10)
pn.sort_values().plot.barh(figsize=(8, 4), title="Top 10 parts: forecast demand, next 4 weeks")"""),
M("""## 8. Business recommendations
Fill in after reading the results above (numbers differ on the real data):
1. **Order from `forecast.csv`** weekly; start with the highest-volume parts (chart above).
2. **Safety stock:** add a buffer on parts where a stockout costs more than holding; the over/under trade-off table gives the dial.
3. **Seasonal parts:** pre-stock ahead of the months the seasonality heatmap flags.
4. **Parts with high WAPE** (per-item table): keep manual review; intermittent demand is inherently noisy.
5. **Data gaps to close:** quantity per line, part number, service-centre ID. They are the largest accuracy and relevance limits today.
6. **Operate:** retrain weekly with `python run.py`, track WAPE, view forecasts in `streamlit run app.py`.

**Limitations:** demand counts lines, not units; centre dimension is a proxy; labour and parts share one text field; ARIMA/Prophet/LSTM not run (one global gradient-boosted model chosen for simplicity)."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "notebooks/spare_parts_demand_forecasting.ipynb")
print("notebook written")
