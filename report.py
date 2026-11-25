"""python report.py  -> outputs/project_report.pdf (numbers come from outputs/metrics.json)"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table

m = json.load(open("outputs/metrics.json"))
tp = pd.read_csv("outputs/test_predictions.csv", parse_dates=["week"])
fi = pd.read_csv("outputs/feature_importance.csv").head(8)
imp = m["improvement_vs_naive"]

wk = tp.groupby("week")[["demand", "lag_4", "pred"]].sum()
ax = wk.rename(columns={"demand": "actual", "lag_4": "naive (lag 4)", "pred": "LightGBM"}).plot(figsize=(7, 3))
ax.set_ylabel("invoice lines / week (all parts)")
plt.tight_layout()
plt.savefig("outputs/test_chart.png", dpi=150)

S = getSampleStyleSheet()
P = lambda t, s="BodyText": Paragraph(t, S[s])
tab = [["model", "MAE", "RMSE", "WAPE", "overstock ev.", "stockout ev.", "overstock u.", "stockout u."]]
for k, v in m["table"].items():
    tab.append([k] + [f"{v[c]:.3f}" for c in ("MAE", "RMSE", "WAPE")] + [f"{v[c]:.0f}" for c in
               ("overstock_events", "stockout_events", "overstock_units", "stockout_units")])
met = "MET" if m["target_met"] else "NOT met"
story = [
    P("Spare-Part Demand Forecasting - NewX Services", "Title"),
    P("1. Objective", "Heading2"),
    P("Forecast weekly spare-part demand per part and service centre, 4 weeks ahead, to support a "
      "just-in-time inventory policy. Target: at least 20% fewer overstock and stockout events than a naive baseline."),
    P("2. Data and method", "Heading2"),
    P(f"Source: service invoice lines (MySQL project_service_data.service_data). Part names are parsed from INVOICE LINE TEXT "
      f"(lower-case, digits removed, first 4 words); the {m['n_items']} most frequent items are modelled across {m['n_centres']} centre group(s). "
      "Demand is the weekly count of invoice lines; the data has no quantity column. "
      f"Features use only data at least {m['horizon_weeks']} weeks old: lags, rolling mean/std, last-known odometer and 10,000-km bin, "
      "shop-wide job volume, and week-of-year seasonality. Model: LightGBM with Poisson loss, one direct model for all horizons. "
      f"Validation: the last {m['test_weeks']} observed weeks held out (time-based split). "
      "Baselines: naive (demand 4 weeks ago) and 4-week moving average. Order quantity = rounded forecast; "
      "overstock = ordered above need, stockout = ordered below need."),
    P("3. Results", "Heading2"),
    Table(tab),
    Spacer(1, 6),
    P(f"Reduction vs naive: overstock events {imp['overstock_events']:.1%}, stockout events {imp['stockout_events']:.1%}, "
      f"overstock units {imp['overstock_units']:.1%}, stockout units {imp['stockout_units']:.1%}. "
      f"The 20% event target is <b>{met}</b>."),
    Image("outputs/test_chart.png", width=16 * cm, height=6.9 * cm),
    P("Top features by gain: " + ", ".join(fi["feature"]) + "."),
    P("4. Recommendations", "Heading2"),
    P("- Order from outputs/forecast.csv (or the Streamlit dashboard); review the highest-volume parts first.<br/>"
      "- Add a safety stock on parts where stockout cost exceeds holding cost; tune it on the stockout/overstock trade-off above.<br/>"
      "- Re-train weekly (python run.py) and track WAPE; retrain sooner if it drifts.<br/>"
      "- Capture quantity, part number and service-centre ID on invoices; they are the biggest accuracy and relevance gaps."),
    P("5. Limitations", "Heading2"),
    P("Demand counts lines, not units. No service-centre column exists, so the centre dimension is a proxy or a single group. "
      "Labour and parts share one text field. Intermittent low-volume parts are inherently noisy; MAPE is reported only on non-zero weeks. "
      "ARIMA/Prophet/LSTM were not run: a single global gradient-boosted model was chosen for simplicity."),
]
SimpleDocTemplate("outputs/project_report.pdf", pagesize=A4).build(story)
print("outputs/project_report.pdf")
