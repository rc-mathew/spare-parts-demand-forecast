"""python report.py  -> outputs/project_report.pdf (every number comes from outputs/)"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

m = json.load(open("outputs/metrics.json"))
tp = pd.read_csv("outputs/test_predictions.csv", parse_dates=["week"])
fi = pd.read_csv("outputs/feature_importance.csv").head(8)
cmp_ = pd.DataFrame(m["model_comparison"]).T
imp = m["improvement_vs_naive"]

wk = tp.groupby("week")[["demand", "lag_4", "pred"]].sum()
ax = wk.rename(columns={"demand": "actual", "lag_4": "naive (lag 4)", "pred": "LightGBM"}).plot(figsize=(7, 3))
ax.set_ylabel("invoice lines / week (modelled parts)")
plt.tight_layout()
plt.savefig("outputs/test_chart.png", dpi=150)

S = getSampleStyleSheet()
P = lambda t, s="BodyText": Paragraph(t, S[s])


def table(rows):
    t = Table(rows, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                           ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    return t


main = [["model", "MAE", "RMSE", "WAPE", "overstock ev.", "stockout ev.", "overstock u.", "stockout u."]]
for k, v in m["table"].items():
    main.append([k] + [f"{v[c]:.3f}" for c in ("MAE", "RMSE", "WAPE")] + [f"{v[c]:.0f}" for c in
                ("overstock_events", "stockout_events", "overstock_units", "stockout_units")])
second = [["model (top 8 parts, summed)", "MAE", "RMSE", "WAPE"]] + [[k] + [f"{v[c]:.3f}" for c in ("MAE", "RMSE", "WAPE")] for k, v in m["model_comparison"].items()]
best = cmp_["WAPE"].idxmin()
met = "MET" if m["target_met"] else "NOT met"

story = [
    P("Spare-Part Demand Forecasting - NewX Services", "Title"),
    P("1. Objective", "Heading2"),
    P("Forecast weekly spare-part demand per part and service centre, 4 weeks ahead, to support a just-in-time inventory policy. "
      "Target: at least 20% fewer overstock and stockout events than a naive baseline."),
    P("2. Data and method", "Heading2"),
    P(f"Source: service invoice lines (MySQL project_service_data.service_data), {m['n_weeks']} full weeks. Part names are parsed from INVOICE LINE TEXT "
      f"(lower-case, digits removed, first 4 words). Service, labour and supply lines (e.g. general service, consumables, polish) are excluded. "
      f"The {m['n_items']} most frequent parts are modelled across {m['n_centres']} centre group(s). Demand is the weekly count of invoice lines; the data has no quantity column. "
      f"Features use only data at least {m['horizon_weeks']} weeks old: lags, rolling mean/std, last-known odometer and 10,000-km bin, shop-wide job volume, week-of-year seasonality. "
      "Model: LightGBM, Poisson loss, one direct model for all horizons. "
      f"Hold-out: last {m['test_weeks']} observed weeks. Baselines: naive (demand 4 weeks ago) and 4-week moving average. "
      "Order quantity = rounded forecast; overstock = ordered above need, stockout = ordered below need."),
    P("3. Results on the hold-out weeks", "Heading2"),
    table(main),
    Spacer(1, 6),
    P(f"Reduction vs naive: overstock events {imp['overstock_events']:.1%}, stockout events {imp['stockout_events']:.1%}, "
      f"overstock units {imp['overstock_units']:.1%}, stockout units {imp['stockout_units']:.1%}. The 20% event target is <b>{met}</b>."),
    Image("outputs/test_chart.png", width=16 * cm, height=6.9 * cm),
    P("Top features by gain: " + ", ".join(fi["feature"]) + "."),
    P("4. Model comparison and cross-validation", "Heading2"),
    table(second),
    Spacer(1, 6),
    P(f"ARIMA (with annual Fourier terms) and Prophet were run on the top parts with rolling 4-week origins over the hold-out. Lowest WAPE: <b>{best}</b>. "
      "LightGBM is the deployed model because it forecasts every part and centre with one model and uses odometer and shop-volume features; "
      "the classical models are per-series and cover only the top parts."),
    P((f"Rolling-origin CV ({m['cv_folds']} folds, sklearn TimeSeriesSplit): LightGBM WAPE {m['cv_mean']['WAPE_lightgbm']:.3f} +/- {m['cv_std']['WAPE_lightgbm']:.3f} "
       f"vs naive {m['cv_mean']['WAPE_naive']:.3f} +/- {m['cv_std']['WAPE_naive']:.3f}. Mean reduction in overstock units {m['cv_mean']['gain_overstock_units']:.1%}, "
       f"stockout units {m['cv_mean']['gain_stockout_units']:.1%}.") if m["cv_folds"] else "Cross-validation skipped: not enough weeks of history."),
    P("5. Business recommendations", "Heading2"),
    *[P(f"{i}. {r}") for i, r in enumerate(m["recommendations"], 1)],
    P("6. Limitations", "Heading2"),
    P("Demand counts invoice lines, not units. No service-centre column exists, so the centre dimension is a proxy or a single group. "
      "Part names come from free text and may merge or split real parts. Intermittent low-volume parts are inherently noisy; MAPE is reported only on non-zero weeks (see metrics.json). "
      "LSTM was not run: with this few weekly points per series a neural sequence model is unlikely to beat gradient boosting and adds heavy dependencies."),
]
SimpleDocTemplate("outputs/project_report.pdf", pagesize=A4).build(story)
print("outputs/project_report.pdf")
