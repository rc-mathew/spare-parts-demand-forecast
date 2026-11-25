# Spare-Part Demand Forecasting (NewX Services, PM-PR-0027)

Forecasts weekly spare-part demand per part and service centre, 4 weeks ahead, and compares against a naive baseline on overstock and stockout events.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env   # fill DB_HOST, DB_USER, DB_PASSWORD from the project brief
export $(grep -v '^#' .env | xargs)
python run.py                      # data -> model -> outputs/forecast.csv, metrics.json
python report.py                   # outputs/project_report.pdf
streamlit run app.py               # dashboard
jupyter notebook notebooks/spare_parts_demand_forecasting.ipynb   # submission notebook: Run All
pytest                             # smoke + leakage test
```
No DB access? Export `service_data` to `data/raw_service_data.csv` and run without `DB_HOST`.
`python scripts/make_sample_data.py` makes a synthetic file with the same schema, for smoke tests only. Never report results from it.

## Layout
| Path | What |
|---|---|
| `src/pipeline.py` | load, clean, weekly panel, features, LightGBM, evaluation, forecast |
| `notebooks/` | EDA + modelling notebook (calls `src/`) |
| `run.py`, `report.py`, `app.py` | CLI, PDF report, Streamlit |
| `models/demand_model.joblib` | trained model (created by `run.py`) |
| `outputs/forecast.csv` | `part_name, service_centre, week, predicted_demand` |

## Method
- Demand = invoice lines per item per week. Item = `INVOICE LINE TEXT` lower-cased, digits removed, first 4 words. Top 25 items modelled.
- Direct 4-week forecast: all features use data at least 4 weeks old (lags 4/5/6/8/12, rolling mean/std, last odometer + 10k-km bin, shop job volume, week-of-year). A test checks no look-ahead.
- Service/labour/supply lines (`NON_PART` regex in `src/pipeline.py`) are excluded. Check the list against your data.
- Model: LightGBM, Poisson loss. Hold-out: last 12 weeks. Baselines: demand 4 weeks ago; 4-week moving average.
- Also run: `TimeSeriesSplit` rolling CV, ARIMA with Fourier seasonality and Prophet on the top 8 parts. Recommendations in the notebook and PDF are generated from the real results.
- Metrics: MAE, RMSE, WAPE, MAPE on non-zero weeks, overstock/stockout events and units. Order quantity = rounded forecast.

## Known limits (read before submitting)
- **No service-centre column** in the schema. Default is one group (`ALL`). `python run.py --centre partner` uses Business Partner Name as a proxy, which is a customer, not a centre.
- **No quantity column**: demand counts lines, not units.
- Labour and parts share one text field, so labour lines appear as items.
- LSTM not run (short series, heavy dependency). Classical models cover only the top 8 parts.
- Event-count metrics are noisy on low-volume parts: compare units too. The 20% target may or may not be met on real data; `metrics.json` says which.
