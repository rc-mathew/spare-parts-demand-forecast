"""streamlit run app.py  (needs outputs/ from `python run.py`)"""
import numpy as np
import pandas as pd
import streamlit as st

st.title("Spare-part demand: 4-week forecast")
hist = pd.read_csv("outputs/weekly_history.csv", parse_dates=["week"])
fc = pd.read_csv("outputs/forecast.csv", parse_dates=["week"])

centre = st.sidebar.selectbox("Service centre", sorted(fc.service_centre.unique()))
part = st.sidebar.selectbox("Part", sorted(fc.part_name.unique()))
h = hist[(hist.item == part) & (hist.centre == centre)].set_index("week")["demand"].tail(26)
f = fc[(fc.part_name == part) & (fc.service_centre == centre)].set_index("week")

st.line_chart(pd.DataFrame({"actual": h, "forecast": f["predicted_demand"]}))
tbl = f[["predicted_demand"]].assign(suggested_order=np.ceil(f["predicted_demand"]).astype(int))
st.dataframe(tbl)
st.caption("Demand = invoice lines per week (the data has no quantity column).")
st.download_button("Download all forecasts (CSV)", fc.to_csv(index=False), "forecast.csv")
