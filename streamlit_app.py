# src/streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import os
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="ARIMA Stock Forecast", layout="wide")
st.title("Stock Forecasting (ARIMA)")

ticker = st.text_input("Ticker", value="GOOGL")
start = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))
end = st.date_input("End date", value=pd.to_datetime(datetime.today().date()))
forecast_steps = st.number_input("Forecast steps (days)", min_value=1, max_value=365, value=30)

if st.button("Run Forecast"):
    with st.spinner("Downloading data..."):
        df = yf.download(ticker, start=start, end=end)['Close'].dropna()
    st.subheader("Historical Close Price")
    st.line_chart(df)

    with st.spinner("Training ARIMA..."):
        model = ARIMA(df, order=(1,1,1)).fit()
        forecast = model.forecast(steps=forecast_steps)
        last_date = df.index[-1]
        future_index = pd.bdate_range(last_date, periods=forecast_steps+1, closed='right')
        forecast_series = pd.Series(forecast, index=future_index)

    st.subheader("Forecast (next {} business days)".format(forecast_steps))
    st.line_chart(pd.concat([df.tail(200), forecast_series]))

    st.download_button("Download forecast CSV", data=forecast_series.to_csv(), file_name=f"{ticker}_forecast.csv")
    st.write("Model summary (truncated):")
    st.text(str(model.summary().as_text())[:1000])
