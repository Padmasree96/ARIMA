
---

# 3) Scripts & files to add

### `src/arima_pipeline.py`
```python
# src/arima_pipeline.py
import argparse
import pandas as pd
import yfinance as yf
import joblib
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

def download_close(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df['Close'].dropna()

def train_arima(series, order=(1,1,1)):
    model = ARIMA(series, order=order)
    result = model.fit()
    return result

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return mae, rmse

def plot_series(original, forecast, out_path='images/forecast.png'):
    plt.figure(figsize=(12,6))
    plt.plot(original.index, original.values, label='Original')
    # forecast is a pandas Series with index
    plt.plot(forecast.index, forecast.values, label='Forecast')
    plt.legend()
    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def main(args):
    series = download_close(args.ticker, args.start, args.end)
    # train on full series (or split into train/test for eval)
    if args.train_test_split:
        split_ix = int(len(series)*(1-args.test_size))
        train, test = series.iloc[:split_ix], series.iloc[split_ix:]
        model = train_arima(train, order=args.order)
        pred = model.forecast(steps=len(test))
        mae, rmse = evaluate(test.values, pred)
        print(f"Evaluation on test set -> MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        # store results
        pd.Series(pred, index=test.index).to_csv('forecast_test.csv', header=['Forecast'])
    else:
        model = train_arima(series, order=args.order)

    # forecast future
    steps = args.forecast_steps
    forecast = model.forecast(steps=steps)
    # create index for forecast (next business days)
    last_date = series.index[-1]
    future_index = pd.bdate_range(last_date, periods=steps+1, closed='right')
    forecast_series = pd.Series(forecast, index=future_index)

    # save model and forecast
    joblib.dump(model, 'arima_model.joblib')
    forecast_series.to_csv('forecast.csv', header=['Forecast'])

    # plot and save
    plot_series(series, forecast_series, out_path='images/output_arima.png')
    print("Saved model -> arima_model.joblib, forecast -> forecast.csv, plot -> images/output_arima.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='GOOGL')
    parser.add_argument('--start', default='2020-01-01')
    parser.add_argument('--end', default='2025-11-01')
    parser.add_argument('--forecast_steps', type=int, default=30)
    parser.add_argument('--order', nargs=3, type=int, default=(1,1,1))
    parser.add_argument('--train_test_split', action='store_true')
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()
    main(args)
