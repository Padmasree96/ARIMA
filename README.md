# 📈 Google Stock Price Forecasting using ARIMA

## 🧠 Project Overview
This project predicts **Google (GOOGL)** stock prices using the **ARIMA (AutoRegressive Integrated Moving Average)** model — a powerful time series forecasting technique.  
By analyzing historical stock data from **Yahoo Finance**, the model generates a **30-day forecast** of future prices.

---

## 🚀 Features
- Fetches live Google stock data using `yfinance`
- Applies ARIMA(1,1,1) model for short-term prediction
- Visualizes both actual and forecasted stock prices
- Easy to run and customize for other companies

---

## 🏗️ Project Workflow
```
User → Yahoo Finance API → Data Extraction → ARIMA Model → Forecast Generation → Visualization
```

---

## 🧩 Technologies Used
- **Programming Language:** Python  
- **Libraries:** yfinance, pandas, statsmodels, matplotlib  
- **IDE:** Jupyter Notebook  

---

## 📊 Implementation Steps
1. **Data Collection**  
   - Downloaded Google stock data from January to November 2025.  
   ```python
   import yfinance as yf
   G = yf.download("GOOGL", start="2025-01-01", end="2025-11-01")
   google = G["Close"]
   ```

2. **Model Training and Forecasting**  
   - Built an ARIMA(1,1,1) model and forecasted 30 days ahead.  
   ```python
   from statsmodels.tsa.arima.model import ARIMA
   model = ARIMA(google, order=(1,1,1))
   result = model.fit()
   forecast = result.forecast(steps=30)
   ```

3. **Visualization**  
   - Displayed original vs. forecasted prices.  
   ```python
   import matplotlib.pyplot as plt
   plt.plot(google, label='Original Data')
   plt.plot(forecast, label='30-Day Forecast', color='red')
   plt.legend()
   plt.show()
   ```

---

## 🖼️ Output
The graph below shows the predicted trend (red line) alongside actual Google stock data.

![Output Plot](images/output_plot.png)

---

## 🔮 Future Enhancements
- Add dashboard (Streamlit/Flask) for live visualization  
- Implement LSTM or Prophet models for deeper forecasting  
- Integrate real-time data refresh  

---

## 📁 Files in This Repository
| File | Description |
|------|--------------|
| `arima_stock_forecast.ipynb` | Main Jupyter Notebook |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
| `images/output_plot.png` | Result visualization |

---

## 🧑‍💻 How to Run
1. Clone the repository:  
   ```bash
   git clone https://github.com/<your-username>/Stock_Forecasting_ARIMA.git
   cd Stock_Forecasting_ARIMA
   ```
2. Install the requirements:  
   ```bash
   pip install -r requirements.txt
   ```
3. Open Jupyter Notebook:  
   ```bash
   jupyter notebook arima_stock_forecast.ipynb
   ```
4. Run all cells to generate forecast and plot.

---

## 🏁 Conclusion
This project demonstrates how time series forecasting can be applied in financial analytics.  
ARIMA offers an interpretable and efficient way to predict short-term stock movements.

---
**Author:** Padmasree Rajavel  
📅 *Created in 2025*
