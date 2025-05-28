# 📈 Sales Demand Forecasting App (SARIMA-powered)

This project provides a **web-based interactive platform** for forecasting sales demand using **SARIMA time series models**. Built with **FastAPI**, **Jinja2**, and **Chart.js**, it offers intuitive visual diagnostics, forecast outputs, and performance metrics from uploaded CSV/Excel datasets.

---

## 🚀 Features

- 📤 Upload sales datasets in CSV or Excel format
- 📅 Select `date` and `target` value columns
- 🔍 Automatic data cleaning & frequency inference
- 🔧 Hyperparameter tuning for SARIMA with AIC optimization
- 📊 Diagnostics: ACF, PACF, and seasonal decomposition
- 📉 Forecasts: 30-day forecast with upper & lower confidence bounds
- 📈 Metrics: MAE, RMSE, and MAPE evaluation
- 🖼️ Visuals powered by Chart.js and Matplotlib
- 💾 Export forecast results as CSV

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Pandas, Statsmodels, Scikit-learn, Joblib
- **Frontend**: HTML5, Bootstrap 5, JavaScript, Chart.js
- **Plots**: Matplotlib, Seaborn
- **Concurrency**: Joblib + Parallel processing
- **Caching**: LRU cache for SARIMA models
- **Deployment**: Uvicorn server

---

## Project Screenshots :

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/forecasting/static/f1.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/forecasting/static/f2.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/forecasting/static/f3.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/forecasting/static/f4.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/forecasting/static/f5.JPG)

## 📂 File Structure

```
.
├── main.py               # FastAPI backend with SARIMA pipeline and frontend rendering
├── static/
│   └── plots/            # Saved forecast and diagnostic plots
├── templates/            # Jinja2 HTML templates (inlined in main.py)
└── forecasting.log       # Logs of operations and errors
```

---

## ▶️ Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/your-username/sales-demand-forecasting.git
cd sales-demand-forecasting
```

### 2. Install dependencies

We recommend using a virtual environment:

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 9000
```

Open your browser and visit: [http://127.0.0.1:9000](http://127.0.0.1:9000)

---

## 📁 Upload Instructions

- Accepts `.csv` or `.xlsx` files
- Must contain at least:
  - A **date/time** column
  - A **numeric target** (sales) column
- Minimum: 30 data points for forecasting

---

## 📤 Output Includes

- Model diagnostics: ACF, PACF, decomposition
- 30-day forecast with interactive charts
- Downloadable forecast results (CSV)
- Model details: SARIMA (p,d,q)(P,D,Q,s), AIC, BIC, LLF

---

## 📌 Example Use Case

Use this tool to:
- Forecast monthly sales for inventory planning
- Evaluate seasonality and trend in your product demand
- Compare model performance using MAE, RMSE, MAPE

---

## 🧠 Model Info

Uses **SARIMA (Seasonal ARIMA)** for time series modeling, with automatic selection of the best parameters based on AIC. Frequency is inferred from the time index or estimated heuristically.

---

## 📝 License

MIT License. See `LICENSE` for more details.

---

## 🙌 Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [Statsmodels](https://www.statsmodels.org/)
- [Chart.js](https://www.chartjs.org/)
- [Bootstrap](https://getbootstrap.com/)
