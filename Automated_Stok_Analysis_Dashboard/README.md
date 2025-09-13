# AI-Powered Stock Analysis Dashboard (FastAPI + Airflow + PostgreSQL)

A complete, end-to-end mini data platform for daily stock ingestion and AI-assisted time-series analysis.

- **Ingestion:** An Apache Airflow DAG pulls historical/daily prices from Yahoo Finance (via `yfinance`) and stores them in **PostgreSQL** with deduping/constraints.  
- **Analytics App:** A **FastAPI** web app reads from PostgreSQL, generates ACF/differencing/moving-average/high-price visuals, and asks **Google Gemini** for a concise, technical interpretation of each chart.

---

## ✨ Highlights

- **Two-stage pipeline**
  - First run: backfills multi-year historical data; subsequent runs: daily refresh.
- **Robust DB writes**
  - Temp table + optional `ON CONFLICT` with a unique constraint on (`Date`, `Symbol`) to avoid duplicates.
- **Interactive dashboard**
  - Autocorrelation (ACF), differencing for stationarity, multi-window moving averages, high-price trend with basic stats.
- **AI Insights**
  - Each chart is piped to **Gemini** with a domain-specific prompt to produce professional, markdown-formatted analysis.

---

## 🧱 Architecture (High-Level)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Automated_Stok_Analysis_Dashboard/static/diagram.JPG)

---

## 🧰 Tech Stack

- **Workflow**: Apache Airflow (daily schedule), Python operators  
- **Data**: `yfinance`, `pandas`, SQLAlchemy, PostgreSQL (unique constraint, temp tables)  
- **App/UI**: FastAPI, Bootstrap front-end, Matplotlib/Seaborn charts  
- **Time-Series**: `statsmodels` (ACF, seasonal tools), NumPy/Pandas pipelines  
- **AI**: Google Gemini (image-grounded chart explanations)

---

## Project Screenshots :

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Automated_Stok_Analysis_Dashboard/static/u1.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Automated_Stok_Analysis_Dashboard/static/u2.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Automated_Stok_Analysis_Dashboard/static/u3.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Automated_Stok_Analysis_Dashboard/static/u4.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Automated_Stok_Analysis_Dashboard/static/u5.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Automated_Stok_Analysis_Dashboard/static/u6.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Automated_Stok_Analysis_Dashboard/static/u7.JPG)

---

## 📦 Repository Structure

```
.
├─ dags/
│  └─ dag3.py                  # Airflow DAG: yfinance → PostgreSQL (historical/daily)
├─ app/
│  └─ main.py                  # FastAPI dashboard + AI insights
├─ requirements.txt            # (suggested) Python deps for DAG & app
└─ README.md                   # You are here
```

> **Note:** In your repo, the app file may be named `main (3).py`. Consider renaming to `main.py` for clarity.

---

## 🔐 Security & Configuration (Important)

- **Move secrets to env vars**: The sample code uses inline DB credentials and an API key. **Replace with environment variables** before deploying.
  - `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`
  - `GEMINI_API_KEY`
- **.env example (don’t commit this file):**
  ```bash
  POSTGRES_USER=appuser
  POSTGRES_PASSWORD=********
  POSTGRES_DB=appdb
  POSTGRES_HOST=localhost
  GEMINI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
  ```

Update the code to read from env (e.g., `os.getenv`) instead of hardcoding.

---

## ⚙️ Setup

### 1) Prerequisites
- Python 3.10+  
- PostgreSQL 14+ running and reachable  
- Airflow 2.x installed and initialized (`airflow db init`)  
- (Optional) A virtualenv/conda environment

### 2) Install Python dependencies

Create a `requirements.txt` similar to:

```txt
fastapi
uvicorn
pandas
numpy
matplotlib
seaborn
sqlalchemy
psycopg2-binary
statsmodels
yfinance
google-generativeai
markdown
Pillow
```

Then:

```bash
pip install -r requirements.txt
```

---

## 🗄️ Database

The DAG creates the `stock_data` table if missing and enforces a unique constraint on (`Date`, `Symbol`). It also logs scraping errors to `stock_errors`. You don’t need to pre-create tables.

---

## ⛲ Airflow: Data Ingestion

1. Copy `dag3.py` into your Airflow DAGs folder (e.g., `~/airflow/dags/dag3.py`).  
2. Ensure your environment (or Airflow connections) has the correct PostgreSQL credentials.  
3. Start Airflow:
   ```bash
   airflow webserver -p 8080
   airflow scheduler
   ```
4. In the UI, enable the DAG **`stock_data_scraper`**:
   - **First run** → backfills historical data from **2020-09-12** to **today**  
   - **Subsequent runs** → fetches **daily** updates  
   (Deduping handled via unique constraint or application fallback.)

**Symbols covered by default**: `AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA, SPY` (edit the list in the DAG as needed).

---

## 🖥️ FastAPI: Analytics Dashboard

1. Set your env vars (esp. `GEMINI_API_KEY` & PostgreSQL creds).  
2. Run the app:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
3. Open **http://localhost:8000**:
   - Select a symbol (auto-populated from DB).
   - Click **Analyze Stock** to generate:
     - **Autocorrelation (ACF)** chart + AI write-up  
     - **Differenced series** (stationarity check) + AI write-up  
     - **Moving Averages** (20/50/200) + AI write-up  
     - **High prices** trend + stats + AI write-up  

### API Endpoints
- `GET /` – HTML dashboard (Bootstrap).  
- `GET /analyze/{symbol}` – Returns plots (as base64) + Gemini analyses, along with date range & datapoint count.  
- `GET /api/symbols` – Lists distinct tickers available in the DB.  

---

## 📈 How the AI Insights Work

- Each Matplotlib figure is saved to a temp file and **sent to Gemini** with a tailored, expert-level prompt (quant/tech analysis framing).  
- The response is converted to HTML via `markdown` and rendered in the dashboard.  

---

## 🔧 Customization Tips

- **Change tickers**: Edit the `symbols` list in the DAG.  
- **Change look-back window**: The app pulls ~3 years by default; adjust in `get_stock_data(years=3)`.  
- **Add more charts**: Follow the pattern in `create_*_plot` functions to generate a figure, encode as base64, and pass it to Gemini with a new, focused prompt.  

---

## 🧪 Local Testing Notes

- Ensure you’ve run the Airflow DAG at least once so the DB has data; the app raises an error if no rows exist for a selected symbol.  
- If a specific date isn’t available on a trading holiday, the DAG falls back to the most recent available row when scraping single-day data.  

---

## 🚀 Roadmap Ideas

- Dockerize Airflow + App + Postgres for one-command spin-up  
- Add TA indicators (RSI/MACD/Bollinger) and send those charts to Gemini  
- Add forecasting (ARIMA/Prophet) with AI-assisted narrative  
- OAuth/API-key guardrails and rate-limit handling for yfinance/Gemini

---

## 📜 License

MIT (or your preferred license).

---

## 🙌 Acknowledgements

- Apache Airflow, FastAPI, PostgreSQL, yfinance, statsmodels, Matplotlib/Seaborn  
- Google Gemini for chart-aware analyses
