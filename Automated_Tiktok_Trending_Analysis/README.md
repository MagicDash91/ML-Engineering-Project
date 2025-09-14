# 🎵 AI-Powered TikTok Trending Analysis Dashboard (FastAPI + Airflow + PostgreSQL + Gemini)  
A complete, end-to-end mini data platform for **daily TikTok trending video ingestion** and **AI-assisted content/engagement analysis**.

---

## 📥 Ingestion  
- **Apache Airflow DAG** scrapes trending TikTok videos daily using **TikTokApi**.  
- Data is processed with **Pandas/NumPy**, validated, deduplicated, and stored in **PostgreSQL**.  
- On conflict, existing rows are updated with the latest engagement stats (likes, comments, shares, views).  

---

## 📊 Analytics App  
- A **FastAPI web app** connects to PostgreSQL and retrieves top trending videos.  
- Generates multiple charts:  
  - Top 10 videos by likes  
  - Engagement distribution (likes, comments, shares, views)  
  - Top creators by engagement  
  - Word cloud of captions  
  - Viral score & engagement rate distributions  
- Each chart is sent to **Google Gemini** with a tailored prompt for **AI-generated insights** (markdown-formatted and displayed in the dashboard).  

---

## ✨ Highlights  

- **Automated Pipeline**  
  - Airflow DAG runs daily to refresh TikTok trending data.  
  - Ensures no duplicates with unique video IDs + synthetic fallback IDs if missing.  

- **Robust DB Writes**  
  - Uses temporary tables + UPSERT (`ON CONFLICT`) logic.  
  - Error logs stored separately in `tiktok_errors`.  

- **Interactive Dashboard**  
  - Bootstrap-based responsive UI with multiple analysis sections.  
  - AI-driven insights embedded alongside each chart.  

- **AI Insights**  
  - Charts piped to **Google Gemini** with domain-specific prompts.  
  - Produces professional, marketing-ready insights.  

---

## 🧱 Architecture (High-Level)  

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Automated_Tiktok_Trending_Analysis/static/dia.JPG)

---

## 🧰 Tech Stack  

- **Workflow:** Apache Airflow (daily schedule, Python operators)  
- **Data:** TikTokApi, Pandas, NumPy, SQLAlchemy, PostgreSQL  
- **App/UI:** FastAPI, Bootstrap front-end, Matplotlib/Seaborn charts, WordCloud  
- **AI:** Google Gemini (chart-aware analyses)  

---

## Project Screenshots :

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Automated_Tiktok_Trending_Analysis/static/u1.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Automated_Tiktok_Trending_Analysis/static/u2.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Automated_Tiktok_Trending_Analysis/static/u3.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Automated_Tiktok_Trending_Analysis/static/u4.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Automated_Tiktok_Trending_Analysis/static/u5.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Automated_Tiktok_Trending_Analysis/static/u6.JPG)

---

## 📦 Repository Structure  

```
.
├─ dag4.py                   # Airflow DAG: TikTokApi → PostgreSQL
├─ tiktok_dashboard.py       # FastAPI dashboard + AI insights
├─ requirements.txt          # Python dependencies
└─ README.md                 # You are here
```

---

## 🔐 Security & Configuration (Important)  

Move secrets to environment variables. The sample code uses inline credentials and an API key—replace with env vars before deploying:  

```env
POSTGRES_USER=appuser
POSTGRES_PASSWORD=********
POSTGRES_DB=appdb
POSTGRES_HOST=localhost
GEMINI_API_KEY=xxxxxxxxxxxxxxxxxxxxxx
```

Update code to read from env (via `os.getenv`).  

---

## ⚙️ Setup  

### 1) Prerequisites  
- Python 3.10+  
- PostgreSQL 14+ running locally or in Docker  
- Airflow 2.x installed (`airflow db init`)  
- (Optional) Virtualenv/conda environment  

### 2) Install dependencies  
```bash
pip install -r requirements.txt
```

### 3) Database  
- `tiktok_data` table is auto-created if missing (with schema).  
- `tiktok_errors` table logs scraping failures.  

### 4) Airflow: Data Ingestion  
- Copy `dag4.py` into your Airflow `~/airflow/dags/`.  
- Start Airflow services:  
  ```bash
  airflow webserver -p 8080
  airflow scheduler
  ```
- Enable DAG `tiktok_trending_scraper` in Airflow UI.  
  - First run → full scrape of trending data.  
  - Subsequent runs → daily refresh.  

### 5) FastAPI: Analytics Dashboard  
- Export env vars (esp. PostgreSQL + Gemini API key).  
- Run the app:  
  ```bash
  uvicorn tiktok_dashboard:app --host 0.0.0.0 --port 8001
  ```
- Open dashboard at: 👉 http://localhost:8001  

---

## 📈 Features in Dashboard  

- **Top Videos Engagement Analysis**  
- **Engagement Distribution** (likes/comments/shares/views)  
- **Top Creators by Engagement**  
- **Word Cloud + Keyword Frequency Analysis**  
- **Trending Insights & Viral Score Distribution**  
- **AI-Powered Insights** (from Gemini, embedded directly in UI)  

---

## 🔧 Customization Tips  

- Change video count scraped → update `TIKTOK_CONFIG['video_count']` in DAG.  
- Add/remove dashboard charts → follow `create_*_plot` functions in `tiktok_dashboard.py`.  
- Adjust AI analysis prompts → modify prompt templates before sending to Gemini.  

---

## 🧪 Local Testing Notes  

- Ensure Airflow DAG has run at least once so DB has data.  
- Word cloud requires non-empty captions; otherwise shows fallback text.  
- Gemini requires a valid API key with sufficient quota.  

---

## 🚀 Roadmap Ideas  

- Dockerize full stack (Airflow + FastAPI + Postgres) for one-command deploy  
- Add sentiment analysis of captions  
- Incorporate trending music analytics  
- Add forecasting models for engagement growth (ARIMA/Prophet)  
- Add authentication (OAuth/API key) for dashboard access  

---

## 📜 License  
MIT (or your preferred license).  

---

## 🙌 Acknowledgements  
- Apache Airflow  
- FastAPI  
- PostgreSQL  
- TikTokApi  
- Pandas, NumPy, Matplotlib, Seaborn, WordCloud  
- Google Gemini AI  
