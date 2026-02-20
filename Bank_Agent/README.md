# Banking Analytics Multi-Agent System

An AI-powered banking data team built with **CrewAI**, **LangGraph**, **FastAPI**, and **PostgreSQL**. The system orchestrates five specialized agents that automatically profile data, clean and normalize datasets, train machine learning models, generate visualizations, and produce executive-ready PDF and PowerPoint reports — all driven from a browser UI with real-time live logs.

---

## Screenshots

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Bank_Agent/static/b1.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Bank_Agent/static/b2.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Bank_Agent/static/b3.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Bank_Agent/static/b4.JPG)

---

## Architecture Overview

```
User Browser
     │
     ▼
FastAPI (app.py)  ──  SSE live log stream
     │
     ▼
Phase 1 — LangGraph Strategic Planning
     │   plan → data_engineering → data_science → data_analysis → reporting
     │   (all nodes powered by NVIDIA LLaMA Nemotron)
     │
     ▼
Phase 2 — CrewAI Multi-Agent Execution (sequential)
     │
     ├── 1. Data Engineer        (NVIDIA LLaMA)
     │       profile → clean columns → normalize dtypes → web search
     │
     ├── 2. Data Scientist       (NVIDIA LLaMA)
     │       label encode → churn model → segmentation → feature importance charts
     │
     ├── 3. Data Preprocessing   (NVIDIA LLaMA)
     │       label encode all categoricals → churn_encoded table
     │
     ├── 4. Data Analyst         (NVIDIA LLaMA reasoning + Gemini vision)
     │       visualizations → dashboard → text report → PDF → PowerPoint
     │
     └── 5. Manager / CDO        (NVIDIA LLaMA)
             executive briefing → top findings → recommendations
```

---

## Agent Responsibilities

| Agent | LLM | Key Tasks |
|---|---|---|
| **Chief Data Officer (Manager)** | NVIDIA LLaMA | Delegates work, synthesizes final executive briefing |
| **Senior Data Engineer** | NVIDIA LLaMA | Profiles table, cleans unused columns, normalizes dtypes, web research |
| **Senior Data Scientist** | NVIDIA LLaMA | Trains churn / fraud / credit risk models, K-Means segmentation, feature importance charts |
| **Data Preprocessing Analyst** | NVIDIA LLaMA | Label-encodes all categorical columns → `churn_encoded` table for full-feature heatmap |
| **Senior Data Analyst** | NVIDIA LLaMA (reasoning) + **Gemini 2.5 Flash** (vision) | Creates visualizations, dashboard, text/PDF/PPT reports |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | [CrewAI](https://crewai.com) |
| Strategic planning | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| LLM — reasoning & SQL | NVIDIA LLaMA 3.3 Nemotron Super 49B |
| LLM — chart vision | Google Gemini 2.5 Flash |
| Web framework | FastAPI + Uvicorn |
| Data warehouse | PostgreSQL (SQLAlchemy) |
| ML models | scikit-learn (GradientBoosting, RandomForest, LogisticRegression, KMeans) |
| Forecasting | statsmodels (Holt-Winters Exponential Smoothing) |
| Visualization | Seaborn + Matplotlib |
| Report generation | ReportLab (PDF), python-pptx (PowerPoint) |
| Observability | LangSmith tracing |
| Web search | Tavily |
| Market data | yfinance |

---

## Features

- **LangGraph pre-planning** — The CDO agent strategically plans data needs, ML tasks, and visualization requirements before any execution begins
- **Automated data cleaning** — Removes identifier columns (`customerID`), all-null columns, and zero-variance columns automatically
- **Dtype normalization** — Detects columns stored as strings but containing numeric values (e.g. `TotalCharges = "200.50"`) and casts them to `float64` / `int64`
- **Label encoding** — Encodes all categorical columns for full-feature correlation heatmaps and K-Means segmentation
- **Top 10 Feature Importance charts** — Auto-generated for every classification model (churn, fraud, credit risk)
- **Gemini vision analysis** — Every chart is sent to Gemini 2.5 Flash for business insight extraction (rate-limited to 3 requests / 60 seconds)
- **Seaborn-based visualizations** — All charts use seaborn (`barplot`, `lineplot`, `scatterplot`, `histplot`, `heatmap`); histplot replaces boxplot for audience-friendly distribution comparisons
- **Real-time SSE log streaming** — The browser receives live agent logs via Server-Sent Events
- **One-click reports** — PDF and PowerPoint executive reports generated automatically

---

## Project Structure

```
Bank_Agent/
├── app.py                  # FastAPI backend + SSE streaming
├── main.py                 # CLI entry point
├── crew.py                 # CrewAI agents, tasks, crew assembly
├── config.py               # API keys, LLM clients, PostgreSQL config
│
├── graphs/
│   └── banking_graph.py    # LangGraph strategic planning pipeline
│
├── tools/
│   ├── engineer_tools.py   # ETL, profiling, cleaning, normalization
│   ├── scientist_tools.py  # ML model training + feature importance
│   ├── analyst_tools.py    # Visualization, Gemini vision, reports
│   └── report_tools.py     # PDF and PowerPoint generation
│
├── static/
│   └── index.html          # Bootstrap browser UI
│
└── outputs/
    ├── charts/             # Generated PNG charts
    ├── reports/            # PDF, PPTX, Markdown reports
    └── models/             # Saved .joblib model files
```

---

## Prerequisites

- Python 3.10+
- PostgreSQL running locally (default: `localhost:5432`, database: `bank_analytics`)
- API keys for NVIDIA, Google Gemini, Tavily, and LangSmith

---

## Installation

```bash
git clone https://github.com/MagicDash91/ML-Engineering-Project.git
cd ML-Engineering-Project/Bank_Agent
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the `Bank_Agent/` directory:

```env
# LLM APIs
NVIDIA_API_KEY=your_nvidia_api_key
GOOGLE_API_KEY=your_google_gemini_api_key

# Web Search
TAVILY_API_KEY=your_tavily_api_key

# LangSmith Observability (optional)
LANGCHAIN_API_KEY=your_langsmith_api_key

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=bank_analytics
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

---

## Running the Application

```bash
# Standard
python main.py

# Custom port
python main.py --port 9000

# Development mode with hot-reload
python main.py --reload
```

Then open **http://localhost:8000** in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Browser UI |
| `POST` | `/api/analyze` | Start a new analysis run |
| `GET` | `/api/stream/{task_id}` | SSE live log stream |
| `GET` | `/api/status/{task_id}` | Task status |
| `GET` | `/api/results/{task_id}` | Full results (report + artifacts) |
| `GET` | `/api/logs/{task_id}` | All buffered log lines |
| `GET` | `/api/charts` | List generated chart PNGs |
| `GET` | `/api/reports` | List PDF / PPTX / MD reports |
| `GET` | `/api/tasks` | List all task runs |

---

## Data Pipeline Flow

```
PostgreSQL (churn table)
        │
        ▼
[Engineer]  Profile → Clean columns → Normalize dtypes → Web research
        │
        ▼
[Scientist] Label encode → Train models → Generate feature importance charts
        │
        ▼
[Preprocessor] Label encode categoricals → Save churn_encoded table
        │
        ▼
[Analyst]  Generate charts (Seaborn) → Gemini vision analysis → Dashboard → PDF + PPT
        │
        ▼
[Manager]  Executive briefing → Top 5 findings → Recommendations
```

---

## ML Models

| Model | Algorithm | Target | Metric |
|---|---|---|---|
| Customer Churn | Gradient Boosting | `Churn` (Yes/No) | AUC-ROC |
| Fraud Detection | Random Forest + Gradient Boosting | `is_fraud` | AUC-ROC |
| Credit Risk | Logistic Regression | `default` | AUC-ROC |
| Customer Segmentation | K-Means (4 clusters) | — | Silhouette |
| Stock Forecasting | Holt-Winters | `Close` price | RMSE |

---

## License

MIT License
