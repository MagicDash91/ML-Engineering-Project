# AI Financial Risk & Compliance Monitor

A two-phase AI pipeline for financial risk analysis and regulatory compliance monitoring. Built with LangGraph, CrewAI, NVIDIA LLaMA, and Ollama — runs locally on port **8003**.

---

## Architecture

```
User Input (question + data source)
        │
        ▼
┌─────────────────────────────────┐
│  Phase 1 — LangGraph Planning   │  ← Ollama qwen3.5:cloud
│  5 nodes: plan → data_eng →     │
│  modelling → analysis →         │
│  reporting                      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Human-in-the-Loop (HITL)       │  ← Review plan, approve or abort
│  Auto-approves after 10 min     │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Phase 2 — CrewAI (5 Agents)    │  ← NVIDIA LLaMA 3.3 Nemotron 49B
│  CRO → Risk Data Engineer →     │
│  Risk Scientist → Label         │
│  Encoder → Risk Analyst         │
└────────────┬────────────────────┘
             │
             ▼
   Risk Report · Charts · Compliance Summary · PDF · PPTX
```

---

## Models

| Role                         | Model                                      |
| ---------------------------- | ------------------------------------------ |
| LangGraph planning (5 nodes) | Ollama `qwen3.5:cloud`                     |
| Chart vision analysis        | Ollama `qwen3.5:cloud`                     |
| CrewAI agents (all 5)        | NVIDIA `llama-3.3-nemotron-super-49b-v1.5` |
| Risk report generation       | NVIDIA `llama-3.3-nemotron-super-49b-v1.5` |
| Chat endpoint                | NVIDIA `llama-3.3-nemotron-super-49b-v1.5` |

---

## Features

- **Credit Risk Modelling** — Logistic Regression + Gradient Boosting (PD/LGD)
- **Market Risk** — VaR / CVaR (Historical Simulation), Monte Carlo stress testing (1000 simulations, 30-day horizon)
- **Risk Segmentation** — K-Means 4-tier clustering (Critical / High / Medium / Low)
- **Fraud Detection** — Random Forest + GBM ensemble
- **Risk Visualizations** — Bar, line, scatter, histogram, heatmap, pie, box charts with Ollama vision analysis per chart
- **Compliance Summary** — Basel III, IFRS 9, FRTB, AML automatically generated
- **Report Exports** — Professional PDF (ReportLab) and PowerPoint (python-pptx)
- **LangSmith Tracing** — Full pipeline observability via LangChain tracing
- **Real-time Console** — SSE streaming of all agent activity to the browser

---

## Screenshots

![Application Screenshot 1](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Financial_Risk_Agent/static/h1.JPG)

![Application Screenshot 2](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Financial_Risk_Agent/static/h2.JPG)

![Application Screenshot 3](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Financial_Risk_Agent/static/h3.JPG)

![Application Screenshot 4](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Financial_Risk_Agent/static/h4.JPG)

![Application Screenshot 4](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Financial_Risk_Agent/static/h5.JPG)

## Inputs

At least one data source is required alongside the risk question:

| Input         | Required     | Description                                                                 |
| ------------- | ------------ | --------------------------------------------------------------------------- |
| Risk Question | **Yes**      | What to analyse (e.g. "Assess credit risk and identify default drivers")    |
| Database URI  | One of these | Any SQLAlchemy-compatible URI (PostgreSQL, MySQL, SQLite, MSSQL, Snowflake) |
| File Upload   | One of these | CSV / Excel → loaded into SQLite · PDF / Word / PPTX → text context         |

Both a Database URI and files can be provided simultaneously — tables are merged into a single SQLite database for the run.

---

## UI Tabs

| Tab          | Contents                                          |
| ------------ | ------------------------------------------------- |
| Console      | Real-time agent activity log + HITL review panel  |
| Risk Report  | Full markdown risk report with executive summary  |
| Charts       | Risk visualizations with AI chart analysis        |
| Compliance   | Basel III / IFRS 9 / FRTB / AML compliance status |
| Downloads    | PDF report · PPTX presentation · raw data         |
| Conversation | Chat with the risk analysis results               |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the `Financial_Risk_Agent/` directory:

```env
NVIDIA_API_KEY="nvapi-..."
GOOGLE_API_KEY="AIzaSy..."
TAVILY_API_KEY="tvly-..."
LANGCHAIN_API_KEY="lsv2_pt_..."   # optional — enables LangSmith tracing
```

### 3. Start Ollama

Ensure Ollama is running locally with the `qwen3.5:cloud` model:

```bash
ollama serve
ollama pull qwen3.5:cloud
```

### 4. Run

```bash
python main.py
```

Open **http://localhost:8003** in your browser.

---

## Project Structure

```
Financial_Risk_Agent/
├── main.py                  # Entry point (uvicorn, port 8003)
├── app.py                   # FastAPI backend — pipeline orchestration, SSE, file upload
├── config.py                # API keys, NVIDIA client, LangSmith tracing
├── crew_risk.py             # CrewAI 5-agent sequential crew definition
├── graphs/
│   └── risk_graph.py        # LangGraph 5-node planning pipeline
├── tools/
│   ├── risk_engineer.py     # DB discovery, profiling, ETL, web search
│   ├── risk_scientist.py    # Credit risk, VaR/CVaR, Monte Carlo, segmentation, fraud
│   ├── risk_analyst.py      # Charts, vision analysis, text report generation
│   └── risk_report.py       # PDF and PowerPoint export
├── static/
│   └── index.html           # Bootstrap 5 dark UI (6 tabs)
└── outputs/
    ├── charts/              # Generated PNG charts + session_charts.json
    ├── reports/             # Markdown, PDF, PPTX reports
    ├── models/              # Saved ML models
    └── uploads/             # Per-run SQLite databases
```

---

## API Keys

| Key                 | Source                                             | Purpose                             |
| ------------------- | -------------------------------------------------- | ----------------------------------- |
| `NVIDIA_API_KEY`    | [build.nvidia.com](https://build.nvidia.com)       | CrewAI agents + report generation   |
| `GOOGLE_API_KEY`    | Google AI Studio                                   | Gemini (optional fallback)          |
| `TAVILY_API_KEY`    | [tavily.com](https://tavily.com)                   | Web research for regulatory context |
| `LANGCHAIN_API_KEY` | [smith.langchain.com](https://smith.langchain.com) | LangSmith tracing (optional)        |

---

## Risk Models Detail

### Credit Risk

- **Logistic Regression** — baseline PD model
- **Gradient Boosting** — enhanced default prediction with feature importance ranking

### Market Risk

- **Historical VaR / CVaR** — 95% and 99% confidence levels
- **Monte Carlo Stress Test** — 1000 simulations, 30-day horizon, 2.5σ shock scenario

### Segmentation

- **K-Means (4 tiers)** — Critical / High / Medium / Low risk clusters

### Fraud Detection

- **Random Forest + GBM** — ensemble fraud scoring with anomaly flags

---

## Compliance Coverage

- **Basel III** — Capital adequacy, LCR, NSFR
- **IFRS 9** — Expected Credit Loss (ECL), staging
- **FRTB** — Fundamental Review of the Trading Book (SA / IMA)
- **AML** — Anti-Money Laundering transaction monitoring
