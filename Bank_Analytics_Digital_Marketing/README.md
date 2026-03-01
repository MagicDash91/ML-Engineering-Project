# Bank Analytics & Digital Marketing AI System

An end-to-end multi-agent AI system that runs **banking customer churn analysis** and automatically hands the results to a **digital marketing team** to design data-driven retention campaigns — all in one pipeline.

Built with **LangGraph**, **CrewAI**, **FastAPI**, and served through a **Bootstrap 5 dark UI** with real-time streaming logs, a Human-in-the-Loop (HITL) approval gate, and a Gemini-powered chat interface.

---

## Screenshots

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Digital_Marketing_Agent/static/b1.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Digital_Marketing_Agent/static/b2.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Digital_Marketing_Agent/static/b3.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Digital_Marketing_Agent/static/b4.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Digital_Marketing_Agent/static/b5.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Digital_Marketing_Agent/static/b6.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Digital_Marketing_Agent/static/b7.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Digital_Marketing_Agent/static/b8.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Digital_Marketing_Agent/static/b9.JPG)

---

## Architecture Overview

```
User Query (Analysis Request)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 1 — LangGraph Banking Planning (Gemini 2.5 Flash) │
│  5 nodes: plan → ETL guidance → ML guidance →           │
│           analytics guidance → preliminary report        │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  HITL Approval Gate   │  ← Human reviews plan (10-min timeout)
            └───────────┬───────────┘
                        │  Approved
                        ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 2 — Banking CrewAI  (NVIDIA LLaMA Nemotron 49B)  │
│  5 agents (sequential):                                  │
│    Data Engineer → Data Scientist → Label Encoder →      │
│    Data Analyst  → CDO / Manager                         │
│                                                          │
│  Outputs: churn model, segments, charts, PDF/PPTX report │
└───────────────────────┬─────────────────────────────────┘
                        │  analyst_output + scientist_output
                        │  + chart analyses passed down
                        ▼
            ┌────────────────────────────┐
            │  Gemini CDO → Marketing    │  ← Translates analyst findings
            │  Handoff Brief             │     into campaign brief with
            │                            │     specific campaign ideas
            └────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 3a — LangGraph Marketing Planning (Gemini Flash)  │
│  5 nodes: plan → research → strategy → content →        │
│           preliminary marketing brief                    │
│  (informed by banking churn context throughout)          │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 3b — Marketing CrewAI  (Gemini 2.5 Flash)        │
│  4 agents (sequential):                                  │
│    Researcher → Planner/Strategist →                     │
│    Content Maker (+ Veo 3) → CMO / Manager               │
│                                                          │
│  Outputs: ad copy, social posts, email, PDF/PPTX,        │
│           Veo 3 AI videos, executive campaign brief      │
└─────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature                      | Details                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------- |
| **Multi-LLM**                | NVIDIA LLaMA 3.3 Nemotron 49B (banking) + Gemini 2.5 Flash (marketing + vision) |
| **LangGraph**                | Two strategic pre-planning pipelines (banking + marketing), each 5 nodes        |
| **CrewAI**                   | 9 agents total — 5 banking + 4 marketing, all sequential                        |
| **HITL**                     | Human approval gate after Phase 1 with 10-min auto-approve timeout              |
| **AI Video**                 | Google Veo 3 (`veo-3.1-fast-generate-preview`) for marketing video content      |
| **ML Models**                | Random Forest churn prediction, customer segmentation (K-Means, 4 clusters)     |
| **Data Analyst → Marketing** | Analyst raw output + ML results + chart analyses sent directly to marketing     |
| **Gemini Vision**            | Every chart automatically analysed by Gemini for business insights              |
| **Security**                 | PII redaction, SQL injection guard, jailbreak guardrails, audit JSONL log       |
| **LangSmith**                | Full observability tracing (optional)                                           |
| **Real-time UI**             | SSE log streaming, 7-tab Bootstrap 5 dark dashboard                             |

---

## Tech Stack

```
Backend      FastAPI + Uvicorn (port 8002)
Agents       CrewAI (sequential process)
Graphs       LangGraph (StateGraph)
LLMs         NVIDIA LLaMA 3.3 Nemotron Super 49B
             Google Gemini 2.5 Flash
Video        Google Veo 3 (veo-3.1-fast-generate-preview)
Web Search   Tavily API
Database     PostgreSQL (churn data warehouse)
ML           scikit-learn (Random Forest, Gradient Boosting, K-Means)
Reports      ReportLab (PDF), python-pptx (PowerPoint)
Charts       Matplotlib, Seaborn
UI           Bootstrap 5 dark theme, marked.js, highlight.js
Tracing      LangSmith (optional)
```

---

## Project Structure

```
Bank_Analytics_Digital_Marketing/
├── app.py                  # FastAPI backend — 3-phase pipeline + SSE + security
├── main.py                 # Entry point (uvicorn, port 8002)
├── config.py               # API keys, LLM clients, PostgreSQL config
├── crew_banking.py         # Banking CrewAI — 5-agent sequential crew
├── crew_marketing.py       # Marketing CrewAI — 4-agent sequential crew
│
├── graphs/
│   ├── banking_graph.py    # LangGraph 5-node banking planning pipeline
│   └── marketing_graph.py  # LangGraph 5-node marketing planning pipeline
│
├── tools/
│   ├── bank_engineer.py    # ETL, PostgreSQL profiling, web search
│   ├── bank_scientist.py   # Churn model, segmentation, credit risk, fraud detection
│   ├── bank_analyst.py     # Charts (Matplotlib/Seaborn), Gemini vision, dashboards
│   ├── bank_report.py      # PDF + PowerPoint report generation (banking)
│   ├── mkt_researcher.py   # Tavily web search, competitor + audience research
│   ├── mkt_planner.py      # Marketing strategy, content calendar, KPIs, budget
│   ├── mkt_content.py      # Veo 3 video, ad copy, social posts, email templates
│   └── mkt_report.py       # PDF + PowerPoint + Markdown reports (marketing)
│
├── static/
│   └── index.html          # Bootstrap 5 dark UI — 7 tabs
│
└── outputs/
    ├── charts/             # PNG chart files + session_charts.json
    ├── reports/            # PDF, PPTX, Markdown reports
    ├── models/             # Trained .joblib ML models
    ├── videos/             # Veo 3 generated MP4 files
    ├── content/            # Ad copy, social posts, email templates, session JSON
    └── audit/              # conversation_audit.jsonl
```

---

## Agent Teams

### Banking Team — NVIDIA LLaMA 3.3 Nemotron 49B

| Agent              | Role                                                  | Key Tools                                                             |
| ------------------ | ----------------------------------------------------- | --------------------------------------------------------------------- |
| **CDO / Manager**  | Synthesises executive brief, coordinates team         | — (synthesis only)                                                    |
| **Data Engineer**  | ETL, data profiling, PostgreSQL, web research         | `profile_database_table`, `run_etl_pipeline`, `web_search_collect`    |
| **Data Scientist** | Churn prediction (AUC-ROC), segmentation, forecasting | `train_churn_model`, `customer_segmentation`, `time_series_forecast`  |
| **Label Encoder**  | Encodes categoricals for full-feature heatmaps        | `label_encode_table`                                                  |
| **Data Analyst**   | Charts, Gemini vision analysis, PDF/PPTX reports      | `generate_visualization`, `generate_dashboard`, `generate_pdf_report` |

### Marketing Team — Gemini 2.5 Flash

| Agent                    | Role                                                | Key Tools                                                                                   |
| ------------------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **CMO / Manager**        | Executive campaign brief (synthesis)                | — (synthesis only)                                                                          |
| **Researcher**           | Market research, competitors, audience, trends      | `web_search_market`, `analyze_competitors`, `analyze_industry_trends`                       |
| **Planner / Strategist** | Strategy, content calendar, KPIs, budget            | `create_marketing_strategy`, `create_content_calendar`, `define_campaign_kpis`              |
| **Content Maker**        | Veo 3 videos, ad copy, social posts, email, reports | `generate_video_content`, `write_ad_copy`, `generate_social_posts`, `create_email_template` |

---

## Prerequisites

- Python 3.10+
- PostgreSQL with a `churn` table (see schema below)
- API keys for NVIDIA, Google, and Tavily

### PostgreSQL Churn Table Schema

The system expects a `churn` table with these columns:

```sql
customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
PhoneService, MultipleLines, InternetService, OnlineSecurity,
OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
MonthlyCharges, TotalCharges, Churn, transaction_date
```

---

## Installation

**1. Clone and navigate:**

```bash
git clone <repo-url>
cd Bank_Analytics_Digital_Marketing
```

**2. Install dependencies:**

```bash
pip install fastapi uvicorn crewai langchain langchain-google-genai
pip install langgraph openai google-generativeai tavily-python
pip install psycopg2-binary sqlalchemy pandas numpy scikit-learn
pip install matplotlib seaborn reportlab python-pptx joblib
pip install python-dotenv litellm langchain-core
```

**3. Create `.env` file:**

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here
NVIDIA_API_KEY=your_nvidia_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional — enables LangSmith tracing
LANGCHAIN_API_KEY=your_langsmith_api_key_here

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=churn
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here
```

**4. Run:**

```bash
python main.py
```

Open **http://localhost:8002** in your browser.

---

## Usage

### Running an Analysis

1. Open **http://localhost:8002**
2. Enter an **Analysis Request** — e.g. _"How to prevent customer churn?"_
   _(leave blank to run the full default banking analytics stack)_
3. Toggle **LangGraph strategic planning** on/off
4. Click **▶ Start Analysis**

### Pipeline Walkthrough

| Step         | What Happens                                                                                                                     |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| **Phase 1**  | LangGraph runs 5 Gemini nodes to create a strategic banking analysis plan                                                        |
| **HITL**     | A modal appears with the preliminary plan — review and click **Approve** (or wait 10 min for auto-approve)                       |
| **Phase 2**  | 5 banking agents run sequentially: ETL → ML models → encoding → charts → CDO brief                                               |
| **Handoff**  | Gemini translates the Data Analyst's findings into a concrete marketing campaign brief (cashback offers, loyalty programs, etc.) |
| **Phase 3a** | Marketing LangGraph runs 5 Gemini nodes informed by churn context                                                                |
| **Phase 3b** | 4 marketing agents run: Researcher → Planner → Content Maker (Veo 3) → CMO                                                       |
| **Complete** | All 7 tabs populated with results, reports, videos, and downloads                                                                |

### UI Tabs

| Tab                  | Content                                                                          |
| -------------------- | -------------------------------------------------------------------------------- |
| **Console**          | Real-time SSE log stream with colour-coded agent messages                        |
| **Banking Report**   | Gemini-enriched CDO executive brief + per-chart blog sections with AI insights   |
| **Charts**           | Responsive gallery of all generated PNG charts (click to enlarge)                |
| **Marketing Report** | CDO→Marketing handoff brief + CMO executive campaign brief                       |
| **Content**          | Veo 3 video players, social post cards, ad copy sections, email template preview |
| **Downloads**        | All PDF / PPTX / MD reports + MP4 videos with file size and download links       |
| **Conversation**     | Gemini Q&A chat over both banking + marketing results (4 security layers)        |

---

## Outputs

After a full pipeline run the following files are generated:

```
outputs/
├── charts/
│   ├── churn_distribution_pie_*.png
│   ├── tenure_histogram_*.png
│   ├── monthly_charges_histplot_*.png
│   ├── contract_bar_*.png
│   ├── tenure_vs_charges_scatter_*.png
│   ├── feature_correlation_heatmap_*.png
│   └── session_charts.json          ← chart metadata + Gemini analyses
│
├── reports/
│   ├── report_*.pdf                 ← banking PDF report
│   ├── report_*.pptx                ← banking PowerPoint
│   ├── campaign_report_*.pdf        ← marketing PDF report
│   ├── marketing_presentation_*.pptx← marketing PowerPoint
│   └── campaign_report_*.md         ← marketing markdown report
│
├── models/
│   └── churn_model_*.joblib         ← trained churn prediction model
│
├── videos/
│   └── *.mp4                        ← Veo 3 generated campaign videos
│
├── content/
│   ├── research_report_*.md
│   ├── marketing_strategy_*.md
│   ├── campaign_brief_*.md
│   ├── ad_copy_*.md                 ← Google / Meta / LinkedIn ads
│   ├── social_posts_*.json
│   ├── email_template_*.html
│   └── session_content.json         ← all content items index
│
└── audit/
    └── conversation_audit.jsonl     ← chat security audit log
```

---

## Security

The chat interface implements 4 security layers:

| Layer             | What it blocks                                                             |
| ----------------- | -------------------------------------------------------------------------- |
| **PII Redaction** | Emails, phone numbers, NIK/account numbers, card numbers                   |
| **SQL Guard**     | `UNION SELECT`, `DROP TABLE`, `INSERT INTO`, `DELETE FROM`, etc.           |
| **Guardrails**    | Jailbreaks, role overrides, prompt injection, off-topic financial requests |
| **Audit Log**     | Every conversation turn logged to `outputs/audit/conversation_audit.jsonl` |

---

## API Reference

| Method | Endpoint                      | Description                        |
| ------ | ----------------------------- | ---------------------------------- |
| `GET`  | `/`                           | Web UI                             |
| `POST` | `/api/analyze`                | Start a new pipeline run           |
| `GET`  | `/api/stream/{task_id}`       | SSE real-time log stream           |
| `GET`  | `/api/status/{task_id}`       | Task status                        |
| `GET`  | `/api/results/{task_id}`      | Full results (banking + marketing) |
| `GET`  | `/api/logs/{task_id}`         | Buffered logs                      |
| `POST` | `/api/approve/{task_id}`      | HITL approve or abort              |
| `POST` | `/api/chat/{task_id}`         | Chat with Gemini over results      |
| `GET`  | `/api/chat/history/{task_id}` | Chat history                       |
| `GET`  | `/api/charts`                 | List generated charts              |
| `GET`  | `/api/reports`                | List generated reports             |
| `GET`  | `/api/videos`                 | List generated videos              |
| `GET`  | `/api/tasks`                  | All task runs                      |

### POST `/api/analyze` — Request body

```json
{
  "analysis_request": "How to prevent customer churn?",
  "use_langgraph": true
}
```

All other fields (`bank_symbols`, `brand_name`, `industry`, etc.) are optional — the system auto-derives them from the analysis results.

---

## Environment Variables

| Variable            | Required | Description                                  |
| ------------------- | -------- | -------------------------------------------- |
| `GOOGLE_API_KEY`    | ✅       | Google AI API key (Gemini 2.5 Flash + Veo 3) |
| `NVIDIA_API_KEY`    | ✅       | NVIDIA API key (LLaMA 3.3 Nemotron 49B)      |
| `TAVILY_API_KEY`    | ✅       | Tavily search API key                        |
| `LANGCHAIN_API_KEY` | ☐        | LangSmith tracing (optional)                 |
| `POSTGRES_HOST`     | ✅       | PostgreSQL host (default: `localhost`)       |
| `POSTGRES_PORT`     | ✅       | PostgreSQL port (default: `5432`)            |
| `POSTGRES_DB`       | ✅       | Database name (default: `churn`)             |
| `POSTGRES_USER`     | ✅       | PostgreSQL user (default: `postgres`)        |
| `POSTGRES_PASSWORD` | ✅       | PostgreSQL password                          |

---

## How the Data Analyst Feeds the Marketing Team

A core design principle of this system is that the Digital Marketing team's input comes **entirely from the Data Analyst's findings**, not from manual user input.

```
Data Analyst agent
    │  ├─ chart analyses (Gemini vision on each PNG)
    │  ├─ churn model results (AUC-ROC, top predictors)
    │  └─ customer segments (4 K-Means clusters)
    │
    ▼
Gemini CDO → Marketing Handoff
    │  Generates a campaign brief with:
    │  ├─ Named at-risk segments with actual numbers
    │  ├─ Specific churn drivers per segment
    │  ├─ 4–6 named campaign ideas (e.g. "Gebyar Berhadiah",
    │  │   "Cashback 5% for autopayment enrollment",
    │  │   "Upgrade & Hemat — 3-month admin fee discount")
    │  ├─ Priority channels per segment
    │  └─ Specific retention targets (e.g. reduce churn 42% → 28%)
    │
    ▼
Marketing LangGraph (5 nodes, informed by churn context)
    │
    ▼
Marketing CrewAI (4 agents)
    Researcher, Planner, Content Maker (Veo 3), CMO
```

---

## License

MIT License — see `LICENSE` for details.

---

## Acknowledgements

- [CrewAI](https://github.com/joaomdmoura/crewAI) — multi-agent orchestration
- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful agent workflows
- [NVIDIA NIM](https://build.nvidia.com/) — LLaMA 3.3 Nemotron inference
- [Google Gemini](https://ai.google.dev/) — Gemini 2.5 Flash + Veo 3
- [Tavily](https://tavily.com/) — real-time web search for agents
- [LangSmith](https://smith.langchain.com/) — LLM observability
