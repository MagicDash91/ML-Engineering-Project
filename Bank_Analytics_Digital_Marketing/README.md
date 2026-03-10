# Bank Analytics & Digital Marketing AI System

An end-to-end autonomous multi-agent AI system that runs **banking customer churn analysis** and automatically hands the findings to a **digital marketing team** to design data-driven retention campaigns — all in one pipeline, with real-time streaming, a Human-in-the-Loop approval gate, and a Gemini-powered chat interface.

Built with **LangGraph**, **CrewAI**, **FastAPI**, and a **Bootstrap 5 dark UI**.

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

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Digital_Marketing_Agent/static/b10.JPG)

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
            │   HITL Approval Gate  │  ← Human reviews plan (10-min auto-approve)
            └───────────┬───────────┘
                        │  Approved
                        ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 2 — Banking CrewAI  (NVIDIA LLaMA Nemotron 49B)  │
│  5 agents (sequential):                                  │
│    Data Engineer → Data Scientist → Label Encoder →      │
│    Data Analyst  → CDO / Manager                         │
│                                                          │
│  Outputs: churn model · segments · charts · PDF/PPTX     │
└───────────────────────┬─────────────────────────────────┘
                        │  analyst_output + scientist_output
                        │  + Gemini chart vision analyses
                        ▼
            ┌──────────────────────────────┐
            │  Gemini CDO → Marketing      │  ← Translates Data Analyst findings
            │  Handoff Brief               │     into concrete campaign brief
            └──────────────────────────────┘
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
│    Content Maker (Gemini Image Gen) → CMO / Manager      │
│                                                          │
│  Outputs: promotional posters · ad copy · social posts   │
│           email templates · PDF/PPTX · campaign brief    │
└─────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature              | Details                                                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Multi-LLM**        | NVIDIA LLaMA 3.3 Nemotron 49B (banking) + Gemini 2.5 Flash (marketing + vision)                                                      |
| **LangGraph**        | Two strategic pre-planning pipelines (banking + marketing), each 5 nodes                                                             |
| **CrewAI**           | 9 agents total — 5 banking + 4 marketing, all sequential                                                                             |
| **HITL**             | Human approval gate after Phase 1 with 10-min auto-approve timeout                                                                   |
| **AI Posters**       | Gemini image generation (`gemini-3.1-flash-image-preview`) — 1K (1024×1024) promotional posters replacing expensive video generation |
| **ML Models**        | Random Forest churn prediction + K-Means customer segmentation (4 clusters)                                                          |
| **Gemini Vision**    | Every chart automatically analysed by Gemini for business insights                                                                   |
| **Data → Marketing** | Analyst output + ML results + chart analyses fed directly into marketing crew                                                        |
| **OJK Compliance**   | Indonesian banking context — PII redaction, audit trail, guardrails                                                                  |
| **Security**         | SQL injection guard, jailbreak detection, PII redaction, JSONL audit log                                                             |
| **LangSmith**        | Full observability tracing (optional)                                                                                                |
| **Real-time UI**     | SSE log streaming, 7-tab Bootstrap 5 dark dashboard                                                                                  |

---

## Tech Stack

```
Backend       FastAPI + Uvicorn (port 8002)
Agents        CrewAI (sequential process)
Graphs        LangGraph (StateGraph)
LLMs          NVIDIA LLaMA 3.3 Nemotron Super 49B   ← banking analytics
              Google Gemini 2.5 Flash                ← planning, vision, marketing
Image Gen     Gemini 3.1 Flash Image Preview          ← 1K promotional posters ($0.067/image)
Web Search    Tavily API
Database      PostgreSQL (churn data warehouse)
ML            scikit-learn (Random Forest, Gradient Boosting, K-Means)
Reports       ReportLab (PDF), python-pptx (PowerPoint)
Charts        Matplotlib, Seaborn
UI            Bootstrap 5 dark theme, marked.js, highlight.js
Tracing       LangSmith (optional)
```

---

## Project Structure

```
Bank_Analytics_Digital_Marketing/
├── app.py                  # FastAPI backend — 3-phase pipeline, SSE, security
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
│   ├── mkt_content.py      # Gemini image posters, ad copy, social posts, email
│   └── mkt_report.py       # PDF + PowerPoint + Markdown reports (marketing)
│
├── static/
│   └── index.html          # Bootstrap 5 dark UI — 7 tabs
│
└── outputs/
    ├── charts/             # PNG chart files + session_charts.json
    ├── reports/            # PDF, PPTX, Markdown reports
    ├── models/             # Trained .joblib ML models
    ├── posters/            # Gemini AI generated promotional poster PNGs (1K)
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

| Agent                    | Role                                                 | Key Tools                                                                                        |
| ------------------------ | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **CMO / Manager**        | Executive campaign brief (synthesis)                 | — (synthesis only)                                                                               |
| **Researcher**           | Market research, competitors, audience, trends       | `web_search_market`, `analyze_competitors`, `analyze_industry_trends`                            |
| **Planner / Strategist** | Strategy, content calendar, KPIs, budget             | `create_marketing_strategy`, `create_content_calendar`, `define_campaign_kpis`                   |
| **Content Maker**        | AI promotional posters, ad copy, social posts, email | `generate_promotional_poster`, `write_ad_copy`, `generate_social_posts`, `create_email_template` |

---

## Cost Optimisation

The system is designed to keep API costs predictable:

| Asset                          | Model                  | Cost per unit          |
| ------------------------------ | ---------------------- | ---------------------- |
| LangGraph planning nodes (×10) | Gemini 2.5 Flash       | ~$0.30/1M input tokens |
| Banking CrewAI agents          | NVIDIA LLaMA Nemotron  | Pay-per-token via NIM  |
| Chart vision analysis          | Gemini 2.5 Flash       | ~$0.30/1M input tokens |
| Promotional poster (1K)        | Gemini 3.1 Flash Image | **$0.067 per image**   |
| Marketing CrewAI agents        | Gemini 2.5 Flash       | ~$0.30/1M input tokens |

> Promotional posters replaced Google Veo 3.1 video generation which cost **$0.75–$3.00 per 5-second clip** — a 10–40× cost reduction per content asset.

---

## Prerequisites

- Python 3.10+
- PostgreSQL with a `churn` table (see schema below)
- API keys for NVIDIA, Google, and Tavily

### PostgreSQL Churn Table Schema

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
pip install langgraph openai google-generativeai google-genai tavily-python
pip install psycopg2-binary sqlalchemy pandas numpy scikit-learn
pip install matplotlib seaborn reportlab python-pptx joblib pillow
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

| Step           | What Happens                                                                                                                       |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Phase 1**    | LangGraph runs 5 Gemini nodes to produce a structured banking analysis plan                                                        |
| **HITL**       | A modal shows the preliminary plan — click **Approve** or wait 10 min for auto-approve                                             |
| **Phase 2**    | 5 banking agents run sequentially: ETL → ML models → encoding → charts + Gemini vision → CDO brief                                 |
| **Enrichment** | Gemini post-processes the CDO report — enriches recommendations with timelines, keeps chart analyses separate to avoid duplication |
| **Handoff**    | Gemini translates Data Analyst findings into a marketing campaign brief with named segments and specific campaign ideas            |
| **Phase 3a**   | Marketing LangGraph runs 5 Gemini nodes informed by churn context                                                                  |
| **Phase 3b**   | 4 marketing agents run: Researcher → Planner → Content Maker (posters + copy + email) → CMO                                        |
| **Complete**   | All 7 tabs populated with results, reports, posters, and downloads                                                                 |

### UI Tabs

| Tab                  | Content                                                                                                  |
| -------------------- | -------------------------------------------------------------------------------------------------------- |
| **Console**          | Real-time SSE log stream with colour-coded agent messages                                                |
| **Banking Report**   | Gemini-enriched CDO executive brief + per-chart sections with AI vision insights (markdown formatted)    |
| **Charts**           | Responsive gallery of all generated PNG charts (click to enlarge)                                        |
| **Marketing Report** | CDO→Marketing handoff brief + CMO executive campaign brief                                               |
| **Content**          | AI promotional posters (1K), social post cards, ad copy (markdown rendered), email template HTML preview |
| **Downloads**        | All PDF / PPTX / MD reports + poster PNGs with file size and download links                              |
| **Conversation**     | Gemini Q&A chat over both banking + marketing results (4 security layers)                                |

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
│   └── session_charts.json          ← chart metadata + Gemini vision analyses
│
├── reports/
│   ├── report_*.pdf                 ← banking PDF report
│   ├── report_*.pptx                ← banking PowerPoint
│   ├── campaign_report_*.pdf        ← marketing PDF report
│   ├── marketing_presentation_*.pptx
│   └── campaign_report_*.md         ← marketing markdown report
│
├── models/
│   └── churn_model_*.joblib         ← trained churn prediction model
│
├── posters/
│   └── poster_*.png                 ← Gemini AI promotional posters (1024×1024)
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

| Layer             | What it blocks                                                                               |
| ----------------- | -------------------------------------------------------------------------------------------- |
| **PII Redaction** | Emails, Indonesian phone numbers (08xx), NIK, account numbers (16-digit), card numbers, NPWP |
| **SQL Guard**     | `UNION SELECT`, `DROP TABLE`, `INSERT INTO`, `DELETE FROM`, `xp_cmdshell`, etc.              |
| **Guardrails**    | Jailbreaks, role overrides, prompt injection, out-of-scope financial action requests         |
| **Audit Log**     | Every conversation turn logged to `outputs/audit/conversation_audit.jsonl`                   |

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
| `GET`  | `/api/tasks`                  | All task runs                      |

### POST `/api/analyze` — Request body

```json
{
  "analysis_request": "How to prevent customer churn?",
  "use_langgraph": true
}
```

---

## Environment Variables

| Variable            | Required | Description                                             |
| ------------------- | -------- | ------------------------------------------------------- |
| `GOOGLE_API_KEY`    | ✅       | Google AI API key (Gemini 2.5 Flash + image generation) |
| `NVIDIA_API_KEY`    | ✅       | NVIDIA API key (LLaMA 3.3 Nemotron 49B)                 |
| `TAVILY_API_KEY`    | ✅       | Tavily search API key                                   |
| `LANGCHAIN_API_KEY` | ☐        | LangSmith tracing (optional)                            |
| `POSTGRES_HOST`     | ✅       | PostgreSQL host (default: `localhost`)                  |
| `POSTGRES_PORT`     | ✅       | PostgreSQL port (default: `5432`)                       |
| `POSTGRES_DB`       | ✅       | Database name (default: `churn`)                        |
| `POSTGRES_USER`     | ✅       | PostgreSQL user (default: `postgres`)                   |
| `POSTGRES_PASSWORD` | ✅       | PostgreSQL password                                     |

---

## How the Data Analyst Feeds the Marketing Team

A core design principle is that the marketing crew's input comes **entirely from the Data Analyst's findings**, not from manual user input.

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
    │  ├─ 4–6 named campaign ideas
    │  │   (e.g. "Cashback 5% autopayment", "Loyalty Upgrade Programme")
    │  ├─ Priority channels per segment
    │  └─ Specific retention targets (e.g. reduce churn 42% → 28%)
    │
    ▼
Marketing LangGraph (5 nodes, informed by churn context)
    │
    ▼
Marketing CrewAI (4 agents)
    Researcher → Planner → Content Maker (Gemini posters + copy) → CMO
```

---

## License

MIT License — see `LICENSE` for details.

---

## Acknowledgements

- [CrewAI](https://github.com/joaomdmoura/crewAI) — multi-agent orchestration
- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful agent workflows
- [NVIDIA NIM](https://build.nvidia.com/) — LLaMA 3.3 Nemotron inference
- [Google Gemini](https://ai.google.dev/) — Gemini 2.5 Flash + image generation
- [Tavily](https://tavily.com/) — real-time web search for agents
- [LangSmith](https://smith.langchain.com/) — LLM observability
