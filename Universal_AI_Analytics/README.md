# Universal Analytics & Digital Marketing AI System

An end-to-end autonomous multi-agent AI platform that connects to **any data source** — database, CSV, Excel, PDF, Word, PowerPoint, or nothing at all — and runs a full analytics + marketing campaign pipeline powered by **NVIDIA LLaMA**, **Google Gemini**, **LangGraph**, and **CrewAI**.

Not limited to banking or churn. Works universally: fraud detection, credit risk, customer segmentation, time-series forecasting, document analysis, or pure web research mode when no data is provided.

---

## Screenshots

![Application Screenshot 1](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Universal_AI_Analytics/static/b1.JPG)

![Application Screenshot 2](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Universal_AI_Analytics/static/b2.JPG)

![Application Screenshot 3](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Universal_AI_Analytics/static/b3.JPG)

![Application Screenshot 4](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Universal_AI_Analytics/static/b4.JPG)

![Application Screenshot 4](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Universal_AI_Analytics/static/b5.JPG)

![Application Screenshot 4](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Universal_AI_Analytics/static/b6.JPG)

## Architecture Overview

```
User Input (optional: DB URI + files + analysis request)
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│  Adaptive Data Source Resolution                           │
│  ─ Database URI (any SQLAlchemy-compatible DB)             │
│  ─ Structured files: CSV / Excel → SQLite                  │
│  ─ Documents: PDF / Word (.docx) / PowerPoint → text       │
│  ─ Nothing → Tavily web research mode                      │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│  PHASE 1 — LangGraph Banking Planning (Ollama qwen3.5)     │
│  5 nodes: plan → ETL → ML → analytics → preliminary report │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │   HITL Approval Gate  │  ← Human reviews plan (10-min auto-approve)
             └───────────┬───────────┘
                         │ Approved
                         ▼
┌────────────────────────────────────────────────────────────┐
│  PHASE 2 — Analytics CrewAI  (NVIDIA LLaMA Nemotron 49B)   │
│  5 agents (sequential):                                    │
│    Data Engineer → Data Scientist → Label Encoder →        │
│    Data Analyst  → CDO / Manager                           │
│                                                            │
│  Outputs: ML models · segments · charts · PDF/PPTX report  │
└────────────────────────┬───────────────────────────────────┘
                         │  analyst findings + chart vision analyses
                         ▼
             ┌──────────────────────────────┐
             │ Qwen3.5 CDO → Marketing      │  ← Translates findings into
             │  Handoff Brief               │     data-driven campaign brief
             └──────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│  PHASE 3a — LangGraph Marketing Planning (Qwen3.5)         | 
│  5 nodes: plan → research → strategy → content → brief     │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│  PHASE 3b — Marketing CrewAI  (Ollama qwen3.5)             │
│  4 agents (sequential):                                    │
│    Researcher → Planner/Strategist →                       │
│    Content Maker (posters · ad copy · social posts) → CMO  │
└────────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature                  | Details                                                                                                                                                                       |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Universal data input** | Database URI, CSV/Excel (→ SQLite), PDF/Word/PPTX (→ text), or no data (Tavily research mode) — all optional and combinable                                                   |
| **Multi-file upload**    | Upload multiple files at once; structured files auto-merged into one SQLite database                                                                                          |
| **Adaptive pipeline**    | Agents receive only what's available — DB+files, files only, DB only, or pure research                                                                                        |
| **Multi-LLM**            | NVIDIA LLaMA 3.3 Nemotron 49B (analytics) + Gemini 2.5 Flash (planning, vision, posters) + Ollama qwen3.5 (marketing agents, chat tools)                                      |
| **LangGraph**            | Two strategic pre-planning pipelines (analytics + marketing), each 5 nodes, powered by Gemini                                                                                 |
| **CrewAI**               | 9 agents total — 5 analytics + 4 marketing, all sequential                                                                                                                    |
| **HITL**                 | Human approval gate after Phase 1 with 10-min auto-approve timeout                                                                                                            |
| **Agentic chat**         | NVIDIA LLaMA ReAct chat — detects visualization intent, generates real matplotlib/seaborn charts on demand                                                                    |
| **AI posters**           | Gemini image generation (`gemini-3.1-flash-image-preview`) — 1K (1024×1024) promotional posters                                                                               |
| **ML models**            | Fraud detection (RF + GradientBoosting), credit risk (Logistic Regression), churn (GradientBoosting), customer segmentation (K-Means), time-series forecasting (Holt-Winters) |
| **Document analysis**    | Full text extraction from PDF (pdfplumber + pypdf fallback), Word (.docx), PowerPoint (.pptx) — no page/character truncation                                                  |
| **Gemini Vision**        | Every chart automatically analysed by Gemini for business insights                                                                                                            |
| **Security**             | SQL injection guard, jailbreak detection, PII redaction, JSONL audit log                                                                                                      |
| **LangSmith**            | Full observability tracing (optional)                                                                                                                                         |
| **Real-time UI**         | SSE log streaming, modern glassmorphism dark dashboard, 7 tabs                                                                                                                |

---

## Use Cases

The system adapts to whatever data you provide — it is **not limited to banking or churn**:

| Data Provided               | What the Agents Do                                                  |
| --------------------------- | ------------------------------------------------------------------- |
| Customer transaction CSV    | Fraud detection, segmentation, churn prediction, charts             |
| Loan application database   | Credit risk model, Basel III analysis, segment dashboards           |
| Time-series CSV             | Holt-Winters forecasting, trend analysis                            |
| PDF / Word / PPTX documents | Document analysis, key insights extraction, marketing strategy      |
| Mix of files + database     | Combined structured + document analysis                             |
| Nothing (no data)           | Tavily web research mode — builds strategy from market intelligence |

---

## Tech Stack

```
Backend        FastAPI + Uvicorn (port 8002)
Agents         CrewAI (sequential process)
Graphs         LangGraph (StateGraph)
LLMs           NVIDIA LLaMA 3.3 Nemotron Super 49B  ← analytics agents + chat
               Google Gemini 3.1 Veo               ← image gen
               Ollama qwen3.5:cloud                  ← marketing agents, chart analysis, planning, vision,
Image Gen      Gemini 3.1 Flash Image Preview         ← 1K promotional posters ($0.067/image)
Web Search     Tavily API
Databases      Any SQLAlchemy-compatible DB (PostgreSQL, MySQL, SQLite, Snowflake, BigQuery)
               + auto SQLite for uploaded CSV/Excel files
Document Parse pdfplumber / pypdf (PDF), python-docx (Word), python-pptx (PPTX)
ML             scikit-learn (Random Forest, Gradient Boosting, Logistic Regression, K-Means)
               statsmodels (Holt-Winters time-series)
Reports        ReportLab (PDF), python-pptx (PowerPoint)
Charts         Matplotlib, Seaborn
UI             Custom glassmorphism dark design system (CSS custom properties, no Bootstrap)
Tracing        LangSmith (optional)
```

---

## Project Structure

```
Bank_Analytics_Digital_Marketing/
├── app.py                  # FastAPI backend — 3-phase pipeline, SSE, security, chat
├── main.py                 # Entry point (uvicorn, port 8002)
├── config.py               # API keys, LLM clients, DB config
├── crew_banking.py         # Analytics CrewAI — 5-agent sequential crew
├── crew_marketing.py       # Marketing CrewAI — 4-agent sequential crew
│
├── graphs/
│   ├── banking_graph.py    # LangGraph 5-node analytics planning pipeline
│   └── marketing_graph.py  # LangGraph 5-node marketing planning pipeline
│
├── tools/
│   ├── bank_engineer.py    # ETL, DB profiling, multi-DB support, Tavily research
│   ├── bank_scientist.py   # Fraud, credit risk, churn, segmentation, forecasting
│   ├── bank_analyst.py     # Charts (Matplotlib/Seaborn), Gemini vision, dashboards, reports
│   ├── bank_report.py      # PDF + PowerPoint report generation (analytics)
│   ├── mkt_researcher.py   # Tavily web search, competitor + audience research
│   ├── mkt_planner.py      # Marketing strategy, content calendar, KPIs, budget
│   ├── mkt_content.py      # Gemini image posters, ad copy, social posts
│   └── mkt_report.py       # PDF + PowerPoint + Markdown reports (marketing)
│
├── static/
│   └── index.html          # Modern glassmorphism dark UI — 7 tabs
│
└── outputs/
    ├── charts/             # PNG chart files + session_charts.json
    ├── reports/            # PDF, PPTX, Markdown reports
    ├── models/             # Trained .joblib ML models
    ├── posters/            # Gemini AI promotional poster PNGs (1K)
    ├── content/            # Ad copy, social posts, session JSON
    ├── uploads/            # Uploaded files + combined SQLite databases
    └── audit/              # conversation_audit.jsonl
```

---

## Agent Teams

### Analytics Team — NVIDIA LLaMA 3.3 Nemotron 49B

| Agent              | Role                                                                      | Key Tools                                                                                                                      |
| ------------------ | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **CDO / Manager**  | Synthesises executive brief, coordinates team                             | — (synthesis only)                                                                                                             |
| **Data Engineer**  | ETL, data profiling, multi-DB support, web research                       | `profile_database_table`, `run_etl_pipeline`, `web_search_collect`                                                             |
| **Data Scientist** | Fraud detection, credit risk, churn prediction, segmentation, forecasting | `train_fraud_detection_model`, `train_credit_risk_model`, `train_churn_model`, `customer_segmentation`, `time_series_forecast` |
| **Label Encoder**  | Encodes categoricals for full-feature heatmaps                            | `label_encode_table`                                                                                                           |
| **Data Analyst**   | Charts, Gemini vision analysis, PDF/PPTX reports                          | `generate_visualization`, `generate_dashboard`, `generate_pdf_report`                                                          |

### Marketing Team — Ollama qwen3.5:cloud

| Agent                    | Role                                           | Key Tools                                                                      |
| ------------------------ | ---------------------------------------------- | ------------------------------------------------------------------------------ |
| **CMO / Manager**        | Executive campaign brief (synthesis)           | — (synthesis only)                                                             |
| **Researcher**           | Market research, competitors, audience, trends | `web_search_market`, `analyze_competitors`, `analyze_industry_trends`          |
| **Planner / Strategist** | Strategy, content calendar, KPIs, budget       | `create_marketing_strategy`, `create_content_calendar`, `define_campaign_kpis` |
| **Content Maker**        | AI promotional posters, ad copy, social posts  | `generate_promotional_poster`, `write_ad_copy`, `generate_social_posts`        |

---

## Agentic Chat (Conversation Tab)

The chat interface is powered by **NVIDIA LLaMA 3.3 Nemotron** with intent-based visualization:

- **Visualization requests** — keywords like "chart", "visualize", "bar chart", "histogram" trigger a dedicated two-pass flow:
  1. NVIDIA LLaMA generates executable matplotlib/seaborn Python code
  2. Code is executed safely (blocked: `os`, `sys`, `subprocess`, etc.) and the PNG is rendered inline in the chat
  3. NVIDIA LLaMA provides a natural-language explanation of the chart and key insights

- **Conversational questions** — answered directly from the analytics report, chart analyses, and marketing brief context

- **4 security layers**: PII redaction, SQL injection guard, jailbreak detection, JSONL audit log

---

## Data Input Options

All three input methods are **fully optional** and can be combined:

| Input                | Format                                                        | How Agents Use It                                            |
| -------------------- | ------------------------------------------------------------- | ------------------------------------------------------------ |
| **Database URI**     | Any SQLAlchemy URI (postgres/mysql/sqlite/snowflake/bigquery) | Direct SQL queries, schema discovery, ML training            |
| **Structured files** | `.csv`, `.xlsx`, `.xls`                                       | Loaded into SQLite → same ML pipeline as database            |
| **Documents**        | `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`                      | Full text extracted (no truncation) → document analysis mode |
| **Nothing**          | —                                                             | Agents use Tavily web search to research the topic           |

Multiple files of any type can be uploaded simultaneously. Structured files are auto-merged into one SQLite database (each file becomes a named table).

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
pip install sqlalchemy pandas numpy scikit-learn statsmodels
pip install matplotlib seaborn reportlab python-pptx joblib pillow
pip install python-dotenv litellm langchain-core langchain-community
pip install pdfplumber pypdf python-docx
pip install psycopg2-binary   # if using PostgreSQL
```

**3. Install and run Ollama (for marketing agents and chart analysis):**

```bash
# Install Ollama: https://ollama.com
ollama pull qwen3.5:cloud
```

**4. Create `.env` file:**

```env
# Required for analytics agents + chat
NVIDIA_API_KEY=your_nvidia_api_key_here

# Required for LangGraph planning, chart vision, AI posters
GOOGLE_API_KEY=your_google_api_key_here

# Required for web research (market research + Tavily mode)
TAVILY_API_KEY=your_tavily_api_key_here

# Optional — enables LangSmith tracing
LANGCHAIN_API_KEY=your_langsmith_api_key_here

# Optional — only needed if using PostgreSQL as default fallback
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=your_database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here
```

**5. Run:**

```bash
python main.py
```

Open **http://localhost:8002** in your browser.

---

## Usage

### Starting an Analysis

1. Open **http://localhost:8002**
2. (Optional) Enter a **Database URI** — e.g. `postgresql://user:pass@localhost:5432/mydb`
3. (Optional) Upload files — drag & drop CSV, Excel, PDF, Word, or PowerPoint (multiple files supported)
4. (Optional) Enter an **Analysis Request** — leave blank to auto-generate based on available data
5. Configure **Campaign Settings** in the sidebar (brand, industry, audience, budget)
6. Click **Run Analysis**

All inputs are optional — the system adapts to whatever is provided.

### Pipeline Walkthrough

| Step                | What Happens                                                                                              |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| **Data Resolution** | Structured files → SQLite; documents → text extraction; DB URI → direct connection; nothing → Tavily mode |
| **Phase 1**         | LangGraph runs 5 Gemini nodes producing a structured analytics plan                                       |
| **HITL**            | Plan preview shown in UI — click **Approve** or wait 10 min for auto-approve                              |
| **Phase 2**         | 5 analytics agents: ETL → ML models → encoding → charts + Gemini vision → CDO brief                       |
| **Handoff**         | Gemini translates analyst findings into a marketing campaign brief                                        |
| **Phase 3a**        | Marketing LangGraph runs 5 Gemini nodes informed by analytics context                                     |
| **Phase 3b**        | 4 marketing agents: Researcher → Planner → Content Maker → CMO                                            |
| **Complete**        | All 7 tabs populated with results, charts, posters, reports, and downloads                                |

### UI Tabs

| Tab                  | Content                                                                          |
| -------------------- | -------------------------------------------------------------------------------- |
| **Console**          | Real-time SSE log stream with colour-coded agent messages (noise-filtered)       |
| **Analytics Report** | Gemini-enriched CDO executive brief + per-chart sections with AI vision insights |
| **Charts**           | Responsive gallery of all generated PNG charts (click to enlarge)                |
| **Marketing Report** | Handoff brief + CMO executive campaign brief (markdown rendered)                 |
| **Content**          | AI promotional posters, social post cards, ad copy                               |
| **Downloads**        | All PDF / PPTX / MD reports + poster PNGs with file size and download links      |
| **Conversation**     | NVIDIA LLaMA agentic chat — ask questions or request charts generated on demand  |

---

## Security

The chat interface implements 4 security layers:

| Layer             | What it blocks                                                                   |
| ----------------- | -------------------------------------------------------------------------------- |
| **PII Redaction** | Emails, phone numbers (08xx format), NIK, 16-digit account numbers, card numbers |
| **SQL Guard**     | `UNION SELECT`, `DROP TABLE`, `INSERT INTO`, `DELETE FROM`, `xp_cmdshell`, etc.  |
| **Guardrails**    | Jailbreaks, role overrides, prompt injection, out-of-scope requests              |
| **Audit Log**     | Every conversation turn logged to `outputs/audit/conversation_audit.jsonl`       |

Visualization code execution is sandboxed — blocked modules: `os`, `sys`, `subprocess`, `socket`, `requests`, `pathlib`, `eval`, `exec`, `open`.

---

## API Reference

| Method | Endpoint                      | Description                                    |
| ------ | ----------------------------- | ---------------------------------------------- |
| `GET`  | `/`                           | Web UI                                         |
| `POST` | `/api/analyze`                | Start a new pipeline run                       |
| `POST` | `/api/upload`                 | Upload one or more files (multipart/form-data) |
| `GET`  | `/api/stream/{task_id}`       | SSE real-time log stream                       |
| `GET`  | `/api/status/{task_id}`       | Task status                                    |
| `GET`  | `/api/results/{task_id}`      | Full results (analytics + marketing)           |
| `GET`  | `/api/logs/{task_id}`         | Buffered logs                                  |
| `POST` | `/api/approve/{task_id}`      | HITL approve or abort                          |
| `POST` | `/api/chat/{task_id}`         | Agentic chat (NVIDIA LLaMA + visualization)    |
| `GET`  | `/api/chat/history/{task_id}` | Chat history                                   |
| `GET`  | `/api/charts`                 | List generated charts                          |
| `GET`  | `/api/reports`                | List generated reports                         |
| `GET`  | `/api/tasks`                  | All task runs                                  |

### POST `/api/analyze` — Request body

```json
{
  "database_uri": "postgresql://user:pass@localhost:5432/mydb",
  "uploaded_file_ids": ["uuid1", "uuid2"],
  "analysis_request": "Analyse fraud patterns and segment customers",
  "use_langgraph": true,
  "brand_name": "MyBrand",
  "industry": "Finance",
  "target_audience": "Retail banking customers",
  "campaign_goals": "Reduce fraud, improve retention",
  "budget": "$50,000",
  "campaign_type": "Retention"
}
```

All fields are optional — provide only what applies.

### POST `/api/upload` — Response

```json
{
  "files": [
    {
      "file_id": "uuid",
      "original_name": "data.csv",
      "file_type": "structured",
      "format": ".csv",
      "table_name": "data",
      "rows": 5000,
      "columns": ["id", "amount", "is_fraud"]
    },
    {
      "file_id": "uuid2",
      "original_name": "report.pdf",
      "file_type": "document",
      "format": ".pdf",
      "char_count": 18423
    }
  ],
  "count": 2
}
```

---

## Environment Variables

| Variable            | Required | Description                                             |
| ------------------- | -------- | ------------------------------------------------------- |
| `NVIDIA_API_KEY`    | ✅       | NVIDIA NIM API key (LLaMA 3.3 Nemotron 49B)             |
| `GOOGLE_API_KEY`    | ✅       | Google AI API key (Gemini 2.5 Flash + image generation) |
| `TAVILY_API_KEY`    | ✅       | Tavily search API key (web research)                    |
| `LANGCHAIN_API_KEY` | ☐        | LangSmith tracing (optional)                            |
| `POSTGRES_HOST`     | ☐        | PostgreSQL host — only needed as default DB fallback    |
| `POSTGRES_PORT`     | ☐        | PostgreSQL port (default: `5432`)                       |
| `POSTGRES_DB`       | ☐        | Database name                                           |
| `POSTGRES_USER`     | ☐        | PostgreSQL user                                         |
| `POSTGRES_PASSWORD` | ☐        | PostgreSQL password                                     |

---

## License

MIT License — see `LICENSE` for details.

---

## Acknowledgements

- [CrewAI](https://github.com/joaomdmoura/crewAI) — multi-agent orchestration
- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful agent workflows
- [NVIDIA NIM](https://build.nvidia.com/) — LLaMA 3.3 Nemotron inference
- [Google Gemini](https://ai.google.dev/) — Gemini 2.5 Flash + image generation
- [Ollama](https://ollama.com/) — local LLM inference (qwen3.5)
- [Tavily](https://tavily.com/) — real-time web search for agents
- [LangSmith](https://smith.langchain.com/) — LLM observability
