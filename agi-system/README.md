# AGI System

A self-directing multi-agent AI platform. You ask a question — the system automatically designs a team of AI agents, executes them step-by-step, evaluates each output, and synthesises a structured final answer in markdown.

Runs locally on **gpt-oss:120b-cloud** via Ollama. No cloud LLM required.

---

## Screenshots

![Application Screenshot 1](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/agi-system/static/a1.JPG)

![Application Screenshot 2](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/agi-system/static/a2.JPG)

![Application Screenshot 3](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/agi-system/static/a3.JPG)

![Application Screenshot 4](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/agi-system/static/a4.JPG)

![Application Screenshot 5](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/agi-system/static/a5.JPG)

![Application Screenshot 6](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/agi-system/static/a6.JPG)

---

## How it works

```
User question
      │
      ▼
 Meta-Planner          ← gpt-oss generates a JSON execution plan (3–5 steps)
      │
      ▼
 Dynamic Agents        ← CrewAI spawns one agent per step with the right role + tools
      │
      ▼
 Step Evaluator        ← gpt-oss scores each output 1–10; retries if score < 7 (max 2×)
      │
      ▼
 Final Synthesis       ← gpt-oss writes a complete markdown answer from all step outputs
```

Each step gets a specific **role**, **goal**, **backstory**, and a set of **tools** — all decided automatically by the meta-planner based on your question.

---

## Quick start

**Requirements:** Python 3.11+, Ollama running `gpt-oss:120b-cloud`, dependencies from `Universal_AI_Analytics`.

```bash
cd d:\Langsmith-main\agi-system
python main.py
```

Open **http://localhost:8003** in your browser.

---

## UI

ChatGPT-like interface:

- **Left sidebar** — conversation history, new chat button
- **Chat area** — user bubbles (right), AI responses (left)
- **Thinking section** — collapsible, shows each step as it runs with live scores
- **Final Answer** — full markdown rendered response (headers, tables, code blocks, etc.)
- **📎 Attach** — upload files (see below)
- **🗄 DB** — optional database URI for direct DB analysis

---

## File upload

Attach files directly in the chat input. The system handles them automatically:

| Format | What happens |
|---|---|
| `.csv` | Loaded into SQLite, agents query it via SQL |
| `.xlsx` / `.xls` | Loaded into SQLite, agents query it via SQL |
| `.pdf` | Text extracted, passed as document context |
| `.docx` / `.doc` | Text extracted, passed as document context |
| `.pptx` / `.ppt` | Text extracted, passed as document context |

Multiple structured files are merged into one SQLite database. Multiple documents are concatenated.

---

## Available tools (33)

The meta-planner selects tools automatically. You never specify them manually.

### Data Engineering
| Tool | Description |
|---|---|
| `list_database_tables` | Discover all tables in the connected database |
| `profile_database_table` | Row count, columns, dtypes, nulls, sample values |
| `query_database` | Execute a SQL SELECT query |
| `clean_table_columns` | Remove ID/UUID and low-value columns |
| `normalize_column_dtypes` | Cast object columns to numeric types |
| `run_etl_pipeline` | Extract, transform, load across tables |
| `web_search_collect` | Search the web and save results to the database |
| `fetch_financial_data` | Fetch historical stock/market data |

### Data Science
| Tool | Description |
|---|---|
| `train_churn_model` | Gradient Boosting churn prediction, auto-detects churn column |
| `train_fraud_detection_model` | Random Forest + GBM fraud detection |
| `train_credit_risk_model` | Logistic Regression credit risk model |
| `customer_segmentation` | K-Means clustering into 4 customer segments |
| `time_series_forecast` | Holt-Winters forecasting from date + value columns |

### Visualization & Reporting
| Tool | Description |
|---|---|
| `generate_visualization` | Bar, pie, scatter, histogram, or heatmap chart |
| `generate_dashboard` | 7-panel analytics dashboard PNG |
| `generate_text_report` | Professional markdown business insight report |
| `label_encode_table` | Label-encode categoricals for ML or heatmaps |
| `generate_pdf_report` | Downloadable PDF report with embedded charts |
| `generate_ppt_report` | 16:9 PowerPoint presentation |

### Marketing Research
| Tool | Description |
|---|---|
| `web_search_market` | Market research, industry trends, news |
| `analyze_competitors` | Competitive landscape matrix |
| `research_target_audience` | Demographics, psychographics, pain points |
| `analyze_industry_trends` | Market size, macro trends, opportunities |
| `exa_web_search` | Neural web search via Exa AI |

### Marketing Planning
| Tool | Description |
|---|---|
| `create_marketing_strategy` | Full go-to-market strategy document |
| `create_content_calendar` | 30-day content publishing schedule |
| `define_campaign_kpis` | Success metrics and measurement framework |
| `create_campaign_brief` | One-page campaign brief |
| `plan_budget_allocation` | Channel-by-channel budget recommendation |

### Marketing Content
| Tool | Description |
|---|---|
| `write_ad_copy` | Platform-specific ad copy (Google, Meta, LinkedIn) |
| `generate_social_posts` | Social media posts with hashtags |
| `create_email_template` | HTML email with subject, body, CTA |
| `generate_promotional_poster` | AI-generated promotional poster image (Gemini) |

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload files; returns `file_id` per file |
| `POST` | `/api/solve` | Start a pipeline; returns `task_id` |
| `GET` | `/api/stream/{task_id}` | SSE stream of real-time pipeline events |
| `GET` | `/api/status/{task_id}` | Full task status + final answer |
| `GET` | `/api/tasks` | List all tasks |
| `GET` | `/api/logs/{task_id}` | Event log for a task |

### POST /api/solve
```json
{
  "question": "Why is our customer churn increasing?",
  "db_uri": "postgresql://user:pass@localhost:5432/mydb",
  "uploaded_file_ids": ["20240518_120000_001"]
}
```

### SSE event types
| Type | Meaning |
|---|---|
| `info` | General pipeline status |
| `plan` | Meta-planner finished; payload contains full plan |
| `step_start` | Agent step beginning |
| `step_output` | Agent step completed |
| `eval` | Evaluator score for the step |
| `retry` | Step failed evaluation, retrying |
| `synthesis_start` | Final synthesis beginning |
| `done` | Pipeline complete; payload contains `final_answer` |
| `error` | Pipeline error |

---

## Configuration

The system loads `.env` from `../Universal_AI_Analytics/.env`. Required keys depend on which tools are used:

| Key | Required for |
|---|---|
| `TAVILY_API_KEY` | Web search tools |
| `EXA_API_KEY` | `exa_web_search` |
| `GOOGLE_API_KEY` | `generate_promotional_poster` (Gemini) |
| `LANGCHAIN_API_KEY` | LangSmith tracing (optional) |
| `ZEP_API_KEY` | Zep memory (optional) |
| `POSTGRES_HOST/DB/USER/PASSWORD` | PostgreSQL database tools |

Ollama must be running locally on port **11434** with `gpt-oss:120b-cloud` available.

---

## Project structure

```
agi-system/
├── main.py            — uvicorn entry point (port 8003)
├── app.py             — FastAPI routes, file upload, pipeline orchestration
├── config.py          — env loading, query_ollama(), Ollama/Gemini setup
├── meta_planner.py    — LLM generates JSON execution plan from user question
├── dynamic_graph.py   — Runs plan steps, retry logic, final synthesis
├── agent_factory.py   — Spawns CrewAI agent per step with role + tools
├── step_evaluator.py  — LLM-as-judge: scores 1–10, pass if ≥ 7
├── tool_registry.py   — 33 tools mapped from Universal_AI_Analytics
└── static/
    └── index.html     — ChatGPT-like frontend (marked.js, SSE)
```
