"""
crew_banking.py – Banking Analytics CrewAI 5-agent sequential crew.
Adapted from Bank_Agent/crew.py for the combined system.

Agents:
  1. Data Engineer    – ETL, profiling, PostgreSQL
  2. Data Scientist   – ML models, segmentation
  3. Label Encoder    – encode categoricals for heatmaps
  4. Data Analyst     – charts, Gemini vision, reports
  5. CDO / Manager    – synthesises executive brief
"""

import os
import sys

os.environ["OTEL_SDK_DISABLED"]         = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import litellm
    litellm.request_timeout = 45
    litellm.drop_params     = True
    litellm.set_verbose     = False
except Exception:
    pass

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

from config import NVIDIA_API_KEY, GOOGLE_API_KEY, NVIDIA_MODEL

# ── Banking tool imports ──────────────────────────────────────────────────────
from tools.bank_engineer import (
    profile_database_table,
    fetch_financial_data,
    fetch_indonesian_bank_data,
    web_search_collect,
    run_etl_pipeline,
    list_database_tables,
    query_database,
    clean_table_columns,
    normalize_column_dtypes,
)
from tools.bank_scientist import (
    train_fraud_detection_model,
    train_credit_risk_model,
    train_churn_model,
    time_series_forecast,
    customer_segmentation,
)
from tools.bank_analyst import (
    generate_visualization,
    generate_dashboard,
    generate_text_report,
    label_encode_table,
)
from tools.bank_report import (
    generate_pdf_report,
    generate_ppt_report,
)


# ── LLM configuration ─────────────────────────────────────────────────────────

nvidia_llm = LLM(
    model=f"openai/{NVIDIA_MODEL}",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
    temperature=0.1,
    max_tokens=4096,
    timeout=45,
    max_retries=2,
)


# ── Agent factory ─────────────────────────────────────────────────────────────

def _build_banking_agents() -> dict:
    manager = Agent(
        role="Chief Data Officer (CDO)",
        goal=(
            "Coordinate the banking data team to deliver comprehensive, actionable analytics. "
            "Synthesise all outputs into an executive-ready brief for banking leadership."
        ),
        backstory=(
            "You are a 15-year veteran CDO at one of Indonesia's top banks. "
            "You have led digital transformation programmes and built regulatory reporting "
            "frameworks. You brief the Board concisely with data-backed insights."
        ),
        llm=nvidia_llm,
        allow_delegation=False,
        verbose=True,
        max_iter=15,
    )

    engineer = Agent(
        role="Senior Data Engineer",
        goal=(
            "Collect, transform, and load all required banking data into PostgreSQL "
            "so that downstream teams have clean, query-ready tables."
        ),
        backstory=(
            "You are a Senior Data Engineer specialising in financial data pipelines. "
            "You have built ETL systems for major Indonesian banks. "
            "You validate data quality and schema design after every load."
        ),
        llm=nvidia_llm,
        tools=[
            profile_database_table,
            fetch_financial_data,
            fetch_indonesian_bank_data,
            web_search_collect,
            run_etl_pipeline,
            list_database_tables,
            query_database,
            clean_table_columns,
            normalize_column_dtypes,
        ],
        allow_delegation=False,
        verbose=True,
        max_iter=15,
        max_retry_limit=3,
    )

    scientist = Agent(
        role="Senior Data Scientist",
        goal=(
            "Build and evaluate ML models covering churn prediction, customer segmentation, "
            "credit risk, fraud detection, and time-series forecasting."
        ),
        backstory=(
            "You are a Senior Data Scientist with a PhD in Applied Statistics. "
            "You have deployed fraud detection and IFRS 9 credit risk models compliant with OJK regulations."
        ),
        llm=nvidia_llm,
        tools=[
            train_fraud_detection_model,
            train_credit_risk_model,
            train_churn_model,
            time_series_forecast,
            customer_segmentation,
            query_database,
            list_database_tables,
        ],
        allow_delegation=False,
        verbose=True,
        max_iter=15,
        max_retry_limit=3,
    )

    label_encoder = Agent(
        role="Data Preprocessing Analyst",
        goal=(
            "Label-encode all categorical columns in the banking dataset so that "
            "visualizations (especially heatmaps) can include every feature."
        ),
        backstory=(
            "You are a Data Preprocessing specialist who transforms raw tables with "
            "mixed data types into fully numeric, analysis-ready versions."
        ),
        llm=nvidia_llm,
        tools=[
            label_encode_table,
            list_database_tables,
            query_database,
        ],
        allow_delegation=False,
        verbose=True,
        max_iter=6,
        max_retry_limit=2,
    )

    analyst = Agent(
        role="Senior Data Analyst",
        goal=(
            "Transform data and model outputs into compelling visualizations, "
            "AI-powered business insights (via Gemini vision), and polished reports."
        ),
        backstory=(
            "You are a Senior Data Analyst with expertise in banking KPI dashboards "
            "(NIM, ROA, ROE, NPL, CAR, BOPO). You use Gemini AI to extract insights from charts."
        ),
        llm=nvidia_llm,
        tools=[
            generate_visualization,
            generate_dashboard,
            generate_text_report,
            generate_pdf_report,
            generate_ppt_report,
            query_database,
            list_database_tables,
        ],
        allow_delegation=False,
        verbose=True,
        max_iter=30,
        max_retry_limit=3,
    )

    return {
        "manager":       manager,
        "engineer":      engineer,
        "scientist":     scientist,
        "label_encoder": label_encoder,
        "analyst":       analyst,
    }


# ── Task factory ──────────────────────────────────────────────────────────────

def _build_banking_tasks(agents: dict, analysis_request: str, langgraph_plan: dict = None) -> list:
    plan_context = ""
    etl_context  = ""
    ml_context   = ""
    viz_context  = ""

    if langgraph_plan:
        plan_context = f"\n\n[Strategic Plan from CDO]:\n{langgraph_plan.get('analysis_plan', '')[:600]}"
        etl_context  = f"\n\n[ETL Guidance]:\n{langgraph_plan.get('etl_guidance', '')[:400]}"
        ml_context   = f"\n\n[ML Guidance]:\n{langgraph_plan.get('ml_guidance', '')[:400]}"
        viz_context  = f"\n\n[Analytics Guidance]:\n{langgraph_plan.get('analytics_guidance', '')[:400]}"

    task_engineer = Task(
        description=(
            f"Inspect and validate all data in the PostgreSQL data warehouse.\n\n"
            f"Analysis Request: {analysis_request}{plan_context}{etl_context}\n\n"
            "STEP 1 — Profile the existing 'churn' table:\n"
            "   - Use profile_database_table with table_name='churn'\n"
            "STEP 2 — Verify with queries:\n"
            "   - query_database: SELECT * FROM churn LIMIT 5\n"
            "   - query_database: SELECT Churn, COUNT(*) FROM churn GROUP BY Churn\n"
            "STEP 3 — List all tables: list_database_tables\n"
            "STEP 4 — Clean unused columns: clean_table_columns with table_name='churn'\n"
            "STEP 5 — Normalize column dtypes: normalize_column_dtypes with table_name='churn'\n"
            "STEP 6 — Web search for churn insights:\n"
            "   - web_search_collect: 'telecom customer churn reduction strategies 2025'\n"
            "   - web_search_collect: 'customer retention banking churn prediction 2025'\n"
            "STEP 7 — Report data quality findings."
        ),
        expected_output=(
            "A data quality report: row count, column names/dtypes, null counts, "
            "Churn distribution, dropped/converted columns, web research summary."
        ),
        agent=agents["engineer"],
    )

    task_scientist = Task(
        description=(
            f"Build ML models on the banking data.\n\n"
            f"Analysis Request: {analysis_request}{ml_context}\n\n"
            "1. list_database_tables to see available tables\n"
            "2. query_database: SELECT * FROM churn LIMIT 5\n"
            "3. CUSTOMER CHURN MODEL: train_churn_model with table_name='churn', target_column='Churn'\n"
            "   (try 'Churn' first, then 'churn' if needed)\n"
            "4. CUSTOMER SEGMENTATION: customer_segmentation on 'churn' with n_clusters=4\n"
            "5. TIME SERIES FORECASTING: if bank stock tables exist, use time_series_forecast\n"
            "6. Provide detailed model performance metrics and business interpretation."
        ),
        expected_output=(
            "ML analysis: churn AUC-ROC, top churn predictors, 4 customer segments, "
            "retention program, model paths in outputs/models/."
        ),
        agent=agents["scientist"],
        context=[task_engineer],
    )

    task_label_encoder = Task(
        description=(
            f"Prepare the dataset for full-feature visual analysis.\n\n"
            "1. list_database_tables to confirm 'churn' table exists\n"
            "2. query_database: SELECT * FROM churn LIMIT 3\n"
            "3. label_encode_table with table_name='churn'\n"
            "   This creates 'churn_encoded' with all categorical columns as integers\n"
            "4. Verify: query_database: SELECT * FROM churn_encoded LIMIT 3\n"
            "5. Report which columns were encoded."
        ),
        expected_output=(
            "Confirmation 'churn_encoded' table created, list of encoded columns, "
            "sample rows showing all-numeric columns."
        ),
        agent=agents["label_encoder"],
        context=[task_engineer],
    )

    task_analyst = Task(
        description=(
            f"Create visualizations and professional reports.\n\n"
            f"Analysis Request: {analysis_request}{viz_context}\n\n"
            "Guidelines: For heatmap use table='churn_encoded'. For all other types use table='churn'.\n"
            "Gemini vision analysis is applied automatically to every chart.\n\n"
            "HOW TO CALL generate_data_visualization:\n"
            "  generate_data_visualization('churn', 'pie', 'Churn', '', 'Churn Distribution', '')\n"
            "  generate_data_visualization('churn', 'histogram', 'tenure', '', '', '')\n"
            "  generate_data_visualization('churn', 'histplot', 'MonthlyCharges', '', '', 'Churn')\n"
            "  generate_data_visualization('churn', 'scatter', 'tenure', 'MonthlyCharges', '', 'Churn')\n"
            "  generate_data_visualization('churn', 'bar', 'Contract', 'MonthlyCharges', '', '')\n"
            "  generate_data_visualization('churn_encoded', 'heatmap', 'Churn', '', '', '')\n\n"
            "Required deliverables:\n"
            "1. Individual charts (at least 5 meaningful charts — DO NOT generate a dashboard)\n"
            "2. A text report: generate_text_report with ALL team analysis results\n"
            "3. A PDF report: generate_pdf_report with all chart paths\n"
            "4. A PowerPoint: generate_ppt_report with all chart paths"
        ),
        expected_output=(
            "Chart PNGs in outputs/charts/, markdown insight report in outputs/reports/, "
            "PDF path (outputs/reports/report_*.pdf), PPTX path."
        ),
        agent=agents["analyst"],
        context=[task_engineer, task_scientist, task_label_encoder],
    )

    task_manager = Task(
        description=(
            f"Review all team outputs and produce a final executive briefing.\n\n"
            f"Analysis Request: {analysis_request}\n\n"
            "Synthesise a structured executive briefing covering:\n"
            "1. Executive Summary (3-4 sentences, data-driven)\n"
            "2. Top 5 Key Findings (with specific metrics)\n"
            "3. Risk Assessment\n"
            "4. Strategic Recommendations (top 5, prioritised)\n"
            "5. Customer Churn Insights (for handoff to Digital Marketing team)\n"
            "6. Investment Outlook\n"
            "7. Next Steps\n"
            "Do NOT include file paths or system locations."
        ),
        expected_output=(
            "CEO/Board-ready executive briefing with quantitative support, "
            "risk assessment, strategic recommendations, and churn insights for marketing handoff."
        ),
        agent=agents["manager"],
        context=[task_engineer, task_scientist, task_label_encoder, task_analyst],
    )

    return [task_engineer, task_scientist, task_label_encoder, task_analyst, task_manager]


# ── Public API ────────────────────────────────────────────────────────────────

def run_banking_crew(analysis_request: str, langgraph_plan: dict = None) -> dict:
    """
    Run the Banking Analytics CrewAI crew (Phase 2).
    Called by app.py after HITL approval.

    Returns:
        dict with 'crew_output' (str) and 'status'.
    """
    agents = _build_banking_agents()
    tasks  = _build_banking_tasks(agents, analysis_request, langgraph_plan)

    crew = Crew(
        agents=[
            agents["engineer"],
            agents["scientist"],
            agents["label_encoder"],
            agents["analyst"],
            agents["manager"],
        ],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=False,
    )

    print("\n[CrewAI-Banking] ══ Starting 5-agent sequential Banking crew ══\n", flush=True)
    result = crew.kickoff()
    print("\n[CrewAI-Banking] ══ Banking crew complete ══\n", flush=True)

    # Extract individual agent outputs for downstream handoff
    # task order: 0=engineer, 1=scientist, 2=label_encoder, 3=analyst, 4=manager(CDO)
    scientist_output = ""
    analyst_output   = ""
    try:
        if hasattr(result, "tasks_output") and len(result.tasks_output) >= 5:
            scientist_output = str(result.tasks_output[1])
            analyst_output   = str(result.tasks_output[3])
            print(f"[CrewAI-Banking] Data Scientist output captured ({len(scientist_output)} chars)",
                  flush=True)
            print(f"[CrewAI-Banking] Data Analyst output captured ({len(analyst_output)} chars)",
                  flush=True)
    except Exception as _e:
        print(f"[CrewAI-Banking] Could not extract task outputs: {_e}", flush=True)

    return {
        "crew_output":       str(result),     # CDO executive brief (last task)
        "analyst_output":    analyst_output,  # Data Analyst's raw output
        "scientist_output":  scientist_output, # Data Scientist's ML results
        "status":            "completed",
    }
