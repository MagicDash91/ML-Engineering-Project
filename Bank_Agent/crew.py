"""
Banking Analytics CrewAI Team
==============================
Implements a hierarchical multi-agent system to automate the banking
data team:

    Manager (CDO)    – delegates and reviews all work
    Data Engineer    – ETL, scraping, PostgreSQL warehouse (NVIDIA LLaMA)
    Data Scientist   – ML models, forecasting, segmentation (NVIDIA LLaMA)
    Data Analyst     – visualizations, insights, PDF/PPT reports (Gemini 2.5 Flash)

LangGraph is called first for strategic pre-planning; the output is
injected into the CrewAI task descriptions so agents have full context.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crewai import Agent, Task, Crew, Process, LLM

from config import NVIDIA_API_KEY, GOOGLE_API_KEY, NVIDIA_MODEL

# ── Tools ─────────────────────────────────────────────────────────────────
from tools.engineer_tools import (
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
from tools.scientist_tools import (
    train_fraud_detection_model,
    train_credit_risk_model,
    train_churn_model,
    time_series_forecast,
    customer_segmentation,
)
from tools.analyst_tools import (
    generate_visualization,
    generate_dashboard,
    generate_text_report,
    label_encode_table,
)
from tools.report_tools import (
    generate_pdf_report,
    generate_ppt_report,
)


# ============================================================
# LLM Configuration
# ============================================================

# NVIDIA Llama Nemotron via OpenAI-compatible endpoint
# CrewAI/LiteLLM uses the "openai/<model>" prefix for custom endpoints
nvidia_llm = LLM(
    model=f"openai/{NVIDIA_MODEL}",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
    temperature=0.1,
    max_tokens=4096,
    timeout=180,
    max_retries=10,
)

# Gemini 2.5 Flash for the Data Analyst (vision capabilities)
gemini_llm_crew = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.1,
    max_tokens=4096,
    timeout=180,
    max_retries=10,
)


# ============================================================
# Agent factory functions
# ============================================================

def create_manager_agent() -> Agent:
    return Agent(
        role="Chief Data Officer (CDO)",
        goal=(
            "Coordinate the banking data team to deliver comprehensive, "
            "actionable analytics. Ensure data quality, oversee model accuracy, "
            "and produce executive-ready insights for banking leadership."
        ),
        backstory=(
            "You are a 15-year veteran CDO at one of Indonesia's top banks. "
            "You have led digital transformation programmes, implemented real-time fraud "
            "detection pipelines, and built regulatory reporting frameworks for Bank Indonesia. "
            "You know exactly how to brief your Data Engineer, Data Scientist, and Data Analyst "
            "to extract maximum business value from data. You communicate findings concisely "
            "to the Board and C-suite."
        ),
        llm=nvidia_llm,
        allow_delegation=True,
        verbose=True,
        max_iter=8,
    )


def create_data_engineer_agent() -> Agent:
    return Agent(
        role="Senior Data Engineer",
        goal=(
            "Collect, transform, and load all required banking and financial data "
            "into the PostgreSQL data warehouse so that downstream teams always have "
            "clean, structured, query-ready tables."
        ),
        backstory=(
            "You are a Senior Data Engineer specialising in financial data pipelines. "
            "You have built batch and streaming ETL systems for major Indonesian banks, "
            "integrating Yahoo Finance, Bloomberg APIs, Bank Indonesia open data, and "
            "real-time web scraping. You are meticulous about data quality, schema design, "
            "and documentation. You always validate row counts and data types after loading."
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


def create_data_scientist_agent() -> Agent:
    return Agent(
        role="Senior Data Scientist",
        goal=(
            "Build, train, and evaluate machine learning models for banking intelligence "
            "covering fraud detection, credit risk, customer churn, time-series forecasting, "
            "and customer segmentation."
        ),
        backstory=(
            "You are a Senior Data Scientist with a PhD in Applied Statistics. "
            "You have deployed fraud detection systems saving Rp 500 billion annually, "
            "built IFRS 9 credit risk models compliant with OJK regulations, and developed "
            "customer lifetime value models for Indonesia's largest retail bank. "
            "You always justify model choices with business impact and regulatory context."
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


def create_data_analyst_agent() -> Agent:
    return Agent(
        role="Senior Data Analyst",
        goal=(
            "Transform data and model outputs into compelling visualizations, "
            "AI-powered business insights (via Gemini vision), and polished "
            "executive reports in PDF and PowerPoint formats."
        ),
        backstory=(
            "You are a Senior Data Analyst with expertise in banking KPI dashboards "
            "(NIM, ROA, ROE, NPL ratio, CAR, BOPO). You have presented to Board members "
            "of BCA, Mandiri, BRI, and BNI. You use Gemini AI to extract nuanced insights "
            "from charts that go beyond the raw numbers, and you craft narratives that "
            "drive strategic decisions. Your reports are always crisp, visual, and actionable."
        ),
        llm=nvidia_llm,               # NVIDIA LLaMA for reasoning/SQL; Gemini used directly inside chart tools for vision
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
        max_iter=20,         # needs 13+ tool calls; 10 was too low
        max_retry_limit=2,
    )


def create_label_encoder_agent() -> Agent:
    return Agent(
        role="Data Preprocessing Analyst",
        goal=(
            "Label-encode all categorical columns in the banking dataset so that "
            "downstream visualizations (especially heatmaps) can include every feature, "
            "not just numeric ones."
        ),
        backstory=(
            "You are a Data Preprocessing specialist who prepares datasets for visual analytics. "
            "Your job is to transform raw tables with mixed data types into fully numeric, "
            "analysis-ready versions stored back in the data warehouse."
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


# ============================================================
# Task factory
# ============================================================

def create_tasks(
    agents: dict,
    analysis_request: str,
    langgraph_plan: dict = None,
) -> list:
    """
    Build the four sequential CrewAI tasks.
    Optionally injects the LangGraph pre-plan into task descriptions.
    """
    manager         = agents["manager"]
    engineer        = agents["engineer"]
    scientist       = agents["scientist"]
    label_encoder   = agents["label_encoder"]
    analyst         = agents["analyst"]

    # Attach LangGraph plan snippets if available
    plan_context = ""
    etl_context  = ""
    ml_context   = ""
    viz_context  = ""
    if langgraph_plan:
        plan_context = f"\n\n[Strategic Plan from CDO]:\n{langgraph_plan.get('analysis_plan', '')[:600]}"
        etl_context  = f"\n\n[ETL Guidance]:\n{langgraph_plan.get('etl_guidance', '')[:400]}"
        ml_context   = f"\n\n[ML Guidance]:\n{langgraph_plan.get('ml_guidance', '')[:400]}"
        viz_context  = f"\n\n[Analytics Guidance]:\n{langgraph_plan.get('analytics_guidance', '')[:400]}"

    # ── Task 1: Data Engineering ──────────────────────────────────
    task_engineer = Task(
        description=(
            f"Inspect and validate all data in the PostgreSQL data warehouse.\n\n"
            f"Analysis Request: {analysis_request}{plan_context}{etl_context}\n\n"
            f"Your step-by-step tasks:\n"
            f"STEP 1 — Profile the existing 'churn' table (PRIORITY):\n"
            f"   - Use profile_database_table with table_name='churn'\n"
            f"   - This table has 22 columns: customerID, gender, SeniorCitizen, Partner,\n"
            f"     Dependents, tenure, PhoneService, MultipleLines, InternetService,\n"
            f"     OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,\n"
            f"     StreamingTV, StreamingMovies, Contract, PaperlessBilling,\n"
            f"     PaymentMethod, MonthlyCharges, TotalCharges, Churn, transaction_date\n"
            f"STEP 2 — Verify the table with a quick query:\n"
            f"   - query_database: SELECT * FROM churn LIMIT 5\n"
            f"   - query_database: SELECT Churn, COUNT(*) FROM churn GROUP BY Churn\n"
            f"STEP 3 — List all tables in the warehouse:\n"
            f"   - Use list_database_tables\n"
            f"STEP 4 — Clean unused columns from the churn table:\n"
            f"   - Call clean_table_columns with table_name='churn'\n"
            f"   - This auto-removes identifier columns (customerID, id, etc.),\n"
            f"     all-null columns, and zero-variance columns.\n"
            f"   - If there are additional irrelevant columns found during profiling\n"
            f"     (e.g. transaction_date if it adds no value), pass them via\n"
            f"     extra_drop_columns='col1,col2'.\n"
            f"   - Confirm which columns were dropped and which were kept.\n"
            f"STEP 5 — Normalize column data types in the churn table:\n"
            f"   - Call normalize_column_dtypes with table_name='churn'\n"
            f"   - This detects columns stored as object/string but containing numeric\n"
            f"     values (e.g. TotalCharges='200.50', SeniorCitizen='1') and casts\n"
            f"     them to float64 or int64 automatically.\n"
            f"   - Report which columns were converted and what their new dtypes are.\n"
            f"STEP 6 — Search for customer churn insights:\n"
            f"   - web_search_collect: 'telecom customer churn reduction strategies 2025'\n"
            f"   - web_search_collect: 'customer retention banking churn prediction 2025'\n"
            f"STEP 7 — Report data quality findings:\n"
            f"   - Row count, remaining column names, final dtypes per column\n"
            f"   - Churn label distribution (Yes vs No counts)\n"
            f"   - Confirm all numeric-looking columns now have correct dtypes\n"
            f"   - Confirm data is clean and ready for ML and visualisation"
        ),
        expected_output=(
            "A data quality report containing:\n"
            "- Full profile of 'churn' table: row count, all column names and dtypes\n"
            "- Null count per column and data quality assessment\n"
            "- Churn label distribution: exact count of Yes vs No\n"
            "- List of columns dropped by clean_table_columns (e.g. customerID)\n"
            "- List of columns converted by normalize_column_dtypes (e.g. TotalCharges → float64)\n"
            "- Final dtype map confirming all numeric-looking columns are now float64/int64\n"
            "- Web research summary (articles collected)\n"
            "- Confirmation all data is clean and ready for ML and analysis"
        ),
        agent=engineer,
    )

    # ── Task 2: Data Science ──────────────────────────────────────
    task_scientist = Task(
        description=(
            f"Build ML models and statistical analyses on the collected banking data.\n\n"
            f"Analysis Request: {analysis_request}{ml_context}\n\n"
            f"Your step-by-step tasks:\n"
            f"1. Call list_database_tables to see all available tables\n"
            f"2. Inspect the data: query_database 'SELECT * FROM churn LIMIT 5'\n"
            f"   - Note which columns are categorical (text) — these will be label-encoded\n"
            f"     automatically inside each modelling tool before training begins.\n"
            f"   - Categorical columns are NEVER dropped; they are always encoded to integers\n"
            f"     so every feature participates in the model.\n"
            f"3. CUSTOMER CHURN MODEL:\n"
            f"   - Run train_churn_model with table_name='churn'\n"
            f"   - Try target_column='Churn' first; if that fails try 'churn' (lowercase)\n"
            f"   - The tool encodes all categorical features before fitting.\n"
            f"4. CUSTOMER SEGMENTATION:\n"
            f"   - Use customer_segmentation on 'churn' with n_clusters=4\n"
            f"   - The tool label-encodes all categorical columns so all features are used,\n"
            f"     not just numeric ones.\n"
            f"5. CREDIT RISK MODEL — if a 'default' column exists in the churn table:\n"
            f"   - Run train_credit_risk_model with table_name='churn', target_column='default'\n"
            f"6. TIME SERIES FORECASTING — if bank stock tables exist:\n"
            f"   - Use time_series_forecast on available bank tables with value_column='Close'\n"
            f"7. Provide detailed model performance metrics and banking business interpretation"
        ),
        expected_output=(
            "Machine learning analysis report containing:\n"
            "- Churn model results: AUC-ROC, accuracy, top churn predictors, retention strategy\n"
            "- Customer segmentation: 4 segment profiles with marketing recommendations\n"
            "- Time series forecast results (if bank data available): 30-day outlook per bank\n"
            "- All model performance metrics with NVIDIA Llama business interpretation\n"
            "- Saved model file paths in outputs/models/\n"
            "- Key risk findings and customer retention recommendations"
        ),
        agent=scientist,
        context=[task_engineer],
    )

    # ── Task 3: Label Encoding ────────────────────────────────────
    task_label_encoder = Task(
        description=(
            f"Prepare the banking dataset for full-feature visual analysis.\n\n"
            f"Analysis Request: {analysis_request}\n\n"
            f"Your step-by-step tasks:\n"
            f"1. Call list_database_tables to confirm the 'churn' table exists\n"
            f"2. Inspect the table: query_database 'SELECT * FROM churn LIMIT 3'\n"
            f"3. Call label_encode_table with table_name='churn'\n"
            f"   - This will create a 'churn_encoded' table where every categorical\n"
            f"     column (gender, Contract, PaymentMethod, Churn, etc.) is replaced\n"
            f"     with its integer label code\n"
            f"4. Verify the output: query_database 'SELECT * FROM churn_encoded LIMIT 3'\n"
            f"5. Report which columns were encoded and confirm the table is ready"
        ),
        expected_output=(
            "Preprocessing report containing:\n"
            "- Confirmation that 'churn_encoded' table was created successfully\n"
            "- List of all categorical columns that were label-encoded\n"
            "- Sample rows from churn_encoded to confirm all columns are now numeric\n"
            "- Row count verification matching the original churn table"
        ),
        agent=label_encoder,
        context=[task_engineer],
    )

    # ── Task 4: Data Analysis & Reporting ────────────────────────
    task_analyst = Task(
        description=(
            f"Create visualizations and professional reports based on the data.\n\n"
            f"Analysis Request: {analysis_request}{viz_context}\n\n"
            f"Guidelines:\n"
            f"- You decide which charts to create based on what the data and ML results suggest.\n"
            f"  Do NOT force a fixed number of charts — only create what adds clear business value.\n"
            f"- For any heatmap, use table='churn_encoded' (all columns are numeric there).\n"
            f"  For all other chart types, use table='churn'.\n"
            f"- Set analyze_with_ai=True on every generate_visualization call.\n\n"
            f"Suggested chart types to consider (pick the most relevant ones):\n"
            f"  - pie / bar   : distributions and comparisons (e.g. Churn, Contract type)\n"
            f"  - histogram   : single-feature distribution (e.g. tenure, MonthlyCharges)\n"
            f"  - histplot    : compare a numeric column across groups using overlapping\n"
            f"                  histograms (e.g. MonthlyCharges split by Churn=Yes/No).\n"
            f"                  Use this instead of boxplot — histplots are intuitive for\n"
            f"                  all audiences, whereas boxplots require statistical training.\n"
            f"  - scatter     : relationship between two numerics (e.g. tenure vs MonthlyCharges)\n"
            f"  - heatmap     : full-feature correlation — use table='churn_encoded'\n"
            f"  - line        : time-series, only if a date/time table is available\n\n"
            f"Required deliverables (always produce these):\n"
            f"1. At least one chart per insight category identified in the ML results\n"
            f"2. A full dashboard: generate_dashboard on table='churn', analysis_focus='customer'\n"
            f"3. A text report: generate_text_report with ALL team analysis results\n"
            f"4. A PDF report: generate_pdf_report with all chart paths\n"
            f"5. A PowerPoint: generate_ppt_report with all chart paths"
        ),
        expected_output=(
            "Complete analytics deliverables package:\n"
            "- Individual chart PNG files in outputs/charts/ — only charts that add value\n"
            "- Heatmap generated from churn_encoded showing all features including categoricals\n"
            "- 1 comprehensive dashboard PNG with full business intelligence analysis\n"
            "- Markdown insight report saved in outputs/reports/\n"
            "- PDF report path (outputs/reports/report_*.pdf)\n"
            "- PowerPoint presentation path (outputs/reports/presentation_*.pptx)\n"
            "- Executive summary with key findings and banking recommendations"
        ),
        agent=analyst,
        context=[task_engineer, task_scientist, task_label_encoder],
    )

    # ── Task 5: Executive Summary (Manager) ──────────────────────
    task_manager = Task(
        description=(
            f"Review all team outputs and produce a final CEO/Board-ready executive briefing.\n\n"
            f"Analysis Request: {analysis_request}\n\n"
            f"Your tasks:\n"
            f"1. Review the Data Engineer's data collection report\n"
            f"2. Review the Data Scientist's model results and forecasts\n"
            f"3. Review the Data Analyst's visualizations and business insights\n"
            f"4. Synthesise into a structured executive briefing covering:\n"
            f"   - Executive Summary (3-4 sentences, data-driven)\n"
            f"   - Top 5 Key Findings (with specific metrics)\n"
            f"   - Risk Assessment (per bank: Low / Medium / High)\n"
            f"   - Strategic Recommendations (top 5, prioritised by impact)\n"
            f"   - Investment Outlook (30-day and 6-month view)\n"
            f"   - Next Steps for the data team\n"
            f"   - Report file paths for board distribution"
        ),
        expected_output=(
            "CEO/Board-ready executive briefing:\n"
            "- Concise executive summary with key numbers\n"
            "- Top 5 findings with quantitative support\n"
            "- Per-bank risk assessment\n"
            "- 5 prioritised strategic recommendations\n"
            "- Investment outlook with forecast data reference\n"
            "- Clear next steps\n"
            "- File paths to all reports for distribution"
        ),
        agent=manager,
        context=[task_engineer, task_scientist, task_label_encoder, task_analyst],
    )

    return [task_engineer, task_scientist, task_label_encoder, task_analyst, task_manager]


# ============================================================
# Crew assembly
# ============================================================

def create_banking_crew(
    analysis_request: str,
    langgraph_plan: dict = None,
) -> Crew:
    """
    Assemble the full banking analytics CrewAI crew.

    Args:
        analysis_request: The banking analysis task.
        langgraph_plan:   Pre-computed plan from LangGraph (injected as context).

    Returns:
        A configured Crew ready to .kickoff().
    """
    agents = {
        "manager":       create_manager_agent(),
        "engineer":      create_data_engineer_agent(),
        "scientist":     create_data_scientist_agent(),
        "label_encoder": create_label_encoder_agent(),
        "analyst":       create_data_analyst_agent(),
    }

    tasks = create_tasks(agents, analysis_request, langgraph_plan)

    crew = Crew(
        agents=[
            agents["engineer"],
            agents["scientist"],
            agents["label_encoder"],
            agents["analyst"],
            agents["manager"],
        ],
        tasks=tasks,
        process=Process.sequential,   # each task runs directly on its assigned agent
        verbose=True,
        memory=False,
    )

    return crew


# ============================================================
# Main entry point (called from main.py)
# ============================================================

def run_banking_analysis_crew(
    analysis_request: str,
    use_langgraph: bool = True,
) -> dict:
    """
    Run the complete banking analytics pipeline:
      Phase 1 – LangGraph strategic planning (optional)
      Phase 2 – CrewAI multi-agent execution

    Args:
        analysis_request: Natural-language analysis task.
        use_langgraph:    Whether to run LangGraph pre-planning first.

    Returns:
        Results dict with keys: langgraph_plan, crew_output, status.
    """
    results = {"analysis_request": analysis_request, "status": "started"}

    # ── Phase 1: LangGraph ────────────────────────────────────────
    langgraph_plan = None
    if use_langgraph:
        print("\n" + "=" * 60)
        print("  PHASE 1 — LangGraph Strategic Planning")
        print("=" * 60)

        from graphs.banking_graph import run_banking_analysis as lg_run
        graph_state = lg_run(analysis_request)

        langgraph_plan = {
            "analysis_plan":      graph_state.get("analysis_plan", ""),
            "etl_guidance":       graph_state.get("etl_guidance", ""),
            "ml_guidance":        graph_state.get("ml_guidance", ""),
            "analytics_guidance": graph_state.get("analytics_guidance", ""),
            "preliminary_report": graph_state.get("report_content", ""),
        }

        results["langgraph_plan"] = langgraph_plan
        print("\n  LangGraph planning complete.\n")
        print(f"  Plan preview:\n{langgraph_plan['analysis_plan'][:400]}...")

    # ── Phase 2: CrewAI ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2 — CrewAI Multi-Agent Execution")
    print("=" * 60)

    crew        = create_banking_crew(analysis_request, langgraph_plan)
    crew_result = crew.kickoff()

    results["crew_output"] = str(crew_result)
    results["status"]      = "completed"

    return results
