"""
crew_banking.py – Universal Data Analytics CrewAI 5-agent sequential crew.

Agents:
  1. Data Engineer    – database discovery, ETL, profiling, quality checks
  2. Data Scientist   – ML models, segmentation, forecasting
  3. Label Encoder    – encode categoricals for heatmaps
  4. Data Analyst     – charts, Ollama vision analysis, reports
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

# ── Analytics tool imports ────────────────────────────────────────────────────
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
    timeout=300,
    max_retries=2,
)


# ── Agent factory ─────────────────────────────────────────────────────────────

def _build_analytics_agents() -> dict:
    manager = Agent(
        role="Chief Data Officer (CDO)",
        goal=(
            "Oversee the entire data analytics team and ensure each agent delivers "
            "high-quality, actionable outputs. Synthesise all findings — engineering "
            "quality report, ML model results, encoded dataset confirmation, and visual "
            "analytics — into a single executive-ready briefing that leadership can act on. "
            "The briefing must be specific, quantitative, and domain-aware."
        ),
        backstory=(
            "You are a seasoned CDO with 15 years leading data transformation programmes "
            "across retail, finance, healthcare, logistics, and e-commerce. You have a track "
            "record of turning raw database tables into board-level decision intelligence. "
            "You know that great analytics starts with data quality and ends with a narrative "
            "that non-technical executives can understand and act on. "
            "You brief leadership with precision: no vague statements, always backed by numbers."
        ),
        llm=nvidia_llm,
        allow_delegation=False,
        verbose=True,
        max_iter=15,
    )

    engineer = Agent(
        role="Senior Data Engineer",
        goal=(
            "Auto-discover the full database schema, profile every relevant table, "
            "validate data quality end-to-end, and deliver a clean, analysis-ready dataset. "
            "Identify the primary analysis table, its target or outcome column, its key "
            "numeric and categorical features, and any data issues (nulls, type mismatches, "
            "ID columns, duplicates). Leave the database in a clean state for downstream ML "
            "and visualization work."
        ),
        backstory=(
            "You are a Senior Data Engineer with deep expertise in building ETL pipelines "
            "for diverse industries. You never assume what the data contains — you always "
            "start by listing tables and inspecting schemas. You have seen databases with "
            "churn data, sales records, medical records, IoT sensor logs, and financial "
            "transactions — and you approach each one the same way: list, profile, sample, "
            "clean, normalize. You are thorough: you check every column for nulls, verify "
            "data types, drop ID columns that add no analytical value, and convert object "
            "columns that should be numeric. You document every finding."
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
            "Select and train the most appropriate ML models given the data schema discovered "
            "by the Data Engineer. Produce quantitative model performance metrics, identify the "
            "top predictive features, and generate customer or entity segments. "
            "Provide clear business interpretations of every model result — not just numbers, "
            "but what they mean for the organisation."
        ),
        backstory=(
            "You are a Senior Data Scientist with a PhD in Applied Statistics and 12 years "
            "of industry experience. You have deployed classification, regression, clustering, "
            "and time-series models across banking, retail, telecom, and healthcare. "
            "You always inspect the data schema before modelling. If you see a binary column "
            "(churn, fraud, default, outcome, label), you build a classification model. "
            "If you see a continuous target (revenue, price, quantity), you explore regression. "
            "If you see time-indexed data, you consider forecasting. "
            "You always run customer_segmentation to uncover hidden audience clusters "
            "that the marketing team can target. "
            "You explain feature importance in plain English: not just 'feature X has importance 0.32' "
            "but 'customers with attribute X are 3× more likely to churn'."
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
            "Transform the primary dataset into a fully numeric version by label-encoding "
            "all categorical columns. This encoded table is required by the Data Analyst "
            "to generate a full-feature correlation heatmap. "
            "Verify the encoded table is correct and report which columns were transformed."
        ),
        backstory=(
            "You are a Data Preprocessing specialist with expertise in feature engineering "
            "for machine learning pipelines. You understand that categorical variables — "
            "like gender, contract type, payment method, product category, or region — "
            "must be converted to integers before correlation analysis and many ML algorithms. "
            "You always verify the result: the encoded table must have zero object-type columns, "
            "only integers and floats."
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
            "Produce a comprehensive visual analytics package: at least 5 meaningful charts "
            "covering distribution, relationships, and trends in the data; "
            "a full-feature correlation heatmap using the encoded table; "
            "a markdown insight report combining all team findings; "
            "a professional PDF report; and a PowerPoint presentation. "
            "Every chart must be analysed by Qwen AI vision to extract key business insights."
        ),
        backstory=(
            "You are a Senior Data Analyst with 10 years of experience building KPI dashboards "
            "and insight reports across industries. You know that the best charts tell a story: "
            "they show the most important patterns, segment comparisons, and risk signals. "
            "You use Qwen AI to add AI-powered interpretation to every chart automatically. "
            "You are disciplined about calling the visualization tool correctly: "
            "one JSON dict per call, never an array, always using exact column names from the data."
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

def _build_analytics_tasks(agents: dict, analysis_request: str, langgraph_plan: dict = None) -> list:
    plan_context = ""
    etl_context  = ""
    ml_context   = ""
    viz_context  = ""

    if langgraph_plan:
        plan_context = f"\n\n[Strategic Plan from CDO]:\n{langgraph_plan.get('analysis_plan', '')[:600]}"
        etl_context  = f"\n\n[ETL Guidance]:\n{langgraph_plan.get('etl_guidance', '')[:400]}"
        ml_context   = f"\n\n[ML Guidance]:\n{langgraph_plan.get('ml_guidance', '')[:400]}"
        viz_context  = f"\n\n[Analytics Guidance]:\n{langgraph_plan.get('analytics_guidance', '')[:400]}"

    # ── Task 1: Data Engineering ───────────────────────────────────────────────
    task_engineer = Task(
        description=(
            f"You are the first agent in the pipeline. Your job is to fully discover, "
            f"profile, and clean the data in the connected database.\n\n"
            f"Analysis Request: {analysis_request}{plan_context}{etl_context}\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 1 — DISCOVER: List all available tables\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: list_database_tables\n"
            "   Goal: Identify which tables exist. The primary analysis table is the one\n"
            "         with the most rows and analytical columns (not just lookup tables).\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 2 — PROFILE: Deep-dive into the primary table\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: profile_database_table with the primary table name\n"
            "   Goal: Get full row count, column names, dtypes, null counts,\n"
            "         numeric stats (mean, std, min, max), and categorical distributions.\n"
            "   Document: How many rows? How many columns? Any columns with >20% nulls?\n"
            "             Which columns are numeric? Which are categorical?\n"
            "             What is the likely target/outcome column?\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 3 — SAMPLE: Inspect actual data\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: query_database\n"
            "   Queries to run:\n"
            "     a) SELECT * FROM <primary_table> LIMIT 5\n"
            "        → See real values, confirm column names and types\n"
            "     b) SELECT <target_col>, COUNT(*) FROM <primary_table> GROUP BY <target_col>\n"
            "        → Understand the distribution of the target/outcome column\n"
            "     c) SELECT COUNT(*) FROM <primary_table>\n"
            "        → Confirm total row count\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 4 — CLEAN: Remove noise columns\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: clean_table_columns with the primary table name\n"
            "   Goal: Drop ID columns, UUID fields, and other non-analytical columns.\n"
            "         These add noise to ML models and visualizations.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 5 — NORMALIZE: Fix data types\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: normalize_column_dtypes with the primary table name\n"
            "   Goal: Convert object columns that contain numeric values to float/int.\n"
            "         This is critical — columns like 'TotalCharges' stored as text\n"
            "         will break ML models and numeric charts if not converted.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 6 — RESEARCH: Find external context\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: web_search_collect\n"
            "   Goal: Run 2 web searches relevant to the data domain discovered.\n"
            "   Examples (adapt to your domain):\n"
            "     - If churn data: 'customer churn reduction strategies 2025'\n"
            "     - If sales data: 'sales forecasting best practices 2025'\n"
            "     - If fraud data: 'fraud detection trends financial services 2025'\n"
            "     - If healthcare: 'patient outcome prediction analytics 2025'\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 7 — REPORT: Summarise all findings\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Write a structured data quality report that includes:\n"
            "   - Primary table name and total row/column count\n"
            "   - Column inventory: name, dtype, null count, example values\n"
            "   - Identified target column and its class distribution\n"
            "   - Columns dropped (step 4) and columns type-converted (step 5)\n"
            "   - Top 3 data quality issues found\n"
            "   - Web research summary (2 key insights from external sources)\n"
            "   - Recommendation for the Data Scientist on which model to use\n"
        ),
        expected_output=(
            "Structured data quality report including: primary table name, row/column counts, "
            "full column inventory with dtypes and null counts, target column distribution, "
            "list of dropped and converted columns, top data quality issues, "
            "web research summary, and model recommendation for the Data Scientist."
        ),
        agent=agents["engineer"],
    )

    # ── Task 2: Data Science ───────────────────────────────────────────────────
    task_scientist = Task(
        description=(
            f"You are the second agent in the pipeline. Your job is to build ML models "
            f"appropriate for the data schema the Data Engineer discovered.\n\n"
            f"Analysis Request: {analysis_request}{ml_context}\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 1 — ORIENT: Understand what data is available\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: list_database_tables\n"
            "   Then: query_database → SELECT * FROM <primary_table> LIMIT 5\n"
            "   Goal: Confirm column names and types. Read the Data Engineer's report\n"
            "         to know which table is primary and what the target column is.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 2 — CHOOSE & TRAIN: Select the right model(s)\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Use this decision logic to choose models:\n\n"
            "   IF the primary table has a binary column named 'Churn', 'churn', 'churned',\n"
            "   'is_churn', or similar:\n"
            "     → train_churn_model(table_name=<table>, target_column='Churn')\n"
            "       Try 'Churn' first; if it fails, try 'churn' (lowercase)\n\n"
            "   IF the primary table has a column named 'fraud', 'is_fraud', 'Fraud',\n"
            "   'isFraud', or similar:\n"
            "     → train_fraud_detection_model(table_name=<table>, target_column='isFraud')\n\n"
            "   IF the primary table has a column named 'default', 'Default', 'risk',\n"
            "   'credit_risk', or similar:\n"
            "     → train_credit_risk_model(table_name=<table>, target_column='default')\n\n"
            "   IF the primary table has a date/time column AND a numeric value column:\n"
            "     → time_series_forecast(table_name=<table>, date_column=<col>, value_column=<col>)\n\n"
            "   IF none of the above match:\n"
            "     → Try train_churn_model on the most likely binary column you can find\n"
            "     → Or run customer_segmentation only if no binary target exists\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 3 — ALWAYS RUN: Customer Segmentation\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: customer_segmentation(table_name=<primary_table>, n_clusters=4)\n"
            "   Goal: Discover 4 distinct audience segments in the data.\n"
            "         These segments are critical for the Digital Marketing team —\n"
            "         they tell us which groups to target with different campaigns.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 4 — INTERPRET: Translate numbers to business language\n"
            "═══════════════════════════════════════════════════════════\n"
            "   For every model trained, provide:\n"
            "   a) Performance metrics: AUC-ROC, accuracy, precision, recall (for classifiers)\n"
            "      OR RMSE, MAE, R² (for regression/forecasting)\n"
            "   b) Top 5 predictive features with plain-English explanation:\n"
            "      NOT 'feature importance: 0.32' — INSTEAD 'Customers on month-to-month\n"
            "      contracts are 2.8× more likely to churn than annual contract holders'\n"
            "   c) Segment profiles: name each cluster (e.g. 'High-Value Loyalists',\n"
            "      'Price-Sensitive Switchers', 'At-Risk Digitals', 'New Onboarders')\n"
            "      and describe their key characteristics\n"
            "   d) Business recommendation: what action does this model recommend?\n"
            "      (e.g. 'Target Segment 2 with personalised retention offers immediately')\n"
        ),
        expected_output=(
            "Detailed ML analysis including: model type and rationale, AUC-ROC or equivalent, "
            "top 5 features with plain-English business explanation, 4 named customer segments "
            "with profiles, specific business recommendations per segment, "
            "and all model file paths in outputs/models/."
        ),
        agent=agents["scientist"],
        context=[task_engineer],
    )

    # ── Task 3: Label Encoding ─────────────────────────────────────────────────
    task_label_encoder = Task(
        description=(
            f"You are the third agent in the pipeline. Your job is to create a fully numeric "
            f"version of the primary dataset so the Data Analyst can generate a complete "
            f"correlation heatmap covering ALL features.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 1 — CONFIRM: Verify the primary table exists\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: list_database_tables\n"
            "   Goal: Confirm the primary table name from the Data Engineer's report.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 2 — INSPECT: Check which columns need encoding\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: query_database → SELECT * FROM <primary_table> LIMIT 3\n"
            "   Goal: Identify all object/string columns that need to be converted\n"
            "         to integers. Common examples: gender, contract type, payment method,\n"
            "         region, category, product type, status.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 3 — ENCODE: Create the numeric version\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: label_encode_table(table_name=<primary_table>)\n"
            "   Goal: This creates '<primary_table>_encoded' in the database.\n"
            "         Every categorical column becomes an integer code.\n"
            "         Example: 'Yes'=1, 'No'=0; 'Male'=1, 'Female'=0;\n"
            "         'Month-to-month'=0, 'One year'=1, 'Two year'=2\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 4 — VERIFY: Confirm all columns are now numeric\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: query_database → SELECT * FROM <primary_table>_encoded LIMIT 3\n"
            "   Goal: Confirm there are NO object-type columns remaining.\n"
            "         All columns should be integers or floats.\n"
            "         Report the exact encoded table name to the Data Analyst.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 5 — REPORT: Document the encoding\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Report:\n"
            "   - Encoded table name (e.g. 'churn_encoded')\n"
            "   - List of columns that were encoded and their mapping logic\n"
            "   - Total column count in the encoded table\n"
            "   - Confirm the table is ready for heatmap generation\n"
        ),
        expected_output=(
            "Confirmation that '<primary_table>_encoded' was successfully created. "
            "Full list of encoded columns with their label mapping. "
            "Sample rows showing all-numeric columns. "
            "Exact table name for the Data Analyst to use in the heatmap."
        ),
        agent=agents["label_encoder"],
        context=[task_engineer],
    )

    # ── Task 4: Data Analysis & Visualization ─────────────────────────────────
    task_analyst = Task(
        description=(
            f"You are the fourth agent in the pipeline. Your job is to produce a complete "
            f"visual analytics package with at least 5 charts, a business insight report, "
            f"a PDF report, and a PowerPoint presentation.\n\n"
            f"Analysis Request: {analysis_request}{viz_context}\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "CRITICAL TOOL CALL RULES — READ BEFORE CALLING ANY TOOL\n"
            "═══════════════════════════════════════════════════════════\n"
            "Tool name: 'Generate Data Visualization'\n"
            "• Call it ONCE per chart — one JSON dict per call\n"
            "• NEVER pass a list or array of dicts\n"
            "• Use EXACT column names from the database (case-sensitive)\n"
            "• For heatmap: always use the '_encoded' table\n"
            "• For all other charts: use the original table\n\n"
            "Correct format (do this):\n"
            '  {"table_name": "<table>", "chart_type": "pie", "x_column": "<target_col>", '
            '"y_column": "", "title": "Target Distribution", "hue_column": ""}\n\n'
            "Wrong format (never do this):\n"
            '  [{"table_name": "..."}, {"table_name": "..."}]\n\n'

            "═══════════════════════════════════════════════════════════\n"
            "STEP 1 — ORIENT: Know your data\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: list_database_tables → confirm table names available\n"
            "   Tool: query_database → SELECT * FROM <primary_table> LIMIT 3\n"
            "   Goal: Know the exact column names before generating charts.\n"
            "         Note: the '_encoded' table is for heatmap only.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 2 — GENERATE CHARTS (at least 5, one tool call each)\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Generate these chart types using real column names from the data:\n\n"
            "   CHART A — Target Distribution (pie chart)\n"
            '     {"table_name": "<table>", "chart_type": "pie",\n'
            '      "x_column": "<target_col>", "y_column": "",\n'
            '      "title": "<Target_Col> Distribution", "hue_column": ""}\n\n'
            "   CHART B — Primary Numeric Distribution (histogram)\n"
            '     {"table_name": "<table>", "chart_type": "histogram",\n'
            '      "x_column": "<most_important_numeric_col>", "y_column": "",\n'
            '      "title": "<col> Distribution", "hue_column": ""}\n\n'
            "   CHART C — Numeric by Target (histplot with hue)\n"
            '     {"table_name": "<table>", "chart_type": "histplot",\n'
            '      "x_column": "<numeric_col>", "y_column": "",\n'
            '      "title": "<numeric_col> by <target_col>", "hue_column": "<target_col>"}\n\n'
            "   CHART D — Relationship between two numerics (scatter)\n"
            '     {"table_name": "<table>", "chart_type": "scatter",\n'
            '      "x_column": "<numeric_col_1>", "y_column": "<numeric_col_2>",\n'
            '      "title": "<col_1> vs <col_2>", "hue_column": "<target_col>"}\n\n'
            "   CHART E — Category vs Numeric (bar chart)\n"
            '     {"table_name": "<table>", "chart_type": "bar",\n'
            '      "x_column": "<categorical_col>", "y_column": "<numeric_col>",\n'
            '      "title": "<cat_col> vs <numeric_col>", "hue_column": ""}\n\n'
            "   CHART F — Full Feature Correlation Heatmap (MUST use encoded table)\n"
            '     {"table_name": "<table>_encoded", "chart_type": "heatmap",\n'
            '      "x_column": "<target_col>", "y_column": "",\n'
            '      "title": "Feature Correlation Heatmap", "hue_column": ""}\n\n'
            "   Feel free to add more charts if the data supports them.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 3 — GENERATE TEXT REPORT\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: generate_text_report (Generate Business Insight Report)\n"
            "   Include ALL findings from the Data Engineer, Data Scientist,\n"
            "   and Label Encoder in the report body.\n"
            "   Structure:\n"
            "   - Executive Summary\n"
            "   - Data Quality Overview (from Engineer)\n"
            "   - ML Model Results (from Scientist)\n"
            "   - Customer Segments (from Scientist)\n"
            "   - Key Visual Insights (from chart Gemini analyses)\n"
            "   - Strategic Recommendations\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 4 — GENERATE PDF REPORT\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: generate_pdf_report\n"
            "   Pass all chart paths generated in Step 2.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 5 — GENERATE POWERPOINT\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: generate_ppt_report\n"
            "   Pass all chart paths generated in Step 2.\n"
        ),
        expected_output=(
            "At least 5 chart PNG files in outputs/charts/ — each with Ollama vision analysis. "
            "A comprehensive markdown business insight report in outputs/reports/. "
            "A PDF report at outputs/reports/report_*.pdf. "
            "A PowerPoint deck at outputs/reports/report_*.pptx. "
            "Summary of all chart insights and key visual patterns found."
        ),
        agent=agents["analyst"],
        context=[task_engineer, task_scientist, task_label_encoder],
    )

    # ── Task 5: Executive Brief ────────────────────────────────────────────────
    task_manager = Task(
        description=(
            f"You are the final agent in the pipeline. Your job is to synthesise all team "
            f"outputs into a polished, CEO-ready executive briefing.\n\n"
            f"Analysis Request: {analysis_request}\n\n"

            "You have access to the full outputs of:\n"
            "• Data Engineer: data quality report, schema overview, web research\n"
            "• Data Scientist: ML model results, feature importance, customer segments\n"
            "• Label Encoder: dataset confirmation\n"
            "• Data Analyst: chart insights, business report, PDF/PPTX deliverables\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "WRITE THE EXECUTIVE BRIEFING — Structure below (600–900 words)\n"
            "═══════════════════════════════════════════════════════════\n\n"

            "## 1. Executive Summary (3-4 sentences)\n"
            "   State: what data was analysed, how many records, the primary business "
            "question, and the single most important finding with a specific number.\n\n"

            "## 2. Data Overview\n"
            "   - Source: database table name, row count, column count\n"
            "   - Data quality: any major issues found and how they were resolved\n"
            "   - Key variables: target column, most important features\n\n"

            "## 3. Top 5 Key Findings\n"
            "   Each finding must include a specific metric or statistic.\n"
            "   Format: 'Finding [N]: [statement with number] — [business implication]'\n"
            "   Example: 'Finding 1: 26.5% of customers churned — representing significant\n"
            "             revenue loss; month-to-month contracts drive 42% of all churn.'\n\n"

            "## 4. ML Model Performance\n"
            "   - Model type used and why it was chosen\n"
            "   - Key metric (AUC-ROC, accuracy, etc.) with value\n"
            "   - Top 3 predictive factors with plain-English explanation\n\n"

            "## 5. Customer / Audience Segments\n"
            "   Name all 4 segments and describe:\n"
            "   - Size (number or % of total)\n"
            "   - Key characteristics\n"
            "   - Risk level or opportunity level\n"
            "   - Recommended action for marketing\n\n"

            "## 6. Risk Assessment\n"
            "   Identify the top 3 business risks revealed by the data.\n"
            "   Rate each: HIGH / MEDIUM / LOW risk with justification.\n\n"

            "## 7. Strategic Recommendations (Top 5, Prioritised)\n"
            "   Format each as:\n"
            "   [Priority N] | Action | Owner | KPI Target | Timeline\n"
            "   These must be specific, measurable, and directly derived from the data.\n\n"

            "## 8. Audience Insights for Digital Marketing Handoff\n"
            "   This section is specifically for the Digital Marketing team.\n"
            "   Provide:\n"
            "   - Which segments are highest priority targets\n"
            "   - What messaging angle will resonate with each segment\n"
            "   - Specific offers or incentives to test based on the data\n"
            "   - Recommended channels per segment\n\n"

            "## 9. Next Steps\n"
            "   Numbered action plan:\n"
            "   [N]. Action — Owner — Deadline — Success Metric\n\n"

            "IMPORTANT: Do NOT mention file paths, system locations, or tool names in the brief."
        ),
        expected_output=(
            "A polished 600–900 word executive briefing in professional markdown. "
            "Includes data overview, 5 key findings with metrics, ML model performance, "
            "4 named audience segments, risk assessment, 5 prioritised recommendations "
            "with owner/KPI/timeline, marketing handoff insights, and numbered next steps."
        ),
        agent=agents["manager"],
        context=[task_engineer, task_scientist, task_label_encoder, task_analyst],
    )

    return [task_engineer, task_scientist, task_label_encoder, task_analyst, task_manager]


# ── Public API ────────────────────────────────────────────────────────────────

def run_banking_crew(analysis_request: str, langgraph_plan: dict = None) -> dict:
    """
    Run the Universal Analytics CrewAI crew (Phase 2).
    Called by app.py after HITL approval.

    Returns:
        dict with 'crew_output' (str), 'analyst_output', 'scientist_output', and 'status'.
    """
    agents = _build_analytics_agents()
    tasks  = _build_analytics_tasks(agents, analysis_request, langgraph_plan)

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

    print("\n[CrewAI-Analytics] ══ Starting 5-agent sequential Analytics crew ══\n", flush=True)
    result = crew.kickoff()
    print("\n[CrewAI-Analytics] ══ Analytics crew complete ══\n", flush=True)

    # Extract individual agent outputs for downstream handoff
    # task order: 0=engineer, 1=scientist, 2=label_encoder, 3=analyst, 4=manager(CDO)
    scientist_output = ""
    analyst_output   = ""
    try:
        if hasattr(result, "tasks_output") and len(result.tasks_output) >= 5:
            scientist_output = str(result.tasks_output[1])
            analyst_output   = str(result.tasks_output[3])
            print(f"[CrewAI-Analytics] Data Scientist output captured ({len(scientist_output)} chars)",
                  flush=True)
            print(f"[CrewAI-Analytics] Data Analyst output captured ({len(analyst_output)} chars)",
                  flush=True)
    except Exception as _e:
        print(f"[CrewAI-Analytics] Could not extract task outputs: {_e}", flush=True)

    return {
        "crew_output":      str(result),
        "analyst_output":   analyst_output,
        "scientist_output": scientist_output,
        "status":           "completed",
    }
