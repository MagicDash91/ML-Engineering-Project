"""
crew_risk.py – Financial Risk & Compliance CrewAI 5-agent sequential crew.

Agents:
  1. Risk Data Engineer     – database discovery, ETL, profiling, data quality
  2. Risk Scientist         – credit risk, VaR/CVaR, Monte Carlo, segmentation
  3. Label Encoder          – encode categoricals for heat map
  4. Risk Analyst           – charts, Ollama vision analysis, compliance report
  5. CRO / Risk Manager     – synthesises executive risk brief + compliance summary
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

from config import NVIDIA_API_KEY, NVIDIA_MODEL

if NVIDIA_API_KEY:
    os.environ["OPENAI_API_KEY"] = NVIDIA_API_KEY

# ── Tool imports ──────────────────────────────────────────────────────────────
from tools.risk_engineer import (
    list_database_tables,
    profile_database_table,
    query_database,
    clean_table_columns,
    normalize_column_dtypes,
    web_search_risk,
)
from tools.risk_scientist import (
    train_credit_risk_model,
    compute_var_cvar,
    monte_carlo_stress_test,
    risk_segmentation,
    train_fraud_model,
)
from tools.risk_analyst import (
    generate_risk_visualization,
    label_encode_table,
    generate_risk_text_report,
)
from tools.risk_report import (
    generate_risk_pdf_report,
    generate_risk_ppt_report,
)


# ── LLM ───────────────────────────────────────────────────────────────────────

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

def _build_risk_agents() -> dict:
    cro = Agent(
        role="Chief Risk Officer (CRO)",
        goal=(
            "Oversee the entire risk analytics team and synthesise all outputs — "
            "data quality findings, risk model results, encoded dataset confirmation, "
            "and visual risk analytics — into a single executive-ready risk briefing. "
            "The briefing must include a compliance status summary referencing applicable "
            "regulatory frameworks (Basel III, IFRS 9, FRTB, AML). "
            "All findings must be specific, quantitative, and actionable."
        ),
        backstory=(
            "You are a seasoned CRO with 20 years leading risk transformation programmes "
            "across global banks, insurance firms, and asset managers. "
            "You have overseen Basel III implementation, IFRS 9 ECL model validation, "
            "FRTB market risk capital calculations, and AML compliance programmes. "
            "You brief boards and regulators with precision: every statement backed by a number, "
            "every recommendation tied to a specific risk driver."
        ),
        llm=nvidia_llm,
        allow_delegation=False,
        verbose=True,
        max_iter=15,
    )

    engineer = Agent(
        role="Senior Risk Data Engineer",
        goal=(
            "Auto-discover the full database schema, profile every relevant table, "
            "validate data quality for financial risk analysis, and deliver a clean, "
            "analysis-ready dataset. Identify the primary risk table, its target or "
            "outcome column (default flag, loss amount, fraud indicator, exposure), "
            "its key numeric and categorical features, and any data issues."
        ),
        backstory=(
            "You are a Senior Data Engineer specialising in financial risk data infrastructure. "
            "You have built ETL pipelines for loan origination systems, trading platforms, "
            "transaction monitoring systems, and regulatory reporting engines. "
            "You never assume what the data contains — you always start by listing tables "
            "and inspecting schemas. You check every column for nulls, verify data types, "
            "drop ID columns, and convert text-stored amounts to numeric."
        ),
        llm=nvidia_llm,
        tools=[
            list_database_tables,
            profile_database_table,
            query_database,
            clean_table_columns,
            normalize_column_dtypes,
            web_search_risk,
        ],
        allow_delegation=False,
        verbose=True,
        max_iter=15,
        max_retry_limit=3,
    )

    scientist = Agent(
        role="Senior Quantitative Risk Scientist",
        goal=(
            "Select and run the most appropriate risk models given the data schema. "
            "Produce quantitative risk metrics (VaR, CVaR, AUC-ROC, PD, LGD), "
            "identify the top risk drivers, and generate risk tier segmentation. "
            "Provide clear regulatory interpretation of every model result."
        ),
        backstory=(
            "You are a PhD-level Quantitative Risk Scientist with 15 years deploying risk "
            "models across credit risk, market risk, and operational risk. "
            "You have built IRB credit scorecards, FRTB internal models, IFRS 9 ECL engines, "
            "and AML transaction monitoring models. "
            "You always inspect data schema before modelling. "
            "If you see a binary default/fraud column → build a classification model. "
            "If you see loss/return/P&L → compute VaR and CVaR. "
            "If you see exposure data → run portfolio risk and concentration analysis. "
            "You always run risk_segmentation to discover 4 risk tiers (Critical/High/Medium/Low)."
        ),
        llm=nvidia_llm,
        tools=[
            train_credit_risk_model,
            compute_var_cvar,
            monte_carlo_stress_test,
            risk_segmentation,
            train_fraud_model,
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
            "Transform the primary risk dataset into a fully numeric version by "
            "label-encoding all categorical columns. This encoded table is required "
            "by the Risk Analyst to generate a full-feature risk correlation heat map."
        ),
        backstory=(
            "You are a Data Preprocessing specialist focused on financial risk feature engineering. "
            "You understand that categorical risk variables — loan grade, industry sector, "
            "rating category, product type, geography — must be integer-encoded before "
            "correlation analysis and ML modelling. "
            "You always verify the result: zero object-type columns in the encoded table."
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
        role="Senior Risk Analyst",
        goal=(
            "Produce a comprehensive risk visualization package: at least 5 meaningful charts "
            "covering risk distribution, loss exposure, risk factor relationships, and trends; "
            "a full-feature risk correlation heat map using the encoded table; "
            "a markdown risk insight report; a professional PDF report; "
            "and a PowerPoint risk presentation. "
            "Every chart must be analysed by Ollama vision to extract key risk insights."
        ),
        backstory=(
            "You are a Senior Risk Analyst with 10 years building risk dashboards and "
            "regulatory reports across credit, market, and operational risk. "
            "You know that the best risk charts tell the story of where capital is at risk. "
            "You use Ollama AI to add AI-powered interpretation to every chart automatically. "
            "You call visualization tools one at a time, never passing arrays."
        ),
        llm=nvidia_llm,
        tools=[
            generate_risk_visualization,
            generate_risk_text_report,
            generate_risk_pdf_report,
            generate_risk_ppt_report,
            query_database,
            list_database_tables,
        ],
        allow_delegation=False,
        verbose=True,
        max_iter=30,
        max_retry_limit=3,
    )

    return {
        "cro":           cro,
        "engineer":      engineer,
        "scientist":     scientist,
        "label_encoder": label_encoder,
        "analyst":       analyst,
    }


# ── Task factory ──────────────────────────────────────────────────────────────

def _build_risk_tasks(agents: dict, analysis_request: str,
                      langgraph_plan: dict = None) -> list:
    plan_context  = ""
    etl_context   = ""
    model_context = ""
    viz_context   = ""

    if langgraph_plan:
        plan_context  = f"\n\n[CRO Strategic Plan]:\n{langgraph_plan.get('analysis_plan', '')[:600]}"
        etl_context   = f"\n\n[ETL Guidance]:\n{langgraph_plan.get('etl_guidance', '')[:400]}"
        model_context = f"\n\n[Risk Model Guidance]:\n{langgraph_plan.get('risk_modelling', '')[:400]}"
        viz_context   = f"\n\n[Risk Analysis Guidance]:\n{langgraph_plan.get('risk_analysis', '')[:400]}"

    # ── Task 1: Risk Data Engineering ─────────────────────────────────────────
    task_engineer = Task(
        description=(
            f"You are the first agent in the pipeline. Fully discover, profile, "
            f"and clean the financial risk data in the connected database.\n\n"
            f"Analysis Request: {analysis_request}{plan_context}{etl_context}\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 1 — DISCOVER: List all available tables\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: list_database_tables\n"
            "   Goal: Identify which tables exist. The primary risk table is the one\n"
            "         with the most rows and analytical columns (not lookup tables).\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 2 — PROFILE: Deep-dive into the primary table\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: profile_database_table\n"
            "   Goal: Get full row count, column names, dtypes, null counts,\n"
            "         numeric stats (mean, std, min, max), and categorical distributions.\n"
            "   Identify: target/risk column (default, fraud, loss, exposure, risk_flag).\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 3 — SAMPLE: Inspect actual data\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: query_database\n"
            "   Run: SELECT * FROM <primary_table> LIMIT 5\n"
            "        SELECT <risk_col>, COUNT(*) FROM <primary_table> GROUP BY <risk_col>\n"
            "        SELECT COUNT(*) FROM <primary_table>\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 4 — CLEAN: Remove noise columns\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: clean_table_columns\n"
            "   Goal: Drop ID/UUID/index columns that add no analytical value.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 5 — NORMALIZE: Fix financial data types\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: normalize_column_dtypes\n"
            "   Goal: Convert text-stored amounts, rates, and scores to float/int.\n"
            "         Critical for financial data: loan amounts, interest rates, LTV ratios.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 6 — RESEARCH: Find regulatory context\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: web_search_risk\n"
            "   Run 2 searches relevant to the risk domain discovered.\n"
            "   Examples:\n"
            "     - 'Basel III capital adequacy requirements 2025'\n"
            "     - 'IFRS 9 expected credit loss best practices'\n"
            "     - 'credit risk default prediction benchmarks'\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 7 — REPORT: Summarise all findings\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Write a structured data quality report:\n"
            "   - Primary table name, row/column count\n"
            "   - Column inventory: name, dtype, null count, example values\n"
            "   - Identified risk/target column and its distribution\n"
            "   - Columns dropped and columns type-converted\n"
            "   - Top 3 data quality issues\n"
            "   - Web research summary (2 key regulatory insights)\n"
            "   - Model recommendation for the Risk Scientist\n"
        ),
        expected_output=(
            "Structured data quality report: primary table name, row/column counts, "
            "full column inventory with dtypes and null counts, risk column distribution, "
            "dropped/converted columns, data quality issues, regulatory research summary, "
            "and model recommendation."
        ),
        agent=agents["engineer"],
    )

    # ── Task 2: Risk Modelling ─────────────────────────────────────────────────
    task_scientist = Task(
        description=(
            f"You are the second agent. Build risk models appropriate for the data schema.\n\n"
            f"Analysis Request: {analysis_request}{model_context}\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 1 — ORIENT: Understand the data\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: list_database_tables → query_database (SELECT * LIMIT 5)\n"
            "   Read the Data Engineer's report for table name and target column.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 2 — CHOOSE & RUN: Select appropriate risk model(s)\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Decision logic:\n\n"
            "   IF binary column found ('default', 'is_default', 'Default',\n"
            "      'credit_risk', 'bad_loan', etc.):\n"
            "     → train_credit_risk_model(table_name=<table>, target_column=<col>)\n\n"
            "   IF binary column found ('fraud', 'is_fraud', 'Fraud', 'isFraud'):\n"
            "     → train_fraud_model(table_name=<table>, target_column=<col>)\n\n"
            "   IF numeric loss/return/P&L column found:\n"
            "     → compute_var_cvar(table_name=<table>, value_column=<col>)\n"
            "     → monte_carlo_stress_test(table_name=<table>, value_column=<col>)\n\n"
            "   IF no clear target → run risk_segmentation only\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 3 — ALWAYS RUN: Risk Segmentation\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: risk_segmentation(table_name=<primary_table>, n_clusters=4)\n"
            "   Produces 4 risk tiers: Critical / High / Medium / Low risk.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 4 — INTERPRET: Regulatory language\n"
            "═══════════════════════════════════════════════════════════\n"
            "   For every model run, provide:\n"
            "   a) Performance metrics (AUC-ROC, accuracy; or VaR/CVaR values)\n"
            "   b) Top 5 risk drivers in plain business language\n"
            "      NOT 'feature importance: 0.32' — INSTEAD 'Borrowers with LTV > 80%\n"
            "      are 3× more likely to default'\n"
            "   c) Risk tier profiles (size, key characteristics, risk treatment)\n"
            "   d) Regulatory reference: Basel III PD/LGD/EAD, IFRS 9 ECL stage, or FRTB VaR\n"
        ),
        expected_output=(
            "Detailed risk model results: model type and rationale, AUC-ROC or VaR/CVaR, "
            "top 5 risk drivers in business language, 4 risk tier profiles with treatment, "
            "regulatory framework commentary, and all model file paths."
        ),
        agent=agents["scientist"],
        context=[task_engineer],
    )

    # ── Task 3: Label Encoding ─────────────────────────────────────────────────
    task_label_encoder = Task(
        description=(
            "You are the third agent. Create a fully numeric version of the primary "
            "risk dataset for heat map generation.\n\n"

            "STEP 1 — CONFIRM: list_database_tables → verify primary table name.\n"
            "STEP 2 — INSPECT: query_database (SELECT * LIMIT 3) → identify categorical columns.\n"
            "STEP 3 — ENCODE: label_encode_table(table_name=<primary_table>)\n"
            "         Creates '<primary_table>_encoded' with all integers.\n"
            "STEP 4 — VERIFY: query_database on the _encoded table → confirm no object columns.\n"
            "STEP 5 — REPORT: List encoded columns and confirm table name for Risk Analyst.\n"
        ),
        expected_output=(
            "Confirmation that '<primary_table>_encoded' was successfully created. "
            "List of encoded columns with mapping logic. "
            "Exact encoded table name for the Risk Analyst."
        ),
        agent=agents["label_encoder"],
        context=[task_engineer],
    )

    # ── Task 4: Risk Visualization & Reports ──────────────────────────────────
    task_analyst = Task(
        description=(
            f"You are the fourth agent. Produce a complete risk visualization package "
            f"with at least 5 charts, a risk insight report, PDF, and PowerPoint.\n\n"
            f"Analysis Request: {analysis_request}{viz_context}\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "CRITICAL TOOL CALL RULES\n"
            "═══════════════════════════════════════════════════════════\n"
            "Tool name: 'Generate Risk Visualization'\n"
            "• Call it ONCE per chart — one JSON dict per call, NEVER an array\n"
            "• Use EXACT column names from the database (case-sensitive)\n"
            "• For heat map: always use the '_encoded' table\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 1 — ORIENT\n"
            "═══════════════════════════════════════════════════════════\n"
            "   list_database_tables → query_database (SELECT * LIMIT 3)\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 2 — GENERATE 5+ CHARTS (one tool call each)\n"
            "═══════════════════════════════════════════════════════════\n"
            '   CHART A — Risk Distribution (pie)\n'
            '     {"table_name":"<table>","chart_type":"pie","x_column":"<risk_col>","y_column":"","title":"Risk Distribution","hue_column":""}\n\n'
            '   CHART B — Exposure/Loss Histogram\n'
            '     {"table_name":"<table>","chart_type":"histogram","x_column":"<amount_col>","y_column":"","title":"Loss/Exposure Distribution","hue_column":""}\n\n'
            '   CHART C — Amount by Risk Category (histplot with hue)\n'
            '     {"table_name":"<table>","chart_type":"histplot","x_column":"<amount_col>","y_column":"","title":"Amount by Risk Category","hue_column":"<risk_col>"}\n\n'
            '   CHART D — Risk Factor Scatter\n'
            '     {"table_name":"<table>","chart_type":"scatter","x_column":"<numeric_col_1>","y_column":"<numeric_col_2>","title":"Risk Factor Relationship","hue_column":"<risk_col>"}\n\n'
            '   CHART E — Category vs Risk Metric (bar)\n'
            '     {"table_name":"<table>","chart_type":"bar","x_column":"<categorical_col>","y_column":"<numeric_col>","title":"Risk by Category","hue_column":""}\n\n'
            '   CHART F — Full Risk Correlation Heat Map (MUST use _encoded table)\n'
            '     {"table_name":"<table>_encoded","chart_type":"heatmap","x_column":"<risk_col>","y_column":"","title":"Risk Factor Correlation Heat Map","hue_column":""}\n\n'

            "═══════════════════════════════════════════════════════════\n"
            "STEP 3 — GENERATE RISK TEXT REPORT\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: generate_risk_text_report\n"
            "   Include ALL findings from Engineer, Scientist, and Label Encoder.\n"
            "   Structure: Executive Risk Summary | Data Quality | Model Results |\n"
            "   Risk Segments | Visual Insights | Compliance Status | Recommendations\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 4 — PDF REPORT\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: generate_risk_pdf_report — pass all chart paths from Step 2.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 5 — POWERPOINT\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: generate_risk_ppt_report — pass all chart paths from Step 2.\n"
        ),
        expected_output=(
            "At least 5 risk chart PNG files with Ollama vision analysis. "
            "A comprehensive markdown risk insight report. "
            "A PDF risk report at outputs/reports/. "
            "A PowerPoint risk presentation at outputs/reports/. "
            "Summary of all chart risk insights."
        ),
        agent=agents["analyst"],
        context=[task_engineer, task_scientist, task_label_encoder],
    )

    # ── Task 5: CRO Executive Brief + Compliance Summary ─────────────────────
    task_cro = Task(
        description=(
            f"You are the final agent. Synthesise all team outputs into a polished, "
            f"board-ready risk briefing and compliance summary.\n\n"
            f"Analysis Request: {analysis_request}\n\n"

            "You have access to outputs from:\n"
            "• Risk Data Engineer: data quality report, schema overview, regulatory research\n"
            "• Risk Scientist: model results, risk factor importance, risk tier profiles\n"
            "• Label Encoder: dataset confirmation\n"
            "• Risk Analyst: chart insights, risk report, PDF/PPTX deliverables\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "WRITE THE EXECUTIVE RISK BRIEFING (700–1000 words)\n"
            "═══════════════════════════════════════════════════════════\n\n"

            "## 1. Executive Risk Summary (3–4 sentences)\n"
            "   State: what data was analysed, total records, primary risk question, "
            "and the single most important risk finding with a specific number.\n\n"

            "## 2. Risk Data Overview\n"
            "   - Source: table name, row count, column count\n"
            "   - Data quality: issues found and how resolved\n"
            "   - Key variables: target/risk column, top risk features\n\n"

            "## 3. Top 5 Key Risk Findings\n"
            "   Each finding must include a specific metric.\n"
            "   Format: 'Finding [N]: [statement with number] — [risk implication]'\n\n"

            "## 4. Risk Model Performance\n"
            "   - Model type and regulatory justification\n"
            "   - Key metric (AUC-ROC / VaR / CVaR) with value\n"
            "   - Top 3 risk drivers in plain business language\n\n"

            "## 5. Risk Tier Profiles\n"
            "   Name all 4 tiers and describe:\n"
            "   - Size (count or % of total portfolio)\n"
            "   - Key risk characteristics\n"
            "   - Risk rating (Critical / High / Medium / Low)\n"
            "   - Recommended risk treatment / mitigation\n\n"

            "## 6. Regulatory Compliance Status\n"
            "   For each applicable framework:\n"
            "   - Basel III: capital adequacy / PD / LGD / EAD assessment\n"
            "   - IFRS 9: ECL stage classification (Stage 1/2/3)\n"
            "   - FRTB: market risk capital (if applicable)\n"
            "   - AML: suspicious activity indicators (if applicable)\n"
            "   Rate each: COMPLIANT / REQUIRES ACTION / NOT APPLICABLE\n\n"

            "## 7. Risk Mitigation Recommendations (Top 5, Prioritised)\n"
            "   Format: [Priority N] | Action | Risk Owner | Target Metric | Timeline\n\n"

            "## 8. Risk Appetite Assessment\n"
            "   Is the current risk profile within risk appetite thresholds?\n"
            "   List any breaches or near-breaches with specific values.\n\n"

            "## 9. Action Plan\n"
            "   [N]. Action — Risk Owner — Deadline — Success Metric\n\n"

            "IMPORTANT: Do NOT mention file paths, system locations, or tool names."
        ),
        expected_output=(
            "A polished 700–1000 word executive risk briefing: data overview, 5 key risk "
            "findings with metrics, model performance, 4 risk tier profiles, regulatory "
            "compliance status per framework, 5 prioritised mitigation recommendations, "
            "risk appetite assessment, and numbered action plan."
        ),
        agent=agents["cro"],
        context=[task_engineer, task_scientist, task_label_encoder, task_analyst],
    )

    return [task_engineer, task_scientist, task_label_encoder, task_analyst, task_cro]


# ── Public API ────────────────────────────────────────────────────────────────

def run_risk_crew(analysis_request: str, langgraph_plan: dict = None) -> dict:
    """
    Run the Financial Risk CrewAI crew (Phase 2).
    Returns dict with 'crew_output', 'analyst_output', 'scientist_output', 'status'.
    """
    agents = _build_risk_agents()
    tasks  = _build_risk_tasks(agents, analysis_request, langgraph_plan)

    crew = Crew(
        agents=[
            agents["engineer"],
            agents["scientist"],
            agents["label_encoder"],
            agents["analyst"],
            agents["cro"],
        ],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=False,
    )

    print("\n[CrewAI-Risk] ══ Starting 5-agent sequential Risk crew ══\n", flush=True)
    result = crew.kickoff()
    print("\n[CrewAI-Risk] ══ Risk crew complete ══\n", flush=True)

    scientist_output = ""
    analyst_output   = ""
    try:
        if hasattr(result, "tasks_output") and len(result.tasks_output) >= 5:
            scientist_output = str(result.tasks_output[1])
            analyst_output   = str(result.tasks_output[3])
    except Exception as _e:
        print(f"[CrewAI-Risk] Could not extract task outputs: {_e}", flush=True)

    return {
        "crew_output":      str(result),
        "analyst_output":   analyst_output,
        "scientist_output": scientist_output,
        "status":           "completed",
    }
