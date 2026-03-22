"""
Banking Analytics — LangGraph Workflow
=======================================
5-node Gemini-only strategic pre-planning pipeline.
Pattern identical to Bank_Agent/graphs/banking_graph.py.

Workflow:
    [plan] → [data_engineering] → [data_science] → [data_analysis] → [reporting] → END
"""

import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ── Ollama helper ─────────────────────────────────────────────────────────────

def _query_gemini(messages: list, label: str = "") -> str:
    """Call Ollama qwen3.5:cloud with hard wall-clock timeout."""
    import concurrent.futures
    from langchain_community.chat_models import ChatOllama

    tag = f"[BankingGraph · {label}]" if label else "[BankingGraph]"

    def _call() -> str:
        import time
        lc_msgs = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lc_msgs.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_msgs.append(AIMessage(content=content))
            else:
                lc_msgs.append(HumanMessage(content=content))
        last_exc = None
        for attempt in range(3):
            try:
                llm = ChatOllama(model="qwen3.5:cloud")
                reply = llm.invoke(lc_msgs).content.strip()
                return reply
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    print(f"  {tag} Ollama error (attempt {attempt+1}/3): {exc} — retrying in 5s", flush=True)
                    time.sleep(5)
        return f"Planning step failed after 3 attempts: {last_exc}"

    print(f"  {tag} querying Ollama qwen3.5:cloud…", flush=True)
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_call)
    try:
        result = fut.result(timeout=130)
        ex.shutdown(wait=False)
        print(f"  {tag} ✓ Ollama responded", flush=True)
        return result
    except concurrent.futures.TimeoutError:
        ex.shutdown(wait=False)
        print(f"  {tag} timeout – using fallback", flush=True)
        return "Planning step timed out – pipeline continues with available context."


# ── State schema ──────────────────────────────────────────────────────────────

class BankingAnalyticsState(TypedDict):
    task_description:   str
    bank_symbols:       Optional[List[str]]
    analysis_plan:      Optional[str]
    etl_guidance:       Optional[str]
    data_collected:     Optional[bool]
    ml_guidance:        Optional[str]
    analytics_guidance: Optional[str]
    report_content:     Optional[str]
    current_step:       Optional[str]
    iteration_count:    Optional[int]
    errors:             Optional[List[str]]
    final_output:       Optional[str]


# ── Node implementations ──────────────────────────────────────────────────────

def plan_node(state: BankingAnalyticsState) -> BankingAnalyticsState:
    task    = state["task_description"]
    symbols = state.get("bank_symbols") or []

    raw = _query_gemini([
        {
            "role": "system",
            "content": (
                "You are a Chief Data Officer with broad industry experience. "
                "Create a detailed, structured analysis plan for a multi-agent data team. "
                "Your team consists of: Data Engineer (database discovery, ETL, profiling, quality checks), "
                "Data Scientist (ML models: classification, clustering, regression, forecasting), "
                "Data Analyst (visualizations, Gemini AI chart analysis, PDF/PPTX reports). "
                "The analysis must be domain-agnostic — adapt to whatever data is in the database. "
                "Respond with a valid JSON object containing:\n"
                "  'summary', 'focus_area', 'data_needs', 'etl_steps',\n"
                "  'ml_tasks', 'visualizations', 'report_sections', 'success_criteria'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task: {task}\n"
                f"Additional context: {', '.join(symbols) if symbols else 'None'}\n\n"
                "The database may contain any type of data (customer records, sales, transactions, "
                "medical records, IoT data, etc.). The team will auto-discover the schema. "
                "Plan for schema discovery first, then adapt modelling and analysis to what is found. "
                "Output ONLY valid JSON."
            ),
        },
    ], label="1/5 plan")

    plan_str = raw
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            plan_obj = json.loads(match.group())
            plan_str = json.dumps(plan_obj, indent=2)
    except (json.JSONDecodeError, TypeError):
        pass

    return {
        **state,
        "analysis_plan":   plan_str,
        "current_step":    "data_engineering",
        "iteration_count": 0,
        "errors":          [],
    }


def data_engineering_node(state: BankingAnalyticsState) -> BankingAnalyticsState:
    task    = state["task_description"]
    plan    = state.get("analysis_plan", "")
    symbols = state.get("bank_symbols") or []

    guidance = _query_gemini([
        {
            "role": "system",
            "content": (
                "You are a Senior Data Engineer with cross-industry expertise. "
                "Given the analysis plan, produce precise data engineering instructions.\n"
                "Specify: 1) Steps to discover and list all database tables "
                "2) How to profile the primary table (row count, schema, nulls, dtypes) "
                "3) Web search queries relevant to the data domain "
                "4) Which columns to clean or normalize "
                "5) Data quality checks specific to this dataset "
                "6) How to identify the target/outcome column from the schema. "
                "The engineer must never assume the schema — always start with list_database_tables."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analysis Plan:\n{plan}\n\n"
                f"Task: {task}\n"
                f"Additional context: {symbols if symbols else 'None'}\n\n"
                "Provide step-by-step data engineering instructions. "
                "Emphasise schema discovery first, then profiling, then cleaning."
            ),
        },
    ], label="2/5 data_engineering")

    return {
        **state,
        "etl_guidance":    guidance,
        "data_collected":  True,
        "current_step":    "data_science",
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def data_science_node(state: BankingAnalyticsState) -> BankingAnalyticsState:
    task     = state["task_description"]
    plan     = state.get("analysis_plan", "")
    etl_info = state.get("etl_guidance", "")

    guidance = _query_gemini([
        {
            "role": "system",
            "content": (
                "You are a Senior Data Scientist with cross-industry expertise. "
                "Based on the data schema discovered, recommend the most appropriate ML approach:\n"
                "- Binary classification (churn, fraud, default, outcome) → "
                "  train_churn_model or train_fraud_detection_model or train_credit_risk_model\n"
                "- Time-series data (date column + numeric value) → time_series_forecast\n"
                "- Any dataset → always run customer_segmentation (n_clusters=4) for audience clusters\n"
                "Specify: 1) Which model to use and why given the schema "
                "2) The exact table name and target column "
                "3) Evaluation metrics to report "
                "4) How to interpret feature importance in business language "
                "5) How to name and profile the 4 customer segments for marketing use."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analysis Plan:\n{plan}\n\n"
                f"ETL Guidance:\n{etl_info[:600]}\n\n"
                f"Task: {task}\n\n"
                "Based on what the Data Engineer found, which ML models should be trained? "
                "Be specific about table names and column names."
            ),
        },
    ], label="3/5 data_science")

    return {
        **state,
        "ml_guidance":     guidance,
        "current_step":    "data_analysis",
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def data_analysis_node(state: BankingAnalyticsState) -> BankingAnalyticsState:
    task     = state["task_description"]
    plan     = state.get("analysis_plan", "")
    ml_info  = state.get("ml_guidance", "")
    etl_info = state.get("etl_guidance", "")

    guidance = _query_gemini([
        {
            "role": "system",
            "content": (
                "You are a Senior Data Analyst with cross-industry expertise in business dashboards. "
                "Design a complete visualization package for the data found:\n"
                "1) Recommend at least 5 charts (type, x_column, y_column, hue_column, purpose)\n"
                "2) Include: pie chart (target distribution), histogram (key numeric), "
                "   histplot with hue (numeric by target), scatter (two numerics), "
                "   bar (category vs numeric), heatmap (use encoded table)\n"
                "3) All column names must match exactly what the Data Engineer found\n"
                "4) Identify the key business KPIs to highlight based on the domain\n"
                "5) Define report sections and what insight each chart should reveal\n"
                "6) Note: heatmap must use the '_encoded' version of the primary table."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analysis Plan:\n{plan}\n\n"
                f"ETL Guidance:\n{etl_info[:400]}\n\nML Guidance:\n{ml_info[:400]}\n\n"
                f"Task: {task}\n\n"
                "Specify the complete visualization package. Use actual column names from the ETL guidance."
            ),
        },
    ], label="4/5 data_analysis")

    return {
        **state,
        "analytics_guidance": guidance,
        "current_step":       "reporting",
        "iteration_count":    state.get("iteration_count", 0) + 1,
    }


def reporting_node(state: BankingAnalyticsState) -> BankingAnalyticsState:
    task      = state["task_description"]
    plan      = state.get("analysis_plan", "")
    etl_info  = state.get("etl_guidance", "")
    ml_info   = state.get("ml_guidance", "")
    analytics = state.get("analytics_guidance", "")

    report = _query_gemini([
        {
            "role": "system",
            "content": (
                "You are a Chief Data Officer writing a pre-analysis strategic brief. "
                "Format as professional markdown with clear headings. "
                "This brief is domain-agnostic — adapt the language and KPIs to whatever "
                "industry and data type the task describes. "
                "Be specific: mention the likely table names, target columns, and model types "
                "the team will use based on the planning documents."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task: {task}\n\n"
                f"Analysis Plan:\n{plan[:800]}\n\n"
                f"ETL Plan:\n{etl_info[:400]}\nML Plan:\n{ml_info[:400]}\n"
                f"Analytics Plan:\n{analytics[:400]}\n\n"
                "Write the preliminary strategic brief:\n"
                "# Preliminary Data Analytics Brief\n"
                "## Executive Summary\n## Scope & Objectives\n"
                "## Data Sources & Methodology\n## Expected Findings\n"
                "## Key Risk Areas\n## Strategic Recommendations (preliminary)\n## Deliverables"
            ),
        },
    ], label="5/5 reporting")

    return {
        **state,
        "report_content": report,
        "current_step":   "completed",
        "final_output":   report,
    }


# ── Router ────────────────────────────────────────────────────────────────────

def _router(state: BankingAnalyticsState) -> str:
    step = state.get("current_step", "data_engineering")
    if state.get("iteration_count", 0) > 5:
        return "end"
    routes = {
        "data_engineering": "data_engineering",
        "data_science":     "data_science",
        "data_analysis":    "data_analysis",
        "reporting":        "reporting",
        "completed":        "end",
    }
    return routes.get(step, "end")


# ── Build graph ───────────────────────────────────────────────────────────────

def _build_graph():
    try:
        from langgraph.graph import StateGraph, END
    except ImportError as exc:
        print(f"[BankingGraph] langgraph not installed – {exc}")
        return None

    g = StateGraph(BankingAnalyticsState)
    g.add_node("plan",             plan_node)
    g.add_node("data_engineering", data_engineering_node)
    g.add_node("data_science",     data_science_node)
    g.add_node("data_analysis",    data_analysis_node)
    g.add_node("reporting",        reporting_node)

    g.set_entry_point("plan")

    _edge_map = {
        "data_engineering": "data_engineering",
        "data_science":     "data_science",
        "data_analysis":    "data_analysis",
        "reporting":        "reporting",
        "end":              END,
    }
    for node in ("plan", "data_engineering", "data_science", "data_analysis"):
        g.add_conditional_edges(node, _router, _edge_map)

    g.add_edge("reporting", END)
    return g.compile()


_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ── Public API ────────────────────────────────────────────────────────────────

def run_banking_analysis(task_description: str, bank_symbols: list = None) -> dict:
    """Run the 5-node LangGraph banking planning pipeline."""
    graph = _get_graph()
    if graph is None:
        return {"error": "langgraph not available"}

    initial: BankingAnalyticsState = {
        "task_description":   task_description,
        "bank_symbols":       bank_symbols or [],
        "analysis_plan":      None,
        "etl_guidance":       None,
        "data_collected":     False,
        "ml_guidance":        None,
        "analytics_guidance": None,
        "report_content":     None,
        "current_step":       "data_engineering",
        "iteration_count":    0,
        "errors":             [],
        "final_output":       None,
    }

    print("\n[BankingGraph] ══ Starting 5-node Gemini planning pipeline ══\n", flush=True)
    final_state = graph.invoke(initial)
    print("\n[BankingGraph] ══ Pipeline complete ══\n", flush=True)
    return final_state
