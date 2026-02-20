"""
Banking Analytics — LangGraph Workflow
=======================================
Implements a structured reasoning pipeline using LangGraph StateGraph
so the LLM "thinks step-by-step" through the banking analysis task
before the CrewAI multi-agent team executes it.

Workflow:
    [plan] → [data_engineering] → [data_science] → [data_analysis] → [reporting] → END

Each node calls NVIDIA Llama Nemotron for domain-specific reasoning,
accumulating structured guidance that the CrewAI agents can reference.
"""

import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END

from config import query_nvidia


# ============================================================
# State schema
# ============================================================
class BankingAnalyticsState(TypedDict):
    # ── Input ──────────────────────────────────────────────
    task_description: str
    bank_symbols:     Optional[List[str]]

    # ── Planning ───────────────────────────────────────────
    analysis_plan:    Optional[str]

    # ── Data engineering guidance ──────────────────────────
    etl_guidance:     Optional[str]
    data_collected:   Optional[bool]

    # ── Data science guidance ──────────────────────────────
    ml_guidance:      Optional[str]

    # ── Analytics guidance ─────────────────────────────────
    analytics_guidance: Optional[str]

    # ── Final report ───────────────────────────────────────
    report_content:   Optional[str]

    # ── Control ────────────────────────────────────────────
    current_step:     Optional[str]
    iteration_count:  Optional[int]
    errors:           Optional[List[str]]
    final_output:     Optional[str]


# ============================================================
# Nodes
# ============================================================

def plan_node(state: BankingAnalyticsState) -> BankingAnalyticsState:
    """
    Planning node – NVIDIA Llama Nemotron creates a structured analysis plan
    covering data needs, ML tasks, and visualization requirements.
    """
    task    = state["task_description"]
    symbols = state.get("bank_symbols") or []

    raw = query_nvidia([
        {
            "role": "system",
            "content": (
                "You are a Chief Data Officer at a major bank. "
                "Create a detailed, structured analysis plan for your data team. "
                "Your team consists of: Data Engineer (ETL/scraping/PostgreSQL), "
                "Data Scientist (ML models/forecasting), Data Analyst (viz/insights/reports). "
                "Respond with a valid JSON object containing:\n"
                "  'summary'          – 2-sentence overview\n"
                "  'focus_area'       – main business focus\n"
                "  'data_needs'       – list of required datasets\n"
                "  'etl_steps'        – ordered list of ETL steps\n"
                "  'ml_tasks'         – list of ML tasks with model types\n"
                "  'visualizations'   – list of charts to produce\n"
                "  'report_sections'  – list of report sections\n"
                "  'success_criteria' – list of measurable outcomes"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task: {task}\n"
                f"Bank symbols to analyse: {', '.join(symbols) if symbols else 'TBD'}\n\n"
                "Output ONLY valid JSON."
            ),
        },
    ], temperature=0.1, max_tokens=1500)

    # Try to extract the JSON block
    plan_str = raw
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            plan_obj = json.loads(match.group())
            plan_str = json.dumps(plan_obj, indent=2)
    except (json.JSONDecodeError, TypeError):
        pass  # Keep raw text as-is

    return {
        **state,
        "analysis_plan":   plan_str,
        "current_step":    "data_engineering",
        "iteration_count": 0,
        "errors":          [],
    }


def data_engineering_node(state: BankingAnalyticsState) -> BankingAnalyticsState:
    """
    Data Engineering node – determines exactly what data to collect,
    which tables to create, and what ETL transformations are needed.
    """
    task    = state["task_description"]
    plan    = state.get("analysis_plan", "")
    symbols = state.get("bank_symbols") or []

    guidance = query_nvidia([
        {
            "role": "system",
            "content": (
                "You are a Senior Data Engineer at a bank. "
                "Given the analysis plan, produce precise data engineering instructions.\n"
                "Specify:\n"
                "1. Exact stock ticker symbols (e.g. BBCA.JK, BMRI.JK)\n"
                "2. yfinance parameters (period, interval)\n"
                "3. Web search queries for banking news\n"
                "4. PostgreSQL table names to create\n"
                "5. Data quality checks to perform\n"
                "6. Any transformations needed before loading"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analysis Plan:\n{plan}\n\n"
                f"Task: {task}\n"
                f"Bank symbols requested: {symbols}\n\n"
                "Provide step-by-step data engineering instructions."
            ),
        },
    ], max_tokens=1200)

    return {
        **state,
        "etl_guidance":  guidance,
        "data_collected": True,
        "current_step":  "data_science",
    }


def data_science_node(state: BankingAnalyticsState) -> BankingAnalyticsState:
    """
    Data Science node – selects appropriate ML models, hyperparameters,
    and evaluation strategies for the banking use-case.
    """
    task     = state["task_description"]
    plan     = state.get("analysis_plan", "")
    etl_info = state.get("etl_guidance", "")

    guidance = query_nvidia([
        {
            "role": "system",
            "content": (
                "You are a Senior Data Scientist at a bank. "
                "Based on the plan and available data, decide:\n"
                "1. Which ML models to train (fraud detection / credit risk / churn / forecasting / segmentation)\n"
                "2. Which table + target column to use for each model\n"
                "3. Appropriate evaluation metrics (AUC-ROC, RMSE, Silhouette, etc.)\n"
                "4. How to interpret results in a banking business context\n"
                "5. Key risk / regulatory considerations (Basel III, GDPR, etc.)"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analysis Plan:\n{plan}\n\n"
                f"ETL Guidance (tables available):\n{etl_info[:600]}\n\n"
                f"Task: {task}\n\n"
                "Which ML models should be trained and how?"
            ),
        },
    ], max_tokens=1200)

    return {
        **state,
        "ml_guidance":  guidance,
        "current_step": "data_analysis",
    }


def data_analysis_node(state: BankingAnalyticsState) -> BankingAnalyticsState:
    """
    Data Analysis node – designs the visualization suite and business
    insight narrative for the Data Analyst to produce.
    """
    task      = state["task_description"]
    plan      = state.get("analysis_plan", "")
    ml_info   = state.get("ml_guidance", "")
    etl_info  = state.get("etl_guidance", "")

    guidance = query_nvidia([
        {
            "role": "system",
            "content": (
                "You are a Senior Data Analyst at a bank specialising in executive dashboards. "
                "Design the full visualization and reporting package:\n"
                "1. List each chart to create (type, x_column, y_column, purpose)\n"
                "2. Which banking KPIs to highlight (NIM, ROA, ROE, NPL, CAR, BOPO)\n"
                "3. Dashboard focus areas\n"
                "4. Report sections and key messages\n"
                "5. How Gemini vision analysis should be framed for each chart"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analysis Plan:\n{plan}\n\n"
                f"ETL Guidance:\n{etl_info[:400]}\n\n"
                f"ML Guidance:\n{ml_info[:400]}\n\n"
                f"Task: {task}\n\n"
                "Specify the complete visualization and analytics package."
            ),
        },
    ], max_tokens=1200)

    return {
        **state,
        "analytics_guidance": guidance,
        "current_step":       "reporting",
    }


def reporting_node(state: BankingAnalyticsState) -> BankingAnalyticsState:
    """
    Reporting node – compiles all guidance into a preliminary executive
    report draft that the Data Analyst can use as a base.
    """
    task      = state["task_description"]
    plan      = state.get("analysis_plan", "")
    etl_info  = state.get("etl_guidance", "")
    ml_info   = state.get("ml_guidance", "")
    analytics = state.get("analytics_guidance", "")

    report = query_nvidia([
        {
            "role": "system",
            "content": (
                "You are a Chief Data Officer writing a pre-analysis strategic brief. "
                "This is a PLANNING report — it outlines what the data team WILL discover "
                "and what the likely findings are based on the task and domain knowledge. "
                "Format as professional banking markdown with clear headings. "
                "Include specific KPI targets and risk metrics where relevant."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task: {task}\n\n"
                f"Analysis Plan (structured):\n{plan[:800]}\n\n"
                f"ETL Plan:\n{etl_info[:400]}\n"
                f"ML Plan:\n{ml_info[:400]}\n"
                f"Analytics Plan:\n{analytics[:400]}\n\n"
                "Write the preliminary strategic brief covering:\n"
                "# Preliminary Banking Analytics Brief\n"
                "## Executive Summary\n"
                "## Scope & Objectives\n"
                "## Data Sources & Methodology\n"
                "## Expected Findings\n"
                "## Key Risk Areas\n"
                "## Strategic Recommendations (preliminary)\n"
                "## Deliverables"
            ),
        },
    ], max_tokens=2500)

    return {
        **state,
        "report_content": report,
        "current_step":   "completed",
        "final_output":   report,
    }


# ============================================================
# Routing
# ============================================================

def _router(state: BankingAnalyticsState) -> str:
    """Route to the next node based on current_step."""
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


# ============================================================
# Build & compile graph
# ============================================================

def build_banking_graph() -> StateGraph:
    g = StateGraph(BankingAnalyticsState)

    g.add_node("plan",             plan_node)
    g.add_node("data_engineering", data_engineering_node)
    g.add_node("data_science",     data_science_node)
    g.add_node("data_analysis",    data_analysis_node)
    g.add_node("reporting",        reporting_node)

    g.set_entry_point("plan")

    # From plan → conditional
    g.add_conditional_edges("plan", _router, {
        "data_engineering": "data_engineering",
        "data_science":     "data_science",
        "data_analysis":    "data_analysis",
        "reporting":        "reporting",
        "end":              END,
    })

    # All intermediate nodes → conditional
    for node in ("data_engineering", "data_science", "data_analysis"):
        g.add_conditional_edges(node, _router, {
            "data_engineering": "data_engineering",
            "data_science":     "data_science",
            "data_analysis":    "data_analysis",
            "reporting":        "reporting",
            "end":              END,
        })

    g.add_edge("reporting", END)

    return g.compile()


# Singleton compiled graph
banking_graph = build_banking_graph()


# ============================================================
# Public API
# ============================================================

def run_banking_analysis(task_description: str, bank_symbols: list = None) -> dict:
    """
    Execute the LangGraph strategic planning pipeline.

    Args:
        task_description: Natural-language description of the banking analysis task.
        bank_symbols:     List of bank ticker codes (without '.JK', e.g. ['BBCA', 'BMRI']).

    Returns:
        Final state dict with keys:
            analysis_plan, etl_guidance, ml_guidance,
            analytics_guidance, report_content, final_output
    """
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

    return banking_graph.invoke(initial)
