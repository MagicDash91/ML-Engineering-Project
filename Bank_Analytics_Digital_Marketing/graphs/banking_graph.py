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


# ── Gemini helper ─────────────────────────────────────────────────────────────

def _query_gemini(messages: list, label: str = "") -> str:
    """Call Gemini 2.5 Flash with hard wall-clock timeout."""
    import concurrent.futures

    tag = f"[BankingGraph · {label}]" if label else "[BankingGraph]"

    def _call() -> str:
        from config import gemini_llm
        if not gemini_llm:
            return "[Gemini unavailable – check GOOGLE_API_KEY]"
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
        reply = gemini_llm.invoke(lc_msgs).content.strip()
        return reply

    print(f"  {tag} querying Gemini 2.5 Flash…", flush=True)
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_call)
    try:
        result = fut.result(timeout=130)
        ex.shutdown(wait=False)
        print(f"  {tag} ✓ Gemini responded", flush=True)
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
                "You are a Chief Data Officer at a major bank. "
                "Create a detailed, structured analysis plan for your data team. "
                "Your team consists of: Data Engineer (ETL/scraping/PostgreSQL), "
                "Data Scientist (ML models/forecasting), Data Analyst (viz/insights/reports). "
                "Respond with a valid JSON object containing:\n"
                "  'summary', 'focus_area', 'data_needs', 'etl_steps',\n"
                "  'ml_tasks', 'visualizations', 'report_sections', 'success_criteria'."
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
                "You are a Senior Data Engineer at a bank. "
                "Given the analysis plan, produce precise data engineering instructions.\n"
                "Specify: 1) Exact stock ticker symbols 2) yfinance parameters 3) Web search queries "
                "4) PostgreSQL table names 5) Data quality checks 6) Transformations needed."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analysis Plan:\n{plan}\n\n"
                f"Task: {task}\nBank symbols requested: {symbols}\n\n"
                "Provide step-by-step data engineering instructions."
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
                "You are a Senior Data Scientist at a bank. "
                "Decide: 1) Which ML models to train (fraud/credit/churn/forecasting/segmentation) "
                "2) Table + target column for each model 3) Evaluation metrics "
                "4) Banking business interpretation 5) Regulatory considerations (Basel III, OJK)."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analysis Plan:\n{plan}\n\n"
                f"ETL Guidance:\n{etl_info[:600]}\n\n"
                f"Task: {task}\n\nWhich ML models should be trained and how?"
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
                "You are a Senior Data Analyst specialising in banking dashboards. "
                "Design: 1) Charts to create (type, columns, purpose) 2) Banking KPIs to highlight "
                "3) Dashboard focus areas 4) Report sections 5) Gemini vision framing per chart."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analysis Plan:\n{plan}\n\n"
                f"ETL Guidance:\n{etl_info[:400]}\n\nML Guidance:\n{ml_info[:400]}\n\n"
                f"Task: {task}\n\nSpecify the complete visualization and analytics package."
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
                "Format as professional banking markdown with clear headings. "
                "Include specific KPI targets and risk metrics where relevant."
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
                "# Preliminary Banking Analytics Brief\n"
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
