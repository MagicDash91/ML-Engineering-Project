"""
Risk Analytics — LangGraph Workflow
=====================================
5-node Ollama strategic pre-planning pipeline for financial risk assessment.

Workflow:
    [plan] → [data_engineering] → [risk_modelling] → [risk_analysis] → [reporting] → END
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

def _query_ollama(messages: list, label: str = "") -> str:
    """Call Ollama qwen3.5:cloud with hard wall-clock timeout."""
    import concurrent.futures
    from langchain_community.chat_models import ChatOllama

    tag = f"[RiskGraph · {label}]" if label else "[RiskGraph]"

    def _call() -> str:
        import time
        lc_msgs = []
        for m in messages:
            role    = m.get("role", "user")
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
                llm   = ChatOllama(model="qwen3.5:cloud")
                reply = llm.invoke(lc_msgs).content.strip()
                return reply
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    print(f"  {tag} Ollama error (attempt {attempt+1}/3): {exc} — retrying in 5s", flush=True)
                    time.sleep(5)
        return f"Planning step failed after 3 attempts: {last_exc}"

    print(f"  {tag} querying Ollama qwen3.5:cloud…", flush=True)
    ex  = concurrent.futures.ThreadPoolExecutor(max_workers=1)
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

class RiskAnalyticsState(TypedDict):
    task_description:   str
    risk_domain:        Optional[str]           # e.g. "credit", "market", "liquidity", "operational"
    analysis_plan:      Optional[str]
    etl_guidance:       Optional[str]
    data_collected:     Optional[bool]
    risk_modelling:     Optional[str]
    risk_analysis:      Optional[str]
    report_content:     Optional[str]
    current_step:       Optional[str]
    iteration_count:    Optional[int]
    errors:             Optional[List[str]]
    final_output:       Optional[str]


# ── Node implementations ──────────────────────────────────────────────────────

def plan_node(state: RiskAnalyticsState) -> RiskAnalyticsState:
    task        = state["task_description"]
    risk_domain = state.get("risk_domain") or "financial risk"

    raw = _query_ollama([
        {
            "role": "system",
            "content": (
                "You are a Chief Risk Officer (CRO) with 20 years of experience across "
                "credit risk, market risk, liquidity risk, and operational risk. "
                "Create a detailed, structured risk analysis plan for a multi-agent risk team. "
                "Your team consists of: "
                "Risk Data Engineer (data discovery, ETL, data quality), "
                "Risk Scientist (VaR, CVaR, stress testing, credit scoring, portfolio risk models), "
                "Risk Analyst (risk dashboards, heat maps, visualizations, compliance reports). "
                "The analysis must adapt to whatever financial data is in the connected source. "
                "Respond with a valid JSON object containing:\n"
                "  'summary', 'risk_domain', 'data_needs', 'etl_steps',\n"
                "  'risk_models', 'visualizations', 'report_sections', 'regulatory_framework'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task: {task}\n"
                f"Risk Domain Context: {risk_domain}\n\n"
                "The data source may contain loan portfolios, transaction records, market prices, "
                "P&L data, exposure tables, or any other financial data. "
                "Plan for schema discovery first, then adapt risk modelling to what is found. "
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


def data_engineering_node(state: RiskAnalyticsState) -> RiskAnalyticsState:
    task    = state["task_description"]
    plan    = state.get("analysis_plan", "")

    guidance = _query_ollama([
        {
            "role": "system",
            "content": (
                "You are a Senior Risk Data Engineer specialising in financial data infrastructure. "
                "Given the risk analysis plan, produce precise data engineering instructions.\n"
                "Specify: 1) Steps to discover all tables in the database "
                "2) How to profile the primary risk table (row count, schema, nulls, dtypes) "
                "3) Web search queries for relevant regulatory/risk benchmarks "
                "4) Which columns to clean or normalize (e.g. amount fields stored as text) "
                "5) Data quality checks specific to financial data "
                "6) How to identify the key risk column (default flag, loss amount, exposure, etc.)."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analysis Plan:\n{plan}\n\n"
                f"Task: {task}\n\n"
                "Provide step-by-step data engineering instructions for financial risk data. "
                "Emphasise schema discovery first, then profiling, then cleaning."
            ),
        },
    ], label="2/5 data_engineering")

    return {
        **state,
        "etl_guidance":    guidance,
        "data_collected":  True,
        "current_step":    "risk_modelling",
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def risk_modelling_node(state: RiskAnalyticsState) -> RiskAnalyticsState:
    task     = state["task_description"]
    plan     = state.get("analysis_plan", "")
    etl_info = state.get("etl_guidance", "")

    guidance = _query_ollama([
        {
            "role": "system",
            "content": (
                "You are a Senior Quantitative Risk Scientist. "
                "Based on the discovered data schema, recommend the most appropriate risk models:\n"
                "- Loan/credit data (default flag, PD, LGD) → credit risk model + credit scoring\n"
                "- Market/price data (returns, prices, P&L) → VaR, CVaR, Monte Carlo simulation\n"
                "- Transaction data (amounts, frequencies)  → anomaly/fraud detection + segmentation\n"
                "- Portfolio data (positions, exposures)    → portfolio risk, concentration risk\n"
                "- Any dataset → always run risk segmentation (4 risk tiers: Low/Medium/High/Critical)\n"
                "Specify: 1) Which risk model to use and why "
                "2) Exact table name and target/key column "
                "3) Risk metrics to compute (VaR 95%, VaR 99%, CVaR, Sharpe, PD, etc.) "
                "4) Stress test scenarios to apply "
                "5) Regulatory framework to reference (Basel III, IFRS 9, FRTB, etc.)."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analysis Plan:\n{plan}\n\n"
                f"ETL Guidance:\n{etl_info[:600]}\n\n"
                f"Task: {task}\n\n"
                "Which risk models should be run? Be specific about table and column names."
            ),
        },
    ], label="3/5 risk_modelling")

    return {
        **state,
        "risk_modelling":  guidance,
        "current_step":    "risk_analysis",
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def risk_analysis_node(state: RiskAnalyticsState) -> RiskAnalyticsState:
    task       = state["task_description"]
    plan       = state.get("analysis_plan", "")
    risk_model = state.get("risk_modelling", "")
    etl_info   = state.get("etl_guidance", "")

    guidance = _query_ollama([
        {
            "role": "system",
            "content": (
                "You are a Senior Risk Analyst specialising in financial risk dashboards. "
                "Design a complete risk visualization package:\n"
                "1) At least 5 charts (type, columns, purpose)\n"
                "2) Must include: risk distribution (pie/bar), loss/exposure histogram, "
                "   risk score scatter, time-trend line chart, risk heat map (encoded table)\n"
                "3) All column names must match the discovered schema exactly\n"
                "4) Key risk KPIs to highlight: PD, LGD, EAD, VaR, NPL ratio, etc.\n"
                "5) Compliance checklist sections to include in the report\n"
                "6) Note: heat map must use the '_encoded' version of the primary table."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analysis Plan:\n{plan}\n\n"
                f"ETL Guidance:\n{etl_info[:400]}\n\nRisk Model Guidance:\n{risk_model[:400]}\n\n"
                f"Task: {task}\n\n"
                "Specify the complete risk visualization and compliance checklist package."
            ),
        },
    ], label="4/5 risk_analysis")

    return {
        **state,
        "risk_analysis": guidance,
        "current_step":  "reporting",
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def reporting_node(state: RiskAnalyticsState) -> RiskAnalyticsState:
    task        = state["task_description"]
    plan        = state.get("analysis_plan", "")
    etl_info    = state.get("etl_guidance", "")
    risk_model  = state.get("risk_modelling", "")
    risk_anal   = state.get("risk_analysis", "")

    report = _query_ollama([
        {
            "role": "system",
            "content": (
                "You are a Chief Risk Officer writing a pre-analysis strategic risk brief. "
                "Format as professional markdown with clear headings. "
                "Be specific: mention likely table names, risk columns, and model types "
                "based on the planning documents. Reference applicable regulatory frameworks."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task: {task}\n\n"
                f"Analysis Plan:\n{plan[:800]}\n\n"
                f"ETL Plan:\n{etl_info[:400]}\nRisk Model Plan:\n{risk_model[:400]}\n"
                f"Risk Analysis Plan:\n{risk_anal[:400]}\n\n"
                "Write the preliminary strategic risk brief:\n"
                "# Preliminary Financial Risk Assessment Brief\n"
                "## Executive Summary\n## Risk Scope & Objectives\n"
                "## Data Sources & Methodology\n## Risk Models Planned\n"
                "## Key Risk Areas\n## Regulatory Considerations\n"
                "## Preliminary Recommendations\n## Deliverables"
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

def _router(state: RiskAnalyticsState) -> str:
    step = state.get("current_step", "data_engineering")
    if state.get("iteration_count", 0) > 5:
        return "end"
    routes = {
        "data_engineering": "data_engineering",
        "risk_modelling":   "modelling",
        "risk_analysis":    "analysis",
        "reporting":        "reporting",
        "completed":        "end",
    }
    return routes.get(step, "end")


# ── Build graph ───────────────────────────────────────────────────────────────

def _build_graph():
    try:
        from langgraph.graph import StateGraph, END
    except ImportError as exc:
        print(f"[RiskGraph] langgraph not installed – {exc}")
        return None

    g = StateGraph(RiskAnalyticsState)
    g.add_node("plan",             plan_node)
    g.add_node("data_engineering", data_engineering_node)
    g.add_node("modelling",        risk_modelling_node)
    g.add_node("analysis",         risk_analysis_node)
    g.add_node("reporting",        reporting_node)

    g.set_entry_point("plan")

    _edge_map = {
        "data_engineering": "data_engineering",
        "modelling":        "modelling",
        "analysis":         "analysis",
        "reporting":        "reporting",
        "end":              END,
    }
    for node in ("plan", "data_engineering", "modelling", "analysis"):
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

def run_risk_planning(task_description: str, risk_domain: str = "") -> dict:
    """Run the 5-node LangGraph risk planning pipeline."""
    graph = _get_graph()
    if graph is None:
        return {"error": "langgraph not available"}

    initial: RiskAnalyticsState = {
        "task_description": task_description,
        "risk_domain":      risk_domain or "financial risk",
        "analysis_plan":    None,
        "etl_guidance":     None,
        "data_collected":   False,
        "risk_modelling":   None,
        "risk_analysis":    None,
        "report_content":   None,
        "current_step":     "data_engineering",
        "iteration_count":  0,
        "errors":           [],
        "final_output":     None,
    }

    print("\n[RiskGraph] ══ Starting 5-node Ollama risk planning pipeline ══\n", flush=True)
    final_state = graph.invoke(initial)
    print("\n[RiskGraph] ══ Pipeline complete ══\n", flush=True)
    return final_state
