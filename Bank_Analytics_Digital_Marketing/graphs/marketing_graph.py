"""
graphs/marketing_graph.py – LangGraph 5-node Gemini-only marketing pre-planning pipeline.
Adapted from Digital_Marketing_Agent/graphs/marketing_graph.py for the combined system.

Receives banking_context (churn analysis) so each node's prompts are informed
by the actual customer segments and risk factors discovered in Phase 2.

Nodes: plan → research → strategy → content → reporting
All nodes: Gemini 2.5 Flash
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage


# ── State schema ──────────────────────────────────────────────────────────────

class MarketingState(TypedDict, total=False):
    task_description:   str
    brand_name:         str
    industry:           str
    target_audience:    str
    campaign_goals:     str
    budget:             str
    competitors:        str
    campaign_type:      str
    banking_context:    Optional[str]   # churn analysis from Phase 2
    # outputs written by each node
    analysis_plan:      Optional[str]
    research_guidance:  Optional[str]
    strategy_guidance:  Optional[str]
    content_guidance:   Optional[str]
    report_content:     Optional[str]
    # control
    current_step:       Optional[str]
    iteration_count:    Optional[int]
    errors:             Optional[List[str]]
    final_output:       Optional[Dict[str, Any]]


# ── Gemini helper ─────────────────────────────────────────────────────────────

def _query_gemini(messages: list, label: str = "", _timeout: int = 130) -> str:
    """Call Gemini 2.5 Flash with a hard wall-clock timeout."""
    import concurrent.futures

    tag = f"[MarketingGraph · {label}]" if label else "[MarketingGraph]"

    def _call() -> str:
        try:
            from config import gemini_llm
            if gemini_llm is None:
                return "[Gemini unavailable – check GOOGLE_API_KEY]"

            lc_messages = []
            for m in messages:
                if m["role"] == "system":
                    lc_messages.append(SystemMessage(content=m["content"]))
                else:
                    lc_messages.append(HumanMessage(content=m["content"]))

            response = gemini_llm.invoke(lc_messages)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            return f"[ERROR: {exc}]"

    print(f"{tag} querying Gemini 2.5 Flash…", flush=True)
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_call)
    try:
        text = fut.result(timeout=_timeout)
        ex.shutdown(wait=False)
        if text.startswith("[ERROR"):
            print(f"{tag} ERROR (fallback used) – {text}", file=sys.stderr, flush=True)
        else:
            print(f"{tag} ✓ Gemini responded ({len(text)} chars)", flush=True)
        return text
    except concurrent.futures.TimeoutError:
        ex.shutdown(wait=False)
        print(f"{tag} hard timeout ({_timeout}s) – returning fallback, pipeline continues",
              file=sys.stderr, flush=True)
        return "Planning step timed out – pipeline will continue with available context."


# ── Node implementations ───────────────────────────────────────────────────────

def plan_node(state: MarketingState) -> MarketingState:
    """CMO: produces a structured JSON campaign plan informed by banking churn data."""
    task    = state.get("task_description", "")
    brand   = state.get("brand_name", "Unknown Brand")
    ind     = state.get("industry", "General")
    goals   = state.get("campaign_goals", "")
    budget  = state.get("budget", "Not specified")
    camp    = state.get("campaign_type", "Brand Awareness")
    churn   = state.get("banking_context", "")

    churn_section = f"\n\n[Banking Churn Analysis - Key Context]:\n{churn[:500]}" if churn else ""

    raw = _query_gemini(
        [
            {
                "role": "system",
                "content": (
                    "You are a Chief Marketing Officer creating a strategic campaign plan "
                    "for a banking client. The campaign must address real customer churn risks "
                    "identified in the banking analytics. "
                    "Respond ONLY with valid JSON containing these keys: "
                    "objectives (list), channels (list), target_segments (list), "
                    "content_types (list), kpis (list), budget_allocation (object with channel→pct)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Brand: {brand}\nIndustry: {ind}\nCampaign type: {camp}\n"
                    f"Goals: {goals}\nBudget: {budget}\nTask: {task}"
                    f"{churn_section}\n\nOutput ONLY valid JSON."
                ),
            },
        ],
        label="1/5 plan",
    )

    clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        plan_obj = json.loads(clean)
        plan_str = json.dumps(plan_obj, indent=2)
    except json.JSONDecodeError:
        plan_str = raw

    return {
        **state,
        "analysis_plan": plan_str,
        "current_step":  "plan",
        "errors":        state.get("errors", []),
    }


def research_node(state: MarketingState) -> MarketingState:
    """Market Research Director: competitor landscape, audience personas, trends."""
    brand       = state.get("brand_name", "Unknown Brand")
    ind         = state.get("industry", "General")
    audience    = state.get("target_audience", "")
    competitors = state.get("competitors", "Not specified")
    plan        = state.get("analysis_plan", "")
    churn       = state.get("banking_context", "")

    churn_section = f"\n\n[Churn Analysis - At-Risk Segments]:\n{churn[:400]}" if churn else ""

    guidance = _query_gemini(
        [
            {
                "role": "system",
                "content": (
                    "You are a Market Research Director for a banking retention campaign. "
                    "Produce a detailed research brief covering: "
                    "1) Competitor landscape & positioning gaps in banking retention. "
                    "2) Target audience personas mapped to churn risk profiles. "
                    "3) Current trend signals in banking loyalty and retention marketing. "
                    "4) Specific search queries the research team should run. "
                    "Format as structured markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Brand: {brand}\nIndustry: {ind}\nAudience: {audience}\n"
                    f"Competitors: {competitors}\n\nStrategic plan:\n{plan[:600]}"
                    f"{churn_section}"
                ),
            },
        ],
        label="2/5 research",
    )

    return {
        **state,
        "research_guidance": guidance,
        "current_step":      "research",
    }


def strategy_node(state: MarketingState) -> MarketingState:
    """Marketing Strategist: positioning, messaging framework, channel mix, timeline."""
    brand    = state.get("brand_name", "Unknown Brand")
    goals    = state.get("campaign_goals", "")
    research = state.get("research_guidance", "")
    plan     = state.get("analysis_plan", "")
    churn    = state.get("banking_context", "")

    churn_section = f"\n\n[Churn Drivers to Address]:\n{churn[:400]}" if churn else ""

    guidance = _query_gemini(
        [
            {
                "role": "system",
                "content": (
                    "You are a Senior Marketing Strategist specialising in banking retention. "
                    "Develop a campaign strategy that includes: "
                    "1) Brand positioning & unique value proposition for at-risk customers. "
                    "2) Messaging framework (headlines, taglines, proof points addressing churn drivers). "
                    "3) Channel mix with rationale (digital-first, personalised). "
                    "4) Campaign timeline (phases & milestones). "
                    "5) Creative direction principles. "
                    "Format as structured markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Brand: {brand}\nGoals: {goals}\n\nResearch brief:\n{research[:600]}\n\n"
                    f"Strategic plan:\n{plan[:400]}"
                    f"{churn_section}"
                ),
            },
        ],
        label="3/5 strategy",
    )

    return {
        **state,
        "strategy_guidance": guidance,
        "current_step":      "strategy",
    }


def content_node(state: MarketingState) -> MarketingState:
    """Content Director: video concepts, ad copy guidelines, social calendar, email sequences."""
    brand    = state.get("brand_name", "Unknown Brand")
    audience = state.get("target_audience", "")
    strategy = state.get("strategy_guidance", "")
    camp     = state.get("campaign_type", "Brand Awareness")
    churn    = state.get("banking_context", "")

    churn_section = f"\n\n[At-Risk Customer Context]:\n{churn[:400]}" if churn else ""

    guidance = _query_gemini(
        [
            {
                "role": "system",
                "content": (
                    "You are a Creative Content Director for banking retention campaigns. "
                    "Produce a detailed content plan with: "
                    "1) Video concepts (2–3 ideas with prompts for Veo 3 AI video generation, "
                    "focused on customer trust, loyalty rewards, personalised service). "
                    "2) Ad copy guidelines for Google, Meta, LinkedIn. "
                    "3) 30-day social media calendar outline (platforms, post types, retention themes). "
                    "4) Email sequence (3-5 subject lines + brief body outlines targeting churn-risk segments). "
                    "5) Key visual & tone-of-voice guidelines (empathetic, reassuring, value-driven). "
                    "Format as structured markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Brand: {brand}\nAudience: {audience}\nCampaign type: {camp}\n\n"
                    f"Strategy:\n{strategy[:600]}"
                    f"{churn_section}"
                ),
            },
        ],
        label="4/5 content",
    )

    return {
        **state,
        "content_guidance": guidance,
        "current_step":     "content",
    }


def reporting_node(state: MarketingState) -> MarketingState:
    """CMO: writes a preliminary strategic brief (markdown) for HITL or handoff to crew."""
    brand    = state.get("brand_name", "Unknown Brand")
    ind      = state.get("industry", "General")
    goals    = state.get("campaign_goals", "")
    budget   = state.get("budget", "Not specified")
    plan     = state.get("analysis_plan", "")
    research = state.get("research_guidance", "")
    strategy = state.get("strategy_guidance", "")
    content  = state.get("content_guidance", "")
    churn    = state.get("banking_context", "")

    churn_section = f"\n\nBanking Churn Context:\n{churn[:300]}" if churn else ""

    report = _query_gemini(
        [
            {
                "role": "system",
                "content": (
                    "You are a Chief Marketing Officer writing a pre-campaign strategic brief for "
                    "banking executive review. Format as markdown with these sections: "
                    "## Executive Summary, ## Banking Churn Context, ## Campaign Objectives, "
                    "## Target Audience, ## Strategy Overview, ## Content Plan Summary, "
                    "## KPIs & Success Metrics, ## Budget Allocation, ## Timeline, ## Deliverables. "
                    "Be concise but comprehensive."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Brand: {brand} | Industry: {ind} | Budget: {budget}\nGoals: {goals}"
                    f"{churn_section}\n\n"
                    f"Campaign Plan:\n{plan[:400]}\n\n"
                    f"Research Brief:\n{research[:400]}\n\n"
                    f"Strategy:\n{strategy[:400]}\n\n"
                    f"Content Plan:\n{content[:400]}"
                ),
            },
        ],
        label="5/5 reporting",
    )

    return {
        **state,
        "report_content": report,
        "current_step":   "completed",
        "final_output": {
            "analysis_plan":      state.get("analysis_plan"),
            "research_guidance":  state.get("research_guidance"),
            "strategy_guidance":  state.get("strategy_guidance"),
            "content_guidance":   state.get("content_guidance"),
            "preliminary_report": report,
        },
    }


# ── Router ─────────────────────────────────────────────────────────────────────

def _router(state: MarketingState) -> str:
    step = state.get("current_step", "plan")
    routing = {
        "plan":     "research",
        "research": "strategy",
        "strategy": "content",
        "content":  "reporting",
    }
    return routing.get(step, "end")


# ── Build graph ────────────────────────────────────────────────────────────────

def _build_graph():
    try:
        from langgraph.graph import StateGraph, END
    except ImportError as exc:
        print(f"[MarketingGraph] langgraph not installed – {exc}", file=sys.stderr)
        return None

    g = StateGraph(MarketingState)
    g.add_node("plan",      plan_node)
    g.add_node("research",  research_node)
    g.add_node("strategy",  strategy_node)
    g.add_node("content",   content_node)
    g.add_node("reporting", reporting_node)

    g.set_entry_point("plan")

    g.add_conditional_edges("plan",      _router, {"research":  "research",  "end": END})
    g.add_conditional_edges("research",  _router, {"strategy":  "strategy",  "end": END})
    g.add_conditional_edges("strategy",  _router, {"content":   "content",   "end": END})
    g.add_conditional_edges("content",   _router, {"reporting": "reporting", "end": END})
    g.add_edge("reporting", END)

    return g.compile()


_graph = None  # lazy singleton


def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ── Public API ────────────────────────────────────────────────────────────────

def run_marketing_analysis(
    task_description: str,
    brand_name:       str = "Unknown Brand",
    industry:         str = "Banking & Financial Services",
    target_audience:  str = "",
    campaign_goals:   str = "",
    budget:           str = "Not specified",
    competitors:      str = "Not specified",
    campaign_type:    str = "Customer Retention",
    banking_context:  str = "",
) -> dict:
    """
    Run the 5-node LangGraph marketing planning pipeline.
    banking_context is the churn analysis from Phase 2 — injected into all nodes.

    Returns dict with: analysis_plan, research_guidance, strategy_guidance,
                       content_guidance, preliminary_report, final_output.
    """
    graph = _get_graph()
    if graph is None:
        return {
            "error":              "langgraph not available",
            "preliminary_report": "",
            "analysis_plan":      "",
            "research_guidance":  "",
            "strategy_guidance":  "",
            "content_guidance":   "",
        }

    initial_state: MarketingState = {
        "task_description": task_description,
        "brand_name":       brand_name,
        "industry":         industry,
        "target_audience":  target_audience,
        "campaign_goals":   campaign_goals,
        "budget":           budget,
        "competitors":      competitors,
        "campaign_type":    campaign_type,
        "banking_context":  banking_context,
        "current_step":     "plan",
        "iteration_count":  0,
        "errors":           [],
    }

    print("\n[MarketingGraph] ══ Starting 5-node Gemini marketing planning pipeline ══\n",
          flush=True)
    final_state = graph.invoke(initial_state)
    print("\n[MarketingGraph] ══ Marketing pipeline complete ══\n", flush=True)

    return {
        "analysis_plan":      final_state.get("analysis_plan", ""),
        "research_guidance":  final_state.get("research_guidance", ""),
        "strategy_guidance":  final_state.get("strategy_guidance", ""),
        "content_guidance":   final_state.get("content_guidance", ""),
        "preliminary_report": final_state.get("report_content", ""),
        "final_output":       final_state.get("final_output", {}),
    }
