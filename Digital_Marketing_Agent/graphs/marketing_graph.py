"""
graphs/marketing_graph.py – LangGraph 5-node Gemini-only strategic pre-planning pipeline.
Pattern mirrors banking_graph.py exactly.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

# ── State schema ─────────────────────────────────────────────────────────────

class MarketingState(TypedDict, total=False):
    task_description:   str
    brand_name:         str
    industry:           str
    target_audience:    str
    campaign_goals:     str
    budget:             str
    competitors:        str
    campaign_type:      str
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
        print(f"{tag} hard timeout ({_timeout}s) – returning fallback, pipeline continues", file=sys.stderr, flush=True)
        return "Planning step timed out – pipeline will continue with available context."


# ── Node implementations ──────────────────────────────────────────────────────

def plan_node(state: MarketingState) -> MarketingState:
    """CMO: produces a structured JSON campaign plan."""
    task   = state.get("task_description", "")
    brand  = state.get("brand_name", "Unknown Brand")
    ind    = state.get("industry", "General")
    goals  = state.get("campaign_goals", "")
    budget = state.get("budget", "Not specified")
    camp   = state.get("campaign_type", "Brand Awareness")

    raw = _query_gemini(
        [
            {
                "role": "system",
                "content": (
                    "You are a Chief Marketing Officer creating a strategic campaign plan. "
                    "Respond ONLY with valid JSON containing these keys: "
                    "objectives (list), channels (list), target_segments (list), "
                    "content_types (list), kpis (list), budget_allocation (object with channel→pct)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Brand: {brand}\nIndustry: {ind}\nCampaign type: {camp}\n"
                    f"Goals: {goals}\nBudget: {budget}\nTask: {task}\n\nOutput ONLY valid JSON."
                ),
            },
        ],
        label="1/5 plan",
    )

    # strip markdown fences if present
    clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        plan_obj = json.loads(clean)
        plan_str = json.dumps(plan_obj, indent=2)
    except json.JSONDecodeError:
        plan_str = raw  # keep raw text if JSON fails

    return {
        **state,
        "analysis_plan": plan_str,
        "current_step":  "plan",
        "errors":        state.get("errors", []),
    }


def research_node(state: MarketingState) -> MarketingState:
    """Market Research Director: competitor landscape, audience personas, trends."""
    brand      = state.get("brand_name", "Unknown Brand")
    ind        = state.get("industry", "General")
    audience   = state.get("target_audience", "")
    competitors= state.get("competitors", "Not specified")
    plan       = state.get("analysis_plan", "")

    guidance = _query_gemini(
        [
            {
                "role": "system",
                "content": (
                    "You are a Market Research Director. Produce a detailed research brief covering: "
                    "1) Competitor landscape & positioning gaps. "
                    "2) Target audience personas (demographics, psychographics, pain points). "
                    "3) Current trend signals and emerging opportunities. "
                    "4) Specific search queries the research team should run. "
                    "Format as structured markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Brand: {brand}\nIndustry: {ind}\nAudience: {audience}\n"
                    f"Competitors: {competitors}\n\nStrategic plan:\n{plan[:600]}"
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

    guidance = _query_gemini(
        [
            {
                "role": "system",
                "content": (
                    "You are a Senior Marketing Strategist. Develop a campaign strategy that includes: "
                    "1) Brand positioning & unique value proposition. "
                    "2) Messaging framework (headlines, taglines, proof points). "
                    "3) Channel mix with rationale. "
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

    guidance = _query_gemini(
        [
            {
                "role": "system",
                "content": (
                    "You are a Creative Content Director. Produce a detailed content plan with: "
                    "1) Video concepts (2–3 ideas with prompts for AI video generation). "
                    "2) Ad copy guidelines for Google, Meta, TikTok, LinkedIn. "
                    "3) 30-day social media calendar outline (platforms, post types, themes). "
                    "4) Email sequence (3-5 email subject lines + brief body outlines). "
                    "5) Key visual & tone-of-voice guidelines. "
                    "Format as structured markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Brand: {brand}\nAudience: {audience}\nCampaign type: {camp}\n\n"
                    f"Strategy:\n{strategy[:600]}"
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
    """CMO: writes a preliminary strategic brief (markdown) for HITL review."""
    brand    = state.get("brand_name", "Unknown Brand")
    ind      = state.get("industry", "General")
    goals    = state.get("campaign_goals", "")
    budget   = state.get("budget", "Not specified")
    plan     = state.get("analysis_plan", "")
    research = state.get("research_guidance", "")
    strategy = state.get("strategy_guidance", "")
    content  = state.get("content_guidance", "")

    report = _query_gemini(
        [
            {
                "role": "system",
                "content": (
                    "You are a Chief Marketing Officer writing a pre-campaign strategic brief for "
                    "executive review. Format as markdown with these sections: "
                    "## Executive Summary, ## Campaign Objectives, ## Target Audience, "
                    "## Strategy Overview, ## Content Plan Summary, ## KPIs & Success Metrics, "
                    "## Budget Allocation, ## Timeline, ## Deliverables. "
                    "Be concise but comprehensive."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Brand: {brand} | Industry: {ind} | Budget: {budget}\nGoals: {goals}\n\n"
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
            "analysis_plan":     state.get("analysis_plan"),
            "research_guidance": state.get("research_guidance"),
            "strategy_guidance": state.get("strategy_guidance"),
            "content_guidance":  state.get("content_guidance"),
            "preliminary_report": report,
        },
    }


# ── Router ────────────────────────────────────────────────────────────────────

def _router(state: MarketingState) -> str:
    step = state.get("current_step", "plan")
    routing = {
        "plan":     "research",
        "research": "strategy",
        "strategy": "content",
        "content":  "reporting",
    }
    return routing.get(step, "end")


# ── Build graph ───────────────────────────────────────────────────────────────

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

    g.add_conditional_edges(
        "plan",
        _router,
        {"research": "research", "end": END},
    )
    g.add_conditional_edges(
        "research",
        _router,
        {"strategy": "strategy", "end": END},
    )
    g.add_conditional_edges(
        "strategy",
        _router,
        {"content": "content", "end": END},
    )
    g.add_conditional_edges(
        "content",
        _router,
        {"reporting": "reporting", "end": END},
    )
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
    industry:         str = "General",
    target_audience:  str = "",
    campaign_goals:   str = "",
    budget:           str = "Not specified",
    competitors:      str = "Not specified",
    campaign_type:    str = "Brand Awareness",
) -> dict:
    """Run the 5-node LangGraph marketing planning pipeline and return a result dict."""
    graph = _get_graph()
    if graph is None:
        return {
            "error":            "langgraph not available",
            "preliminary_report": "",
            "analysis_plan":    "",
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
        "current_step":     "plan",
        "iteration_count":  0,
        "errors":           [],
    }

    print("\n[MarketingGraph] ══ Starting 5-node Gemini planning pipeline ══\n", flush=True)
    final_state = graph.invoke(initial_state)
    print("\n[MarketingGraph] ══ Pipeline complete ══\n", flush=True)

    return {
        "analysis_plan":     final_state.get("analysis_plan", ""),
        "research_guidance": final_state.get("research_guidance", ""),
        "strategy_guidance": final_state.get("strategy_guidance", ""),
        "content_guidance":  final_state.get("content_guidance", ""),
        "preliminary_report": final_state.get("report_content", ""),
        "final_output":      final_state.get("final_output", {}),
    }
