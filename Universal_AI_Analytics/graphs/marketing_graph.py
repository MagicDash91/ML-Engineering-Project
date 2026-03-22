"""
graphs/marketing_graph.py – LangGraph 5-node Gemini-only marketing pre-planning pipeline.
Adapted from Digital_Marketing_Agent/graphs/marketing_graph.py for the combined system.

Receives analytics_context (churn analysis) so each node's prompts are informed
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
    analytics_context:    Optional[str]   # data analytics findings from Phase 2
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


# ── Ollama helper ─────────────────────────────────────────────────────────────

def _query_gemini(messages: list, label: str = "", _timeout: int = 130) -> str:
    """Call Ollama qwen3.5:cloud with a hard wall-clock timeout."""
    import concurrent.futures
    from langchain_community.chat_models import ChatOllama

    tag = f"[MarketingGraph · {label}]" if label else "[MarketingGraph]"

    def _call() -> str:
        import time
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            else:
                lc_messages.append(HumanMessage(content=m["content"]))
        last_exc = None
        for attempt in range(3):
            try:
                llm = ChatOllama(model="qwen3.5:cloud")
                response = llm.invoke(lc_messages)
                return response.content if hasattr(response, "content") else str(response)
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    print(f"{tag} Ollama error (attempt {attempt+1}/3): {exc} — retrying in 5s", flush=True)
                    time.sleep(5)
        return f"[ERROR: {last_exc}]"

    print(f"{tag} querying Ollama qwen3.5:cloud…", flush=True)
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_call)
    try:
        text = fut.result(timeout=_timeout)
        ex.shutdown(wait=False)
        if text.startswith("[ERROR"):
            print(f"{tag} ERROR (fallback used) – {text}", file=sys.stderr, flush=True)
        else:
            print(f"{tag} ✓ Ollama responded ({len(text)} chars)", flush=True)
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
    churn   = state.get("analytics_context", "")

    analytics_section = f"\n\n[Data Analytics Findings - Key Context]:\n{churn[:500]}" if churn else ""

    raw = _query_gemini(
        [
            {
                "role": "system",
                "content": (
                    "You are a Chief Marketing Officer creating a strategic campaign plan "
                    "informed by data analytics findings. The campaign must address real "
                    "audience risks and opportunities identified in the analytics. "
                    "Adapt your strategy to the industry and data domain provided. "
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
                    f"{analytics_section}\n\nOutput ONLY valid JSON."
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
    churn       = state.get("analytics_context", "")

    analytics_section = f"\n\n[Analytics Findings - At-Risk Segments]:\n{churn[:400]}" if churn else ""

    guidance = _query_gemini(
        [
            {
                "role": "system",
                "content": (
                    "You are a Market Research Director for a data-driven marketing campaign. "
                    "Produce a detailed research brief covering: "
                    "1) Competitor landscape & positioning gaps relevant to the industry provided. "
                    "2) Target audience personas mapped to the risk/opportunity profiles from analytics. "
                    "3) Current trend signals in marketing and customer engagement for this industry. "
                    "4) Specific search queries the research team should run (tailored to the domain). "
                    "Adapt all research to the industry and audience segments identified in analytics. "
                    "Format as structured markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Brand: {brand}\nIndustry: {ind}\nAudience: {audience}\n"
                    f"Competitors: {competitors}\n\nStrategic plan:\n{plan[:600]}"
                    f"{analytics_section}"
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
    churn    = state.get("analytics_context", "")

    analytics_section = f"\n\n[Key Risk & Opportunity Drivers from Analytics]:\n{churn[:400]}" if churn else ""

    guidance = _query_gemini(
        [
            {
                "role": "system",
                "content": (
                    "You are a Senior Marketing Strategist with cross-industry expertise. "
                    "Develop a campaign strategy that addresses the audience risks and opportunities "
                    "identified in the analytics findings. Include: "
                    "1) Brand positioning & unique value proposition for the at-risk or high-opportunity segments. "
                    "2) Messaging framework (headlines, taglines, proof points addressing key drivers from data). "
                    "3) Specific campaign names and mechanics — creative, named initiatives "
                    "   (e.g. loyalty programmes, upgrade incentives, win-back flows, referral rewards). "
                    "4) Channel mix with rationale tailored to each segment's behaviour. "
                    "5) Campaign timeline (phases & milestones). "
                    "6) Creative direction principles (tone, visual style, messaging tone). "
                    "Format as structured markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Brand: {brand}\nGoals: {goals}\n\nResearch brief:\n{research[:600]}\n\n"
                    f"Strategic plan:\n{plan[:400]}"
                    f"{analytics_section}"
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
    churn    = state.get("analytics_context", "")

    analytics_section = f"\n\n[At-Risk / High-Opportunity Segment Context]:\n{churn[:400]}" if churn else ""

    guidance = _query_gemini(
        [
            {
                "role": "system",
                "content": (
                    "You are a Creative Content Director producing a multi-channel content plan "
                    "informed by data analytics findings. Produce a detailed content plan with: "
                    "1) Promotional poster concepts (2–3 ideas for Gemini AI image generation — "
                    "   describe the visual style, headline, and key message for each poster). "
                    "2) Ad copy guidelines for Google, Meta, and LinkedIn "
                    "   (tailored to the audience segments identified in analytics). "
                    "3) 30-day social media calendar outline (platforms, post types, themes per week). "
                    "4) Email sequence (3–5 subject lines + brief body outlines "
                    "   personalised to each audience segment). "
                    "5) Key visual and tone-of-voice guidelines "
                    "   (empathetic, benefit-driven, segment-specific). "
                    "Adapt all content to the industry and domain identified in the analytics. "
                    "Format as structured markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Brand: {brand}\nAudience: {audience}\nCampaign type: {camp}\n\n"
                    f"Strategy:\n{strategy[:600]}"
                    f"{analytics_section}"
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
    churn    = state.get("analytics_context", "")

    analytics_section = f"\n\nData Analytics Context:\n{churn[:300]}" if churn else ""

    report = _query_gemini(
        [
            {
                "role": "system",
                "content": (
                    "You are a Chief Marketing Officer writing a pre-campaign strategic brief "
                    "for executive review. Adapt the brief to the industry and domain provided. "
                    "Format as markdown with these sections: "
                    "## Executive Summary, ## Analytics Context & Why This Campaign, "
                    "## Campaign Objectives, ## Target Audience & Segments, "
                    "## Strategy Overview, ## Content Plan Summary, "
                    "## KPIs & Success Metrics, ## Budget Allocation, ## Timeline, ## Deliverables. "
                    "Be concise but comprehensive. Reference specific data findings where available."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Brand: {brand} | Industry: {ind} | Budget: {budget}\nGoals: {goals}"
                    f"{analytics_section}\n\n"
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
    task_description:  str,
    brand_name:        str = "Brand",
    industry:          str = "General",
    target_audience:   str = "",
    campaign_goals:    str = "",
    budget:            str = "Not specified",
    competitors:       str = "Not specified",
    campaign_type:     str = "Retention & Growth",
    analytics_context: str = "",
    banking_context:   str = "",   # legacy alias
) -> dict:
    """
    Run the 5-node LangGraph marketing planning pipeline.
    analytics_context is the data analytics findings from Phase 2 — injected into all nodes.

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

    effective_context = analytics_context or banking_context

    initial_state: MarketingState = {
        "task_description":  task_description,
        "brand_name":        brand_name,
        "industry":          industry,
        "target_audience":   target_audience,
        "campaign_goals":    campaign_goals,
        "budget":            budget,
        "competitors":       competitors,
        "campaign_type":     campaign_type,
        "analytics_context": effective_context,
        "current_step":      "plan",
        "iteration_count":   0,
        "errors":            [],
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
