"""
tools/planner_tools.py – Marketing strategy & planning tools for the Planner agent.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from crewai.tools import tool


def _gemini(prompt: str, system: str = "", _timeout: int = 130) -> str:
    """Call Gemini 2.5 Flash.
    Hard wall-clock cap = 130s (covers 4 × 30s attempts).
    On timeout/error returns a fallback string so the agent keeps moving –
    the pipeline never crashes due to a single slow LLM call.
    """
    import concurrent.futures

    def _call() -> str:
        try:
            from config import gemini_llm
            from langchain_core.messages import HumanMessage, SystemMessage

            if gemini_llm is None:
                return "Gemini unavailable – proceeding with available information."

            messages = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(HumanMessage(content=prompt))

            resp = gemini_llm.invoke(messages)
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as exc:
            print(f"[PlannerTools] Gemini error (will return fallback): {exc}", flush=True)
            return "Unable to generate response at this time – proceeding with available context."

    # NOTE: do NOT use `with` here – its __exit__ calls shutdown(wait=True)
    # which blocks forever even after TimeoutError is raised on the future.
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_call)
    try:
        result = fut.result(timeout=_timeout)
        ex.shutdown(wait=False)
        return result
    except concurrent.futures.TimeoutError:
        ex.shutdown(wait=False)  # release immediately, don't join the blocked thread
        print(f"[PlannerTools] Gemini hard timeout ({_timeout}s) – returning fallback", flush=True)
        return "Response timed out – proceeding with available context from prior steps."


def _save_md(content: str, prefix: str, brand: str) -> str:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() else "_" for c in brand)
    out  = Path(__file__).parent.parent / "outputs" / "content"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{prefix}_{safe}_{ts}.md"
    path.write_text(content, encoding="utf-8")
    print(f"[PlannerTools] Saved → {path}", flush=True)
    return str(path)


# ── tools ─────────────────────────────────────────────────────────────────────

@tool("create_marketing_strategy")
def create_marketing_strategy(
    brand: str,
    audience: str,
    goals: str,
    research_summary: str,
) -> str:
    """Synthesise research into a full marketing strategy document.

    Args:
        brand: Brand name.
        audience: Target audience description.
        goals: Campaign goals.
        research_summary: Summary of research findings.

    Returns:
        Path to saved strategy markdown file.
    """
    strategy = _gemini(
        prompt=(
            f"Brand: {brand}\nAudience: {audience}\nGoals: {goals}\n\n"
            f"Research:\n{research_summary[:1500]}\n\n"
            "Write a comprehensive marketing strategy (800–1200 words) with: "
            "positioning, unique value proposition, messaging framework, channel mix, "
            "creative direction, and phased rollout plan."
        ),
        system="You are a Senior Marketing Strategist. Format output as structured markdown.",
    )
    path = _save_md(strategy, "marketing_strategy", brand)
    return f"Strategy saved to {path}\n\n{strategy[:800]}"


@tool("create_content_calendar")
def create_content_calendar(strategy: str, duration_days: int = 30) -> str:
    """Create a content calendar for the specified duration.

    Args:
        strategy: Marketing strategy summary.
        duration_days: Calendar duration in days (30, 60, or 90).

    Returns:
        Content calendar as a formatted markdown table.
    """
    calendar = _gemini(
        prompt=(
            f"Duration: {duration_days} days\nStrategy summary:\n{strategy[:1000]}\n\n"
            f"Create a {duration_days}-day content calendar with weekly breakdown. "
            "Include: Week number, Platform, Post type, Topic/Theme, Hashtags/CTA. "
            "Format as a markdown table."
        ),
        system="You are a Social Media Content Manager. Be specific and actionable.",
    )
    return calendar


@tool("define_campaign_kpis")
def define_campaign_kpis(goals: str, budget: str) -> str:
    """Define a KPI framework for the campaign.

    Args:
        goals: Campaign goals and objectives.
        budget: Total campaign budget.

    Returns:
        KPI framework as a formatted string.
    """
    kpis = _gemini(
        prompt=(
            f"Campaign goals: {goals}\nBudget: {budget}\n\n"
            "Define a complete KPI framework including: "
            "CTR (target range), ROAS, CAC, LTV, Engagement Rate, Conversion Rate, "
            "Reach, Impressions, CPC, CPM. "
            "For each KPI provide: definition, target value, measurement method, reporting frequency."
        ),
        system="You are a Marketing Analytics Director. Format as a markdown table.",
    )
    return kpis


@tool("create_campaign_brief")
def create_campaign_brief(
    brand: str,
    campaign_type: str,
    audience: str,
    strategy: str,
) -> str:
    """Create a one-page campaign brief.

    Args:
        brand: Brand name.
        campaign_type: Type of campaign (e.g., Brand Awareness, Lead Generation).
        audience: Target audience.
        strategy: Strategy summary.

    Returns:
        Path to saved brief markdown file.
    """
    brief = _gemini(
        prompt=(
            f"Brand: {brand}\nCampaign type: {campaign_type}\nAudience: {audience}\n\n"
            f"Strategy:\n{strategy[:800]}\n\n"
            "Write a concise one-page campaign brief with: "
            "Campaign Name, Objective, Target Audience, Key Message, "
            "Channels, Creative Requirements, Timeline, Budget Split, Success Metrics."
        ),
        system="You are a Campaign Manager writing an internal brief. Use clean markdown formatting.",
    )
    path = _save_md(brief, "campaign_brief", brand)
    return f"Campaign brief saved to {path}\n\n{brief}"


@tool("plan_budget_allocation")
def plan_budget_allocation(total_budget: str, channels: str) -> str:
    """Recommend budget allocation across marketing channels.

    Args:
        total_budget: Total campaign budget (e.g., '$50,000').
        channels: Comma-separated list of channels to allocate budget to.

    Returns:
        Budget allocation plan as a formatted string.
    """
    allocation = _gemini(
        prompt=(
            f"Total budget: {total_budget}\nChannels: {channels}\n\n"
            "Recommend percentage budget split across these channels with rationale. "
            "Consider: audience reach, cost efficiency, conversion potential, brand goals. "
            "Format as a markdown table: Channel | % Allocation | $ Amount | Rationale"
        ),
        system="You are a Media Buying Director. Be specific and data-driven.",
    )
    return allocation
