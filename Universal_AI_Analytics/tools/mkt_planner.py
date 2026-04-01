"""
tools/mkt_planner.py – Marketing strategy and planning tools.
Mirrors Digital_Marketing_Agent/tools/planner_tools.py with paths adjusted.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai.tools import tool

_BASE    = Path(__file__).parent.parent / "outputs"
_CONTENT = _BASE / "content"
_CONTENT.mkdir(parents=True, exist_ok=True)


def _gemini(prompt: str, system: str = "", _timeout: int = 130) -> str:
    """Call Ollama qwen3.5:cloud with hard wall-clock timeout."""
    import concurrent.futures

    def _call() -> str:
        import time
        from langchain_community.chat_models import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage
        msgs = []
        if system:
            msgs.append(SystemMessage(content=system))
        msgs.append(HumanMessage(content=prompt))
        last_exc = None
        for attempt in range(3):
            try:
                llm = ChatOllama(model="qwen3.5:cloud")
                resp = llm.invoke(msgs)
                return resp.content if hasattr(resp, "content") else str(resp)
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    print(f"[PlannerTools] Ollama error (attempt {attempt+1}/3): {exc} — retrying in 5s", flush=True)
                    time.sleep(5)
        print(f"[PlannerTools] Ollama failed after 3 attempts: {last_exc}", flush=True)
        return "Unable to generate response at this time."

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_call)
    try:
        result = fut.result(timeout=_timeout)
        ex.shutdown(wait=False)
        return result
    except concurrent.futures.TimeoutError:
        ex.shutdown(wait=False)
        print(f"[PlannerTools] Ollama hard timeout ({_timeout}s) – returning fallback", flush=True)
        return "Response timed out – proceeding with available context from prior steps."


def _save_md(content: str, prefix: str, brand: str) -> str:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() else "_" for c in brand)
    path = _CONTENT / f"{prefix}_{safe}_{ts}.md"
    path.write_text(content, encoding="utf-8")
    print(f"[PlannerTools] Saved → {path}", flush=True)
    return str(path)


@tool("create_marketing_strategy")
def create_marketing_strategy(brand: str, audience: str, goals: str, research_summary: str) -> str:
    """
    Synthesise research into a full marketing strategy document.

    Args:
        brand:            Brand name.
        audience:         Target audience description.
        goals:            Campaign goals.
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
    """
    Create a content calendar for the specified duration.

    Args:
        strategy:      Marketing strategy summary.
        duration_days: Calendar duration in days (30, 60, or 90).

    Returns:
        Content calendar as a formatted markdown table.
    """
    return _gemini(
        prompt=(
            f"Duration: {duration_days} days\nStrategy summary:\n{strategy[:1000]}\n\n"
            f"Create a {duration_days}-day content calendar with weekly breakdown. "
            "Include: Week number, Platform, Post type, Topic/Theme, Hashtags/CTA. "
            "Format as a markdown table."
        ),
        system="You are a Social Media Content Manager. Be specific and actionable.",
    )


@tool("define_campaign_kpis")
def define_campaign_kpis(goals: str, budget: str) -> str:
    """
    Define a KPI framework for the campaign.

    Args:
        goals:  Campaign goals and objectives.
        budget: Total campaign budget.

    Returns:
        KPI framework as a formatted string.
    """
    return _gemini(
        prompt=(
            f"Campaign goals: {goals}\nBudget: {budget}\n\n"
            "Define a complete KPI framework including: "
            "CTR (target range), ROAS, CAC, LTV, Engagement Rate, Conversion Rate, "
            "Reach, Impressions, CPC, CPM. "
            "For each KPI provide: definition, target value, measurement method, reporting frequency."
        ),
        system="You are a Marketing Analytics Director. Format as a markdown table.",
    )


@tool("create_campaign_brief")
def create_campaign_brief(brand: str, campaign_type: str, audience: str, strategy: str) -> str:
    """
    Create a one-page campaign brief.

    Args:
        brand:         Brand name.
        campaign_type: Type of campaign (e.g., Brand Awareness, Lead Generation).
        audience:      Target audience.
        strategy:      Strategy summary.

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
    """
    Recommend budget allocation across marketing channels.

    Args:
        total_budget: Total campaign budget (e.g., '$50,000').
        channels:     Comma-separated list of channels to allocate budget to.

    Returns:
        Budget allocation plan as a formatted string.
    """
    return _gemini(
        prompt=(
            f"Total budget: {total_budget}\nChannels: {channels}\n\n"
            "Recommend percentage budget split across these channels with rationale. "
            "Consider: audience reach, cost efficiency, conversion potential, brand goals. "
            "Format as a markdown table: Channel | % Allocation | $ Amount | Rationale"
        ),
        system="You are a Media Buying Director. Be specific and data-driven.",
    )
