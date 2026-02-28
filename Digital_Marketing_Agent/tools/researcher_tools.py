"""
tools/researcher_tools.py – Market research tools for the Researcher agent.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from crewai.tools import tool

# ── lazy imports ──────────────────────────────────────────────────────────────
def _get_tavily():
    try:
        from config import TAVILY_API_KEY
        from tavily import TavilyClient
        return TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
    except ImportError:
        return None


def _search(query: str, max_results: int = 5) -> str:
    """Helper: run Tavily search and return formatted string."""
    client = _get_tavily()
    if client is None:
        return f"[Tavily unavailable] Query: {query}"
    try:
        results = client.search(query=query, max_results=max_results)
        items = results.get("results", [])
        lines = []
        for r in items:
            lines.append(f"**{r.get('title','')}**\n{r.get('url','')}\n{r.get('content','')[:300]}")
        return "\n\n---\n\n".join(lines) if lines else "[No results found]"
    except Exception as exc:
        return f"[Search error: {exc}]"


# ── tools ─────────────────────────────────────────────────────────────────────

@tool("web_search_market")
def web_search_market(query: str) -> str:
    """Search the web for market intelligence, news, and industry data.

    Args:
        query: The search query string.

    Returns:
        Formatted search results as a string.
    """
    return _search(query, max_results=6)


@tool("analyze_competitors")
def analyze_competitors(competitors: str, industry: str) -> str:
    """Research competitors and build a competitive matrix.

    Args:
        competitors: Comma-separated list of competitor brand names.
        industry: The industry or market vertical.

    Returns:
        Competitive analysis matrix as a formatted string.
    """
    names = [c.strip() for c in competitors.split(",") if c.strip()]
    if not names:
        return "No competitors provided."

    sections = []
    for name in names[:5]:  # cap at 5 to avoid token bloat
        result = _search(f"{name} {industry} marketing strategy positioning 2025", max_results=3)
        sections.append(f"### {name}\n{result}")

    matrix = "\n\n".join(sections)
    return f"# Competitive Analysis – {industry}\n\n{matrix}"


@tool("research_target_audience")
def research_target_audience(audience: str, industry: str) -> str:
    """Research target audience demographics, psychographics, and pain points.

    Args:
        audience: Description of the target audience.
        industry: The industry or market vertical.

    Returns:
        Audience research report as a formatted string.
    """
    demos  = _search(f"{audience} demographics statistics {industry} 2025", max_results=4)
    psycho = _search(f"{audience} pain points motivations buying behaviour {industry}", max_results=3)
    return (
        f"# Audience Research: {audience}\n\n"
        f"## Demographics & Market Size\n{demos}\n\n"
        f"## Psychographics & Behaviour\n{psycho}"
    )


@tool("analyze_industry_trends")
def analyze_industry_trends(industry: str) -> str:
    """Research latest industry trends, market size, and growth signals.

    Args:
        industry: The industry or market vertical to analyse.

    Returns:
        Industry trends report as a formatted string.
    """
    trends  = _search(f"{industry} industry trends 2025 2026", max_results=4)
    size    = _search(f"{industry} market size growth forecast 2025", max_results=3)
    digital = _search(f"{industry} digital marketing trends social media 2025", max_results=3)
    return (
        f"# Industry Trends: {industry}\n\n"
        f"## Market Trends\n{trends}\n\n"
        f"## Market Size & Growth\n{size}\n\n"
        f"## Digital & Social Trends\n{digital}"
    )


@tool("save_research_report")
def save_research_report(content: str, brand_name: str) -> str:
    """Save the research report to the outputs/content directory.

    Args:
        content: Markdown content of the research report.
        brand_name: Brand name (used in filename).

    Returns:
        Path to the saved file.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() else "_" for c in brand_name)
    out_dir = Path(__file__).parent.parent / "outputs" / "content"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / f"research_report_{safe}_{ts}.md"
    filepath.write_text(content, encoding="utf-8")
    print(f"[ResearchTools] Report saved → {filepath}", flush=True)
    return str(filepath)
