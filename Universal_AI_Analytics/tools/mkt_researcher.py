"""
tools/mkt_researcher.py – Marketing research tools for the combined system.
Mirrors Digital_Marketing_Agent/tools/researcher_tools.py with paths adjusted.
"""
from __future__ import annotations

import json
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai.tools import tool

_BASE    = Path(__file__).parent.parent / "outputs"
_CONTENT = _BASE / "content"
_CONTENT.mkdir(parents=True, exist_ok=True)


def _get_tavily():
    try:
        from config import TAVILY_API_KEY
        from tavily import TavilyClient
        return TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
    except ImportError:
        return None


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
                    print(f"[MktResearcher] Ollama error (attempt {attempt+1}/3): {exc} — retrying in 5s", flush=True)
                    time.sleep(5)
        return f"Unable to generate response: {last_exc}"

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_call)
    try:
        result = fut.result(timeout=_timeout)
        ex.shutdown(wait=False)
        return result
    except concurrent.futures.TimeoutError:
        ex.shutdown(wait=False)
        return "Response timed out – proceeding with available context."


@tool("web_search_market")
def web_search_market(query: str) -> str:
    """
    Search the web for market research, trends, and industry information.

    Args:
        query: Search query string (e.g. 'Indonesian banking customer churn marketing 2025').

    Returns:
        Formatted search results string.
    """
    tavily = _get_tavily()
    if tavily is None:
        return _gemini(
            prompt=f"Provide market research information about: {query}",
            system="You are a market research analyst. Provide detailed market insights.",
        )

    try:
        results = tavily.search(query=query, max_results=5)
        formatted = []
        for r in results.get("results", []):
            formatted.append(f"**{r.get('title', '')}**\n{r.get('url', '')}\n{r.get('content', '')[:500]}\n")
        return "\n---\n".join(formatted) if formatted else "No results found."
    except Exception as exc:
        return f"Search failed: {exc}. Using Gemini fallback."


@tool("analyze_competitors")
def analyze_competitors(competitors: str, industry: str) -> str:
    """
    Analyse competitors and produce a competitive landscape matrix.

    Args:
        competitors: Comma-separated list of competitor names.
        industry:    Industry or market segment.

    Returns:
        Competitive analysis as a formatted markdown string.
    """
    competitor_list = [c.strip() for c in competitors.split(",") if c.strip()]
    results = []

    for comp in competitor_list[:5]:
        tavily = _get_tavily()
        info = ""
        if tavily:
            try:
                res = tavily.search(query=f"{comp} {industry} marketing strategy 2025", max_results=3)
                info = " ".join(r.get("content", "")[:300] for r in res.get("results", []))
            except Exception:
                pass

        analysis = _gemini(
            prompt=(
                f"Competitor: {comp}\nIndustry: {industry}\n"
                f"Available info: {info[:600] if info else 'No data available'}\n\n"
                "Analyse: 1) Market positioning 2) Target segments 3) Key messages "
                "4) Channel mix 5) Strengths & weaknesses 6) Differentiation opportunities."
            ),
            system="You are a competitive intelligence analyst. Be specific and actionable.",
        )
        results.append(f"### {comp}\n{analysis}")

    return "\n\n".join(results) if results else "No competitors to analyse."


@tool("research_target_audience")
def research_target_audience(audience: str, industry: str) -> str:
    """
    Research target audience demographics, psychographics, and pain points.

    Args:
        audience: Target audience description (e.g. 'banking customers aged 25-45 at risk of churn').
        industry: Industry context.

    Returns:
        Audience persona research as a formatted string.
    """
    return _gemini(
        prompt=(
            f"Target Audience: {audience}\nIndustry: {industry}\n\n"
            "Create detailed audience personas covering:\n"
            "1) Demographics (age, income, location, education)\n"
            "2) Psychographics (values, lifestyle, interests, attitudes)\n"
            "3) Pain points & frustrations\n"
            "4) Goals & motivations\n"
            "5) Preferred communication channels\n"
            "6) Purchase decision factors\n"
            "7) Content preferences\n"
            "Create 3 distinct personas."
        ),
        system="You are a market research specialist. Create detailed, realistic audience personas.",
    )


@tool("analyze_industry_trends")
def analyze_industry_trends(industry: str) -> str:
    """
    Analyse current industry trends, market size, and emerging opportunities.

    Args:
        industry: Industry to analyse (e.g. 'Indonesian retail banking', 'fintech').

    Returns:
        Industry trend analysis as a formatted string.
    """
    tavily = _get_tavily()
    web_data = ""
    if tavily:
        try:
            res = tavily.search(query=f"{industry} trends market size 2025", max_results=5)
            web_data = " ".join(r.get("content", "")[:400] for r in res.get("results", []))
        except Exception:
            pass

    return _gemini(
        prompt=(
            f"Industry: {industry}\nWeb research: {web_data[:1200] if web_data else 'No data'}\n\n"
            "Analyse:\n"
            "1) Current market size and growth rate\n"
            "2) Top 5 emerging trends\n"
            "3) Technology disruptions\n"
            "4) Consumer behaviour shifts\n"
            "5) Regulatory changes\n"
            "6) Untapped opportunities\n"
            "7) Key success factors for campaigns"
        ),
        system="You are an industry analyst. Provide specific, data-backed insights.",
    )


@tool("save_research_report")
def save_research_report(content: str, brand_name: str) -> str:
    """
    Save research findings to a markdown report file.

    Args:
        content:    Research content to save.
        brand_name: Brand name for the filename.

    Returns:
        Path to the saved report file.
    """
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe     = "".join(c if c.isalnum() else "_" for c in brand_name)
    filename = f"research_report_{safe}_{ts}.md"
    path     = _CONTENT / filename
    path.write_text(content, encoding="utf-8")
    print(f"[MktResearcher] Research report saved → {path}", flush=True)
    return f"Research report saved to outputs/content/{filename}"
