"""
crew.py – CrewAI 4-agent sequential Digital Marketing crew.
"""
from __future__ import annotations

import os
import sys

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

# ── LiteLLM global timeout (applied to ALL internal CrewAI LLM calls) ────────
# This is the reliable way to enforce timeouts regardless of CrewAI version,
# because CrewAI calls litellm.completion() internally and respects these globals.
try:
    import litellm
    litellm.request_timeout = 45          # 45s hard cap per HTTP attempt
    litellm.drop_params     = True        # ignore unsupported params silently
    litellm.set_verbose     = False       # suppress litellm internal logs
except Exception:
    pass

# ── LLM configuration (Gemini 2.5 Flash via LiteLLM) ─────────────────────────
_GOOGLE_KEY = os.getenv("GOOGLE_API_KEY", "")

# Fast LLM for tool-calling agents (Researcher, Planner, Content Maker)
# Shorter max_tokens keeps each reasoning step quick.
gemini_llm_crew = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=_GOOGLE_KEY,
    temperature=0.1,
    max_tokens=1500,
    timeout=45,
    max_retries=2,
)

# Writer LLM for the Manager – needs more tokens to produce a full executive brief
gemini_llm_writer = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=_GOOGLE_KEY,
    temperature=0.2,
    max_tokens=4096,   # full brief can be 800-1000 words ≈ 1200-1500 tokens
    timeout=90,        # allow more time for long synthesis
    max_retries=2,
)

# ── Tool imports ──────────────────────────────────────────────────────────────
from tools.researcher_tools import (
    web_search_market,
    analyze_competitors,
    research_target_audience,
    analyze_industry_trends,
    save_research_report,
)
from tools.planner_tools import (
    create_marketing_strategy,
    create_content_calendar,
    define_campaign_kpis,
    create_campaign_brief,
    plan_budget_allocation,
)
from tools.content_tools import (
    generate_video_content,
    write_ad_copy,
    generate_social_posts,
    create_email_template,
    save_content_session,
    generate_content_report,
)
from tools.report_tools import (
    generate_text_report,
    generate_pdf_report,
    generate_ppt_report,
)


# ── Agent definitions ─────────────────────────────────────────────────────────

def _build_agents() -> dict[str, Agent]:
    researcher = Agent(
        role="Marketing Research Analyst",
        goal=(
            "Conduct comprehensive market research including competitive analysis, "
            "audience insights, and trend identification to inform campaign strategy."
        ),
        backstory=(
            "You are a seasoned marketing research analyst with 10 years of experience "
            "uncovering market opportunities and consumer insights for global brands. "
            "You specialise in translating data into actionable intelligence."
        ),
        tools=[
            web_search_market,
            analyze_competitors,
            research_target_audience,
            analyze_industry_trends,
            save_research_report,
        ],
        llm=gemini_llm_crew,
        verbose=True,
        max_iter=8,   # 5 tools + 3 reasoning steps max
    )

    planner = Agent(
        role="Senior Marketing Strategist",
        goal=(
            "Develop a comprehensive marketing strategy, content calendar, KPI framework, "
            "campaign brief, and budget allocation plan based on research insights."
        ),
        backstory=(
            "You are a strategic marketing expert who has planned campaigns for Fortune 500 "
            "companies across multiple industries. You excel at turning research into "
            "executable strategies with measurable outcomes."
        ),
        tools=[
            create_marketing_strategy,
            create_content_calendar,
            define_campaign_kpis,
            create_campaign_brief,
            plan_budget_allocation,
        ],
        llm=gemini_llm_crew,
        verbose=True,
        max_iter=8,   # 5 tools + 3 reasoning steps max
    )

    content_maker = Agent(
        role="Creative Content Director",
        goal=(
            "Produce all campaign content assets: AI-generated videos (Veo 3), "
            "ad copy for all platforms, social media posts, and email templates. "
            "Generate comprehensive PDF, PPTX, and markdown reports."
        ),
        backstory=(
            "You are an award-winning creative director who leads content production for "
            "major brand campaigns. You leverage the latest AI tools including Google Veo 3 "
            "for video generation, combined with expert copywriting and design direction."
        ),
        tools=[
            generate_video_content,
            write_ad_copy,
            generate_social_posts,
            create_email_template,
            save_content_session,
            generate_content_report,
            generate_text_report,
            generate_pdf_report,
            generate_ppt_report,
        ],
        llm=gemini_llm_crew,
        verbose=True,
        max_iter=12,  # 9 tools + 3 reasoning steps max
    )

    manager = Agent(
        role="Chief Marketing Officer",
        goal=(
            "Synthesise all research, strategy, and content outputs into a comprehensive "
            "executive campaign brief that is ready for client presentation."
        ),
        backstory=(
            "You are a visionary CMO with a track record of launching successful multi-million "
            "dollar campaigns. You excel at distilling complex marketing programmes into "
            "clear, compelling executive narratives."
        ),
        tools=[],  # synthesis only
        llm=gemini_llm_writer,  # higher token budget for full executive brief
        verbose=True,
        max_iter=4,
    )

    return {
        "researcher":    researcher,
        "planner":       planner,
        "content_maker": content_maker,
        "manager":       manager,
    }


# ── Task definitions ──────────────────────────────────────────────────────────

def _build_tasks(
    agents: dict[str, Agent],
    campaign_request: str,
    langgraph_plan: dict | None = None,
) -> list[Task]:

    # Inject LangGraph guidance into task context
    plan_ctx     = ""
    research_ctx = ""
    strategy_ctx = ""
    content_ctx  = ""

    if langgraph_plan:
        plan_ctx     = f"\n\n[Strategic Plan from CMO]:\n{langgraph_plan.get('analysis_plan', '')[:600]}"
        research_ctx = f"\n\n[Research Guidance]:\n{langgraph_plan.get('research_guidance', '')[:400]}"
        strategy_ctx = f"\n\n[Strategy Guidance]:\n{langgraph_plan.get('strategy_guidance', '')[:400]}"
        content_ctx  = f"\n\n[Content Direction]:\n{langgraph_plan.get('content_guidance', '')[:400]}"

    task_researcher = Task(
        description=(
            f"Conduct thorough market research for this campaign:\n{campaign_request}"
            f"{plan_ctx}{research_ctx}\n\n"
            "1. Search for industry trends and market size data.\n"
            "2. Analyse all competitors mentioned in the brief.\n"
            "3. Research the target audience in depth (demographics, psychographics, pain points).\n"
            "4. Identify the top 5 emerging trends relevant to this campaign.\n"
            "5. Save a comprehensive research report to outputs/content/."
        ),
        expected_output=(
            "A comprehensive research report covering competitive landscape, "
            "audience personas, market trends, and strategic opportunities. "
            "Include specific data points, competitor weaknesses, and audience insights."
        ),
        agent=agents["researcher"],
    )

    task_planner = Task(
        description=(
            f"Develop the full marketing strategy for this campaign:\n{campaign_request}"
            f"{plan_ctx}{strategy_ctx}\n\n"
            "Using the researcher's findings:\n"
            "1. Create a comprehensive marketing strategy document.\n"
            "2. Build a 30-day content calendar.\n"
            "3. Define the KPI framework with specific targets.\n"
            "4. Write a one-page campaign brief.\n"
            "5. Recommend budget allocation across all channels."
        ),
        expected_output=(
            "A complete marketing strategy package including: strategy document, "
            "content calendar, KPI framework, campaign brief, and budget allocation. "
            "All files saved to outputs/content/."
        ),
        agent=agents["planner"],
        context=[task_researcher],
    )

    task_content = Task(
        description=(
            f"Create all content assets for this campaign:\n{campaign_request}"
            f"{plan_ctx}{content_ctx}\n\n"
            "Using both the research and strategy outputs:\n"
            "1. Generate at least 1 AI video using Veo 3 (brand story or product showcase).\n"
            "2. Write ad copy for Google, Meta, and LinkedIn.\n"
            "3. Generate 3 social posts each for Instagram, Twitter/X, and LinkedIn.\n"
            "4. Create an email template for the campaign.\n"
            "5. Generate a content summary report.\n"
            "6. Save a full markdown report AND a PDF report AND a PPTX presentation."
        ),
        expected_output=(
            "All campaign content assets: video file(s), ad copy, social posts, "
            "email template, plus PDF report, PPTX deck, and markdown report. "
            "All files saved to appropriate outputs/ directories."
        ),
        agent=agents["content_maker"],
        context=[task_researcher, task_planner],
    )

    task_manager = Task(
        description=(
            f"Synthesise all team outputs into an executive campaign brief:\n{campaign_request}"
            f"{plan_ctx}\n\n"
            "Review the researcher's, planner's, and content maker's outputs then write "
            "a comprehensive executive brief (800–1000 words) that includes:\n"
            "1. Executive Summary (3-4 sentences)\n"
            "2. Campaign Objectives & KPIs\n"
            "3. Target Audience Profile\n"
            "4. Strategy & Positioning\n"
            "5. Content & Channel Mix\n"
            "6. Budget Overview\n"
            "7. Timeline & Milestones\n"
            "8. Risk Factors & Mitigation\n"
            "9. Expected ROI & Success Metrics\n"
            "10. Next Steps\n"
            "Format as professional markdown."
        ),
        expected_output=(
            "A polished executive campaign brief (800–1000 words, markdown) that a "
            "C-suite executive could present to stakeholders. "
            "Includes concrete metrics, clear strategic direction, and actionable next steps."
        ),
        agent=agents["manager"],
        context=[task_researcher, task_planner, task_content],
    )

    return [task_researcher, task_planner, task_content, task_manager]


# ── Crew runner ───────────────────────────────────────────────────────────────

def run_crewai_phase(
    campaign_request: str,
    langgraph_plan: dict | None = None,
) -> str:
    """Run CrewAI Phase 2 (called by app.py after HITL approval).

    Returns the Manager's final executive brief as a string.
    """
    # Ensure Google key is loaded
    if not _GOOGLE_KEY:
        try:
            from config import GOOGLE_API_KEY
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        except Exception:
            pass

    agents = _build_agents()
    tasks  = _build_tasks(agents, campaign_request, langgraph_plan)

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    print("\n[CrewAI] ══ Starting 4-agent sequential Digital Marketing crew ══\n", flush=True)
    result = crew.kickoff()
    print("\n[CrewAI] ══ Crew execution complete ══\n", flush=True)

    return str(result)


def run_marketing_analysis_crew(
    campaign_request: str,
    use_langgraph: bool = True,
) -> str:
    """Legacy CLI entry – runs LangGraph planning then CrewAI execution."""
    langgraph_plan = None
    if use_langgraph:
        from graphs.marketing_graph import run_marketing_analysis
        # parse a minimal request into params
        langgraph_plan = run_marketing_analysis(
            task_description=campaign_request,
            brand_name="Campaign",
        )
    return run_crewai_phase(campaign_request, langgraph_plan)
