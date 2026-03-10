"""
crew_marketing.py – Digital Marketing CrewAI 4-agent sequential crew.
Adapted from Digital_Marketing_Agent/crew.py for the combined system.

Agents:
  1. Researcher     – market research, competitors, audience, trends
  2. Planner        – strategy, content calendar, KPIs, budget
  3. Content Maker  – video (Veo 3), ad copy, social posts, email, reports
  4. Manager (CMO)  – synthesises executive campaign brief

The crew receives banking_context (churn analysis) as additional guidance,
allowing the marketing team to create targeted retention campaigns.
"""

import os
import sys

os.environ["OTEL_SDK_DISABLED"]         = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import litellm
    litellm.request_timeout = 45
    litellm.drop_params     = True
    litellm.set_verbose     = False
except Exception:
    pass

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

from config import GOOGLE_API_KEY

# ── Marketing tool imports ────────────────────────────────────────────────────
from tools.mkt_researcher import (
    web_search_market,
    analyze_competitors,
    research_target_audience,
    analyze_industry_trends,
    save_research_report,
)
from tools.mkt_planner import (
    create_marketing_strategy,
    create_content_calendar,
    define_campaign_kpis,
    create_campaign_brief,
    plan_budget_allocation,
)
from tools.mkt_content import (
    generate_promotional_poster,   # replaces generate_video_content (Veo 3.1)
    write_ad_copy,
    generate_social_posts,
    create_email_template,
    save_content_session,
    generate_content_report,
)
from tools.mkt_report import (
    generate_text_report,
    generate_pdf_report,
    generate_ppt_report,
)


# ── LLM configuration ─────────────────────────────────────────────────────────

gemini_llm_crew = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.1,
    max_tokens=1500,
    timeout=45,
    max_retries=2,
)

gemini_llm_writer = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.2,
    max_tokens=4096,
    timeout=90,
    max_retries=2,
)


# ── Agent factory ─────────────────────────────────────────────────────────────

def _build_marketing_agents() -> dict:
    researcher = Agent(
        role="Marketing Research Analyst",
        goal=(
            "Conduct comprehensive market research including competitive analysis, "
            "audience insights, and trend identification to inform campaign strategy. "
            "Focus specifically on banking customers most at risk of churn."
        ),
        backstory=(
            "You are a seasoned marketing research analyst with 10 years of experience "
            "uncovering market opportunities for financial services brands. "
            "You translate data into actionable intelligence."
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
        max_iter=8,
    )

    planner = Agent(
        role="Senior Marketing Strategist",
        goal=(
            "Develop a comprehensive marketing strategy, content calendar, KPI framework, "
            "campaign brief, and budget allocation plan based on banking churn research."
        ),
        backstory=(
            "You are a strategic marketing expert who has planned campaigns for Fortune 500 "
            "companies across financial services. You excel at turning research data into "
            "retention-focused campaigns with measurable outcomes."
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
        max_iter=8,
    )

    content_maker = Agent(
        role="Creative Content Director",
        goal=(
            "Produce all campaign content assets: AI-generated promotional posters (Gemini image), "
            "ad copy, social media posts, email templates, and reports "
            "targeting at-risk banking customers."
        ),
        backstory=(
            "You are an award-winning creative director who leads content production for "
            "major financial brand campaigns. You leverage Gemini AI image generation for "
            "eye-catching promotional posters combined with expert copywriting."
        ),
        tools=[
            generate_promotional_poster,   # Gemini image gen — replaces Veo 3.1 video
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
        max_iter=12,
    )

    manager = Agent(
        role="Chief Marketing Officer",
        goal=(
            "Synthesise all research, strategy, and content outputs into a comprehensive "
            "executive campaign brief ready for client presentation."
        ),
        backstory=(
            "You are a visionary CMO who launches successful multi-million dollar campaigns "
            "for financial services brands. You excel at distilling complex retention programmes "
            "into clear, compelling executive narratives."
        ),
        tools=[],  # synthesis only
        llm=gemini_llm_writer,
        verbose=True,
        max_iter=4,
    )

    return {
        "researcher":    researcher,
        "planner":       planner,
        "content_maker": content_maker,
        "manager":       manager,
    }


# ── Task factory ──────────────────────────────────────────────────────────────

def _build_marketing_tasks(
    agents: dict,
    campaign_request: str,
    brand_name: str,
    industry: str,
    target_audience: str,
    campaign_goals: str,
    budget: str,
    competitors: str,
    campaign_type: str,
    banking_context: str,
    langgraph_plan: dict = None,
) -> list:
    # Banking churn context injected into every task
    churn_ctx = f"\n\n[Banking Churn Analysis Results]:\n{banking_context[:800]}" if banking_context else ""

    # LangGraph marketing pre-planning context (structured guidance per agent)
    plan_ctx     = ""
    research_ctx = ""
    strategy_ctx = ""
    content_ctx  = ""
    if langgraph_plan:
        if langgraph_plan.get("analysis_plan"):
            plan_ctx     = f"\n\n[CMO Strategic Plan]:\n{langgraph_plan['analysis_plan'][:600]}"
        if langgraph_plan.get("research_guidance"):
            research_ctx = f"\n\n[Research Guidance]:\n{langgraph_plan['research_guidance'][:400]}"
        if langgraph_plan.get("strategy_guidance"):
            strategy_ctx = f"\n\n[Strategy Guidance]:\n{langgraph_plan['strategy_guidance'][:400]}"
        if langgraph_plan.get("content_guidance"):
            content_ctx  = f"\n\n[Content Guidance]:\n{langgraph_plan['content_guidance'][:400]}"

    task_researcher = Task(
        description=(
            f"Conduct thorough market research for this campaign:\n{campaign_request}"
            f"{churn_ctx}{plan_ctx}{research_ctx}\n\n"
            f"Brand: {brand_name} | Industry: {industry} | Audience: {target_audience}\n"
            f"Competitors: {competitors}\n\n"
            "1. Search for industry trends and market size data relevant to banking churn retention.\n"
            "2. Analyse all competitors mentioned in the brief.\n"
            "3. Research the target audience in depth — focus on demographics matching churn risk profiles.\n"
            "4. Identify the top 5 emerging trends for retention marketing in banking.\n"
            "5. Save a comprehensive research report to outputs/content/."
        ),
        expected_output=(
            "A comprehensive research report covering competitive landscape, "
            "audience personas aligned with churn profiles, market trends, and strategic opportunities."
        ),
        agent=agents["researcher"],
    )

    task_planner = Task(
        description=(
            f"Develop the full marketing strategy:\n{campaign_request}"
            f"{churn_ctx}{plan_ctx}{strategy_ctx}\n\n"
            f"Brand: {brand_name} | Goals: {campaign_goals} | Budget: {budget}\n"
            f"Campaign type: {campaign_type}\n\n"
            "Using the researcher's findings:\n"
            "1. Create a comprehensive marketing strategy document targeted at churn-risk segments.\n"
            "2. Build a 30-day content calendar.\n"
            "3. Define the KPI framework with specific retention targets.\n"
            "4. Write a one-page campaign brief.\n"
            "5. Recommend budget allocation across all channels."
        ),
        expected_output=(
            "A complete marketing strategy package: strategy document, content calendar, "
            "KPI framework, campaign brief, budget allocation. All files in outputs/content/."
        ),
        agent=agents["planner"],
        context=[task_researcher],
    )

    task_content = Task(
        description=(
            f"Create all content assets for this campaign:\n{campaign_request}"
            f"{churn_ctx}{plan_ctx}{content_ctx}\n\n"
            f"Brand: {brand_name} | Audience: {target_audience} | Campaign: {campaign_type}\n\n"
            "Using both research and strategy outputs:\n"
            "1. Generate at least 1 AI promotional poster using Gemini image generation "
            "(brand retention campaign poster — banking theme, premium feel).\n"
            "2. Write ad copy for Google, Meta, and LinkedIn.\n"
            "3. Generate 3 social posts each for Instagram, Twitter/X, and LinkedIn.\n"
            "4. Create an email template for the campaign.\n"
            "5. Generate a content summary report.\n"
            "6. Save a full markdown report AND a PDF report AND a PPTX presentation."
        ),
        expected_output=(
            "All campaign content: video file(s), ad copy, social posts, email template, "
            "PDF report, PPTX deck, markdown report. All files in appropriate outputs/ directories."
        ),
        agent=agents["content_maker"],
        context=[task_researcher, task_planner],
    )

    task_manager = Task(
        description=(
            f"Synthesise all team outputs into an executive campaign brief:\n{campaign_request}"
            f"{churn_ctx}{plan_ctx}\n\n"
            "Review the researcher's, planner's, and content maker's outputs then write "
            "a comprehensive executive brief (800–1000 words) including:\n"
            "1. Executive Summary (3-4 sentences)\n"
            "2. Banking Churn Context & Why This Campaign Matters\n"
            "3. Campaign Objectives & KPIs\n"
            "4. Target Audience Profile (mapped to churn segments)\n"
            "5. Strategy & Positioning\n"
            "6. Content & Channel Mix\n"
            "7. Budget Overview\n"
            "8. Timeline & Milestones\n"
            "9. Risk Factors & Mitigation\n"
            "10. Expected ROI & Churn Reduction Targets\n"
            "11. Next Steps\n"
            "Format as professional markdown."
        ),
        expected_output=(
            "A polished executive campaign brief (800–1000 words, markdown) that a C-suite executive "
            "could present to stakeholders. Includes concrete metrics and churn reduction targets."
        ),
        agent=agents["manager"],
        context=[task_researcher, task_planner, task_content],
    )

    return [task_researcher, task_planner, task_content, task_manager]


# ── Public API ────────────────────────────────────────────────────────────────

def run_marketing_crew(
    campaign_request: str,
    brand_name: str       = "Bank Brand",
    industry: str         = "Banking & Financial Services",
    target_audience: str  = "Banking customers at risk of churn",
    campaign_goals: str   = "Reduce customer churn, improve retention",
    budget: str           = "Not specified",
    competitors: str      = "Not specified",
    campaign_type: str    = "Customer Retention",
    banking_context: str  = "",
    langgraph_plan: dict  = None,
) -> str:
    """
    Run the Digital Marketing CrewAI crew (Phase 3b).
    Called by app.py after the marketing LangGraph (Phase 3a) completes.

    Args:
        campaign_request: The marketing campaign task description.
        banking_context:  Banking churn analysis results passed from Phase 2.
        langgraph_plan:   Marketing LangGraph plan dict (analysis_plan, research_guidance,
                          strategy_guidance, content_guidance) from Phase 3a.

    Returns:
        str – Marketing Manager's executive campaign brief.
    """
    agents = _build_marketing_agents()
    tasks  = _build_marketing_tasks(
        agents, campaign_request, brand_name, industry,
        target_audience, campaign_goals, budget, competitors, campaign_type,
        banking_context, langgraph_plan,
    )

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    print("\n[CrewAI-Marketing] ══ Starting 4-agent sequential Marketing crew ══\n", flush=True)
    result = crew.kickoff()
    print("\n[CrewAI-Marketing] ══ Marketing crew complete ══\n", flush=True)

    return str(result)
