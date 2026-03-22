"""
crew_marketing.py – Universal Digital Marketing CrewAI 4-agent sequential crew.

Agents:
  1. Researcher     – market research, competitors, audience, trends
  2. Planner        – strategy, content calendar, KPIs, budget
  3. Content Maker  – posters (Gemini image), ad copy, social posts, email, reports
  4. Manager (CMO)  – synthesises executive campaign brief

The crew receives analytics_context (data analysis findings) as additional guidance,
allowing the marketing team to create targeted, data-driven campaigns.
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

from config import GOOGLE_API_KEY  # still needed for poster generation

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
    generate_promotional_poster,
    write_ad_copy,
    generate_social_posts,
    save_content_session,
    generate_content_report,
)
from tools.mkt_report import (
    generate_text_report,
    generate_pdf_report,
    generate_ppt_report,
)


# ── LLM configuration ─────────────────────────────────────────────────────────

ollama_llm_crew = LLM(
    model="ollama/qwen3.5:cloud",
    temperature=0.1,
    max_tokens=1500,
    timeout=300,
    max_retries=2,
)

ollama_llm_writer = LLM(
    model="ollama/qwen3.5:cloud",
    temperature=0.2,
    max_tokens=4096,
    timeout=300,
    max_retries=2,
)


# ── Agent factory ─────────────────────────────────────────────────────────────

def _build_marketing_agents() -> dict:
    researcher = Agent(
        role="Marketing Research Analyst",
        goal=(
            "Conduct thorough market research that is directly informed by the data analytics "
            "findings handed off from the analytics team. "
            "Identify who the at-risk or high-opportunity segments are, what competitors are doing, "
            "and what trends the brand can leverage. "
            "Every insight must be grounded in the analytics context — not generic advice."
        ),
        backstory=(
            "You are a seasoned Marketing Research Analyst with 10 years of experience across "
            "retail, finance, telecom, healthcare, and e-commerce. "
            "You specialise in translating data science outputs into actionable market intelligence. "
            "When the analytics team hands you customer segments and churn/risk drivers, "
            "you research exactly what marketing approaches work for each segment. "
            "You never produce generic research — every finding maps back to a specific "
            "audience group identified in the data."
        ),
        tools=[
            web_search_market,
            analyze_competitors,
            research_target_audience,
            analyze_industry_trends,
            save_research_report,
        ],
        llm=ollama_llm_crew,
        verbose=True,
        max_iter=8,
    )

    planner = Agent(
        role="Senior Marketing Strategist",
        goal=(
            "Develop a comprehensive, data-driven marketing strategy that directly addresses "
            "the at-risk or high-opportunity audience segments identified in the analytics. "
            "The strategy must include: specific campaign names and mechanics tied to "
            "real data findings, a 30-day content calendar, a KPI framework with measurable "
            "targets, and a budget allocation plan. "
            "Every recommendation must trace back to a specific segment or finding."
        ),
        backstory=(
            "You are a Senior Marketing Strategist with 12 years of experience planning "
            "data-driven campaigns for major brands. "
            "You have a gift for designing campaigns that feel personalised because they "
            "are built on real audience data — not demographic assumptions. "
            "You believe a great campaign brief is one where every proposed action is "
            "justified by a number from the analytics. "
            "You design campaigns with creative names and concrete mechanics: "
            "cashback offers, loyalty tiers, upgrade incentives, referral programmes, "
            "educational content sequences, or re-engagement flows — whatever the data demands."
        ),
        tools=[
            create_marketing_strategy,
            create_content_calendar,
            define_campaign_kpis,
            create_campaign_brief,
            plan_budget_allocation,
        ],
        llm=ollama_llm_crew,
        verbose=True,
        max_iter=8,
    )

    content_maker = Agent(
        role="Creative Content Director",
        goal=(
            "Produce all campaign content assets needed to execute the strategy: "
            "AI-generated promotional posters (Gemini image generation), "
            "multi-platform ad copy (Google, Meta, LinkedIn), "
            "social media posts (Instagram, Twitter/X, LinkedIn), "
            "a content summary report, and final PDF and PowerPoint deliverables. "
            "All content must reflect the specific audience segments and messaging angles "
            "from the analytics and strategy outputs."
        ),
        backstory=(
            "You are an award-winning Creative Content Director who has led content production "
            "for major brand campaigns across industries. "
            "You believe creativity and data are not opposites — the best creative work "
            "is grounded in a deep understanding of who you are speaking to. "
            "You use Gemini AI image generation to produce eye-catching promotional posters "
            "that align with the brand and campaign theme. "
            "Your ad copy speaks directly to each segment's motivations and pain points. "
            "Your email templates feel personal, not broadcast. "
            "You always save your work to the session so the UI can display it."
        ),
        tools=[
            generate_promotional_poster,
            write_ad_copy,
            generate_social_posts,
            save_content_session,
            generate_content_report,
            generate_text_report,
            generate_pdf_report,
            generate_ppt_report,
        ],
        llm=ollama_llm_crew,
        verbose=True,
        max_iter=12,
    )

    manager = Agent(
        role="Chief Marketing Officer (CMO)",
        goal=(
            "Review all research, strategy, and content outputs and synthesise them into "
            "a polished, board-ready executive campaign brief. "
            "The brief must show a clear logical chain: analytics findings → "
            "audience segments → campaign strategy → content assets → measurable ROI. "
            "It should be something a C-suite executive can present to stakeholders."
        ),
        backstory=(
            "You are a visionary CMO with a track record of launching successful "
            "multi-million dollar campaigns across industries. "
            "You have seen too many marketing briefs that are full of buzzwords and "
            "light on substance. You write briefs that are different: "
            "every claim is backed by data, every recommendation has an owner and a deadline, "
            "every KPI is specific and measurable. "
            "You are particularly good at connecting the analytics team's findings to "
            "the marketing team's executions — making the handoff from data to creativity seamless."
        ),
        tools=[],  # synthesis only
        llm=ollama_llm_writer,
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
    analytics_context: str,
    langgraph_plan: dict = None,
) -> list:
    # Analytics context injected into every task
    analytics_ctx = (
        f"\n\n[Data Analytics Findings — use these to ground every decision]:\n"
        f"{analytics_context[:800]}"
    ) if analytics_context else ""

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

    # ── Task 1: Research ───────────────────────────────────────────────────────
    task_researcher = Task(
        description=(
            f"You are the first marketing agent. Your job is to research the market, "
            f"competitors, and target audience — grounded in the analytics findings.\n\n"
            f"Campaign Brief: {campaign_request}"
            f"{analytics_ctx}{plan_ctx}{research_ctx}\n\n"
            f"Brand: {brand_name} | Industry: {industry}\n"
            f"Target Audience: {target_audience} | Competitors: {competitors}\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 1 — INDUSTRY TRENDS: Search for current market trends\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: web_search_market\n"
            f"  Search for: '{industry} marketing trends 2025' and\n"
            f"              '{industry} customer retention strategies 2025'\n"
            "   Goal: Understand what is working right now in this market.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 2 — COMPETITOR ANALYSIS: Map the competitive landscape\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: analyze_competitors\n"
            f"  Analyse: {competitors if competitors != 'Not specified' else 'major competitors in the ' + industry + ' space'}\n"
            "   Focus on:\n"
            "   - What retention/loyalty programmes do they offer?\n"
            "   - What messaging and channels are they using?\n"
            "   - Where are the gaps our brand can exploit?\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 3 — AUDIENCE RESEARCH: Profile each segment from analytics\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: research_target_audience\n"
            "   Goal: Research each audience segment identified in the analytics findings.\n"
            "         For each segment:\n"
            "         - What are their key pain points and motivations?\n"
            "         - What channels do they use most?\n"
            "         - What offers or messages are most likely to resonate?\n"
            "         - What is the best time and context to reach them?\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 4 — TRENDS: Identify emerging opportunities\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: analyze_industry_trends\n"
            "   Goal: Find the top 5 emerging trends the campaign should leverage.\n"
            "         Think: personalisation, AI, loyalty tech, omnichannel, sustainability.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 5 — SAVE: Store research report\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: save_research_report\n"
            "   Include all findings from Steps 1–4.\n"
        ),
        expected_output=(
            "A comprehensive research report (saved to outputs/content/) covering: "
            "top industry trends with sources, competitive landscape analysis with gaps identified, "
            "audience personas mapped to the analytics segments, "
            "top 5 emerging trends to leverage, and strategic opportunities for the campaign."
        ),
        agent=agents["researcher"],
    )

    # ── Task 2: Strategy & Planning ────────────────────────────────────────────
    task_planner = Task(
        description=(
            f"You are the second marketing agent. Your job is to turn the research "
            f"and analytics findings into a concrete, actionable campaign strategy.\n\n"
            f"Campaign Brief: {campaign_request}"
            f"{analytics_ctx}{plan_ctx}{strategy_ctx}\n\n"
            f"Brand: {brand_name} | Goals: {campaign_goals}\n"
            f"Budget: {budget} | Campaign Type: {campaign_type}\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 1 — STRATEGY: Build the full marketing strategy\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: create_marketing_strategy\n"
            "   The strategy must include:\n"
            "   a) Brand positioning for this campaign: what unique value does the brand\n"
            "      offer each at-risk or high-opportunity segment?\n"
            "   b) Messaging framework: headline, subheadline, proof points, and CTA\n"
            "      for each segment — anchored to the analytics findings\n"
            "   c) Campaign names (4–6 specific campaigns with creative names):\n"
            "      Example: 'VIP Loyalty Upgrade', 'Win-Back Sprint', 'Digital First Bonus',\n"
            "      'New Member Welcome Series', 'Premium Tier Invitation'\n"
            "      Each campaign must directly address a risk factor or opportunity\n"
            "      found in the analytics data\n"
            "   d) Channel mix: which channels for which segments and why\n"
            "      (e.g. push notifications for mobile-active users, email for older segments,\n"
            "      Instagram for younger demographics, LinkedIn for B2B audiences)\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 2 — CONTENT CALENDAR: Plan the 30-day execution\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: create_content_calendar\n"
            "   Build a week-by-week 30-day calendar:\n"
            "   - Week 1: Awareness and teaser content\n"
            "   - Week 2: Core campaign launch (main offers)\n"
            "   - Week 3: Engagement deepening (social proof, testimonials)\n"
            "   - Week 4: Urgency and conversion push\n"
            "   Assign: platform, content type, target segment per post.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 3 — KPIs: Define measurable success metrics\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: define_campaign_kpis\n"
            "   For each campaign, define:\n"
            "   - Primary KPI (e.g. retention rate, conversion rate, CLV increase)\n"
            "   - Baseline (current value from analytics)\n"
            "   - Target (goal after campaign)\n"
            "   - Measurement method and timeline\n"
            "   Example: 'Reduce at-risk segment churn from 42% to 28% in 90 days'\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 4 — BRIEF: Write the one-page campaign brief\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: create_campaign_brief\n"
            "   Concise one-pager covering: objective, audience, core message,\n"
            "   channels, budget, KPIs, and timeline.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 5 — BUDGET: Allocate across channels\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: plan_budget_allocation\n"
            f"  Total budget: {budget}\n"
            "   Recommend % split across channels with rationale.\n"
            "   Justify every allocation with data from research or analytics.\n"
        ),
        expected_output=(
            "A complete marketing strategy package (saved to outputs/content/) including: "
            "brand positioning and messaging framework per segment, "
            "4–6 named campaign concepts with mechanics tied to data findings, "
            "30-day content calendar with weekly themes, "
            "KPI framework with baselines and targets from analytics, "
            "one-page campaign brief, and channel budget allocation with rationale."
        ),
        agent=agents["planner"],
        context=[task_researcher],
    )

    # ── Task 3: Content Production ─────────────────────────────────────────────
    task_content = Task(
        description=(
            f"You are the third marketing agent. Your job is to produce ALL campaign "
            f"content assets — posters, copy, social posts, emails, and reports.\n\n"
            f"Campaign Brief: {campaign_request}"
            f"{analytics_ctx}{plan_ctx}{content_ctx}\n\n"
            f"Brand: {brand_name} | Audience: {target_audience} | Campaign: {campaign_type}\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 1 — PROMOTIONAL POSTER: AI-generated visual\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: generate_promotional_poster\n"
            "   Generate at least 1 promotional poster using Gemini image generation.\n"
            "   The poster prompt should:\n"
            f"  - Reflect the brand '{brand_name}' and industry '{industry}'\n"
            "   - Match the campaign theme (retention, loyalty, engagement, or growth)\n"
            "   - Have a professional, premium look with strong visual hierarchy\n"
            "   - Include a clear headline and call-to-action visible in the image\n"
            "   Example prompt: 'Professional marketing poster for a [industry] brand campaign.\n"
            "   Premium dark design with gold accents. Headline: [campaign name]. "
            "   Tagline: [key message]. Clean modern typography. High-end brand feel.'\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 2 — AD COPY: Multi-platform advertising copy\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: write_ad_copy\n"
            "   Write distinct ad copy for each platform:\n"
            "   a) Google Search Ads:\n"
            "      - 3 headlines (max 30 chars each)\n"
            "      - 2 descriptions (max 90 chars each)\n"
            "      - Focus: intent-driven, problem-solution framing\n"
            "   b) Meta (Facebook/Instagram) Ads:\n"
            "      - Primary text (125 chars for feed)\n"
            "      - Headline (40 chars)\n"
            "      - Description (30 chars)\n"
            "      - Focus: visual storytelling, social proof\n"
            "   c) LinkedIn Ads:\n"
            "      - Introductory text (150 chars)\n"
            "      - Headline (70 chars)\n"
            "      - Focus: professional value proposition, ROI\n"
            "   For each platform, tailor tone to the primary at-risk segment.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 3 — SOCIAL POSTS: Platform-native content\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: generate_social_posts\n"
            "   Generate 3 posts per platform (9 posts total):\n"
            "   a) Instagram (3 posts):\n"
            "      - Visual-first captions with emojis\n"
            "      - Storytelling format: hook → value → CTA\n"
            "      - Include relevant hashtags (8–12 per post)\n"
            "   b) Twitter/X (3 posts):\n"
            "      - Punchy, under 280 characters\n"
            "      - Include 1–2 hashtags\n"
            "      - Mix: tip, question, announcement\n"
            "   c) LinkedIn (3 posts):\n"
            "      - Professional, insight-driven\n"
            "      - Include data points from analytics\n"
            "      - End with a thought-provoking question\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 4 — CONTENT REPORT: Save session content\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: generate_content_report\n"
            "   Tool: save_content_session\n"
            "   Save all content assets to the session so the UI can display them.\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "STEP 5 — REPORTS: PDF and PowerPoint deliverables\n"
            "═══════════════════════════════════════════════════════════\n"
            "   Tool: generate_text_report → full markdown campaign report\n"
            "   Tool: generate_pdf_report  → PDF with all assets\n"
            "   Tool: generate_ppt_report  → PowerPoint deck\n"
        ),
        expected_output=(
            "All campaign content assets saved and ready for display: "
            "at least 1 AI-generated promotional poster (PNG in outputs/posters/), "
            "platform-specific ad copy for Google, Meta, and LinkedIn, "
            "9 social media posts (3 per platform), "
            "content summary report, PDF report, and PowerPoint deck."
        ),
        agent=agents["content_maker"],
        context=[task_researcher, task_planner],
    )

    # ── Task 4: Executive Campaign Brief ──────────────────────────────────────
    task_manager = Task(
        description=(
            f"You are the final marketing agent. Synthesise all team outputs into "
            f"a polished executive campaign brief ready for client or board presentation.\n\n"
            f"Campaign Brief: {campaign_request}"
            f"{analytics_ctx}{plan_ctx}\n\n"
            f"Brand: {brand_name} | Industry: {industry}\n\n"

            "Review the researcher's market intelligence, the planner's strategy documents, "
            "and the content maker's creative assets, then write the following:\n\n"

            "═══════════════════════════════════════════════════════════\n"
            "EXECUTIVE CAMPAIGN BRIEF (800–1000 words, professional markdown)\n"
            "═══════════════════════════════════════════════════════════\n\n"

            "## 1. Executive Summary (3-4 sentences)\n"
            "   What problem does this campaign solve? What is the single most important\n"
            "   insight from the analytics that drives this campaign? What is the expected outcome?\n\n"

            "## 2. Analytics Context — Why This Campaign\n"
            "   Summarise the key data findings that justify this campaign:\n"
            "   - What did the analytics reveal about at-risk or high-opportunity segments?\n"
            "   - What are the top predictors of churn, conversion, or growth?\n"
            "   - What segments were identified and what is their business impact?\n\n"

            "## 3. Campaign Objectives & KPIs\n"
            "   List 3–5 objectives, each with:\n"
            "   - Specific KPI (e.g. retention rate, CTR, conversion rate)\n"
            "   - Baseline value (from analytics)\n"
            "   - Target value\n"
            "   - Measurement timeline (30/60/90 days)\n\n"

            "## 4. Target Audience Profile\n"
            "   Describe each segment (from analytics) with:\n"
            "   - Segment name and size\n"
            "   - Key characteristics and behaviours\n"
            "   - Primary pain point or opportunity\n"
            "   - Recommended campaign approach\n\n"

            "## 5. Campaign Strategy & Positioning\n"
            "   - Core message and value proposition\n"
            "   - Campaign names and mechanics (from planner)\n"
            "   - Why this strategy works for this specific audience\n\n"

            "## 6. Content & Channel Mix\n"
            "   - Summary of content assets produced (posters, ads, social posts, emails)\n"
            "   - Channel allocation and rationale per segment\n"
            "   - Key creative direction\n\n"

            "## 7. Budget Overview\n"
            "   - Total budget and allocation across channels\n"
            "   - Expected cost per acquisition or retention\n"
            "   - ROI projection\n\n"

            "## 8. Timeline & Milestones\n"
            "   - Week 1: Launch activities\n"
            "   - Week 2–3: Core execution\n"
            "   - Week 4: Optimisation and review\n"
            "   - 30/60/90-day review checkpoints\n\n"

            "## 9. Risk Factors & Mitigation\n"
            "   Identify 3 risks and how to mitigate each.\n\n"

            "## 10. Expected ROI & Target Outcomes\n"
            "    Project specific outcomes with numbers:\n"
            "    - Churn reduction / retention improvement\n"
            "    - Revenue impact\n"
            "    - Engagement uplift\n\n"

            "## 11. Next Steps\n"
            "    Numbered action plan:\n"
            "    [N]. Action — Owner — Deadline — Success Metric\n"
        ),
        expected_output=(
            "A polished 800–1000 word executive campaign brief in professional markdown. "
            "Includes analytics context, campaign objectives with KPIs from baselines, "
            "audience profiles mapped to segments, campaign strategy with named initiatives, "
            "content/channel mix summary, budget overview with ROI projection, "
            "30/60/90-day timeline, risk mitigation, and numbered next steps."
        ),
        agent=agents["manager"],
        context=[task_researcher, task_planner, task_content],
    )

    return [task_researcher, task_planner, task_content, task_manager]


# ── Public API ────────────────────────────────────────────────────────────────

def run_marketing_crew(
    campaign_request: str,
    brand_name: str       = "Brand",
    industry: str         = "General",
    target_audience: str  = "General audience",
    campaign_goals: str   = "Grow retention and engagement",
    budget: str           = "Not specified",
    competitors: str      = "Not specified",
    campaign_type: str    = "Retention & Growth",
    analytics_context: str = "",
    # legacy alias accepted from app.py
    banking_context: str  = "",
    langgraph_plan: dict  = None,
) -> str:
    """
    Run the Universal Digital Marketing CrewAI crew (Phase 3b).
    Called by app.py after the marketing LangGraph (Phase 3a) completes.

    Returns:
        str – CMO executive campaign brief.
    """
    # Support legacy banking_context kwarg from app.py
    effective_context = analytics_context or banking_context

    agents = _build_marketing_agents()
    tasks  = _build_marketing_tasks(
        agents, campaign_request, brand_name, industry,
        target_audience, campaign_goals, budget, competitors, campaign_type,
        effective_context, langgraph_plan,
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
