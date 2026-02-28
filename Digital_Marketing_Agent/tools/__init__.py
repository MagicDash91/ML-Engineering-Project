# tools package – expose all tools for CrewAI agents
from .researcher_tools import (
    web_search_market,
    analyze_competitors,
    research_target_audience,
    analyze_industry_trends,
    save_research_report,
)
from .planner_tools import (
    create_marketing_strategy,
    create_content_calendar,
    define_campaign_kpis,
    create_campaign_brief,
    plan_budget_allocation,
)
from .content_tools import (
    generate_video_content,
    write_ad_copy,
    generate_social_posts,
    create_email_template,
    save_content_session,
    generate_content_report,
)
from .report_tools import (
    generate_text_report,
    generate_pdf_report,
    generate_ppt_report,
)

__all__ = [
    "web_search_market", "analyze_competitors", "research_target_audience",
    "analyze_industry_trends", "save_research_report",
    "create_marketing_strategy", "create_content_calendar", "define_campaign_kpis",
    "create_campaign_brief", "plan_budget_allocation",
    "generate_video_content", "write_ad_copy", "generate_social_posts",
    "create_email_template", "save_content_session", "generate_content_report",
    "generate_text_report", "generate_pdf_report", "generate_ppt_report",
]
