import os
import sys

UA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Universal_AI_Analytics')
if UA_PATH not in sys.path:
    sys.path.insert(0, UA_PATH)

# Registry: tool_name → metadata
TOOL_REGISTRY = {
    # ── Data Engineering ──────────────────────────────────────────────────────
    "list_database_tables": {
        "module": "tools.bank_engineer", "function": "list_database_tables",
        "description": "List all tables in the connected database",
        "category": "data_engineering",
        "when_to_use": "First step of any data analysis to discover available tables",
    },
    "profile_database_table": {
        "module": "tools.bank_engineer", "function": "profile_database_table",
        "description": "Profile a database table: row count, columns, dtypes, nulls, sample values",
        "category": "data_engineering",
        "when_to_use": "After discovering tables, to understand data structure and quality",
    },
    "query_database": {
        "module": "tools.bank_engineer", "function": "query_database",
        "description": "Execute a SQL SELECT query against the database",
        "category": "data_engineering",
        "when_to_use": "To fetch specific data, aggregations, or counts from any table",
    },
    "clean_table_columns": {
        "module": "tools.bank_engineer", "function": "clean_table_columns",
        "description": "Remove ID/UUID and low-value columns from a database table",
        "category": "data_engineering",
        "when_to_use": "Before ML modelling or visualization to remove noise columns",
    },
    "normalize_column_dtypes": {
        "module": "tools.bank_engineer", "function": "normalize_column_dtypes",
        "description": "Cast object columns that contain numeric values to float/int",
        "category": "data_engineering",
        "when_to_use": "After cleaning, to fix data types for downstream ML and charts",
    },
    "run_etl_pipeline": {
        "module": "tools.bank_engineer", "function": "run_etl_pipeline",
        "description": "Run a full ETL pipeline: extract from source, transform, load to target table",
        "category": "data_engineering",
        "when_to_use": "When data needs to be moved or transformed across tables",
    },
    "web_search_collect": {
        "module": "tools.bank_engineer", "function": "web_search_collect",
        "description": "Search the web for information and save results to the database",
        "category": "data_engineering",
        "when_to_use": "When external context or web data is needed to enrich analysis",
    },
    "fetch_financial_data": {
        "module": "tools.bank_engineer", "function": "fetch_financial_data",
        "description": "Fetch historical financial market data for a stock symbol",
        "category": "data_engineering",
        "when_to_use": "When financial market or stock data is needed",
    },
    # ── Data Science ──────────────────────────────────────────────────────────
    "train_churn_model": {
        "module": "tools.bank_scientist", "function": "train_churn_model",
        "description": "Train a Gradient Boosting churn prediction model, auto-detects churn column",
        "category": "data_science",
        "when_to_use": "When data has a binary churn/retention outcome column",
    },
    "train_fraud_detection_model": {
        "module": "tools.bank_scientist", "function": "train_fraud_detection_model",
        "description": "Train Random Forest + Gradient Boosting fraud detection models",
        "category": "data_science",
        "when_to_use": "When data has a binary fraud/is_fraud outcome column",
    },
    "train_credit_risk_model": {
        "module": "tools.bank_scientist", "function": "train_credit_risk_model",
        "description": "Train a Logistic Regression credit risk model",
        "category": "data_science",
        "when_to_use": "When data has a default/credit_risk outcome column",
    },
    "customer_segmentation": {
        "module": "tools.bank_scientist", "function": "customer_segmentation",
        "description": "Segment customers into 4 groups using K-Means clustering",
        "category": "data_science",
        "when_to_use": "To discover hidden customer clusters for targeting or analysis",
    },
    "time_series_forecast": {
        "module": "tools.bank_scientist", "function": "time_series_forecast",
        "description": "Forecast future values using Holt-Winters Exponential Smoothing",
        "category": "data_science",
        "when_to_use": "When data has a date/time column and a numeric value to forecast",
    },
    # ── Visualization & Reporting ─────────────────────────────────────────────
    "generate_visualization": {
        "module": "tools.bank_analyst", "function": "generate_visualization",
        "description": "Generate a chart (bar, pie, scatter, histogram, heatmap) from a database table",
        "category": "visualization",
        "when_to_use": "To create visual representations of data patterns and distributions",
    },
    "generate_dashboard": {
        "module": "tools.bank_analyst", "function": "generate_dashboard",
        "description": "Generate a 7-panel analytics dashboard PNG from a database table",
        "category": "visualization",
        "when_to_use": "For a comprehensive visual overview of all data metrics at once",
    },
    "generate_text_report": {
        "module": "tools.bank_analyst", "function": "generate_text_report",
        "description": "Generate a professional markdown business insight report",
        "category": "visualization",
        "when_to_use": "To produce a written report summarizing all analysis findings",
    },
    "label_encode_table": {
        "module": "tools.bank_analyst", "function": "label_encode_table",
        "description": "Label-encode all categorical columns in a table for ML or heatmap use",
        "category": "visualization",
        "when_to_use": "Before generating a correlation heatmap or running ML on categorical data",
    },
    "generate_pdf_report": {
        "module": "tools.bank_report", "function": "generate_pdf_report",
        "description": "Generate a professional PDF report with embedded charts",
        "category": "reporting",
        "when_to_use": "To produce a downloadable PDF from analysis results",
    },
    "generate_ppt_report": {
        "module": "tools.bank_report", "function": "generate_ppt_report",
        "description": "Generate a 16:9 PowerPoint presentation from analysis results",
        "category": "reporting",
        "when_to_use": "To produce a downloadable PowerPoint for stakeholder presentations",
    },
    # ── Marketing Research ────────────────────────────────────────────────────
    "web_search_market": {
        "module": "tools.mkt_researcher", "function": "web_search_market",
        "description": "Search the web for market research, industry trends, and news",
        "category": "marketing_research",
        "when_to_use": "To gather external market intelligence and competitor news",
    },
    "analyze_competitors": {
        "module": "tools.mkt_researcher", "function": "analyze_competitors",
        "description": "Analyse competitors and produce a competitive landscape matrix",
        "category": "marketing_research",
        "when_to_use": "To understand competitive positioning and differentiation opportunities",
    },
    "research_target_audience": {
        "module": "tools.mkt_researcher", "function": "research_target_audience",
        "description": "Research target audience demographics, psychographics, and pain points",
        "category": "marketing_research",
        "when_to_use": "To deeply understand who the target customers are",
    },
    "analyze_industry_trends": {
        "module": "tools.mkt_researcher", "function": "analyze_industry_trends",
        "description": "Analyse current industry trends, market size, and emerging opportunities",
        "category": "marketing_research",
        "when_to_use": "To understand macro trends shaping the industry",
    },
    "exa_web_search": {
        "module": "tools.mkt_researcher", "function": "exa_web_search",
        "description": "Neural web search using Exa AI for deep market and competitor intelligence",
        "category": "marketing_research",
        "when_to_use": "For high-quality, semantically relevant web research results",
    },
    # ── Marketing Planning ────────────────────────────────────────────────────
    "create_marketing_strategy": {
        "module": "tools.mkt_planner", "function": "create_marketing_strategy",
        "description": "Synthesise research into a full marketing strategy document",
        "category": "marketing_planning",
        "when_to_use": "To build a comprehensive go-to-market strategy from research findings",
    },
    "create_content_calendar": {
        "module": "tools.mkt_planner", "function": "create_content_calendar",
        "description": "Create a 30-day content calendar for a marketing campaign",
        "category": "marketing_planning",
        "when_to_use": "To plan content publishing schedule and cadence",
    },
    "define_campaign_kpis": {
        "module": "tools.mkt_planner", "function": "define_campaign_kpis",
        "description": "Define measurable KPIs for a marketing campaign",
        "category": "marketing_planning",
        "when_to_use": "To set success metrics and measurement framework",
    },
    "create_campaign_brief": {
        "module": "tools.mkt_planner", "function": "create_campaign_brief",
        "description": "Create a one-page campaign brief",
        "category": "marketing_planning",
        "when_to_use": "To document the campaign strategy in a concise brief format",
    },
    "plan_budget_allocation": {
        "module": "tools.mkt_planner", "function": "plan_budget_allocation",
        "description": "Recommend budget allocation across marketing channels",
        "category": "marketing_planning",
        "when_to_use": "To optimally distribute marketing budget across channels",
    },
    # ── Marketing Content ─────────────────────────────────────────────────────
    "write_ad_copy": {
        "module": "tools.mkt_content", "function": "write_ad_copy",
        "description": "Write platform-specific advertising copy (Google, Meta, LinkedIn, etc.)",
        "category": "marketing_content",
        "when_to_use": "To create compelling ad text for specific platforms",
    },
    "generate_social_posts": {
        "module": "tools.mkt_content", "function": "generate_social_posts",
        "description": "Generate social media posts with hashtags for multiple platforms",
        "category": "marketing_content",
        "when_to_use": "To create social media content for a campaign",
    },
    "create_email_template": {
        "module": "tools.mkt_content", "function": "create_email_template",
        "description": "Create an HTML email template with subject line, body, and CTA",
        "category": "marketing_content",
        "when_to_use": "To create email marketing campaigns",
    },
    "generate_promotional_poster": {
        "module": "tools.mkt_content", "function": "generate_promotional_poster",
        "description": "Generate an AI promotional poster image using Gemini image generation",
        "category": "marketing_content",
        "when_to_use": "To create visual promotional materials for campaigns",
    },
}

# Cache for imported tool objects
_tool_cache: dict = {}


def get_tool(name: str):
    if name in _tool_cache:
        return _tool_cache[name]
    meta = TOOL_REGISTRY.get(name)
    if not meta:
        return None
    try:
        module_path = meta["module"]
        func_name   = meta["function"]
        parts       = module_path.split(".")
        module = __import__(module_path, fromlist=[func_name])
        tool_fn = getattr(module, func_name, None)
        if tool_fn:
            _tool_cache[name] = tool_fn
        return tool_fn
    except Exception as e:
        print(f"[ToolRegistry] Could not load {name}: {e}", flush=True)
        return None


def get_tools_for_names(names: list) -> list:
    tools = []
    for n in names:
        t = get_tool(n)
        if t:
            tools.append(t)
        else:
            print(f"[ToolRegistry] Skipping unavailable tool: {n}", flush=True)
    return tools


# Human-readable tool descriptions for the meta-planner prompt
TOOL_DESCRIPTIONS_FOR_PLANNER = "\n".join(
    f"  - {name}: {meta['description']} [use when: {meta['when_to_use']}]"
    for name, meta in TOOL_REGISTRY.items()
)
