import json
import re
from config import query_ollama
from tool_registry import TOOL_DESCRIPTIONS_FOR_PLANNER

_SYSTEM_PROMPT = """You are an AGI meta-planner. Your job is to analyze a user's problem and create a precise execution plan.

Given a user's question or problem, you will:
1. Classify the problem type
2. Break it into 3-5 sequential steps
3. For each step, choose the right agent role and tools
4. Define clear success criteria for each step

AVAILABLE TOOLS:
{tools}

OUTPUT FORMAT — respond with ONLY a valid JSON object, no other text:
{{
  "problem_type": "data_analytics | marketing | research | mixed | general",
  "domain": "brief domain e.g. banking, ecommerce, sales, telecom",
  "summary": "one sentence: what this plan will do",
  "steps": [
    {{
      "id": "step_1",
      "name": "Human-readable step name",
      "objective": "Specific objective this step must achieve",
      "agent_role": "Job title e.g. Senior Data Engineer",
      "agent_goal": "What the agent must accomplish in this step",
      "agent_backstory": "One sentence about the agent's expertise",
      "tools": ["tool_name_1", "tool_name_2"],
      "success_criteria": "What a good output looks like — be specific",
      "max_retries": 2
    }}
  ],
  "final_synthesis_prompt": "Instructions for the final coordinator to synthesize all step results into a final answer for the user"
}}

RULES:
- Maximum 5 steps
- Only use tool names from the AVAILABLE TOOLS list above
- {db_rule}
- If the question is about marketing → include web_search_market and create_marketing_strategy
- If no data tools are needed → use web research tools only
- Each step should build on the previous step's results
- The final_synthesis_prompt must reference the user's original question
- When a database or data file is available, at least one step MUST include generate_visualization (preferred) or generate_dashboard
- Visualization steps must produce BUSINESS-FRIENDLY charts (bar by category, pie by segment, trend over time) — never ROC curves, AUC plots, confusion matrices, or feature importance charts
"""

_VIZ_TOOLS = {"generate_visualization", "generate_dashboard"}
_VIZ_INJECT_STEP = {
    "id": "step_viz",
    "name": "Business Charts",
    "objective": (
        "Generate 2-3 clear business charts for a non-technical manager. "
        "STEP 1: Use profile_database_table to get the EXACT column names available. "
        "STEP 2: Call generate_visualization using only column names that actually exist in the table. "
        "Choose: a categorical column (e.g. Contract, PaymentMethod, Region) for x_column on bar/pie charts, "
        "a numeric column (e.g. MonthlyCharges, tenure) for y_column when needed. "
        "DO NOT guess column names — always profile first. "
        "DO NOT create ROC, AUC, feature importance, or confusion matrix charts."
    ),
    "agent_role": "Business Intelligence Analyst",
    "agent_goal": (
        "Create 2-3 charts a business executive can instantly understand. "
        "Always call profile_database_table first to see exact column names, then call generate_visualization. "
        "Good charts: churn count by contract type (bar), revenue distribution by segment (pie), monthly charges histogram. "
        "Use chart_type='bar' or chart_type='pie'. For count-based bars, leave y_column empty."
    ),
    "agent_backstory": "Expert BI analyst who always inspects the data schema before building charts.",
    "tools": ["profile_database_table", "generate_visualization"],
    "success_criteria": (
        "At least one business-friendly chart generated using real column names from the database "
        "(verified via profile_database_table), showing a key business metric by a meaningful category"
    ),
    "max_retries": 2,
}

_DB_TOOLS = {
    "list_database_tables", "profile_database_table", "query_database",
    "clean_table_columns", "normalize_column_dtypes", "run_etl_pipeline",
    "train_churn_model", "train_fraud_detection_model", "train_credit_risk_model",
    "customer_segmentation", "time_series_forecast",
    "generate_visualization", "generate_dashboard", "label_encode_table",
}


def generate_plan(user_question: str, context: dict = None, document_context: str = "") -> dict:
    has_db = bool((context or {}).get("db_uri"))

    ctx_str = ""
    if context:
        if context.get("db_uri"):
            ctx_str += f"\nAvailable database URI: {context['db_uri']}"
        else:
            ctx_str += "\nNo database connected."
        if context.get("has_files"):
            ctx_str += "\nUser uploaded data files (structured files loaded into SQLite, documents extracted to text)."
        if context.get("document_context"):
            ctx_str += f"\nDocument content preview:\n{context['document_context'][:1500]}"

    db_rule = (
        "If the question needs data analysis → start with list_database_tables and profile_database_table"
        if has_db else
        "NO database is connected — do NOT use list_database_tables, profile_database_table, "
        "query_database, train_churn_model, train_fraud_detection_model, train_credit_risk_model, "
        "customer_segmentation, time_series_forecast, generate_visualization, generate_dashboard, "
        "label_encode_table, or any other database/ML tool"
    )

    doc_hint = ""
    if document_context:
        doc_hint = f"\n\nDOCUMENT CONTENT (use this as primary data source for analysis):\n{document_context[:3000]}"

    prompt = f"""User's problem/question: {user_question}{ctx_str}{doc_hint}

Create the execution plan now. Output ONLY the JSON object."""

    try:
        raw = query_ollama(
            [
                {"role": "system", "content": _SYSTEM_PROMPT.format(
                    tools=TOOL_DESCRIPTIONS_FOR_PLANNER, db_rule=db_rule)},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        plan = _extract_json(raw)
        _validate_plan(plan)

        # Hard safeguard: strip DB/ML tools from every step if no database is connected
        if not has_db:
            for step in plan.get("steps", []):
                step["tools"] = [t for t in step.get("tools", []) if t not in _DB_TOOLS]

        # Ensure at least one visualization step when a database is available
        if has_db:
            has_viz = any(
                bool(_VIZ_TOOLS & set(step.get("tools", [])))
                for step in plan.get("steps", [])
            )
            if not has_viz and len(plan["steps"]) < 5:
                # Insert viz step before the last step
                plan["steps"].insert(-1, dict(_VIZ_INJECT_STEP))
            elif not has_viz:
                # Replace last step's tools to add viz (plan is already at max)
                plan["steps"][-1].setdefault("tools", []).append("generate_visualization")

        return plan
    except Exception as e:
        print(f"[MetaPlanner] Plan generation failed: {e}. Using fallback.", flush=True)
        return _fallback_plan(user_question)


def _extract_json(text: str) -> dict:
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    # Find first { ... } block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def _validate_plan(plan: dict):
    assert "steps" in plan and len(plan["steps"]) >= 1, "Plan must have at least 1 step"
    for step in plan["steps"]:
        for key in ("id", "name", "objective", "agent_role", "agent_goal", "tools", "success_criteria"):
            assert key in step, f"Step missing key: {key}"


def _fallback_plan(user_question: str) -> dict:
    return {
        "problem_type": "research",
        "domain": "general",
        "summary": f"Research and answer: {user_question[:80]}",
        "steps": [
            {
                "id": "step_1",
                "name": "Web Research",
                "objective": f"Research the topic: {user_question}",
                "agent_role": "Research Analyst",
                "agent_goal": f"Find comprehensive information to answer: {user_question}",
                "agent_backstory": "Expert researcher with deep web research skills.",
                "tools": ["web_search_market", "analyze_industry_trends"],
                "success_criteria": "Relevant findings gathered from multiple sources",
                "max_retries": 1,
            },
            {
                "id": "step_2",
                "name": "Analysis & Synthesis",
                "objective": "Synthesize research into actionable insights",
                "agent_role": "Business Analyst",
                "agent_goal": "Analyse research findings and produce a structured report",
                "agent_backstory": "Senior analyst with expertise in synthesizing complex information.",
                "tools": ["generate_text_report"],
                "success_criteria": "Structured markdown report with key findings and recommendations",
                "max_retries": 1,
            },
        ],
        "final_synthesis_prompt": f"Synthesize all step results into a comprehensive answer for the user's question: {user_question}",
    }


def format_plan_for_display(plan: dict) -> str:
    lines = [
        f"**Problem Type:** {plan.get('problem_type', 'unknown').replace('_', ' ').title()}",
        f"**Domain:** {plan.get('domain', '')}",
        f"**Plan:** {plan.get('summary', '')}",
        "",
        f"**{len(plan['steps'])} Steps:**",
    ]
    for i, step in enumerate(plan["steps"], 1):
        lines.append(f"{i}. **{step['name']}** — {step['objective']}")
        lines.append(f"   Agent: {step['agent_role']} | Tools: {', '.join(step['tools'])}")
    return "\n".join(lines)
