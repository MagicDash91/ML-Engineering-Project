import os
import re
import glob
import time
import json
from pathlib import Path
from tool_registry import get_tools_for_names

_UA_CHARTS_DIR = Path(__file__).parent.parent / "Universal_AI_Analytics" / "outputs" / "charts"


_EXCLUDED_CHART_PREFIXES = ("feature_importance_", "roc_", "auc_", "confusion_")


def _fix_chart_paths(text: str, step_start: float) -> str:
    """Replace hallucinated/fake image paths with real business chart URLs created during this step."""
    # Only consider charts created AFTER this step started; exclude ML-metric charts
    new_charts = sorted(
        [
            f for f in _UA_CHARTS_DIR.glob("*.png")
            if f.stat().st_mtime >= step_start
            and not any(f.name.startswith(p) for p in _EXCLUDED_CHART_PREFIXES)
        ],
        key=lambda f: f.stat().st_mtime,
    )
    real_names = {f.name for f in new_charts}
    chart_iter = iter(new_charts)

    def _replace(m):
        alt  = m.group(1)
        path = m.group(2)
        # /ua-outputs path: keep only if the file actually exists
        if path.startswith("/ua-outputs/charts/"):
            filename = path.rsplit("/", 1)[-1]
            if filename in real_names:
                return m.group(0)
        elif path.startswith("/ua-outputs"):
            return m.group(0)   # non-chart /ua-outputs path — leave alone
        # Hallucinated or non-existent path — assign next real chart
        chart = next(chart_iter, None)
        if chart is None:
            return ""           # no more charts; remove broken reference
        return f"![{alt}](/ua-outputs/charts/{chart.name})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|gif|svg|webp))\)", _replace, text)


def run_agent_step(
    step: dict,
    user_question: str,
    previous_results: list,
    db_uri: str = None,
    document_context: str = "",
) -> str:
    try:
        import litellm
        litellm.drop_params = True
        litellm.set_verbose = False
    except Exception:
        pass

    from crewai import Agent, Task, Crew, Process
    from crewai.llm import LLM

    # Set (or clear) active DB URI — must always be explicit so a previous run's value never leaks
    if db_uri:
        os.environ["ACTIVE_DB_URI"] = db_uri
    else:
        os.environ.pop("ACTIVE_DB_URI", None)

    ollama_llm = LLM(
        model="ollama_chat/gpt-oss:120b-cloud",
        base_url="http://localhost:11434",
        temperature=0,
        max_tokens=4096,
        timeout=300,
        num_ctx=32768,
    )

    tools = get_tools_for_names(step.get("tools", []))

    # Build context from previous results (last 2, truncated)
    prev_ctx = ""
    if previous_results:
        recent = previous_results[-2:]
        parts = []
        for r in recent:
            parts.append(f"[{r['step_name']}]: {str(r['output'])[:800]}")
        prev_ctx = "\n\n".join(parts)

    agent = Agent(
        role=step["agent_role"],
        goal=step["agent_goal"],
        backstory=step.get("agent_backstory", f"Expert {step['agent_role']} solving complex problems."),
        llm=ollama_llm,
        tools=tools,
        allow_delegation=False,
        verbose=True,
        max_iter=12,
        max_retry_limit=2,
    )

    task_description = f"""USER QUESTION: {user_question}

YOUR OBJECTIVE FOR THIS STEP: {step['objective']}

SUCCESS CRITERIA: {step['success_criteria']}

IMPORTANT DATA RULES:
- When writing SQL queries, NEVER use LIMIT values smaller than 10000. Use the full dataset.
- Do NOT write SELECT * FROM table LIMIT 100 — always use at least LIMIT 10000 or no LIMIT at all.
- The uploaded dataset may have thousands of rows; analysis must cover all of them, not just a sample.
"""
    if document_context:
        task_description += f"\nDOCUMENT CONTEXT (extracted from uploaded files — use as primary data):\n{document_context[:4000]}\n"

    if prev_ctx:
        task_description += f"\nPREVIOUS STEP RESULTS (use as context):\n{prev_ctx}"

    task = Task(
        description=task_description,
        expected_output=step["success_criteria"],
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
        memory=False,
    )

    try:
        step_start = time.time()
        result = crew.kickoff()
        output = str(result)
        return _fix_chart_paths(output, step_start)
    except Exception as e:
        return f"[AgentFactory] Step failed: {e}"
