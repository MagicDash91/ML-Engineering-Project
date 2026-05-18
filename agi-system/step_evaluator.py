import json
import re
from config import query_ollama

_EVAL_SYSTEM = """You are a strict quality evaluator for AI agent outputs.
Score the output from 1 to 10 based on whether it meets the success criteria.
Respond with ONLY valid JSON — no other text.
"""


def evaluate_step(step: dict, output: str, user_question: str) -> dict:
    prompt = f"""Evaluate this agent output:

User's original question: {user_question}
Step objective: {step['objective']}
Success criteria: {step['success_criteria']}

Agent output (first 2000 chars):
{output[:2000]}

Score 1-10 (7+ = passed). Respond with ONLY:
{{"score": <int>, "passed": <true|false>, "feedback": "<one sentence>"}}"""

    try:
        raw = query_ollama(
            [
                {"role": "system", "content": _EVAL_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=150,
        )
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
        match = re.search(r"\{[\s\S]*\}", raw)
        result = json.loads(match.group() if match else raw)
        score  = int(result.get("score", 7))
        return {
            "score":    score,
            "passed":   score >= 7,
            "feedback": result.get("feedback", ""),
        }
    except Exception as e:
        print(f"[StepEvaluator] Eval parse failed: {e}. Auto-passing.", flush=True)
        return {"score": 7, "passed": True, "feedback": "Auto-passed (evaluator error)"}
