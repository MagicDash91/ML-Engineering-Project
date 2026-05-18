import time
import asyncio
from agent_factory import run_agent_step, _fix_chart_paths
from step_evaluator import evaluate_step
from config import query_ollama


async def _emit(queue, event_type: str, message: str, payload: dict = None):
    if queue:
        await queue.put({"type": event_type, "message": message, "payload": payload or {}})


async def build_and_run(
    user_question: str,
    plan: dict,
    db_uri: str = None,
    document_context: str = "",
    log_queue=None,
) -> dict:
    pipeline_start = time.time()
    step_results = []

    for step in plan["steps"]:
        step_id   = step["id"]
        step_name = step["name"]
        max_retries = int(step.get("max_retries", 2))
        retry_count = 0
        passed = False

        await _emit(log_queue, "step_start", f"▶ Starting: {step_name}",
                    {"step_id": step_id, "objective": step["objective"], "tools": step.get("tools", [])})

        while retry_count <= max_retries and not passed:
            if retry_count > 0:
                await _emit(log_queue, "retry",
                            f"↩ Retrying {step_name} (attempt {retry_count + 1}/{max_retries + 1})",
                            {"step_id": step_id, "retry": retry_count})

            # Run agent in thread so we don't block the event loop
            output = await asyncio.to_thread(
                run_agent_step, step, user_question, step_results, db_uri, document_context
            )

            await _emit(log_queue, "step_output",
                        f"✓ {step_name} completed — evaluating quality...",
                        {"step_id": step_id, "output_preview": output[:300]})

            # Evaluate output quality
            eval_result = await asyncio.to_thread(
                evaluate_step, step, output, user_question
            )

            score    = eval_result["score"]
            passed   = eval_result["passed"]
            feedback = eval_result["feedback"]

            status_icon = "✅" if passed else "⚠️"
            await _emit(log_queue, "eval",
                        f"{status_icon} Evaluation: {score}/10 — {feedback}",
                        {"step_id": step_id, "score": score, "passed": passed, "feedback": feedback})

            if passed:
                break

            retry_count += 1

        step_results.append({
            "step_id":   step_id,
            "step_name": step_name,
            "output":    output,
            "score":     score,
            "passed":    passed,
            "feedback":  feedback,
        })

    # Final synthesis
    await _emit(log_queue, "synthesis_start", "🧠 Synthesizing final answer...", {})

    final_answer = await asyncio.to_thread(
        _synthesize, user_question, plan, step_results
    )

    # Fix any hallucinated image paths using all charts generated during this pipeline
    final_answer = _fix_chart_paths(final_answer, pipeline_start)

    await _emit(log_queue, "final", "🎯 Complete — final answer ready.", {"answer": final_answer})

    return {
        "user_question": user_question,
        "plan":          plan,
        "step_results":  step_results,
        "final_answer":  final_answer,
        "status":        "completed",
    }


def _synthesize(user_question: str, plan: dict, step_results: list) -> str:
    results_text = "\n\n".join(
        f"### Step {i+1}: {r['step_name']} (Score: {r['score']}/10)\n{r['output'][:3000]}"
        for i, r in enumerate(step_results)
    )

    synthesis_instructions = plan.get(
        "final_synthesis_prompt",
        f"Synthesize all findings into a comprehensive answer for: {user_question}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior coordinator synthesizing outputs from a multi-agent team. "
                "Write a clear, structured, and comprehensive final answer for the user. "
                "Use markdown formatting with headers, bullet points, and bold key insights."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User's original question: {user_question}\n\n"
                f"Synthesis instructions: {synthesis_instructions}\n\n"
                f"Agent team outputs:\n{results_text}\n\n"
                "Write the final answer now:"
            ),
        },
    ]

    try:
        return query_ollama(messages, temperature=0.2, max_tokens=4096)
    except Exception as e:
        # Fallback: concatenate step outputs
        return f"## Summary\n\n" + "\n\n".join(
            f"**{r['step_name']}:** {r['output'][:500]}" for r in step_results
        )
