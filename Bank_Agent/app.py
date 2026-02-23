"""
Banking Analytics — FastAPI Backend
=====================================
Serves the Bootstrap UI and exposes REST + SSE endpoints that drive
the multi-agent crew from the browser.

Endpoints
---------
GET  /                        → static/index.html
POST /api/analyze             → start analysis (background thread)
GET  /api/stream/{task_id}    → SSE live log stream
GET  /api/status/{task_id}    → task status JSON
GET  /api/results/{task_id}   → full results (markdown report + artifacts)
GET  /api/logs/{task_id}      → all buffered log lines
GET  /api/charts              → list chart PNGs
GET  /api/reports             → list PDF / PPTX / MD files
GET  /api/tasks               → list all task runs
GET  /outputs/...             → static serving of generated files

Run:
    cd D:/Langsmith-main/Bank_Agent
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys

# ── Disable CrewAI / OpenTelemetry signal handlers before any crewai import.
# CrewAI tries to register SIGTERM/SIGINT handlers which fail in background
# threads (non-main thread). Setting these vars prevents that entirely.
os.environ["OTEL_SDK_DISABLED"]          = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"]  = "true"

import re
import json
import uuid
import queue
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

for _d in ("outputs/charts", "outputs/reports", "outputs/models", "static"):
    os.makedirs(BASE_DIR / _d, exist_ok=True)

# ── FastAPI ──────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Banking Analytics Multi-Agent System",
    description="AI-powered banking data team: Manager · Data Engineer · Data Scientist · Data Analyst",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static",  StaticFiles(directory=str(BASE_DIR / "static")),  name="static")
app.mount("/outputs", StaticFiles(directory=str(BASE_DIR / "outputs")), name="outputs")

# ── In-memory store ──────────────────────────────────────────────────────────
#  tasks[tid]  = { status, started_at, completed_at, result, error, logs, charts, reports }
#  log_queues[tid] = queue.Queue  (streams new items to SSE clients)
tasks: dict      = {}
log_queues: dict = {}

_analysis_lock = threading.Lock()  # enforce one concurrent run


# ── Pydantic models ──────────────────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    analysis_request: str  = ""
    use_langgraph:    bool = True


# ── Stdout capture ───────────────────────────────────────────────────────────
class _TeeWriter:
    """Writes to the original stdout AND appends to the task log queue."""

    def __init__(self, task_id: str, original):
        self._tid      = task_id
        self._original = original

    def write(self, text: str):
        self._original.write(text)
        self._original.flush()
        stripped = text.rstrip()
        if stripped:
            entry = {
                "type":      "log",
                "message":   stripped,
                "timestamp": datetime.now().isoformat(),
            }
            tasks[self._tid]["logs"].append(entry)
            try:
                log_queues[self._tid].put_nowait(entry)
            except Exception:
                pass

    def flush(self):
        self._original.flush()

    def isatty(self):
        return False


# ── Background worker ────────────────────────────────────────────────────────
def _push(task_id: str, msg: str, kind: str = "log"):
    """Helper: push a structured log message."""
    entry = {"type": kind, "message": msg, "timestamp": datetime.now().isoformat()}
    tasks[task_id]["logs"].append(entry)
    try:
        log_queues[task_id].put_nowait(entry)
    except Exception:
        pass


def _strip_md_fences(text: str) -> str:
    """
    Gemini sometimes wraps its entire response in ```markdown ... ``` or
    appends the original report again inside such a fence for 'reference'.
    This removes both the outer wrapper and any inner fenced blocks.
    """
    # Strip outer ```markdown / ```md / ``` wrapper
    text = re.sub(r'^```(?:markdown|md)?\s*\n', '', text.strip())
    text = re.sub(r'\n```\s*$', '', text.strip())
    # Remove inner ```markdown...``` blocks Gemini sometimes appends
    text = re.sub(r'\n```(?:markdown|md)?\n[\s\S]*?```(?:\n|$)', '\n', text)
    return text.strip()


def _run_analysis(task_id: str, req: AnalysisRequest):
    original_stdout = sys.stdout
    sys.stdout = _TeeWriter(task_id, original_stdout)

    try:
        _push(task_id, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")
        _push(task_id, "  BANKING ANALYTICS MULTI-AGENT SYSTEM  —  starting", "info")
        _push(task_id, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")
        _push(task_id, f"  LangGraph    : {req.use_langgraph}", "info")
        _push(task_id, f"  Started at   : {tasks[task_id]['started_at']}", "info")
        _push(task_id, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "info")

        from crew import run_banking_analysis_crew

        # Clear previous-run chart metadata so each run starts with a clean blog report
        _sess_meta = BASE_DIR / "outputs" / "charts" / "session_charts.json"
        try:
            if _sess_meta.exists():
                _sess_meta.unlink()
        except Exception:
            pass

        analysis_request = req.analysis_request.strip() or _default_request()

        result = run_banking_analysis_crew(
            analysis_request=analysis_request,
            use_langgraph=req.use_langgraph,
        )

        charts  = _list_files("outputs/charts",  [".png"])
        reports = _list_files("outputs/reports", [".pdf", ".pptx", ".md"])

        # Extract the markdown report from crew output or langgraph plan
        markdown_report = _extract_markdown_report(result)

        # Load per-chart blog sections written by generate_visualization / generate_dashboard
        blog_sections: list = []
        _meta_file = BASE_DIR / "outputs" / "charts" / "session_charts.json"
        if _meta_file.exists():
            try:
                with open(_meta_file, "r", encoding="utf-8") as _mf:
                    blog_sections = json.load(_mf)
                _push(task_id, f"  Blog sections loaded from metadata: {len(blog_sections)}", "info")
            except Exception as _meta_err:
                _push(task_id, f"  Blog sections metadata read failed: {_meta_err}", "warn")
        else:
            _push(task_id, "  session_charts.json not found — using PNG fallback", "warn")

        # Fallback: if metadata is missing/empty, build blog sections from session PNGs
        if not blog_sections and charts:
            _run_start_ts = datetime.fromisoformat(tasks[task_id]["started_at"]).timestamp()
            for _c in charts:
                try:
                    _mod_ts = datetime.fromisoformat(_c["modified"]).timestamp()
                except Exception:
                    _mod_ts = 0
                # 30-second grace period to catch charts written just before run_start
                if _mod_ts >= _run_start_ts - 30:
                    blog_sections.append({
                        "title":                  _c["name"].replace(".png", "").replace("_", " ").title(),
                        "chart_type":             "chart",
                        "chart_url":              _c["url"],
                        "gemini_analysis":        "",
                        "nvidia_recommendations": "",
                    })
            if blog_sections:
                _push(task_id, f"  Blog sections (fallback): {len(blog_sections)} PNG(s) included", "info")

        # ── Gemini post-processing: enhance report with actionable strategies ──
        try:
            from config import gemini_llm
            from langchain_core.messages import HumanMessage

            if gemini_llm and markdown_report and len(markdown_report) > 100:
                chart_analyses = "\n\n".join(
                    f"**{s['title']}**:\n{s['gemini_analysis']}"
                    for s in blog_sections
                    if s.get("gemini_analysis")
                )
                enhance_prompt = (
                    "You are a senior banking strategy consultant. "
                    "You have received an executive summary report and detailed chart analyses "
                    "from a data team. Your task is to produce a final enhanced version.\n\n"
                    "## Current Executive Report:\n"
                    f"{markdown_report}\n\n"
                    + (f"## Chart Analyses from Data Team:\n{chart_analyses}\n\n" if chart_analyses else "")
                    + "## Enhancement Instructions:\n"
                    "Rewrite and enrich the report using the chart analyses as additional evidence. "
                    "Focus on making these sections substantially more detailed:\n"
                    "- **Business Recommendations** / **Strategic Recommendations**: For each point, "
                    "add the responsible team, a specific KPI target, and a 30/60/90-day milestone.\n"
                    "- **Segment-Specific Strategies** (if present): Provide 3-5 sentences per "
                    "segment with concrete retention campaigns, discount thresholds, and outreach channels.\n"
                    "- **Next Steps**: Replace vague bullet points with a numbered action plan "
                    "that includes owner, deadline, and success metric.\n"
                    "Enrich Key Findings and Risk Assessment with specific percentages or figures "
                    "drawn from the chart analyses where available. "
                    "Keep the same markdown section headers. Output the complete enhanced report."
                )
                _push(task_id, "  Gemini post-processing: enhancing report with actionable strategies...", "info")
                response = gemini_llm.invoke([HumanMessage(content=enhance_prompt)])
                enhanced = _strip_md_fences(response.content)
                if enhanced and len(enhanced) > 200:
                    markdown_report = enhanced
                    _push(task_id, "  Gemini post-processing: report successfully enhanced", "info")
                else:
                    _push(task_id, "  Gemini post-processing: response empty, keeping original", "warn")
        except Exception as _gem_err:
            _push(task_id, f"  Gemini post-processing failed (non-critical): {_gem_err}", "warn")

        tasks[task_id].update({
            "status":          "completed",
            "completed_at":    datetime.now().isoformat(),
            "result":          result,
            "markdown_report": markdown_report,
            "charts":          charts,
            "reports":         reports,
            "blog_sections":   blog_sections,
        })

        _push(task_id, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "success")
        _push(task_id, f"  Analysis complete! Charts: {len(charts)}  Blog sections: {len(blog_sections)}  Reports: {len(reports)}", "success")
        _push(task_id, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "success")

    except Exception as exc:
        import traceback
        err_trace = traceback.format_exc()
        tasks[task_id].update({
            "status":       "error",
            "error":        str(exc),
            "completed_at": datetime.now().isoformat(),
        })
        _push(task_id, f"ERROR: {exc}", "error")
        _push(task_id, err_trace, "error")

    finally:
        sys.stdout = original_stdout
        # Signal SSE clients that the stream is done
        done_type = "complete" if tasks[task_id]["status"] == "completed" else "error"
        try:
            log_queues[task_id].put_nowait({"type": done_type, "task_id": task_id,
                                            "timestamp": datetime.now().isoformat()})
        except Exception:
            pass


def _run_with_lock(task_id: str, req: AnalysisRequest):
    with _analysis_lock:
        _run_analysis(task_id, req)


def _default_request() -> str:
    return (
        "Perform comprehensive banking customer analytics on the 'churn' table in PostgreSQL.\n\n"
        "1. DATA ENGINEERING: Profile the existing 'churn' table (22 columns: customerID, gender, "
        "SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, "
        "OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, "
        "Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn, "
        "transaction_date). Validate data quality and report null counts and Churn distribution.\n"
        "2. DATA SCIENCE: Train a customer churn prediction model (target column='Churn'), "
        "perform customer segmentation (4 clusters), report AUC-ROC and top churn predictors.\n"
        "3. DATA ANALYSIS: Create individual charts — churn distribution pie, tenure histogram, "
        "monthly charges histplot by Churn, contract type bar chart, tenure vs charges scatter, "
        "feature correlation heatmap (use churn_encoded table). "
        "Analyse every chart with Gemini AI vision. Do NOT generate a dashboard.\n"
        "4. REPORTS: Produce PDF and PowerPoint executive reports.\n"
        "5. EXECUTIVE SUMMARY: Top 5 churn risk findings and customer retention recommendations. "
        "Do NOT include file paths or folder locations in the summary."
    )


def _extract_markdown_report(result: dict) -> str:
    """
    Pull the best available markdown text from the analysis result.
    Priority: crew output → LangGraph preliminary report → summary.
    """
    # Try crew output (the manager's final executive briefing)
    crew_out = result.get("crew_output", "")
    if crew_out and len(crew_out) > 50:   # lowered from 200 — any real output is valid
        return crew_out

    # Fall back to LangGraph preliminary report
    lg = result.get("langgraph_plan", {})
    prelim = lg.get("preliminary_report", "")
    if prelim and len(prelim) > 200:
        return prelim

    return "# Analysis Complete\n\nPlease check the console log for detailed output."


def _list_files(directory: str, extensions: list) -> list:
    p = BASE_DIR / directory
    if not p.exists():
        return []
    items = []
    for ext in extensions:
        for f in sorted(p.glob(f"*{ext}"), key=os.path.getmtime, reverse=True):
            rel = f.relative_to(BASE_DIR).as_posix()
            items.append({
                "name":     f.name,
                "url":      f"/{rel}",
                "size":     f.stat().st_size,
                "ext":      ext.lstrip("."),
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
    return items


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.post("/api/analyze")
async def start_analysis(req: AnalysisRequest):
    """Start a new banking analysis run (background thread)."""
    if _analysis_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="An analysis is already running. Please wait for it to finish.",
        )

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status":          "running",
        "started_at":      datetime.now().isoformat(),
        "completed_at":    None,
        "result":          None,
        "markdown_report": None,
        "error":           None,
        "logs":            [],
        "charts":          [],
        "reports":         [],
        "blog_sections":   [],
    }
    log_queues[task_id] = queue.Queue()

    thread = threading.Thread(target=_run_with_lock, args=(task_id, req), daemon=True)
    thread.start()

    return {"task_id": task_id, "status": "started"}


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """Poll task status without streaming."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found.")
    t = tasks[task_id]
    return {
        "task_id":       task_id,
        "status":        t["status"],
        "started_at":    t["started_at"],
        "completed_at":  t.get("completed_at"),
        "error":         t.get("error"),
        "charts_count":  len(t.get("charts", [])),
        "reports_count": len(t.get("reports", [])),
        "log_lines":     len(t.get("logs", [])),
    }


@app.get("/api/results/{task_id}")
async def get_results(task_id: str):
    """Return full analysis results once completed."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found.")
    t = tasks[task_id]
    if t["status"] == "running":
        raise HTTPException(status_code=202, detail="Analysis still running.")
    if t["status"] == "error":
        raise HTTPException(status_code=500, detail=t.get("error", "Unknown error."))

    return {
        "task_id":         task_id,
        "status":          "completed",
        "markdown_report": t.get("markdown_report", ""),
        "charts":          t.get("charts", []),
        "reports":         t.get("reports", []),
        "blog_sections":   t.get("blog_sections", []),
        "langgraph_plan":  (t.get("result") or {}).get("langgraph_plan", {}),
        "completed_at":    t.get("completed_at"),
    }


@app.get("/api/logs/{task_id}")
async def get_logs(task_id: str):
    """Return all buffered log lines for a task (useful after reconnect)."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found.")
    return {"task_id": task_id, "logs": tasks[task_id].get("logs", [])}


@app.get("/api/stream/{task_id}")
async def stream_events(task_id: str):
    """
    Server-Sent Events endpoint — streams live log messages to the browser.
    Each event: data: {"type": "log"|"info"|"success"|"error"|"complete", "message": "..."}
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found.")

    async def generator():
        q = log_queues[task_id]
        # Replay already-buffered logs first (handles slow browser connect).
        # Track how many we replayed so we skip them in the queue below.
        buffered = list(tasks[task_id].get("logs", []))
        replayed = len(buffered)
        for entry in buffered:
            yield f"data: {json.dumps(entry)}\n\n"

        # Drain any queue entries that were already replayed above
        drained = 0
        while drained < replayed:
            try:
                q.get_nowait()
                drained += 1
            except queue.Empty:
                break

        # Stream new entries
        while True:
            try:
                msg = q.get_nowait()
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ("complete", "error"):
                    return
            except queue.Empty:
                # Task finished but no terminal event in queue
                if tasks[task_id]["status"] in ("completed", "error"):
                    final = "complete" if tasks[task_id]["status"] == "completed" else "error"
                    yield f"data: {json.dumps({'type': final, 'task_id': task_id})}\n\n"
                    return
                await asyncio.sleep(0.35)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


@app.get("/api/charts")
async def list_charts():
    return {"charts": _list_files("outputs/charts", [".png"])}


@app.get("/api/reports")
async def list_reports():
    return {"reports": _list_files("outputs/reports", [".pdf", ".pptx", ".md"])}


@app.get("/api/tasks")
async def list_tasks():
    return [
        {
            "task_id":      tid,
            "status":       t["status"],
            "started_at":   t["started_at"],
            "completed_at": t.get("completed_at"),
        }
        for tid, t in tasks.items()
    ]


# ── Dev server entry-point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
