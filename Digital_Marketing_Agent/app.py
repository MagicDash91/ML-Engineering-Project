"""
app.py – FastAPI + SSE backend for the Digital Marketing Multi-Agent System.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
import uuid
from datetime import datetime
from io import StringIO
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── load env early ────────────────────────────────────────────────────────────
from pathlib import Path as _PL
from dotenv import load_dotenv
load_dotenv(_PL(__file__).parent / ".env", override=True)

# ── output directories ────────────────────────────────────────────────────────
_BASE    = Path(__file__).parent / "outputs"
_VIDEOS  = _BASE / "videos"
_CONTENT = _BASE / "content"
_REPORTS = _BASE / "reports"
_AUDIT   = _BASE / "audit"

for _d in (_VIDEOS, _CONTENT, _REPORTS, _AUDIT):
    _d.mkdir(parents=True, exist_ok=True)

_AUDIT_LOG = _AUDIT / "conversation_audit.jsonl"

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Digital Marketing Multi-Agent System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC  = Path(__file__).parent / "static"
_OUTPUTS = Path(__file__).parent / "outputs"

app.mount("/static",  StaticFiles(directory=str(_STATIC)),  name="static")
app.mount("/outputs", StaticFiles(directory=str(_OUTPUTS)), name="outputs")

# ── in-memory task store ──────────────────────────────────────────────────────
tasks:       Dict[str, dict]  = {}
log_queues:  Dict[str, Queue] = {}
hitl_events: Dict[str, threading.Event] = {}
hitl_abort:  Dict[str, bool]  = {}

# ── Pydantic models ───────────────────────────────────────────────────────────

class MarketingRequest(BaseModel):
    brand_name:      str
    industry:        str
    target_audience: str
    campaign_goals:  str
    budget:          str = "Not specified"
    competitors:     str = "Not specified"
    campaign_type:   str = "Brand Awareness"

class ChatRequest(BaseModel):
    message: str

class ApproveRequest(BaseModel):
    abort: bool = False

# ── Security patterns ─────────────────────────────────────────────────────────
_PII_PATTERNS = [
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), '[EMAIL]'),
    (re.compile(r'(\+?\d[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b'), '[PHONE]'),
    (re.compile(r'\b\d{16}\b'), '[CARD]'),
]

_SQL_RE = re.compile(
    r'(union\s+(all\s+)?select)|(drop\s+(table|database|schema))'
    r'|(insert\s+into\s+\w)|(delete\s+from\s+\w)|(update\s+\w+\s+set\s)'
    r'|(exec(\s|\())|(\bxp_)',
    re.IGNORECASE,
)

_GUARDRAIL_RE = re.compile(
    r'\b(ignore|forget|disregard)\s+(previous|all|above|your)\s+(instruction|rule|prompt)'
    r'|\b(jailbreak|prompt.injection|bypass.filter|bypass.security)'
    r'|\bact\s+as\s+(a\s+)?(different|another|new)\s+(ai|model|assistant)'
    r'|\bspend\s+(my\s+)?\$[\d,]+\s+on'
    r'|\bcharge\s+(my\s+)?(card|account|credit)',
    re.IGNORECASE,
)

def _redact_pii(text: str) -> tuple[str, bool]:
    redacted = False
    for pattern, replacement in _PII_PATTERNS:
        new_text = pattern.sub(replacement, text)
        if new_text != text:
            redacted = True
        text = new_text
    return text, redacted

def _check_sql(text: str) -> bool:
    return bool(_SQL_RE.search(text))

def _check_guardrails(text: str) -> bool:
    return bool(_GUARDRAIL_RE.search(text))

def _audit_log(entry: dict) -> None:
    with _AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ── stdout capture / SSE streaming ───────────────────────────────────────────

class _TeeWriter:
    """Tees stdout to both the original stream and the task log queue for SSE."""
    def __init__(self, original, task_id: str):
        self._orig = original
        self._tid  = task_id

    def write(self, text: str):
        self._orig.write(text)
        stripped = text.strip()
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
        self._orig.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)

# ── main analysis pipeline ────────────────────────────────────────────────────

def _run_analysis(task_id: str, req: MarketingRequest) -> None:
    task = tasks[task_id]
    orig_stdout = sys.stdout
    sys.stdout  = _TeeWriter(orig_stdout, task_id)

    def _log(msg: str, msg_type: str = "log"):
        entry = {"type": msg_type, "message": msg, "timestamp": datetime.now().isoformat()}
        task["logs"].append(entry)
        log_queues[task_id].put_nowait(entry)

    try:
        task["status"]     = "running"
        task["started_at"] = datetime.now().isoformat()

        # ── Phase 1: LangGraph Gemini planning ────────────────────────────────
        _log("═══ Phase 1: LangGraph Strategic Planning (Gemini 2.5 Flash) ═══")

        from graphs.marketing_graph import run_marketing_analysis

        langgraph_plan = run_marketing_analysis(
            task_description=(
                f"Create a {req.campaign_type} campaign for {req.brand_name} "
                f"in the {req.industry} industry. Goals: {req.campaign_goals}. "
                f"Target audience: {req.target_audience}. Budget: {req.budget}. "
                f"Competitors: {req.competitors}."
            ),
            brand_name      = req.brand_name,
            industry        = req.industry,
            target_audience = req.target_audience,
            campaign_goals  = req.campaign_goals,
            budget          = req.budget,
            competitors     = req.competitors,
            campaign_type   = req.campaign_type,
        )

        task["langgraph_plan"] = langgraph_plan
        _log("Phase 1 complete – preliminary brief ready for review.")

        # ── HITL pause ────────────────────────────────────────────────────────
        hitl_entry = {
            "type":            "hitl_pause",
            "task_id":         task_id,
            "plan_preview":    langgraph_plan.get("analysis_plan", "")[:800],
            "preliminary_report": langgraph_plan.get("preliminary_report", "")[:3000],
            "auto_approve_secs": 600,
        }
        task["status"] = "awaiting_approval"
        log_queues[task_id].put_nowait(hitl_entry)

        _log("Waiting for human approval (10 min auto-approve)…", "hitl_wait")
        approved = hitl_events[task_id].wait(timeout=600)

        if hitl_abort.get(task_id, False):
            task["status"]       = "aborted"
            task["hitl_aborted"] = True
            task["completed_at"] = datetime.now().isoformat()
            log_queues[task_id].put_nowait({"type": "aborted", "message": "Campaign aborted by user."})
            return

        if not approved:
            _log("HITL timeout – auto-approving Phase 2.")

        task["status"] = "running"

        # ── Phase 2: CrewAI 4-agent execution ─────────────────────────────────
        _log("═══ Phase 2: CrewAI Multi-Agent Execution ═══")

        campaign_request = (
            f"Campaign for {req.brand_name} ({req.industry})\n"
            f"Type: {req.campaign_type}\n"
            f"Goals: {req.campaign_goals}\n"
            f"Target Audience: {req.target_audience}\n"
            f"Budget: {req.budget}\n"
            f"Competitors: {req.competitors}"
        )

        from crew import run_crewai_phase
        manager_report = run_crewai_phase(campaign_request, langgraph_plan)

        task["markdown_report"] = str(manager_report)

        # ── Post-processing: Gemini enrichment ────────────────────────────────
        _log("Enriching executive brief with Gemini…")
        try:
            from config import gemini_llm
            from langchain_core.messages import HumanMessage, SystemMessage

            if gemini_llm:
                enrich_resp = gemini_llm.invoke([
                    SystemMessage(content=(
                        "You are a Chief Marketing Officer. Rewrite and enrich the following "
                        "campaign brief. Target 700–900 words. Keep all sections but add "
                        "industry-specific insights, sharper KPI targets, and a compelling "
                        "executive narrative. Format as polished markdown."
                    )),
                    HumanMessage(content=str(manager_report)[:3000]),
                ])
                task["enhanced_report"] = enrich_resp.content
                _log("Executive brief enrichment complete.")
        except Exception as _e:
            _log(f"Enrichment skipped – {_e}")
            task["enhanced_report"] = task["markdown_report"]

        # ── Load session content sidecar ──────────────────────────────────────
        _session_file = _CONTENT / "session_content.json"
        if _session_file.exists():
            try:
                raw_entries = json.loads(_session_file.read_text(encoding="utf-8"))
                seen: set   = set()
                deduped: list = []
                for entry in raw_entries:
                    key = f"{entry.get('type','')}::{entry.get('title','')}"
                    if key not in seen:
                        seen.add(key)
                        deduped.append(entry)
                task["content_items"] = deduped
            except Exception as _e:
                _log(f"Could not load session content – {_e}")

        # ── Collect report file paths ─────────────────────────────────────────
        task["reports"] = [
            f"/outputs/reports/{p.name}"
            for p in sorted(_REPORTS.iterdir())
            if p.suffix.lower() in {".pdf", ".pptx", ".md"}
        ]
        task["videos"] = [
            f"/outputs/videos/{p.name}"
            for p in sorted(_VIDEOS.iterdir())
            if p.suffix.lower() == ".mp4"
        ]

        task["status"]       = "completed"
        task["completed_at"] = datetime.now().isoformat()
        _log("═══ Campaign analysis complete ═══")
        log_queues[task_id].put_nowait({"type": "complete", "task_id": task_id})

    except Exception as exc:
        import traceback
        err_msg = f"Pipeline error: {exc}\n{traceback.format_exc()}"
        task["status"] = "failed"
        task["error"]  = err_msg
        log_queues[task_id].put_nowait({"type": "error", "message": err_msg})
        print(err_msg, file=orig_stdout)

    finally:
        sys.stdout = orig_stdout


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = _STATIC / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>UI not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/analyze")
async def start_analysis(req: MarketingRequest):
    task_id = str(uuid.uuid4())

    tasks[task_id] = {
        "status":          "pending",
        "brand_name":      req.brand_name,
        "industry":        req.industry,
        "campaign_type":   req.campaign_type,
        "started_at":      "",
        "completed_at":    "",
        "logs":            [],
        "videos":          [],
        "content_items":   [],
        "reports":         [],
        "markdown_report": "",
        "enhanced_report": "",
        "conversation":    [],
        "hitl_aborted":    False,
        "langgraph_plan":  {},
    }
    log_queues[task_id]  = Queue()
    hitl_events[task_id] = threading.Event()
    hitl_abort[task_id]  = False

    thread = threading.Thread(target=_run_analysis, args=(task_id, req), daemon=True)
    thread.start()

    return {"task_id": task_id, "status": "started"}


@app.get("/api/stream/{task_id}")
async def stream_logs(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    async def _event_generator():
        q = log_queues[task_id]
        while True:
            try:
                entry = q.get(timeout=1)
                yield f"data: {json.dumps(entry)}\n\n"
                if entry.get("type") in {"complete", "error", "aborted"}:
                    break
            except Empty:
                # heartbeat
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                status = tasks[task_id].get("status", "")
                if status in {"completed", "failed", "aborted"}:
                    break
            await asyncio.sleep(0)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    t = tasks[task_id]
    return {
        "task_id":    task_id,
        "status":     t["status"],
        "brand_name": t["brand_name"],
        "started_at": t["started_at"],
        "log_count":  len(t["logs"]),
    }


@app.get("/api/results/{task_id}")
async def get_results(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    t = tasks[task_id]
    return {
        "task_id":        task_id,
        "status":         t["status"],
        "brand_name":     t["brand_name"],
        "markdown_report": t.get("markdown_report", ""),
        "enhanced_report": t.get("enhanced_report", ""),
        "content_items":  t.get("content_items", []),
        "videos":         t.get("videos", []),
        "reports":        t.get("reports", []),
        "hitl_aborted":   t.get("hitl_aborted", False),
    }


@app.get("/api/logs/{task_id}")
async def get_logs(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "logs": tasks[task_id]["logs"]}


@app.get("/api/videos")
async def list_videos():
    files = sorted(_VIDEOS.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {
            "name":     p.name,
            "url":      f"/outputs/videos/{p.name}",
            "size_mb":  round(p.stat().st_size / 1_048_576, 2),
            "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
        }
        for p in files
    ]


@app.get("/api/reports")
async def list_reports():
    files = sorted(
        [p for p in _REPORTS.iterdir() if p.suffix.lower() in {".pdf", ".pptx", ".md"}],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    ext_icons = {".pdf": "📄", ".pptx": "📊", ".md": "📝"}
    return [
        {
            "name":     p.name,
            "url":      f"/outputs/reports/{p.name}",
            "type":     p.suffix.lstrip(".").upper(),
            "icon":     ext_icons.get(p.suffix.lower(), "📁"),
            "size_kb":  round(p.stat().st_size / 1024, 1),
            "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
        }
        for p in files
    ]


@app.get("/api/tasks")
async def list_tasks():
    return [
        {
            "task_id":    tid,
            "status":     t["status"],
            "brand_name": t["brand_name"],
            "started_at": t["started_at"],
        }
        for tid, t in tasks.items()
    ]


@app.post("/api/approve/{task_id}")
async def approve_task(task_id: str, req: ApproveRequest):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    if tasks[task_id]["status"] != "awaiting_approval":
        raise HTTPException(status_code=400, detail="Task is not awaiting approval")

    hitl_abort[task_id] = req.abort
    hitl_events[task_id].set()

    action = "aborted" if req.abort else "approved"
    return {"task_id": task_id, "action": action}


@app.post("/api/chat/{task_id}")
async def chat(task_id: str, req: ChatRequest):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    message = req.message.strip()
    security_flags: list[str] = []

    # ── Security layer 1: SQL injection ──────────────────────────────────────
    if _check_sql(message):
        _audit_log({
            "event":   "sql_blocked",
            "task_id": task_id,
            "message": message,
            "ts":      datetime.now().isoformat(),
        })
        return {
            "reply":          "Blocked: SQL injection detected.",
            "security_flags": ["SQL Injection Blocked"],
        }

    # ── Security layer 2: Guardrails ──────────────────────────────────────────
    if _check_guardrails(message):
        _audit_log({
            "event":   "guardrail_blocked",
            "task_id": task_id,
            "message": message,
            "ts":      datetime.now().isoformat(),
        })
        return {
            "reply":          "Blocked: Request violates marketing assistant guidelines.",
            "security_flags": ["Guardrails Triggered"],
        }

    # ── Security layer 3: PII redaction ──────────────────────────────────────
    clean_msg, pii_found = _redact_pii(message)
    if pii_found:
        security_flags.append("PII Redacted")

    # ── Security layer 4: Audit log ───────────────────────────────────────────
    _audit_log({
        "event":   "chat_message",
        "task_id": task_id,
        "message": clean_msg,
        "ts":      datetime.now().isoformat(),
    })

    # ── Gemini chat ───────────────────────────────────────────────────────────
    t = tasks[task_id]
    context = (
        t.get("enhanced_report") or t.get("markdown_report") or
        t.get("langgraph_plan", {}).get("preliminary_report", "")
    )

    try:
        from config import gemini_llm
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        if gemini_llm is None:
            raise RuntimeError("Gemini not available")

        system_msg = SystemMessage(content=(
            "You are a Digital Marketing AI Assistant helping analyse a marketing campaign. "
            "Answer questions based on the campaign context provided. "
            "Be concise, professional, and marketing-focused. "
            "Do NOT discuss topics outside digital marketing, campaigns, and strategy. "
            f"\n\nCampaign Context:\n{context[:3000]}"
        ))

        # build history
        history_msgs = [system_msg]
        for turn in t["conversation"][-6:]:
            if turn["role"] == "user":
                history_msgs.append(HumanMessage(content=turn["content"]))
            else:
                history_msgs.append(AIMessage(content=turn["content"]))
        history_msgs.append(HumanMessage(content=clean_msg))

        resp  = gemini_llm.invoke(history_msgs)
        reply = resp.content if hasattr(resp, "content") else str(resp)

    except Exception as exc:
        reply = f"[Chat error: {exc}]"

    # store conversation
    t["conversation"].append({"role": "user",      "content": clean_msg})
    t["conversation"].append({"role": "assistant",  "content": reply})

    _audit_log({
        "event":   "chat_reply",
        "task_id": task_id,
        "reply":   reply[:500],
        "ts":      datetime.now().isoformat(),
    })

    return {"reply": reply, "security_flags": security_flags}


@app.get("/api/chat/history/{task_id}")
async def chat_history(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"conversation": tasks[task_id]["conversation"]}
