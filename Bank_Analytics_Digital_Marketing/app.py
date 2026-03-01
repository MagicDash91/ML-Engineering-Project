"""
app.py – Combined Banking Analytics + Digital Marketing FastAPI Backend
=======================================================================

Three-phase pipeline:
  Phase 1 → LangGraph banking planning (5 nodes, Gemini)
  HITL    → Human approval checkpoint
  Phase 2 → Banking CrewAI (5 agents: Engineer, Scientist, LabelEncoder, Analyst, CDO)
             + Gemini enrichment of banking report
  Phase 3 → Digital Marketing CrewAI (4 agents) fed banking churn context

7-tab Bootstrap 5 dark UI:
  Console | Banking Churn Report | Charts | Marketing Report | Content | Downloads | Conversation

Run:
    python main.py   →  http://localhost:8002
"""

import os
import sys

os.environ["OTEL_SDK_DISABLED"]         = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

import re
import json
import uuid
import queue
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

for _d in ("outputs/charts", "outputs/reports", "outputs/models",
           "outputs/videos", "outputs/content", "outputs/audit", "static"):
    (BASE_DIR / _d).mkdir(parents=True, exist_ok=True)

AUDIT_DIR = BASE_DIR / "outputs" / "audit"

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Bank Analytics + Digital Marketing Multi-Agent System",
    description="Combined banking analytics and digital marketing AI crew",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

app.mount("/static",  StaticFiles(directory=str(BASE_DIR / "static")),  name="static")
app.mount("/outputs", StaticFiles(directory=str(BASE_DIR / "outputs")), name="outputs")

# ── In-memory store ───────────────────────────────────────────────────────────
tasks: dict       = {}
log_queues: dict  = {}
hitl_events: dict = {}
_analysis_lock    = threading.Lock()

# ── Security patterns ────────────────────────────────────────────────────────
_PII_PATTERNS = [
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), '[EMAIL]'),
    (re.compile(r'(\+62|62|0)[\s.\-]?8[1-9][0-9]{6,10}\b'), '[PHONE]'),
    (re.compile(r'\b\d{16}\b'), '[NIK/ACCT]'),
    (re.compile(r'\b(?:\d{4}[\s\-]?){3}\d{4}\b'), '[CARD]'),
    (re.compile(r'\b\d{10,15}\b'), '[ACCT]'),
]

_SQL_RE = re.compile(
    r'(union\s+(all\s+)?select)|(drop\s+(table|database))'
    r'|(insert\s+into\s+\w)|(delete\s+from\s+\w)|(update\s+\w+\s+set\s)'
    r'|(exec\s*\(|xp_cmdshell)|(\bor\b\s+[\'"]?\d[\'"]?\s*=\s*[\'"]?\d)'
    r'|(--\s*[\r\n]|;\s*--)|(\/\*[\s\S]*?\*\/)',
    re.IGNORECASE,
)

_GUARDRAIL_RE = re.compile(
    r'\b(transfer|send|wire)\s+(money|fund|cash|rp|usd|idr)'
    r'|\b(withdraw|deposit)\s+(from|to|my)'
    r'|\bmy\s+(account\s+(number|balance|pin|password)|card\s+number)'
    r'|\b(ignore|forget|disregard)\s+(previous|all|above|your)\s+(instruction|rule|prompt)'
    r'|\b(jailbreak|prompt.injection|bypass.filter|bypass.security)'
    r'|\b(act\s+as\s+a\s+|you\s+are\s+now\s+a|pretend\s+(you\s+are|to\s+be))'
    r'|\b(override|reset)\s+your\s+(system|instruction|role)',
    re.IGNORECASE,
)


def _redact_pii(text: str):
    found = False
    for pattern, placeholder in _PII_PATTERNS:
        new_text = pattern.sub(placeholder, text)
        if new_text != text:
            found = True
            text = new_text
    return text, found


def _is_sql_injection(text: str) -> bool:
    return bool(_SQL_RE.search(text))


def _is_guardrail_violation(text: str) -> bool:
    return bool(_GUARDRAIL_RE.search(text))


def _audit_log(entry: dict):
    try:
        with open(AUDIT_DIR / "conversation_audit.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ── Pydantic models ───────────────────────────────────────────────────────────

class CombinedRequest(BaseModel):
    # Banking inputs
    bank_symbols:     str  = "BBCA,BMRI"
    analysis_request: str  = ""
    use_langgraph:    bool = True
    # Marketing inputs
    brand_name:       str = "Bank Brand"
    industry:         str = "Banking & Financial Services"
    target_audience:  str = "Banking customers aged 25-55 at risk of churn"
    campaign_goals:   str = "Reduce customer churn by 20%, improve retention and loyalty"
    budget:           str = "Not specified"
    competitors:      str = "Not specified"
    campaign_type:    str = "Customer Retention"


class ChatRequest(BaseModel):
    message: str


class ApproveRequest(BaseModel):
    abort: bool = False


# ── Stdout capture ────────────────────────────────────────────────────────────

class _TeeWriter:
    def __init__(self, task_id: str, original):
        self._tid      = task_id
        self._original = original

    def write(self, text: str):
        self._original.write(text)
        self._original.flush()
        stripped = text.rstrip()
        if stripped:
            entry = {"type": "log", "message": stripped, "timestamp": datetime.now().isoformat()}
            tasks[self._tid]["logs"].append(entry)
            try:
                log_queues[self._tid].put_nowait(entry)
            except Exception:
                pass

    def flush(self):
        self._original.flush()

    def isatty(self):
        return False


def _push(task_id: str, msg: str, kind: str = "log"):
    entry = {"type": kind, "message": msg, "timestamp": datetime.now().isoformat()}
    tasks[task_id]["logs"].append(entry)
    try:
        log_queues[task_id].put_nowait(entry)
    except Exception:
        pass


def _strip_md_fences(text: str) -> str:
    text = re.sub(r'^```(?:markdown|md)?\s*\n', '', text.strip())
    text = re.sub(r'\n```\s*$', '', text.strip())
    text = re.sub(r'\n```(?:markdown|md)?\n[\s\S]*?```(?:\n|$)', '\n', text)
    return text.strip()


def _default_banking_request() -> str:
    return (
        "Perform comprehensive banking customer analytics on the 'churn' table in PostgreSQL.\n\n"
        "1. DATA ENGINEERING: Profile the existing 'churn' table (22 columns: customerID, gender, "
        "SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, "
        "OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, "
        "Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn, "
        "transaction_date). Validate data quality.\n"
        "2. DATA SCIENCE: Train a customer churn prediction model (target='Churn'), "
        "perform customer segmentation (4 clusters), report AUC-ROC and top churn predictors.\n"
        "3. DATA ANALYSIS: Create individual charts — churn distribution pie, tenure histogram, "
        "monthly charges histplot by Churn, contract type bar chart, tenure vs charges scatter, "
        "feature correlation heatmap (use churn_encoded table). "
        "Analyse every chart with Gemini AI vision. Do NOT generate a dashboard.\n"
        "4. REPORTS: Produce PDF and PowerPoint executive reports.\n"
        "5. EXECUTIVE SUMMARY: Top 5 churn risk findings and customer retention recommendations."
    )


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


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _run_analysis(task_id: str, req: CombinedRequest):
    original_stdout = sys.stdout
    sys.stdout = _TeeWriter(task_id, original_stdout)

    try:
        _push(task_id, "━" * 62, "info")
        _push(task_id, "  BANK ANALYTICS + DIGITAL MARKETING SYSTEM  —  starting", "info")
        _push(task_id, "━" * 62, "info")

        # ── Clear session sidecar files ────────────────────────────────────────
        for _sidecar in (
            BASE_DIR / "outputs" / "charts" / "session_charts.json",
            BASE_DIR / "outputs" / "content" / "session_content.json",
        ):
            try:
                if _sidecar.exists():
                    _sidecar.unlink()
            except Exception:
                pass

        analysis_request = req.analysis_request.strip() or _default_banking_request()
        bank_symbols     = [s.strip() for s in req.bank_symbols.split(",") if s.strip()]

        # Derive brand_name from bank_symbols when the user hasn't set one
        _brand_name = req.brand_name
        if _brand_name == "Bank Brand" and bank_symbols:
            _brand_name = " & ".join(bank_symbols[:2])
        tasks[task_id]["brand_name"] = _brand_name

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 1 — LangGraph Banking Planning
        # ══════════════════════════════════════════════════════════════════════
        langgraph_plan = None
        if req.use_langgraph:
            print("\n" + "=" * 62)
            print("  PHASE 1 — LangGraph Banking Strategic Planning")
            print("=" * 62)

            from graphs.banking_graph import run_banking_analysis as _lg_run
            graph_state = _lg_run(analysis_request, bank_symbols)

            langgraph_plan = {
                "analysis_plan":      graph_state.get("analysis_plan", ""),
                "etl_guidance":       graph_state.get("etl_guidance", ""),
                "ml_guidance":        graph_state.get("ml_guidance", ""),
                "analytics_guidance": graph_state.get("analytics_guidance", ""),
                "preliminary_report": graph_state.get("report_content", ""),
            }
            tasks[task_id]["langgraph_plan"] = langgraph_plan
            print("\n  LangGraph banking planning complete.\n")

            # ── HITL Pause ─────────────────────────────────────────────────────
            _push(task_id, "━" * 58, "info")
            _push(task_id, "  HUMAN IN THE LOOP — Review the strategic plan", "info")
            _push(task_id, "  Approve in the UI to continue to Phase 2.", "info")
            _push(task_id, "  Auto-approving in 10 minutes if no response.", "info")
            _push(task_id, "━" * 58, "info")

            hitl_entry = {
                "type":               "hitl_pause",
                "task_id":            task_id,
                "plan_preview":       langgraph_plan.get("analysis_plan", "")[:800],
                "preliminary_report": langgraph_plan.get("preliminary_report", "")[:2000],
                "timestamp":          datetime.now().isoformat(),
                "auto_approve_secs":  600,
            }
            tasks[task_id]["logs"].append(hitl_entry)
            try:
                log_queues[task_id].put_nowait(hitl_entry)
            except Exception:
                pass

            tasks[task_id]["status"] = "awaiting_approval"
            approved = hitl_events[task_id].wait(timeout=600)

            if tasks[task_id].get("hitl_aborted", False):
                raise RuntimeError("Analysis aborted by user during HITL review.")

            if not approved:
                _push(task_id, "  HITL: 10-minute timeout — auto-approving.", "warn")
            else:
                _push(task_id, "  HITL: Approved — proceeding to Phase 2.", "success")

            tasks[task_id]["status"] = "running"

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 2 — Banking CrewAI
        # ══════════════════════════════════════════════════════════════════════
        print("\n" + "=" * 62)
        print("  PHASE 2 — Banking CrewAI Multi-Agent Execution")
        print("=" * 62)

        from crew_banking import run_banking_crew
        banking_result      = run_banking_crew(analysis_request, langgraph_plan)
        banking_crew_output = banking_result.get("crew_output", "")
        analyst_output      = banking_result.get("analyst_output", "")
        scientist_output    = banking_result.get("scientist_output", "")

        # Load banking blog sections (charts with Gemini analysis)
        banking_blog_sections: list = []
        _meta_file = BASE_DIR / "outputs" / "charts" / "session_charts.json"
        if _meta_file.exists():
            try:
                banking_blog_sections = json.loads(_meta_file.read_text(encoding="utf-8"))
                # Deduplicate
                _seen: set = set()
                _deduped: list = []
                for _s in banking_blog_sections:
                    _url = _s.get("chart_url", "")
                    if _url not in _seen:
                        _seen.add(_url)
                        _deduped.append(_s)
                banking_blog_sections = _deduped
                _push(task_id, f"  Banking charts loaded: {len(banking_blog_sections)}", "info")
            except Exception as _e:
                _push(task_id, f"  Banking chart metadata read failed: {_e}", "warn")

        banking_charts  = _list_files("outputs/charts", [".png"])
        banking_reports = _list_files("outputs/reports", [".pdf", ".pptx", ".md"])

        # Build banking markdown report
        banking_report = banking_crew_output if len(banking_crew_output) > 50 else \
            langgraph_plan.get("preliminary_report", "# Banking Analysis Complete") if langgraph_plan else \
            "# Banking Analysis Complete"

        # ── Gemini enrichment of banking report ──────────────────────────────
        banking_enhanced = banking_report
        try:
            from config import gemini_llm
            from langchain_core.messages import HumanMessage

            if gemini_llm and banking_report and len(banking_report) > 100:
                chart_analyses = "\n\n".join(
                    f"**{s['title']}**:\n{s['gemini_analysis']}"
                    for s in banking_blog_sections
                    if s.get("gemini_analysis")
                )
                enhance_prompt = (
                    "You are a senior banking strategy consultant preparing a C-suite briefing.\n\n"
                    "## Draft Banking Report:\n" + banking_report + "\n\n"
                    + (f"## Chart Evidence:\n{chart_analyses}\n\n" if chart_analyses else "")
                    + "## Your Task — Targeted Enrichment:\n"
                    "Produce the final CEO-ready version (600–900 words). "
                    "Keep Executive Summary max 4 bullets. "
                    "Keep Key Findings max 5 points with specific numbers. "
                    "Enrich Recommendations with Owner|KPI Target|30-day|60-day|90-day table. "
                    "Replace Next Steps with numbered action plan [N]. Action — Owner — Deadline — Metric. "
                    "Output the complete report only — no preamble."
                )
                _push(task_id, "  Gemini: enriching banking report…", "info")
                response = gemini_llm.invoke([HumanMessage(content=enhance_prompt)])
                enhanced = _strip_md_fences(response.content)
                if enhanced and len(enhanced) > 200:
                    banking_enhanced = enhanced
                    _push(task_id, "  Gemini: banking report enhanced ✓", "success")
        except Exception as _gem_err:
            _push(task_id, f"  Gemini enrichment (non-critical): {_gem_err}", "warn")

        tasks[task_id].update({
            "banking_report":        banking_enhanced,
            "banking_blog_sections": banking_blog_sections,
            "banking_charts":        banking_charts,
            "analyst_output":        analyst_output,
            "scientist_output":      scientist_output,
        })

        _push(task_id, "━" * 62, "success")
        _push(task_id, f"  Phase 2 complete! Charts: {len(banking_charts)}  Reports: {len(banking_reports)}", "success")

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 3 — Marketing LangGraph Planning + Digital Marketing CrewAI
        # ══════════════════════════════════════════════════════════════════════
        print("\n" + "=" * 62)
        print("  PHASE 3 — Marketing LangGraph + CrewAI Execution")
        print("=" * 62)
        _push(task_id, "━" * 62, "info")
        _push(task_id, "  PHASE 3 — CDO handing analyst findings to Marketing team…", "info")
        _push(task_id, "━" * 62, "info")

        # Banking context passed to all marketing agents.
        # Priority: Data Analyst raw output first, then CDO executive summary as supplement.
        if analyst_output:
            banking_context = (
                f"[DATA ANALYST FINDINGS]:\n{analyst_output[:2000]}\n\n"
                f"[CDO EXECUTIVE SUMMARY]:\n{banking_enhanced[:1000]}"
            )
        else:
            banking_context = banking_enhanced[:2000] if banking_enhanced else ""

        # ── Build chart analysis summary (Data Analyst's findings) ───────────
        _chart_analyses = "\n\n".join(
            f"**{s['title']}**:\n{s.get('gemini_analysis', '')}"
            for s in banking_blog_sections
            if s.get("gemini_analysis")
        )[:2500]

        # ── Generate data-driven campaign brief from Data Analyst output ─────
        # The campaign request is NOT user-typed — it comes entirely from the
        # banking analytics: churn model findings, customer segments, and the
        # Data Analyst's chart insights. The marketing team uses this to design
        # specific, named campaigns (cashback, loyalty programs, etc.).
        _push(task_id, "  Translating analyst findings → data-driven campaign brief…", "info")
        campaign_request = ""
        derived_audience = req.target_audience or "Banking customers at risk of churn"
        derived_goals    = req.campaign_goals   or "Reduce customer churn, improve retention"

        try:
            from config import gemini_llm
            from langchain_core.messages import HumanMessage as _HMsg

            if gemini_llm and banking_enhanced:
                # Build the analyst evidence block — use raw analyst output first,
                # then chart analyses, then scientist ML results as supporting data
                _analyst_section   = f"## Data Analyst Report:\n{analyst_output[:2000]}\n\n"   if analyst_output   else ""
                _scientist_section = f"## Data Scientist ML Results:\n{scientist_output[:1000]}\n\n" if scientist_output else ""
                _chart_section     = f"## Data Analyst Chart Insights:\n{_chart_analyses}\n\n"  if _chart_analyses  else ""

                _handoff_prompt = (
                    f"You are a Chief Data Officer handing off customer analytics results "
                    f"to the Digital Marketing team.\n\n"
                    f"Brand: {_brand_name} | Industry: {req.industry} | Budget: {req.budget}\n\n"
                    "Write a detailed **Digital Marketing Campaign Brief** based strictly on "
                    "the banking Data Analyst's findings below. The brief must include:\n\n"
                    "1. **At-Risk Segments** — name each segment with specific numbers from the data "
                    "(e.g. '34% of month-to-month contract holders churned', "
                    "'senior citizens with low tech-support uptake — 41% churn rate')\n\n"
                    "2. **Top Churn Drivers** — list the main predictors for each segment "
                    "(contract type, charges, tenure, service uptake, etc.)\n\n"
                    "3. **Specific Campaign Ideas (4–6 ideas)** — give each campaign a concrete name "
                    "and mechanic. Use creative, actionable campaign concepts such as:\n"
                    "   - *'Cashback 5% untuk nasabah yang mendaftar autopayment'*\n"
                    "   - *'Gebyar Berhadiah — program poin loyalitas untuk nasabah setia >12 bulan'*\n"
                    "   - *'Upgrade & Hemat — diskon biaya admin 3 bulan untuk upgrade ke kontrak tahunan'*\n"
                    "   - *'Paket Peduli Senior — layanan prioritas + cashback tagihan untuk nasabah senior'*\n"
                    "   - *'Digital Adoption Bonus — reward untuk nasabah yang aktifkan layanan digital security'*\n"
                    "   Invent campaigns that directly address the actual churn drivers found in the data.\n\n"
                    "4. **Priority Channels** — which channels for which segment "
                    "(e.g. WhatsApp blast for seniors, Instagram retargeting for young churners)\n\n"
                    "5. **Retention Targets** — specific measurable goals "
                    "(e.g. 'Reduce month-to-month churn from 42% to 28% in 90 days')\n\n"
                    + _analyst_section
                    + _scientist_section
                    + _chart_section
                    + f"## CDO Executive Summary:\n{banking_enhanced[:1500]}\n\n"
                    + "Write the brief in English. Be specific, data-driven, and creative. "
                    "This document is the primary input for the Digital Marketing creative team."
                )
                _resp = gemini_llm.invoke([_HMsg(content=_handoff_prompt)])
                _raw_brief = _resp.content if hasattr(_resp, "content") else ""
                campaign_request = _strip_md_fences(_raw_brief)
                if campaign_request and len(campaign_request) > 200:
                    _push(task_id, "  Analyst → Marketing handoff brief generated ✓", "success")
                    # Derive audience and goals from the analysis-generated brief
                    derived_audience = "At-risk banking customers identified by churn model (see brief)"
                    derived_goals    = "Reduce customer churn via data-driven targeted campaigns"
                else:
                    campaign_request = ""
        except Exception as _br_err:
            _push(task_id, f"  Campaign brief generation (non-critical): {_br_err}", "warn")

        # Fallback — if Gemini brief generation failed, build a structured fallback
        if not campaign_request or len(campaign_request) < 200:
            campaign_request = (
                f"Create a comprehensive digital marketing campaign for {_brand_name} "
                f"in the {req.industry} industry.\n\n"
                f"Campaign Type: {req.campaign_type} | Budget: {req.budget}\n\n"
                "Design specific, named campaign initiatives targeting the at-risk customer "
                "segments identified in the banking churn analysis below. Include concrete "
                "campaign mechanics (cashback offers, loyalty programs, upgrade incentives, "
                "digital adoption rewards). Each campaign should directly address a churn driver.\n\n"
                f"Banking Analysis Context:\n{banking_context[:3000]}"
            )
            _push(task_id, "  Using structured fallback campaign brief.", "warn")

        tasks[task_id]["campaign_brief"] = campaign_request

        # ── Phase 3a: Marketing LangGraph pre-planning (5 nodes, Gemini) ─────
        _push(task_id, "  Phase 3a — Marketing LangGraph: 5-node Gemini planning…", "info")
        from graphs.marketing_graph import run_marketing_analysis as _mkt_lg_run
        mkt_plan = _mkt_lg_run(
            task_description = campaign_request,
            brand_name       = _brand_name,
            industry         = req.industry,
            target_audience  = derived_audience,
            campaign_goals   = derived_goals,
            budget           = req.budget,
            competitors      = req.competitors,
            campaign_type    = req.campaign_type,
            banking_context  = banking_context,
        )
        tasks[task_id]["marketing_langgraph_plan"] = mkt_plan
        _push(task_id, "  Phase 3a complete — Marketing LangGraph planning done ✓", "success")

        # ── Phase 3b: Marketing CrewAI (4 agents) ────────────────────────────
        _push(task_id, "  Phase 3b — Marketing CrewAI: 4-agent sequential execution…", "info")
        from crew_marketing import run_marketing_crew
        marketing_result = run_marketing_crew(
            campaign_request  = campaign_request,
            brand_name        = _brand_name,
            industry          = req.industry,
            target_audience   = derived_audience,
            campaign_goals    = derived_goals,
            budget            = req.budget,
            competitors       = req.competitors,
            campaign_type     = req.campaign_type,
            banking_context   = banking_context,
            langgraph_plan    = mkt_plan,
        )

        # Load marketing content sidecar
        marketing_content_items: list = []
        _content_file = BASE_DIR / "outputs" / "content" / "session_content.json"
        if _content_file.exists():
            try:
                all_entries = json.loads(_content_file.read_text(encoding="utf-8"))
                _seen_keys: set = set()
                for entry in all_entries:
                    key = f"{entry.get('type', '')}-{entry.get('title', '')}"
                    if key not in _seen_keys:
                        _seen_keys.add(key)
                        marketing_content_items.append(entry)
                _push(task_id, f"  Marketing content items loaded: {len(marketing_content_items)}", "info")
            except Exception as _e:
                _push(task_id, f"  Marketing content load failed: {_e}", "warn")

        # All downloads (banking + marketing reports + videos)
        all_reports = _list_files("outputs/reports", [".pdf", ".pptx", ".md"])
        all_videos  = [item for item in marketing_content_items if item.get("type") == "video"]

        tasks[task_id].update({
            "status":                  "completed",
            "completed_at":            datetime.now().isoformat(),
            "marketing_report":        marketing_result,
            "marketing_content_items": marketing_content_items,
            "all_reports":             all_reports,
            "all_videos":              all_videos,
        })

        _push(task_id, "━" * 62, "success")
        _push(task_id, "  All 3 phases complete! Banking + Marketing analysis done.", "success")
        _push(task_id, f"  Reports: {len(all_reports)}  Videos: {len(all_videos)}  Content items: {len(marketing_content_items)}", "success")
        _push(task_id, "━" * 62, "success")

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
        done_type = "complete" if tasks[task_id]["status"] == "completed" else (
            "aborted" if tasks[task_id].get("hitl_aborted") else "error"
        )
        try:
            log_queues[task_id].put_nowait({
                "type": done_type, "task_id": task_id,
                "timestamp": datetime.now().isoformat()
            })
        except Exception:
            pass
        hitl_events.pop(task_id, None)


def _run_with_lock(task_id: str, req: CombinedRequest):
    with _analysis_lock:
        _run_analysis(task_id, req)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.post("/api/analyze")
async def start_analysis(req: CombinedRequest):
    """Start a new combined analysis run (background thread)."""
    if _analysis_lock.locked():
        raise HTTPException(status_code=409, detail="An analysis is already running.")

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status":                  "running",
        "started_at":              datetime.now().isoformat(),
        "completed_at":            None,
        "brand_name":              req.brand_name,
        "bank_symbols":            req.bank_symbols,
        "error":                   None,
        "logs":                    [],
        "langgraph_plan":            {},
        "marketing_langgraph_plan": {},
        "campaign_brief":           "",
        "hitl_aborted":            False,
        # Banking outputs
        "banking_report":          "",
        "banking_blog_sections":   [],
        "banking_charts":          [],
        # Marketing outputs
        "marketing_report":        "",
        "marketing_content_items": [],
        "all_reports":             [],
        "all_videos":              [],
        # Conversation
        "conversation":            [],
    }
    log_queues[task_id]  = queue.Queue()
    hitl_events[task_id] = threading.Event()

    thread = threading.Thread(target=_run_with_lock, args=(task_id, req), daemon=True)
    thread.start()

    return {"task_id": task_id, "status": "started"}


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found.")
    t = tasks[task_id]
    return {
        "task_id":      task_id,
        "status":       t["status"],
        "started_at":   t["started_at"],
        "completed_at": t.get("completed_at"),
        "error":        t.get("error"),
        "log_lines":    len(t.get("logs", [])),
    }


@app.get("/api/results/{task_id}")
async def get_results(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found.")
    t = tasks[task_id]
    if t["status"] == "running":
        raise HTTPException(status_code=202, detail="Analysis still running.")
    if t["status"] == "error":
        raise HTTPException(status_code=500, detail=t.get("error", "Unknown error."))

    return {
        "task_id":                 task_id,
        "status":                  "completed",
        "banking_report":          t.get("banking_report", ""),
        "banking_blog_sections":   t.get("banking_blog_sections", []),
        "banking_charts":          t.get("banking_charts", []),
        "marketing_report":        t.get("marketing_report", ""),
        "campaign_brief":          t.get("campaign_brief", ""),
        "marketing_content_items": t.get("marketing_content_items", []),
        "all_reports":             t.get("all_reports", []),
        "all_videos":              t.get("all_videos", []),
        "langgraph_plan":          t.get("langgraph_plan", {}),
        "completed_at":            t.get("completed_at"),
    }


@app.get("/api/logs/{task_id}")
async def get_logs(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found.")
    return {"task_id": task_id, "logs": tasks[task_id].get("logs", [])}


@app.get("/api/stream/{task_id}")
async def stream_events(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found.")

    async def generator():
        q = log_queues[task_id]
        buffered = list(tasks[task_id].get("logs", []))
        replayed = len(buffered)
        for entry in buffered:
            yield f"data: {json.dumps(entry)}\n\n"

        drained = 0
        while drained < replayed:
            try:
                q.get_nowait()
                drained += 1
            except queue.Empty:
                break

        while True:
            try:
                msg = q.get_nowait()
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ("complete", "error", "aborted"):
                    return
            except queue.Empty:
                if tasks[task_id]["status"] in ("completed", "error", "aborted"):
                    final = "complete" if tasks[task_id]["status"] == "completed" else "error"
                    yield f"data: {json.dumps({'type': final, 'task_id': task_id})}\n\n"
                    return
                await asyncio.sleep(0.35)

    return StreamingResponse(generator(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no",
                                       "Connection": "keep-alive"})


@app.get("/api/charts")
async def list_charts():
    return {"charts": _list_files("outputs/charts", [".png"])}


@app.get("/api/reports")
async def list_reports():
    return {"reports": _list_files("outputs/reports", [".pdf", ".pptx", ".md"])}


@app.get("/api/videos")
async def list_videos():
    return {"videos": _list_files("outputs/videos", [".mp4"])}


@app.get("/api/tasks")
async def list_tasks():
    return [
        {"task_id": tid, "status": t["status"],
         "started_at": t["started_at"], "completed_at": t.get("completed_at"),
         "brand_name": t.get("brand_name", "")}
        for tid, t in tasks.items()
    ]


@app.post("/api/approve/{task_id}")
async def approve_hitl(task_id: str, req: ApproveRequest = None):
    if req is None:
        req = ApproveRequest()
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found.")
    if tasks[task_id]["status"] != "awaiting_approval":
        raise HTTPException(status_code=400, detail="Task is not awaiting approval.")
    if task_id not in hitl_events:
        raise HTTPException(status_code=400, detail="No HITL event found for this task.")

    if req.abort:
        tasks[task_id]["hitl_aborted"] = True

    hitl_events[task_id].set()
    return {"task_id": task_id, "aborted": req.abort}


@app.post("/api/chat/{task_id}")
async def chat(task_id: str, req: ChatRequest):
    """Chat with Gemini using context from BOTH banking AND marketing results."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found.")
    t = tasks[task_id]
    if t["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not yet completed.")

    message = req.message.strip()
    if len(message) > 2000:
        raise HTTPException(status_code=400, detail="Message too long (max 2000 characters).")
    if _is_sql_injection(message):
        _audit_log({"timestamp": datetime.now().isoformat(), "task_id": task_id,
                    "flags": ["sql_injection"], "blocked": True})
        raise HTTPException(status_code=400, detail="Blocked: SQL injection pattern detected.")
    if _is_guardrail_violation(message):
        _audit_log({"timestamp": datetime.now().isoformat(), "task_id": task_id,
                    "flags": ["guardrail_violation"], "blocked": True})
        raise HTTPException(status_code=400, detail="Blocked: out of scope or policy violation.")

    clean_message, pii_in_input = _redact_pii(message)

    banking_report    = (t.get("banking_report") or "")[:3000]
    marketing_report  = (t.get("marketing_report") or "")[:2000]
    chart_analyses    = "\n\n".join(
        f"**{s['title']}**:\n{s.get('gemini_analysis', '')}"
        for s in (t.get("banking_blog_sections") or [])
        if s.get("gemini_analysis")
    )[:2000]

    system_content = (
        "You are an AI assistant helping a banking and marketing analyst understand their combined "
        "analytics results. You must ONLY answer questions based on the analysis results provided. "
        "Do NOT provide real financial advice, execute transactions, or discuss off-topic subjects.\n\n"
        "## Banking Churn Analysis Report:\n" + banking_report + "\n\n"
        + (f"## Chart Analyses:\n{chart_analyses}\n\n" if chart_analyses else "")
        + "## Digital Marketing Campaign Brief:\n" + marketing_report + "\n\n"
    )

    history = t["conversation"][-10:]
    messages = [{"role": "system", "content": system_content}]
    for turn in history:
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": clean_message})

    try:
        from config import gemini_llm
        from langchain_core.messages import HumanMessage, SystemMessage
        lc_msgs = []
        for m in messages:
            if m["role"] == "system":
                lc_msgs.append(SystemMessage(content=m["content"]))
            else:
                lc_msgs.append(HumanMessage(content=m["content"]))
        reply_raw = await asyncio.to_thread(lambda: gemini_llm.invoke(lc_msgs).content.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gemini error: {exc}")

    reply, pii_in_output = _redact_pii(reply_raw)
    security_flags = []
    if pii_in_input:
        security_flags.append("pii_redacted_input")
    if pii_in_output:
        security_flags.append("pii_redacted_output")

    _audit_log({"timestamp": datetime.now().isoformat(), "task_id": task_id,
                "input_len": len(message), "flags": security_flags, "blocked": False,
                "output_len": len(reply)})

    t["conversation"].append({"user": clean_message, "assistant": reply})
    return {"reply": reply, "security_flags": security_flags}


@app.get("/api/chat/history/{task_id}")
async def chat_history(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found.")
    return {"history": tasks[task_id].get("conversation", [])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8002, reload=False)
