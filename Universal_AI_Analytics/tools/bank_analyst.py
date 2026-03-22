"""
tools/bank_analyst.py – Banking data visualization, chart analysis, and reporting tools.
Mirrors Bank_Agent/tools/analyst_tools.py with paths adjusted to this system.
"""

import os
import sys
import json
import base64
import time
import threading
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai.tools import tool
from sqlalchemy import create_engine

from config import POSTGRES_URI, query_nvidia


def _get_db_uri() -> str:
    """Return the active database URI (set per-run via ACTIVE_DB_URI env var)."""
    return os.environ.get("ACTIVE_DB_URI") or POSTGRES_URI

sns.set_theme(style="whitegrid", palette="husl")

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR  = os.path.join(BASE_DIR, "outputs", "charts")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
os.makedirs(CHARTS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ── Gemini rate limiter ───────────────────────────────────────────────────────

class _GeminiRateLimiter:
    def __init__(self, max_requests: int = 3, pause_seconds: int = 60):
        self._count = 0
        self._max   = max_requests
        self._pause = pause_seconds
        self._lock  = threading.Lock()

    def throttle(self):
        with self._lock:
            if self._count > 0 and self._count % self._max == 0:
                print(f"\n[GeminiRateLimiter] {self._count} requests completed. "
                      f"Pausing {self._pause}s to avoid rate limits...")
                time.sleep(self._pause)
                print("[GeminiRateLimiter] Resuming.\n")
            self._count += 1


_gemini_rate_limiter = _GeminiRateLimiter(max_requests=3, pause_seconds=60)


def _analyze_chart_gemini(image_path: str, context: str = "") -> str:
    import time
    from langchain_community.chat_models import ChatOllama
    from langchain_core.messages import HumanMessage

    with open(image_path, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode()

    ext  = os.path.splitext(image_path)[-1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"

    prompt_text = (
        "You are a senior banking business intelligence analyst. "
        "Analyse this chart and provide:\n"
        "1. **Key Findings** – what the chart reveals\n"
        "2. **Business Insights** – implications for banking operations / strategy\n"
        "3. **Trends & Anomalies** – notable patterns\n"
        "4. **Risk Factors** – any risks visible in the data\n"
        "5. **Recommendations** – 3 actionable next steps\n\n"
        f"Context: {context}\n\nBe specific, quantitative where possible."
    )
    message = HumanMessage(content=[
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
    ])

    last_exc = None
    for attempt in range(3):
        try:
            llm = ChatOllama(model="qwen3.5:cloud")
            response = llm.invoke([message])
            return response.content
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                print(f"[BankAnalyst] Ollama chart analysis error (attempt {attempt+1}/3): {exc} — retrying in 5s", flush=True)
                time.sleep(5)
    return f"Ollama chart analysis failed after 3 attempts: {last_exc}"


def _save_fig(chart_type: str, table_name: str) -> str:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(CHARTS_DIR, f"{chart_type}_{table_name}_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


@tool("Generate Data Visualization")
def generate_visualization(
    table_name: str,
    chart_type: str,
    x_column: str,
    y_column: str = "",
    title: str = "",
    hue_column: str = "",
) -> str:
    """
    Generate a single chart from a PostgreSQL table and analyse it with Gemini vision.

    Args:
        table_name:  PostgreSQL source table (e.g. 'churn' or 'churn_encoded').
        chart_type:  Chart type: 'bar', 'line', 'scatter', 'histogram', 'histplot', 'heatmap', 'pie'.
        x_column:    Column for X-axis or grouping column for pie/heatmap.
        y_column:    Column for Y-axis; pass '' for histogram/heatmap.
        title:       Chart title; pass '' to auto-generate.
        hue_column:  Column for colour grouping; pass '' if not needed.

    Returns:
        JSON with chart_path, gemini_vision_analysis, and nvidia_recommendations.
    """
    try:
        engine = create_engine(_get_db_uri())
        df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", engine)

        fig, ax = plt.subplots(figsize=(12, 7))
        auto_title = title or f"{chart_type.title()}: {y_column or x_column} by {x_column}"
        hue = hue_column if hue_column and hue_column in df.columns else None

        if chart_type == "bar":
            sns.barplot(data=df, x=x_column, y=y_column, hue=hue, palette="husl", errorbar=None, ax=ax)
        elif chart_type == "line":
            sns.lineplot(data=df.sort_values(x_column), x=x_column, y=y_column, hue=hue, palette="husl", ax=ax)
        elif chart_type == "scatter":
            sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue, palette="husl", alpha=0.6, ax=ax)
        elif chart_type == "histogram":
            sns.histplot(data=df, x=x_column, bins=30, kde=True, edgecolor="white", alpha=0.85, ax=ax)
        elif chart_type == "heatmap":
            nums = df.select_dtypes(include=np.number).columns.tolist()
            corr = df[nums].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax, center=0, linewidths=0.5, annot_kws={"size": 8})
        elif chart_type in ("box", "histplot"):
            col = y_column or x_column
            sns.histplot(data=df, x=col, hue=hue, palette="husl", bins=30, kde=True, alpha=0.55, multiple="layer", ax=ax)
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel("Number of Customers", fontsize=12)
        elif chart_type == "pie":
            pdata = (df.groupby(x_column)[y_column].sum() if y_column else df[x_column].value_counts())
            ax.pie(pdata.values, labels=pdata.index, autopct="%1.1f%%", startangle=90,
                   colors=sns.color_palette("husl", len(pdata)))
            ax.axis("equal")

        ax.set_title(auto_title, fontsize=14, fontweight="bold", pad=20)
        if chart_type not in ("histogram", "heatmap", "pie", "box", "histplot"):
            ax.set_xlabel(x_column, fontsize=12)
            if y_column:
                ax.set_ylabel(y_column, fontsize=12)
        if chart_type not in ("heatmap", "pie"):
            plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        chart_path = _save_fig(chart_type, table_name)

        gemini_analysis = _analyze_chart_gemini(
            chart_path,
            context=f"'{chart_type}' chart showing '{y_column or x_column}' from banking table '{table_name}'."
        )

        nvidia_recs = ""
        try:
            nvidia_recs = query_nvidia([
                {"role": "system", "content": "You are a banking analytics expert. Be brief."},
                {"role": "user",   "content": f"Chart: {auto_title}. Gemini analysis:\n{gemini_analysis}\n\nGive 3 concise actionable recommendations."},
            ], max_tokens=512)
        except Exception as _nv_err:
            print(f"[BankAnalyst] NVIDIA recommendations skipped: {_nv_err}", flush=True)

        # Persist metadata for session blog
        _meta_path = os.path.join(CHARTS_DIR, "session_charts.json")
        try:
            _existing: list = []
            if os.path.exists(_meta_path):
                with open(_meta_path, "r", encoding="utf-8") as _fh:
                    _existing = json.load(_fh)
            _existing.append({
                "title":                  auto_title,
                "chart_type":             chart_type,
                "chart_url":              "/outputs/charts/" + os.path.basename(chart_path),
                "gemini_analysis":        gemini_analysis,
                "nvidia_recommendations": nvidia_recs,
            })
            with open(_meta_path, "w", encoding="utf-8") as _fh:
                json.dump(_existing, _fh, indent=2, ensure_ascii=False)
        except Exception:
            pass

        return json.dumps({
            "status":                 "success",
            "chart_path":             chart_path,
            "title":                  auto_title,
            "gemini_vision_analysis": gemini_analysis,
            "nvidia_recommendations": nvidia_recs,
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Generate Analytics Dashboard")
def generate_dashboard(table_name: str, analysis_focus: str = "general") -> str:
    """
    Generate a comprehensive 7-panel banking analytics dashboard PNG.

    Args:
        table_name:      PostgreSQL source table.
        analysis_focus:  Narrative context passed to Gemini (e.g. 'fraud', 'churn').

    Returns:
        JSON with dashboard PNG path and Gemini full-dashboard analysis.
    """
    try:
        engine = create_engine(_get_db_uri())
        df   = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", engine)
        nums = df.select_dtypes(include=np.number).columns.tolist()

        if len(nums) < 2:
            return json.dumps({"status": "error", "message": "Need ≥ 2 numeric columns for a dashboard."})

        fig = plt.figure(figsize=(22, 17))
        fig.suptitle(f"Banking Analytics Dashboard — {table_name.replace('_', ' ').title()}",
                     fontsize=18, fontweight="bold", y=0.99)
        gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.38)

        ax1 = fig.add_subplot(gs[0, 0])
        df[nums[0]].hist(ax=ax1, bins=30, color="#2196F3", edgecolor="white", alpha=0.85)
        ax1.set_title(f"Distribution: {nums[0]}", fontweight="bold")

        ax2 = fig.add_subplot(gs[0, 1:])
        top_cols = nums[:min(8, len(nums))]
        sns.heatmap(df[top_cols].corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                    ax=ax2, center=0, linewidths=0.5, annot_kws={"size": 7})
        ax2.set_title("Correlation Matrix", fontweight="bold")

        ax3 = fig.add_subplot(gs[1, :2])
        date_cols = [c for c in df.columns if any(d in c.lower() for d in ("date", "time", "month", "year"))]
        if date_cols:
            try:
                dts = pd.to_datetime(df[date_cols[0]], errors="coerce")
                sdf = df.assign(_dt=dts).dropna(subset=["_dt"]).sort_values("_dt")
                ax3.plot(sdf["_dt"], sdf[nums[0]], linewidth=2, color="#4CAF50")
                ax3.fill_between(sdf["_dt"], sdf[nums[0]], alpha=0.2, color="#4CAF50")
                ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
                ax3.set_title(f"Trend: {nums[0]} over Time", fontweight="bold")
            except Exception:
                df[nums[:3]].plot(ax=ax3, linewidth=2)
                ax3.set_title("Multi-Metric Trend", fontweight="bold")
        else:
            df[nums[:3]].plot(ax=ax3, linewidth=2)
            ax3.set_title("Multi-Metric Overview", fontweight="bold")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 2])
        plot_cols = nums[:min(5, len(nums))]
        df_norm = df[plot_cols].apply(lambda x: (x - x.mean()) / (x.std() + 1e-9))
        df_norm.boxplot(ax=ax4, grid=False)
        ax4.set_title("Normalised Distributions", fontweight="bold")
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha="right", fontsize=8)

        ax5 = fig.add_subplot(gs[2, 0])
        top10 = df[nums[0]].nlargest(10)
        ax5.barh(range(len(top10)), top10.values, color=sns.color_palette("husl", 10))
        ax5.set_title(f"Top 10: {nums[0]}", fontweight="bold")

        ax6 = fig.add_subplot(gs[2, 1])
        if len(nums) >= 2:
            ax6.scatter(df[nums[0]], df[nums[1]], alpha=0.4, color="#9C27B0", edgecolors="white", s=30)
            ax6.set_xlabel(nums[0], fontsize=9)
            ax6.set_ylabel(nums[1], fontsize=9)
            ax6.set_title(f"{nums[0]} vs {nums[1]}", fontweight="bold")

        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis("off")
        stats = df[nums[:4]].describe().round(2)
        tbl = ax7.table(cellText=stats.values, rowLabels=stats.index,
                        colLabels=stats.columns, cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        ax7.set_title("Statistical Summary", fontweight="bold", pad=10)

        dashboard_path  = _save_fig("dashboard", table_name)
        gemini_analysis = _analyze_chart_gemini(
            dashboard_path,
            context=f"Full banking analytics dashboard for '{table_name}'. Focus: {analysis_focus}."
        )

        # Persist metadata
        _meta_path = os.path.join(CHARTS_DIR, "session_charts.json")
        try:
            _existing: list = []
            if os.path.exists(_meta_path):
                with open(_meta_path, "r", encoding="utf-8") as _fh:
                    _existing = json.load(_fh)
            _existing.append({
                "title":                  f"Analytics Dashboard — {table_name.replace('_', ' ').title()}",
                "chart_type":             "dashboard",
                "chart_url":              "/outputs/charts/" + os.path.basename(dashboard_path),
                "gemini_analysis":        gemini_analysis,
                "nvidia_recommendations": "",
            })
            with open(_meta_path, "w", encoding="utf-8") as _fh:
                json.dump(_existing, _fh, indent=2, ensure_ascii=False)
        except Exception:
            pass

        return json.dumps({"status": "success", "dashboard_path": dashboard_path,
                           "gemini_vision_analysis": gemini_analysis}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Generate Business Insight Report")
def generate_text_report(analysis_results: str, report_title: str = "Banking Analytics Report") -> str:
    """
    Generate a professional markdown business-insight report using NVIDIA LLaMA.

    Args:
        analysis_results: JSON or text summarising all analysis outputs.
        report_title:     Title for the report.

    Returns:
        JSON with markdown report content and saved file path.
    """
    try:
        try:
            content = query_nvidia([
                {
                    "role": "system",
                    "content": (
                        "You are a senior banking business analyst writing executive reports. "
                        "Be concise — each section should be 3-5 bullet points maximum."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f'Create a banking analytics report titled "{report_title}".\n\n'
                        f"Analysis Data:\n{analysis_results[:6000]}\n\n"
                        f"# {report_title}\n"
                        "## Executive Summary\n## Key Findings\n## Risk Assessment\n"
                        "## Business Recommendations\n## Next Steps\n\n"
                        "Be specific with metrics. Keep total response under 800 words."
                    ),
                },
            ], max_tokens=1200)
        except Exception as _nv_err:
            print(f"[BankAnalyst] NVIDIA report generation skipped: {_nv_err}", flush=True)
            content = f"# {report_title}\n\nReport generation timed out. Analysis data:\n\n{analysis_results[:3000]}"

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(REPORTS_DIR, f"insight_report_{ts}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

        return json.dumps({"status": "success", "report_path": path, "report_content": content}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Label Encode Categorical Columns")
def label_encode_table(table_name: str, output_table: str = "") -> str:
    """
    Label-encode every categorical column in a PostgreSQL table and save as a new table.

    Args:
        table_name:   Source PostgreSQL table (e.g. 'churn').
        output_table: Destination table name (defaults to '<table_name>_encoded').

    Returns:
        JSON with the output table name, encoded column list, and row count.
    """
    from sklearn.preprocessing import LabelEncoder

    try:
        dest   = output_table.strip() or f"{table_name}_encoded"
        engine = create_engine(_get_db_uri())
        df     = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", engine)

        cat_cols     = df.select_dtypes(include=["object", "category"]).columns.tolist()
        encoded_cols = []
        le           = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))
            encoded_cols.append(col)

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.to_sql(dest, engine, if_exists="replace", index=False)

        return json.dumps({"status": "success", "source_table": table_name, "output_table": dest,
                           "encoded_columns": encoded_cols, "row_count": len(df)}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
