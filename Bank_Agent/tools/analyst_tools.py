"""
Data Analyst Tools
------------------
Visualization, AI-powered chart analysis (Gemini 2.5 Flash vision),
business insight generation, and text report creation.

LLMs used:
  - Gemini 2.5 Flash  → chart / image understanding (vision)
  - NVIDIA Llama      → business recommendations
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


# ──────────────────────────────────────────────
# Gemini rate limiter
# Pauses 60 seconds after every 3 Gemini requests
# to stay within free-tier / low-quota limits.
# ──────────────────────────────────────────────

class _GeminiRateLimiter:
    """Thread-safe counter that pauses before the (3n+1)-th request."""

    def __init__(self, max_requests: int = 3, pause_seconds: int = 60):
        self._count        = 0
        self._max          = max_requests
        self._pause        = pause_seconds
        self._lock         = threading.Lock()

    def throttle(self):
        """Call this immediately before each Gemini API request."""
        with self._lock:
            if self._count > 0 and self._count % self._max == 0:
                print(
                    f"\n[GeminiRateLimiter] {self._count} Gemini requests completed. "
                    f"Pausing {self._pause}s to avoid rate limits..."
                )
                time.sleep(self._pause)
                print("[GeminiRateLimiter] Resuming.\n")
            self._count += 1


_gemini_rate_limiter = _GeminiRateLimiter(max_requests=3, pause_seconds=60)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from crewai.tools import tool
from sqlalchemy import create_engine

from config import POSTGRES_URI, query_nvidia, gemini_llm

sns.set_theme(style="whitegrid", palette="husl")

# Absolute base directory (Bank_Agent/)
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR   = os.path.join(BASE_DIR, "outputs", "charts")
REPORTS_DIR  = os.path.join(BASE_DIR, "outputs", "reports")
os.makedirs(CHARTS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _analyze_chart_gemini(image_path: str, context: str = "") -> str:
    """
    Send a PNG chart to Gemini 2.5 Flash for business insight extraction.
    Falls back gracefully if Gemini is unavailable.
    """
    if gemini_llm is None:
        return "Gemini vision not available (GOOGLE_API_KEY missing)."

    try:
        with open(image_path, "rb") as fh:
            img_b64 = base64.b64encode(fh.read()).decode()

        from langchain_core.messages import HumanMessage

        _gemini_rate_limiter.throttle()   # enforce 3-req / 60-s limit

        prompt_text = (
            "You are a senior banking business intelligence analyst. "
            "Analyse this chart and provide:\n"
            "1. **Key Findings** – what the chart reveals\n"
            "2. **Business Insights** – implications for banking operations / strategy\n"
            "3. **Trends & Anomalies** – notable patterns\n"
            "4. **Risk Factors** – any risks visible in the data\n"
            "5. **Recommendations** – 3 actionable next steps\n\n"
            f"Context: {context}\n\n"
            "Be specific, quantitative where possible, and focused on banking value."
        )

        message = HumanMessage(content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ])
        response = gemini_llm.invoke([message])
        return response.content

    except Exception as e:
        return f"Gemini chart analysis failed: {e}"


def _save_fig(chart_type: str, table_name: str) -> str:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(CHARTS_DIR, f"{chart_type}_{table_name}_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


# ============================================================
# Tool 1: Single chart generator
# ============================================================
@tool("Generate Data Visualization")
def generate_visualization(
    table_name: str,
    chart_type: str,
    x_column: str,
    y_column: str = "",
    title: str = "",
    hue_column: str = "",
    analyze_with_ai: bool = True,
) -> str:
    """
    Generate a single chart from a PostgreSQL table and (optionally) analyse it with Gemini vision.

    Args:
        table_name:     PostgreSQL source table.
        chart_type:     One of: 'bar', 'line', 'scatter', 'histogram', 'histplot', 'heatmap', 'pie', 'area'.
                        Use 'histplot' to compare a numeric column's distribution across groups (replaces
                        boxplot — overlapping histograms are easier for all audiences to read).
        x_column:       Column for X-axis (or grouping column for heatmap/pie).
        y_column:       Column for Y-axis (leave empty for histogram/heatmap).
        title:          Chart title (auto-generated if empty).
        hue_column:     Optional colour-grouping column.
        analyze_with_ai: Pass the chart to Gemini 2.5 Flash for vision analysis.

    Returns:
        JSON with chart file path, Gemini analysis, and NVIDIA business recommendations.
    """
    try:
        engine = create_engine(POSTGRES_URI)
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

        fig, ax = plt.subplots(figsize=(12, 7))
        auto_title = title or f"{chart_type.title()}: {y_column or x_column} by {x_column}"

        # ── Draw chart (all types use seaborn) ───────────────────
        hue = hue_column if hue_column and hue_column in df.columns else None

        if chart_type == "bar":
            sns.barplot(
                data=df, x=x_column, y=y_column,
                hue=hue, palette="husl",
                errorbar=None, ax=ax,
            )
            if hue:
                ax.legend(title=hue_column, fontsize=10)

        elif chart_type == "line":
            sns.lineplot(
                data=df.sort_values(x_column),
                x=x_column, y=y_column,
                hue=hue, palette="husl",
                markers=True, dashes=False, ax=ax,
            )
            if hue:
                ax.legend(title=hue_column, fontsize=10)

        elif chart_type == "scatter":
            sns.scatterplot(
                data=df, x=x_column, y=y_column,
                hue=hue, palette="husl",
                alpha=0.6, edgecolor="white", linewidth=0.4, ax=ax,
            )
            if hue:
                ax.legend(title=hue_column, fontsize=10)

        elif chart_type == "histogram":
            sns.histplot(
                data=df, x=x_column,
                bins=30, kde=True,
                color=sns.color_palette("husl")[0],
                edgecolor="white", alpha=0.85, ax=ax,
            )
            ax.set_xlabel(x_column, fontsize=12)
            ax.set_ylabel("Count", fontsize=12)

        elif chart_type == "heatmap":
            nums = df.select_dtypes(include=np.number).columns.tolist()
            corr = df[nums].corr()
            sns.heatmap(
                corr, annot=True, fmt=".2f", cmap="RdYlGn",
                ax=ax, center=0, square=True,
                linewidths=0.5, annot_kws={"size": 8},
            )

        elif chart_type in ("box", "histplot"):
            # Overlapping sns.histplot replaces boxplot — shows the full distribution
            # shape per group and is immediately readable without statistical training.
            col = y_column or x_column
            sns.histplot(
                data=df, x=col,
                hue=hue, palette="husl",
                bins=30, kde=True,
                alpha=0.55, edgecolor="white",
                multiple="layer", ax=ax,
            )
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel("Number of Customers", fontsize=12)
            if hue:
                ax.legend(title=hue_column, fontsize=10)

        elif chart_type == "pie":
            # Seaborn has no pie chart — use matplotlib with seaborn palette
            pdata = (df.groupby(x_column)[y_column].sum()
                     if y_column else df[x_column].value_counts())
            ax.pie(
                pdata.values, labels=pdata.index,
                autopct="%1.1f%%", startangle=90,
                colors=sns.color_palette("husl", len(pdata)),
            )
            ax.axis("equal")

        elif chart_type == "area":
            # Seaborn has no area chart — use sns.lineplot + matplotlib fill_between
            sdf = df.sort_values(x_column).reset_index(drop=True)
            sns.lineplot(data=sdf, x=sdf.index, y=y_column, linewidth=2, ax=ax)
            ax.fill_between(sdf.index, sdf[y_column], alpha=0.25)
            step = max(1, len(sdf) // 10)
            ax.set_xticks(range(0, len(sdf), step))
            ax.set_xticklabels(sdf[x_column].iloc[::step].astype(str).tolist(), rotation=45)
            ax.set_xlabel(x_column, fontsize=12)
            ax.set_ylabel(y_column, fontsize=12)

        # ── Decoration ──────────────────────────────────────────
        ax.set_title(auto_title, fontsize=14, fontweight="bold", pad=20)
        if chart_type not in ("histogram", "heatmap", "pie", "area", "box", "histplot"):
            ax.set_xlabel(x_column, fontsize=12)
            if y_column:
                ax.set_ylabel(y_column, fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        chart_path = _save_fig(chart_type, table_name)

        # ── AI analysis ──────────────────────────────────────────
        gemini_analysis = ""
        if analyze_with_ai:
            gemini_analysis = _analyze_chart_gemini(
                chart_path,
                context=(f"'{chart_type}' chart showing '{y_column or x_column}' "
                         f"from banking table '{table_name}'."),
            )

        nvidia_recs = query_nvidia([
            {"role": "system", "content": "You are a banking analytics expert. Be brief."},
            {"role": "user",   "content": (
                f"Chart: {auto_title} (table: {table_name}). "
                f"Gemini vision summary: {gemini_analysis[:200]}. "
                "Give exactly 3 concise actionable recommendations in 1 sentence each."
            )},
        ], max_tokens=256)

        return json.dumps({
            "status":                    "success",
            "chart_path":                chart_path,
            "chart_type":                chart_type,
            "title":                     auto_title,
            "gemini_vision_analysis":    gemini_analysis,
            "nvidia_recommendations":    nvidia_recs,
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ============================================================
# Tool 2: Multi-panel dashboard
# ============================================================
@tool("Generate Analytics Dashboard")
def generate_dashboard(table_name: str, analysis_focus: str = "general") -> str:
    """
    Generate a comprehensive 7-panel banking analytics dashboard PNG.

    Args:
        table_name:      PostgreSQL source table.
        analysis_focus:  Narrative context passed to Gemini (e.g. 'fraud', 'market', 'credit_risk').

    Returns:
        JSON with dashboard PNG path and Gemini full-dashboard analysis.
    """
    try:
        engine = create_engine(POSTGRES_URI)
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        nums = df.select_dtypes(include=np.number).columns.tolist()

        if len(nums) < 2:
            return json.dumps({"status": "error", "message": "Need ≥ 2 numeric columns for a dashboard."})

        fig = plt.figure(figsize=(22, 17))
        fig.suptitle(
            f"Banking Analytics Dashboard — {table_name.replace('_', ' ').title()}",
            fontsize=18, fontweight="bold", y=0.99,
        )
        gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.38)

        # Panel 1 – Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        df[nums[0]].hist(ax=ax1, bins=30, color="#2196F3", edgecolor="white", alpha=0.85)
        ax1.set_title(f"Distribution: {nums[0]}", fontweight="bold")
        ax1.set_xlabel(nums[0])

        # Panel 2 – Correlation heatmap
        ax2 = fig.add_subplot(gs[0, 1:])
        top_cols = nums[:min(8, len(nums))]
        sns.heatmap(df[top_cols].corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                    ax=ax2, center=0, linewidths=0.5, annot_kws={"size": 7})
        ax2.set_title("Correlation Matrix", fontweight="bold")

        # Panel 3 – Time-series or multi-line
        ax3 = fig.add_subplot(gs[1, :2])
        date_cols = [c for c in df.columns if any(d in c.lower() for d in ("date", "time", "month", "year"))]
        if date_cols:
            try:
                dts = pd.to_datetime(df[date_cols[0]], errors="coerce")
                sdf = df.assign(_dt=dts).dropna(subset=["_dt"]).sort_values("_dt")
                ax3.plot(sdf["_dt"], sdf[nums[0]], linewidth=2, color="#4CAF50")
                ax3.fill_between(sdf["_dt"], sdf[nums[0]], alpha=0.2, color="#4CAF50")
                ax3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
                ax3.set_title(f"Trend: {nums[0]} over Time", fontweight="bold")
            except Exception:
                df[nums[:3]].plot(ax=ax3, linewidth=2)
                ax3.set_title("Multi-Metric Trend", fontweight="bold")
        else:
            df[nums[:3]].plot(ax=ax3, linewidth=2)
            ax3.set_title("Multi-Metric Overview", fontweight="bold")
        ax3.grid(True, alpha=0.3)

        # Panel 4 – Normalised box plots
        ax4 = fig.add_subplot(gs[1, 2])
        plot_cols = nums[:min(5, len(nums))]
        df_norm = df[plot_cols].apply(lambda x: (x - x.mean()) / (x.std() + 1e-9))
        df_norm.boxplot(ax=ax4, grid=False)
        ax4.set_title("Normalised Distributions", fontweight="bold")
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha="right", fontsize=8)

        # Panel 5 – Horizontal bar (top 10)
        ax5 = fig.add_subplot(gs[2, 0])
        top10 = df[nums[0]].nlargest(10)
        ax5.barh(range(len(top10)), top10.values, color=sns.color_palette("husl", 10))
        ax5.set_title(f"Top 10: {nums[0]}", fontweight="bold")

        # Panel 6 – Scatter
        ax6 = fig.add_subplot(gs[2, 1])
        if len(nums) >= 2:
            ax6.scatter(df[nums[0]], df[nums[1]], alpha=0.4,
                        color="#9C27B0", edgecolors="white", linewidths=0.3, s=30)
            ax6.set_xlabel(nums[0], fontsize=9)
            ax6.set_ylabel(nums[1], fontsize=9)
            ax6.set_title(f"{nums[0]} vs {nums[1]}", fontweight="bold")

        # Panel 7 – Summary stats table
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis("off")
        stats = df[nums[:4]].describe().round(2)
        tbl = ax7.table(cellText=stats.values, rowLabels=stats.index,
                        colLabels=stats.columns, cellLoc="center",
                        loc="center", bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        ax7.set_title("Statistical Summary", fontweight="bold", pad=10)

        dashboard_path = _save_fig("dashboard", table_name)

        gemini_analysis = _analyze_chart_gemini(
            dashboard_path,
            context=(f"Full banking analytics dashboard for dataset '{table_name}'. "
                     f"Analysis focus: {analysis_focus}."),
        )

        return json.dumps({
            "status":                  "success",
            "dashboard_path":          dashboard_path,
            "analysis_focus":          analysis_focus,
            "gemini_vision_analysis":  gemini_analysis,
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ============================================================
# Tool 3: Text business insights report
# ============================================================
@tool("Generate Business Insight Report")
def generate_text_report(analysis_results: str, report_title: str = "Banking Analytics Report") -> str:
    """
    Generate a professional markdown business-insight report using NVIDIA Llama Nemotron.

    Args:
        analysis_results: JSON or text summarising all analysis outputs to include.
        report_title:     Title for the report.

    Returns:
        JSON with markdown report content and saved file path.
    """
    try:
        content = query_nvidia([
            {
                "role": "system",
                "content": (
                    "You are a senior banking business analyst writing executive reports. "
                    "Your output is structured, data-driven, and uses markdown formatting. "
                    "Always include specific numbers and percentages where available. "
                    "Be concise — each section should be 3-5 bullet points maximum."
                ),
            },
            {
                "role": "user",
                "content": (
                    f'Create a concise banking analytics report titled "{report_title}".\n\n'
                    f"Analysis Data:\n{analysis_results[:2000]}\n\n"
                    "Structure (keep each section brief):\n"
                    f"# {report_title}\n"
                    "## Executive Summary\n"
                    "## Key Findings\n"
                    "## Risk Assessment\n"
                    "## Business Recommendations\n"
                    "## Next Steps\n\n"
                    "Be specific with metrics. Keep total response under 800 words."
                ),
            },
        ], max_tokens=1200)

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(REPORTS_DIR, f"insight_report_{ts}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

        return json.dumps({
            "status":         "success",
            "report_path":    path,
            "report_content": content,
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ============================================================
# Tool 4: Label-encode categorical columns
# ============================================================
@tool("Label Encode Categorical Columns")
def label_encode_table(table_name: str, output_table: str = "") -> str:
    """
    Label-encode every categorical (object/string) column in a PostgreSQL table
    and save the result as a new table so that visualizations like heatmaps can
    include all features, not just numeric ones.

    Args:
        table_name:   Source PostgreSQL table (e.g. 'churn').
        output_table: Destination table name. Defaults to '<table_name>_encoded'.

    Returns:
        JSON with the output table name, encoded column list, and row count.
    """
    from sklearn.preprocessing import LabelEncoder
    from sqlalchemy import text

    try:
        dest = output_table.strip() or f"{table_name}_encoded"
        engine = create_engine(POSTGRES_URI)

        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        encoded_cols = []

        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))
            encoded_cols.append(col)

        # Coerce any remaining numeric-strings to float
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.to_sql(dest, engine, if_exists="replace", index=False)

        return json.dumps({
            "status":            "success",
            "source_table":      table_name,
            "output_table":      dest,
            "encoded_columns":   encoded_cols,
            "row_count":         len(df),
            "total_columns":     len(df.columns),
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
