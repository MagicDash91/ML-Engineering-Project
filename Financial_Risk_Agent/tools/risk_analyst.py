"""
tools/risk_analyst.py – Risk visualization, chart analysis, and reporting tools.

Provides: risk charts (distribution, heat map, VaR, trend, scatter),
label encoding for heat maps, Ollama vision analysis, and text reports.
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
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine

from config import query_nvidia


def _get_db_uri() -> str:
    uri = os.environ.get("ACTIVE_DB_URI", "")
    if not uri:
        raise RuntimeError("No active database URI.")
    return uri


sns.set_theme(style="whitegrid", palette="husl")

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR  = os.path.join(BASE_DIR, "outputs", "charts")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
os.makedirs(CHARTS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ── Ollama chart analysis ─────────────────────────────────────────────────────

def _analyze_chart_ollama(image_path: str, context: str = "") -> str:
    import time as _time
    from langchain_community.chat_models import ChatOllama
    from langchain_core.messages import HumanMessage

    with open(image_path, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode()

    ext  = os.path.splitext(image_path)[-1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"

    prompt_text = (
        "You are a senior financial risk analyst. "
        "Analyse this risk chart and provide:\n"
        "1. **Key Risk Findings** — what the chart reveals about the risk profile\n"
        "2. **Risk Metrics** — quantitative observations (percentages, ratios, concentrations)\n"
        "3. **Risk Trends** — notable patterns, outliers, or anomalies\n"
        "4. **Regulatory Implications** — Basel III / IFRS 9 / FRTB relevance\n"
        "5. **Risk Recommendations** — 3 actionable mitigation steps\n\n"
        f"Context: {context}\n\nBe specific and quantitative."
    )
    message = HumanMessage(content=[
        {"type": "text",      "text": prompt_text},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
    ])

    last_exc = None
    for attempt in range(3):
        try:
            llm      = ChatOllama(model="qwen3.5:cloud")
            response = llm.invoke([message])
            return response.content
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                print(f"[RiskAnalyst] Ollama chart analysis error (attempt {attempt+1}/3): "
                      f"{exc} — retrying in 5s", flush=True)
                _time.sleep(5)
    return f"Ollama chart analysis failed after 3 attempts: {last_exc}"


def _save_fig(chart_type: str, table_name: str) -> str:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(CHARTS_DIR, f"{chart_type}_{table_name}_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def _persist_chart_meta(title: str, chart_type: str, chart_path: str,
                        analysis: str, recommendations: str = ""):
    meta_path = os.path.join(CHARTS_DIR, "session_charts.json")
    try:
        existing: list = []
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
        existing.append({
            "title":                  title,
            "chart_type":             chart_type,
            "chart_url":              "/outputs/charts/" + os.path.basename(chart_path),
            "gemini_analysis":        analysis,
            "nvidia_recommendations": recommendations,
        })
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool("Generate Risk Visualization")
def generate_risk_visualization(
    table_name: str,
    chart_type: str,
    x_column: str,
    y_column: str = "",
    title: str = "",
    hue_column: str = "",
) -> str:
    """
    Generate a single risk chart from a database table and analyse it with Ollama vision.

    Args:
        table_name:  Source table (e.g. 'loans' or 'loans_encoded').
        chart_type:  One of: 'bar', 'line', 'scatter', 'histogram', 'histplot',
                     'heatmap', 'pie', 'box'.
        x_column:    Column for X-axis / grouping / heatmap target.
        y_column:    Column for Y-axis; pass '' for histogram/heatmap/pie.
        title:       Chart title; pass '' to auto-generate.
        hue_column:  Column for colour grouping; pass '' if not needed.

    Returns:
        JSON with chart_path, ollama_vision_analysis, and nvidia_recommendations.
    """
    try:
        engine = create_engine(_get_db_uri())
        df     = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", engine)

        fig, ax   = plt.subplots(figsize=(12, 7))
        auto_title = title or f"{chart_type.title()}: {y_column or x_column} by {x_column}"
        hue        = hue_column if hue_column and hue_column in df.columns else None

        if chart_type == "bar":
            sns.barplot(data=df, x=x_column, y=y_column, hue=hue,
                        palette="husl", errorbar=None, ax=ax)
        elif chart_type == "line":
            sns.lineplot(data=df.sort_values(x_column), x=x_column, y=y_column,
                         hue=hue, palette="husl", ax=ax)
        elif chart_type == "scatter":
            sns.scatterplot(data=df, x=x_column, y=y_column,
                            hue=hue, palette="husl", alpha=0.6, ax=ax)
        elif chart_type == "histogram":
            sns.histplot(data=df, x=x_column, bins=30, kde=True,
                         edgecolor="white", alpha=0.85, ax=ax)
        elif chart_type == "heatmap":
            nums = df.select_dtypes(include=np.number).columns.tolist()
            corr = df[nums].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                        center=0, linewidths=0.5, annot_kws={"size": 8})
        elif chart_type in ("box", "histplot"):
            col = y_column or x_column
            sns.histplot(data=df, x=col, hue=hue, palette="husl", bins=30,
                         kde=True, alpha=0.55, multiple="layer", ax=ax)
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
        elif chart_type == "pie":
            pdata = (df.groupby(x_column)[y_column].sum()
                     if y_column else df[x_column].value_counts())
            ax.pie(pdata.values, labels=pdata.index, autopct="%1.1f%%",
                   startangle=90, colors=sns.color_palette("husl", len(pdata)))
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

        ollama_analysis = _analyze_chart_ollama(
            chart_path,
            context=f"'{chart_type}' risk chart showing '{y_column or x_column}' "
                    f"from table '{table_name}'."
        )

        nvidia_recs = ""
        try:
            nvidia_recs = query_nvidia([
                {"role": "system",
                 "content": "You are a financial risk expert. Be concise."},
                {"role": "user",
                 "content": (f"Risk chart: {auto_title}. Analysis:\n{ollama_analysis}\n\n"
                             "Give 3 concise actionable risk mitigation recommendations.")},
            ], max_tokens=512)
        except Exception as _e:
            print(f"[RiskAnalyst] NVIDIA recommendations skipped: {_e}", flush=True)

        _persist_chart_meta(auto_title, chart_type, chart_path,
                            ollama_analysis, nvidia_recs)

        return json.dumps({
            "status":                 "success",
            "chart_path":             chart_path,
            "title":                  auto_title,
            "ollama_vision_analysis": ollama_analysis,
            "nvidia_recommendations": nvidia_recs,
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Label Encode Categorical Columns")
def label_encode_table(table_name: str, output_table: str = "") -> str:
    """
    Label-encode every categorical column in a table and save as a new table.
    Required before generating a correlation heat map.

    Args:
        table_name:   Source table (e.g. 'loans').
        output_table: Destination table (defaults to '<table_name>_encoded').

    Returns:
        JSON with output table name, encoded column list, and row count.
    """
    try:
        dest   = output_table.strip() or f"{table_name}_encoded"
        engine = create_engine(_get_db_uri())
        df     = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", engine)

        cat_cols     = df.select_dtypes(include=["object", "category"]).columns.tolist()
        encoded_cols = []
        le           = LabelEncoder()
        for col in cat_cols:
            df[col]       = le.fit_transform(df[col].astype(str))
            encoded_cols.append(col)

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.to_sql(dest, engine, if_exists="replace", index=False)

        return json.dumps({
            "status":           "success",
            "source_table":     table_name,
            "output_table":     dest,
            "encoded_columns":  encoded_cols,
            "row_count":        len(df),
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Generate Risk Insight Report")
def generate_risk_text_report(analysis_results: str,
                               report_title: str = "Financial Risk Assessment Report") -> str:
    """
    Generate a professional markdown risk insight report using NVIDIA LLaMA.

    Args:
        analysis_results: JSON or text summarising all risk analysis outputs.
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
                        "You are a Chief Risk Officer writing executive risk reports. "
                        "Reference Basel III, IFRS 9, or FRTB where relevant. "
                        "Be concise — each section should be 3-5 bullet points maximum."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f'Create a financial risk report titled "{report_title}".\n\n'
                        f"Analysis Data:\n{analysis_results[:6000]}\n\n"
                        f"# {report_title}\n"
                        "## Executive Risk Summary\n## Risk Profile Overview\n"
                        "## Key Risk Findings\n## Model Performance\n"
                        "## Regulatory Compliance Status\n"
                        "## Risk Mitigation Recommendations\n## Action Plan\n\n"
                        "Be specific with metrics. Keep total response under 900 words."
                    ),
                },
            ], max_tokens=1400)
        except Exception as _e:
            print(f"[RiskAnalyst] NVIDIA report generation skipped: {_e}", flush=True)
            content = (f"# {report_title}\n\n"
                       f"Report generation timed out. Analysis data:\n\n{analysis_results[:3000]}")

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(REPORTS_DIR, f"risk_report_{ts}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

        return json.dumps({
            "status":         "success",
            "report_path":    path,
            "report_content": content,
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
