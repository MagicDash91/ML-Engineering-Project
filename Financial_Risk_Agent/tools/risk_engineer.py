"""
tools/risk_engineer.py – Risk Data Engineering tools.

Provides: database discovery, table profiling, data cleaning,
dtype normalisation, web research, and ETL for financial risk data.
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai.tools import tool
from sqlalchemy import create_engine, text

from config import TAVILY_API_KEY, query_nvidia


def _get_db_uri() -> str:
    uri = os.environ.get("ACTIVE_DB_URI", "")
    if not uri:
        raise RuntimeError("No active database URI. Provide a Database URI or upload a file.")
    return uri


def _db_dialect(uri: str) -> str:
    uri = uri.lower()
    if uri.startswith("snowflake"):  return "snowflake"
    if uri.startswith("mysql") or uri.startswith("mariadb"): return "mysql"
    if uri.startswith("sqlite"):     return "sqlite"
    if uri.startswith("mssql") or uri.startswith("sqlserver"): return "mssql"
    if uri.startswith("bigquery"):   return "bigquery"
    return "postgresql"


def _get_tavily():
    if not TAVILY_API_KEY:
        return None
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        return TavilySearchResults(k=5)
    except ImportError:
        return None


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool("List Database Tables")
def list_database_tables() -> str:
    """
    List all user tables in the connected database.
    Supports PostgreSQL, MySQL, SQLite, MSSQL, Snowflake, BigQuery.

    Returns:
        JSON array of {table, size} objects.
    """
    try:
        uri     = _get_db_uri()
        dialect = _db_dialect(uri)
        engine  = create_engine(uri)

        if dialect == "snowflake":
            sql = "SELECT TABLE_NAME, TABLE_SCHEMA FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = CURRENT_SCHEMA() ORDER BY TABLE_NAME"
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                tables = [{"table": row[0], "size": f"schema:{row[1]}"} for row in result]

        elif dialect == "mysql":
            sql = "SELECT TABLE_NAME, ROUND((DATA_LENGTH+INDEX_LENGTH)/1024,1) FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() ORDER BY TABLE_NAME"
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                tables = [{"table": row[0], "size": f"{row[1]} KB"} for row in result]

        elif dialect == "sqlite":
            sql = "SELECT name,'n/a' FROM sqlite_master WHERE type='table' ORDER BY name"
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                tables = [{"table": row[0], "size": row[1]} for row in result]

        elif dialect == "mssql":
            sql = "SELECT TABLE_NAME, TABLE_SCHEMA FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE' ORDER BY TABLE_NAME"
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                tables = [{"table": row[0], "size": f"schema:{row[1]}"} for row in result]

        else:  # PostgreSQL
            sql = """
                SELECT table_name,
                       pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) AS size
                FROM information_schema.tables
                WHERE table_schema='public'
                ORDER BY table_name
            """
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                tables = [{"table": row[0], "size": row[1]} for row in result]

        return json.dumps({"status": "success", "dialect": dialect,
                           "count": len(tables), "tables": tables}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Profile Database Table")
def profile_database_table(table_name: str) -> str:
    """
    Profile a database table: row count, schema, null counts, numeric stats,
    and categorical distributions. Essential first step for any risk analysis.

    Args:
        table_name: Name of the table to profile.

    Returns:
        JSON with full data quality profile.
    """
    try:
        engine = create_engine(_get_db_uri())
        df     = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", engine)

        profile: dict = {
            "status":      "success",
            "table":       table_name,
            "rows":        len(df),
            "columns":     list(df.columns),
            "dtypes":      {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": {col: int(df[col].isna().sum()) for col in df.columns},
            "numeric_stats": {},
            "categorical_distributions": {},
        }

        for col in df.select_dtypes(include=np.number).columns:
            profile["numeric_stats"][col] = {
                "mean":   round(float(df[col].mean()), 4),
                "median": round(float(df[col].median()), 4),
                "std":    round(float(df[col].std()), 4),
                "min":    round(float(df[col].min()), 4),
                "max":    round(float(df[col].max()), 4),
            }

        for col in df.select_dtypes(include="object").columns:
            profile["categorical_distributions"][col] = (
                df[col].value_counts().head(10).to_dict()
            )

        return json.dumps(profile, indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Query Database")
def query_database(sql_query: str) -> str:
    """
    Execute a SQL SELECT query against the connected database.

    Args:
        sql_query: A valid SELECT statement (no DDL/DML).

    Returns:
        JSON with column names and up to 200 rows of results.
    """
    if not sql_query.strip().upper().startswith("SELECT"):
        return json.dumps({"status": "error", "message": "Only SELECT queries are allowed."})
    try:
        if "LIMIT" not in sql_query.upper():
            sql_query = sql_query.rstrip().rstrip(";") + " LIMIT 10000"
        engine = create_engine(_get_db_uri())
        df     = pd.read_sql(sql_query, engine)
        return json.dumps({
            "status":  "success",
            "rows":    len(df),
            "columns": list(df.columns),
            "data":    df.head(200).to_dict(orient="records"),
        }, indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Clean Table Columns")
def clean_table_columns(table_name: str, extra_drop_columns: str = "") -> str:
    """
    Remove identifier and low-value columns from a table (ID, UUID, row index, constant).

    Args:
        table_name:         Table to clean (overwritten in place).
        extra_drop_columns: Optional comma-separated additional columns to drop.

    Returns:
        JSON with dropped columns, kept columns, and row count.
    """
    try:
        engine       = create_engine(_get_db_uri())
        df           = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", engine)
        original_cols = list(df.columns)
        dropped       = []

        id_patterns = {"id", "row_id", "rowid", "index", "uuid",
                       "customerid", "customer_id", "loanid", "loan_id",
                       "accountid", "account_id", "transactionid", "transaction_id"}
        for col in original_cols:
            if col.lower() in id_patterns:
                dropped.append(col)

        for col in original_cols:
            if col not in dropped and df[col].isna().all():
                dropped.append(col)

        for col in original_cols:
            if col not in dropped and df[col].nunique(dropna=False) <= 1:
                dropped.append(col)

        if extra_drop_columns.strip():
            for col in [c.strip() for c in extra_drop_columns.split(",")]:
                if col and col in df.columns and col not in dropped:
                    dropped.append(col)

        df.drop(columns=dropped, errors="ignore", inplace=True)
        df.to_sql(table_name, engine, if_exists="replace", index=False)

        return json.dumps({
            "status":          "success",
            "table":           table_name,
            "dropped_columns": dropped,
            "kept_columns":    list(df.columns),
            "row_count":       len(df),
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Normalize Column Data Types")
def normalize_column_dtypes(table_name: str, numeric_threshold: float = 0.90) -> str:
    """
    Detect object/string columns whose values are actually numeric and cast them.
    Critical for financial data where amounts are often stored as text.

    Args:
        table_name:        Table to normalize (overwritten in place).
        numeric_threshold: Min fraction of non-null values that must be numeric (default 0.90).

    Returns:
        JSON with per-column conversion results and final dtype map.
    """
    try:
        engine = create_engine(_get_db_uri())
        df     = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", engine)

        converted = {}
        unchanged = {}

        for col in df.select_dtypes(include=["object"]).columns:
            series         = df[col].str.strip() if df[col].dtype == object else df[col]
            numeric_series = pd.to_numeric(series, errors="coerce")
            non_null_count = series.notna().sum()

            if non_null_count == 0:
                unchanged[col] = "all null"
                continue

            ratio = numeric_series.notna().sum() / non_null_count

            if ratio >= numeric_threshold:
                has_decimal = series.dropna().str.contains(r"\.", regex=True).any()
                if has_decimal:
                    df[col]       = numeric_series.astype("float64")
                    converted[col] = "float64"
                else:
                    df[col]       = pd.to_numeric(numeric_series, errors="coerce").astype("float64")
                    if df[col].isna().sum() == 0:
                        df[col]       = df[col].astype("int64")
                        converted[col] = "int64"
                    else:
                        converted[col] = "float64"
            else:
                unchanged[col] = f"only {ratio:.0%} numeric — kept as string"

        df.to_sql(table_name, engine, if_exists="replace", index=False)

        return json.dumps({
            "status":            "success",
            "table":             table_name,
            "columns_converted": converted,
            "columns_unchanged": unchanged,
            "row_count":         len(df),
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Search Risk & Regulatory Web Data")
def web_search_risk(query: str) -> str:
    """
    Search the web for risk management, regulatory, and financial benchmarks.

    Args:
        query: Search query string (e.g. 'Basel III capital requirements 2025',
               'IFRS 9 expected credit loss methodology', 'VaR stress testing best practices').

    Returns:
        JSON with result titles, URLs, and content snippets.
    """
    tavily = _get_tavily()
    if tavily is None:
        return json.dumps({"status": "error",
                           "message": "Tavily not available. Check TAVILY_API_KEY."})
    try:
        results = tavily.invoke(query)
        if not results:
            return json.dumps({"status": "success", "message": "No results found",
                               "results_count": 0})

        records = []
        for r in results:
            records.append({
                "query":        query,
                "url":          r.get("url", ""),
                "title":        r.get("title", ""),
                "content":      r.get("content", "")[:2000],
                "score":        float(r.get("score", 0)),
                "collected_at": datetime.now().isoformat(),
            })

        return json.dumps({
            "status":        "success",
            "results_count": len(results),
            "preview": [{"title": r.get("title", ""), "url": r.get("url", ""),
                         "snippet": r.get("content", "")[:300]} for r in results[:3]],
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
