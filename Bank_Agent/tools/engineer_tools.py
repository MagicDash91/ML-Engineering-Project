"""
Data Engineer Tools
-------------------
ETL / ELT pipelines, web scraping, financial data collection,
and PostgreSQL data warehouse management.

LLM used: NVIDIA Llama Nemotron (for ETL planning decisions)
External APIs: yfinance, Tavily search
Storage: PostgreSQL via SQLAlchemy
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any

warnings.filterwarnings("ignore")

# Add parent dir so config is importable when running from Bank_Agent/
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai.tools import tool
from sqlalchemy import create_engine, text

from config import POSTGRES_URI, TAVILY_API_KEY, query_nvidia


# ── lazy imports (avoid hard crash if package missing) ──────────────────────
def _get_yfinance():
    try:
        import yfinance as yf
        return yf
    except ImportError:
        return None


def _get_tavily():
    if not TAVILY_API_KEY:
        return None
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        return TavilySearchResults(k=5)
    except ImportError:
        return None


# ============================================================
# Tool 0: Inspect & profile a PostgreSQL table
# ============================================================
@tool("Profile Database Table")
def profile_database_table(table_name: str) -> str:
    """
    Read a table from PostgreSQL, profile its quality, and return a
    full data summary — without modifying any data.

    Reports: row count, column types, null counts per column,
    numeric stats (mean/median/std/min/max), and categorical
    value distributions (top 10 values per object column).

    Args:
        table_name: Name of the PostgreSQL table to profile.

    Returns:
        JSON with the full quality profile.
    """
    try:
        engine = create_engine(POSTGRES_URI)
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

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


# ============================================================
# Tool 1: Fetch stock / market data via yfinance
# ============================================================
@tool("Fetch Financial Market Data")
def fetch_financial_data(symbol: str, period: str = "1y", interval: str = "1d") -> str:
    """
    Fetch historical financial market data for a given ticker and store it in PostgreSQL.

    Args:
        symbol:   Stock ticker (e.g. 'BBCA.JK', 'BMRI.JK', 'JPM', 'BAC').
        period:   Date range – '1d','5d','1mo','3mo','6mo','1y','2y','5y','max'.
        interval: Bar size – '1d','1wk','1mo'.

    Returns:
        JSON summary: table name, row count, date range, latest close, price change %.
    """
    yf = _get_yfinance()
    if yf is None:
        return json.dumps({"status": "error", "message": "yfinance not installed. Run: pip install yfinance"})

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            return json.dumps({"status": "error", "message": f"No data for symbol '{symbol}'"})

        df.reset_index(inplace=True)
        # Normalize date column
        for col in df.columns:
            if "date" in str(col).lower() or str(df[col].dtype).startswith("datetime"):
                df[col] = df[col].astype(str)
        # Clean column names
        df.columns = [str(c).split(" ")[0] for c in df.columns]
        df.dropna(axis=1, how="all", inplace=True)

        table_name = f"market_{symbol.replace('.', '_').replace('-', '_').lower()}"
        engine = create_engine(POSTGRES_URI)
        df.to_sql(table_name, engine, if_exists="replace", index=False)

        latest_close  = float(df["Close"].iloc[-1])
        initial_close = float(df["Close"].iloc[0])
        pct_change    = round((latest_close - initial_close) / initial_close * 100, 2)

        return json.dumps({
            "status":         "success",
            "symbol":         symbol,
            "table":          table_name,
            "rows":           len(df),
            "columns":        list(df.columns),
            "date_range":     f"{df['Date'].iloc[0]} to {df['Date'].iloc[-1]}",
            "latest_close":   latest_close,
            "avg_volume":     float(df["Volume"].mean()),
            "price_change_%": pct_change,
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ============================================================
# Tool 2: Indonesian bank data (adds company fundamentals)
# ============================================================
@tool("Fetch Indonesian Bank Data")
def fetch_indonesian_bank_data(bank_code: str = "BBCA") -> str:
    """
    Fetch historical prices + company fundamentals for a major Indonesian bank.

    Common codes: BBCA (BCA), BMRI (Mandiri), BBRI (BRI), BBNI (BNI), BRIS (BRI Syariah).

    Args:
        bank_code: Bank ticker without the '.JK' suffix.

    Returns:
        JSON with historical data summary and key fundamentals (P/E, P/B, market cap, etc.).
    """
    yf = _get_yfinance()
    if yf is None:
        return json.dumps({"status": "error", "message": "yfinance not installed."})

    try:
        symbol = f"{bank_code.upper()}.JK"
        ticker = yf.Ticker(symbol)

        hist = ticker.history(period="1y")
        hist.reset_index(inplace=True)
        hist["Date"] = hist["Date"].astype(str)
        hist.columns = [str(c).split(" ")[0] for c in hist.columns]

        info = {}
        try:
            info = ticker.info or {}
        except Exception:
            pass

        table_name = f"bank_{bank_code.lower()}_historical"
        engine = create_engine(POSTGRES_URI)
        hist.to_sql(table_name, engine, if_exists="replace", index=False)

        return json.dumps({
            "status":           "success",
            "bank":             bank_code,
            "symbol":           symbol,
            "table":            table_name,
            "historical_rows":  len(hist),
            "company_info": {
                "name":           info.get("longName", bank_code),
                "sector":         info.get("sector", "Banking"),
                "market_cap":     info.get("marketCap", 0),
                "pe_ratio":       info.get("trailingPE", 0),
                "pb_ratio":       info.get("priceToBook", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "52w_high":       info.get("fiftyTwoWeekHigh", 0),
                "52w_low":        info.get("fiftyTwoWeekLow", 0),
            },
        }, indent=2, default=str)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ============================================================
# Tool 3: Web search + collect (Tavily)
# ============================================================
@tool("Search and Collect Web Data")
def web_search_collect(query: str) -> str:
    """
    Search the web for banking/financial information and save results to PostgreSQL.

    Args:
        query: Search query string (e.g. 'Indonesian bank performance 2025 BCA Mandiri').

    Returns:
        JSON with result titles, URLs, and storage confirmation.
    """
    tavily = _get_tavily()
    if tavily is None:
        return json.dumps({"status": "error", "message": "Tavily not available. Check TAVILY_API_KEY."})

    try:
        results = tavily.invoke(query)
        if not results:
            return json.dumps({"status": "success", "message": "No results found", "results_count": 0})

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

        df = pd.DataFrame(records)
        engine = create_engine(POSTGRES_URI)
        df.to_sql("web_research_data", engine, if_exists="append", index=False)

        return json.dumps({
            "status":        "success",
            "results_count": len(results),
            "saved_table":   "web_research_data",
            "preview": [{"title": r.get("title", ""), "url": r.get("url", "")} for r in results[:3]],
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ============================================================
# Tool 4: Full ETL pipeline
# ============================================================
@tool("Run ETL Pipeline")
def run_etl_pipeline(data_source: str, target_table: str, transformations: str = "{}") -> str:
    """
    Execute a complete ETL pipeline: Extract → Transform → Load into PostgreSQL.

    Args:
        data_source:     Source descriptor.
                         Formats:
                           'yfinance:<TICKER>'            – single ticker
                           'multiple_stocks:<T1>,<T2>,...'– multiple tickers combined
                           'csv:<absolute_path>'          – local CSV file
        target_table:    Destination PostgreSQL table name.
        transformations: JSON string with optional transform rules:
                         {
                           "rename_columns": {"OldName": "new_name"},
                           "drop_columns":   ["col1", "col2"],
                           "add_timestamp":  true
                         }

    Returns:
        JSON summary: rows loaded, columns, dtypes.
    """
    yf = _get_yfinance()

    try:
        source_type, source_value = data_source.split(":", 1)
        df = None

        # ── EXTRACT ─────────────────────────────────────────────
        if source_type == "yfinance":
            if yf is None:
                return json.dumps({"status": "error", "message": "yfinance not installed."})
            ticker = yf.Ticker(source_value.strip())
            df = ticker.history(period="1y")
            df.reset_index(inplace=True)

        elif source_type == "multiple_stocks":
            if yf is None:
                return json.dumps({"status": "error", "message": "yfinance not installed."})
            symbols = [s.strip() for s in source_value.split(",")]
            dfs = []
            for sym in symbols:
                t = yf.Ticker(sym)
                temp = t.history(period="1y")
                temp["symbol"] = sym
                temp.reset_index(inplace=True)
                dfs.append(temp)
            df = pd.concat(dfs, ignore_index=True)

        elif source_type == "csv":
            df = pd.read_csv(source_value)

        else:
            return json.dumps({"status": "error", "message": f"Unknown source type: {source_type}"})

        if df is None or df.empty:
            return json.dumps({"status": "error", "message": "Extraction returned empty data."})

        # ── TRANSFORM ────────────────────────────────────────────
        # Normalise date/datetime columns to strings
        for col in df.columns:
            if str(df[col].dtype).startswith("datetime"):
                df[col] = df[col].astype(str)

        # Flatten column names (remove timezone suffixes)
        df.columns = [str(c).split(" ")[0] for c in df.columns]
        df.dropna(axis=1, how="all", inplace=True)

        trans = {}
        try:
            trans = json.loads(transformations) if transformations else {}
        except json.JSONDecodeError:
            pass

        if trans.get("rename_columns"):
            df.rename(columns=trans["rename_columns"], inplace=True)
        if trans.get("drop_columns"):
            df.drop(columns=trans["drop_columns"], errors="ignore", inplace=True)
        if trans.get("add_timestamp"):
            df["etl_timestamp"] = datetime.now().isoformat()

        # ── LOAD ─────────────────────────────────────────────────
        engine = create_engine(POSTGRES_URI)
        df.to_sql(target_table, engine, if_exists="replace", index=False)

        return json.dumps({
            "status":       "success",
            "source":       data_source,
            "target_table": target_table,
            "rows_loaded":  len(df),
            "columns":      list(df.columns),
            "dtypes":       {col: str(dtype) for col, dtype in df.dtypes.items()},
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ============================================================
# Tool 4b: Clean unused columns from a table
# ============================================================
@tool("Clean Table Columns")
def clean_table_columns(table_name: str, extra_drop_columns: str = "") -> str:
    """
    Remove identifier and low-value columns from a PostgreSQL table, then overwrite
    the table in place with only the analytically useful columns.

    Auto-drops:
      - Known identifier columns (customerid, id, customer_id, index, uuid, row_id)
      - Columns where ALL values are null
      - Columns with only 1 unique value (zero variance / no information)

    Args:
        table_name:         PostgreSQL table to clean (overwritten in place).
        extra_drop_columns: Optional comma-separated additional column names to drop
                            (e.g. 'transaction_date,etl_timestamp').

    Returns:
        JSON with list of dropped columns, kept columns, and row count.
    """
    try:
        engine = create_engine(POSTGRES_URI)
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

        original_cols = list(df.columns)
        dropped = []

        # 1. Known identifier column names (case-insensitive)
        id_patterns = {"customerid", "id", "customer_id", "index", "uuid", "row_id", "rowid"}
        for col in original_cols:
            if col.lower() in id_patterns:
                dropped.append(col)

        # 2. All-null columns
        for col in original_cols:
            if col not in dropped and df[col].isna().all():
                dropped.append(col)

        # 3. Single-value columns (zero analytical value)
        for col in original_cols:
            if col not in dropped and df[col].nunique(dropna=False) <= 1:
                dropped.append(col)

        # 4. Caller-specified extra columns
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


# ============================================================
# Tool 4c: Normalize string columns that contain numeric values
# ============================================================
@tool("Normalize Column Data Types")
def normalize_column_dtypes(table_name: str, numeric_threshold: float = 0.90) -> str:
    """
    Detect object/string columns whose values are actually numeric (e.g. '100', '95.5',
    ' 200.00 ') and cast them to the correct dtype (int64 or float64), then overwrite
    the table in place.

    A column is considered numeric when at least `numeric_threshold` fraction of its
    non-null values successfully convert to a number (default 90 %).  Columns that
    are genuinely text (names, categories, free-text) are left unchanged.

    Args:
        table_name:         PostgreSQL table to normalize (overwritten in place).
        numeric_threshold:  Minimum fraction of non-null values that must be numeric
                            for the column to be cast (0.0–1.0, default 0.90).

    Returns:
        JSON with per-column conversion results and the final dtype map.
    """
    try:
        engine = create_engine(POSTGRES_URI)
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

        converted   = {}   # col -> new dtype
        unchanged   = {}   # col -> reason kept as-is

        for col in df.select_dtypes(include=["object"]).columns:
            # Strip whitespace before attempting conversion
            series = df[col].str.strip() if df[col].dtype == object else df[col]

            numeric_series = pd.to_numeric(series, errors="coerce")
            non_null_count = series.notna().sum()

            if non_null_count == 0:
                unchanged[col] = "all null"
                continue

            converted_count = numeric_series.notna().sum()
            ratio = converted_count / non_null_count

            if ratio >= numeric_threshold:
                # Cast: use int64 if no decimal point appears in any value, else float64
                has_decimal = series.dropna().str.contains(r"\.", regex=True).any()
                if has_decimal:
                    df[col] = numeric_series.astype("float64")
                    converted[col] = "float64"
                else:
                    # Use float first to handle NaN, then downcast to Int64 (nullable int)
                    df[col] = pd.to_numeric(numeric_series, errors="coerce").astype("float64")
                    if df[col].isna().sum() == 0:
                        df[col] = df[col].astype("int64")
                        converted[col] = "int64"
                    else:
                        converted[col] = "float64 (kept float due to NaN)"
            else:
                unchanged[col] = f"only {ratio:.0%} numeric — kept as string"

        df.to_sql(table_name, engine, if_exists="replace", index=False)

        return json.dumps({
            "status":           "success",
            "table":            table_name,
            "columns_converted": converted,
            "columns_unchanged": unchanged,
            "final_dtypes":     {col: str(dt) for col, dt in df.dtypes.items()},
            "row_count":        len(df),
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ============================================================
# Tool 5: List all tables in the data warehouse
# ============================================================
@tool("List Database Tables")
def list_database_tables() -> str:
    """
    List all tables currently in the PostgreSQL data warehouse with their sizes.

    Returns:
        JSON array of {table, size} objects.
    """
    try:
        engine = create_engine(POSTGRES_URI)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name,
                       pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) AS size
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """))
            tables = [{"table": row[0], "size": row[1]} for row in result]

        return json.dumps({"status": "success", "count": len(tables), "tables": tables}, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ============================================================
# Tool 6: Query the data warehouse
# ============================================================
@tool("Query Database")
def query_database(sql_query: str) -> str:
    """
    Execute a SQL SELECT query against the PostgreSQL data warehouse.

    Args:
        sql_query: A valid SELECT statement (no DDL/DML).

    Returns:
        JSON with column names and up to 200 rows of results.
    """
    if not sql_query.strip().upper().startswith("SELECT"):
        return json.dumps({"status": "error", "message": "Only SELECT queries are allowed."})

    try:
        engine = create_engine(POSTGRES_URI)
        df = pd.read_sql(sql_query, engine)

        return json.dumps({
            "status":  "success",
            "rows":    len(df),
            "columns": list(df.columns),
            "data":    df.head(200).to_dict(orient="records"),
        }, indent=2, default=str)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
