"""
Data Scientist Tools
--------------------
ML model training and evaluation for banking use-cases:
  - Fraud detection
  - Credit risk scoring
  - Customer churn prediction
  - Time-series forecasting (stock prices, revenue)
  - Customer segmentation

LLM used: NVIDIA Llama Nemotron (for model interpretation and recommendations)
"""

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", palette="husl")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai.tools import tool
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    mean_squared_error, r2_score,
)

from config import POSTGRES_URI, query_nvidia

# Absolute path so models save correctly regardless of CWD
_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_BASE_DIR, "outputs", "models")
_CHARTS_DIR = os.path.join(_BASE_DIR, "outputs", "charts")
os.makedirs(_MODELS_DIR, exist_ok=True)
os.makedirs(_CHARTS_DIR, exist_ok=True)


def _save_model(artifact: dict, name: str) -> str:
    path = os.path.join(_MODELS_DIR, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
    joblib.dump(artifact, path)
    return path


def _plot_feature_importance(
    feature_names: list,
    importances: list,
    model_name: str,
    top_n: int = 10,
) -> str:
    """
    Save a horizontal bar chart of the top-N most important features.
    Returns the saved PNG path.
    """
    # Sort descending, take top N
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_n]
    names  = [p[0] for p in reversed(pairs)]   # reversed so highest is on top
    values = [p[1] for p in reversed(pairs)]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("husl", len(names))
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.6)

    # Value labels on each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, max(values) * 1.15)
    plt.tight_layout()

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_CHARTS_DIR, f"feature_importance_{model_name.lower().replace(' ', '_')}_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def _preprocess(df: pd.DataFrame, target_col: str):
    """
    Drop target col, encode all categoricals (including string y labels),
    drop pure-ID columns, coerce TotalCharges to numeric, fill NaN.
    Returns X (numeric DataFrame) and y (numeric Series).
    """
    df = df.copy()

    # Drop known identifier columns
    id_cols = [c for c in df.columns if c.lower() in ("customerid", "id", "customer_id")]
    df.drop(columns=id_cols, errors="ignore", inplace=True)

    # Coerce TotalCharges (sometimes stored as string with spaces)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Encode string target: "Yes"→1, "No"→0 (alphabetical: No=0, Yes=1)
    if y.dtype == object or str(y.dtype) == "category":
        le_y = LabelEncoder()
        y = pd.Series(le_y.fit_transform(y.astype(str)), index=y.index, name=target_col)

    # Encode categorical features
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Keep only numeric columns and fill NaN
    X = X.select_dtypes(include=np.number).fillna(X.mean(numeric_only=True))
    return X, y


# ============================================================
# Tool 1: Fraud Detection
# ============================================================
@tool("Train Fraud Detection Model")
def train_fraud_detection_model(table_name: str, target_column: str = "is_fraud") -> str:
    """
    Train Random Forest + Gradient Boosting fraud detection models on banking transaction data.

    Args:
        table_name:    PostgreSQL table containing transaction records.
        target_column: Binary label column (0 = legitimate, 1 = fraud).

    Returns:
        JSON with per-model metrics, best model path, and NVIDIA-generated analysis.
    """
    try:
        engine = create_engine(POSTGRES_URI)
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

        if target_column not in df.columns:
            return json.dumps({
                "status":  "error",
                "message": f"Column '{target_column}' not found. Available: {list(df.columns)}",
            })

        X, y = _preprocess(df, target_column)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        candidates = {
            "RandomForest":     RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }

        results = {}
        best_name, best_model, best_auc = None, None, 0.0

        for name, clf in candidates.items():
            clf.fit(X_tr_s, y_tr)
            y_pred = clf.predict(X_te_s)
            y_prob = clf.predict_proba(X_te_s)[:, 1]
            auc = roc_auc_score(y_te, y_prob)
            results[name] = {
                "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
                "roc_auc":  round(float(auc), 4),
                "report":   classification_report(y_te, y_pred, output_dict=True),
            }
            if auc > best_auc:
                best_auc, best_name, best_model = auc, name, clf

        model_path = _save_model(
            {"model": best_model, "scaler": scaler, "features": list(X.columns)},
            "fraud_detection",
        )

        # Top 10 Feature Importance chart
        chart_path = _plot_feature_importance(
            feature_names=list(X.columns),
            importances=best_model.feature_importances_.tolist(),
            model_name="Fraud Detection",
        )

        analysis = query_nvidia([
            {"role": "system", "content": "You are a banking fraud analytics expert."},
            {"role": "user",   "content": (
                f"Fraud detection model results:\n{json.dumps(results, indent=2)}\n\n"
                "Provide: 1) Best model choice rationale, 2) Key fraud patterns to monitor, "
                "3) Business impact estimate, 4) Recommendations to reduce false positives."
            )},
        ])

        return json.dumps({
            "status":                  "success",
            "best_model":              best_name,
            "best_roc_auc":            round(best_auc, 4),
            "model_path":              model_path,
            "feature_importance_chart": chart_path,
            "all_results":             results,
            "ai_analysis":             analysis,
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ============================================================
# Tool 2: Credit Risk Scoring
# ============================================================
@tool("Train Credit Risk Model")
def train_credit_risk_model(table_name: str, target_column: str = "default") -> str:
    """
    Train a Logistic Regression credit-risk model and produce risk segments (Low/Medium/High/Very High).

    Args:
        table_name:    PostgreSQL table with loan / applicant data.
        target_column: Binary column – 1 = default, 0 = no default.

    Returns:
        JSON with AUC, risk segment distribution, model path, and NVIDIA analysis.
    """
    try:
        engine = create_engine(POSTGRES_URI)
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

        if target_column not in df.columns:
            return json.dumps({
                "status":  "error",
                "message": f"Column '{target_column}' not found. Available: {list(df.columns)}",
            })

        X, y = _preprocess(df, target_column)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(scaler.fit_transform(X_tr), y_tr)

        y_pred = clf.predict(scaler.transform(X_te))
        y_prob = clf.predict_proba(scaler.transform(X_te))[:, 1]
        auc = roc_auc_score(y_te, y_prob)

        # Credit scores 0-1000 (high score = low risk)
        credit_scores = (1 - y_prob) * 1000
        segments = pd.cut(
            credit_scores,
            bins=[0, 300, 500, 700, 1000],
            labels=["Very High Risk", "High Risk", "Medium Risk", "Low Risk"],
        )
        segment_dist = segments.value_counts().to_dict()

        model_path = _save_model(
            {"model": clf, "scaler": scaler, "features": list(X.columns)},
            "credit_risk",
        )

        # Top 10 Feature Importance chart (abs logistic regression coefficients)
        coef_importances = np.abs(clf.coef_[0]).tolist()
        chart_path = _plot_feature_importance(
            feature_names=list(X.columns),
            importances=coef_importances,
            model_name="Credit Risk",
        )

        analysis = query_nvidia([
            {"role": "system", "content": "You are a banking credit risk officer."},
            {"role": "user",   "content": (
                f"Credit risk model AUC: {auc:.4f}. "
                f"Risk segments: {str(segment_dist)}. "
                "Provide Basel III implications, approval policy recommendations, "
                "and portfolio risk management strategies."
            )},
        ])

        return json.dumps({
            "status":                   "success",
            "roc_auc":                  round(float(auc), 4),
            "accuracy":                 round(float(accuracy_score(y_te, y_pred)), 4),
            "risk_segments":            {str(k): int(v) for k, v in segment_dist.items()},
            "model_path":               model_path,
            "feature_importance_chart": chart_path,
            "ai_analysis":              analysis,
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ============================================================
# Tool 3: Customer Churn Prediction
# ============================================================
@tool("Train Customer Churn Model")
def train_churn_model(table_name: str, target_column: str = "churn") -> str:
    """
    Train a Gradient Boosting churn prediction model and identify top retention drivers.

    Args:
        table_name:    PostgreSQL table with customer records.
        target_column: Binary column – 1 = churned, 0 = retained.

    Returns:
        JSON with AUC, top churn predictors, model path, and NVIDIA retention strategy.
    """
    try:
        engine = create_engine(POSTGRES_URI)
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

        if target_column not in df.columns:
            return json.dumps({
                "status":  "error",
                "message": f"Column '{target_column}' not found. Available: {list(df.columns)}",
            })

        X, y = _preprocess(df, target_column)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
        clf.fit(scaler.fit_transform(X_tr), y_tr)

        y_pred = clf.predict(scaler.transform(X_te))
        y_prob = clf.predict_proba(scaler.transform(X_te))[:, 1]
        auc = roc_auc_score(y_te, y_prob)

        # Top 10 churn drivers
        feature_importance = dict(zip(list(X.columns), clf.feature_importances_.tolist()))
        top_features = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        model_path = _save_model(
            {"model": clf, "scaler": scaler, "features": list(X.columns)},
            "churn_model",
        )

        # Top 10 Feature Importance chart
        chart_path = _plot_feature_importance(
            feature_names=list(X.columns),
            importances=clf.feature_importances_.tolist(),
            model_name="Customer Churn",
        )

        analysis = query_nvidia([
            {"role": "system", "content": "You are a banking customer retention strategist."},
            {"role": "user",   "content": (
                f"Churn model AUC: {auc:.4f}. "
                f"Top churn predictors: {json.dumps(top_features)}. "
                "Design a 90-day customer retention program with targeted interventions."
            )},
        ])

        return json.dumps({
            "status":                   "success",
            "roc_auc":                  round(float(auc), 4),
            "accuracy":                 round(float(accuracy_score(y_te, y_pred)), 4),
            "top_churn_predictors":     {k: round(v, 4) for k, v in top_features.items()},
            "model_path":               model_path,
            "feature_importance_chart": chart_path,
            "ai_analysis":              analysis,
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ============================================================
# Tool 4: Time-Series Forecasting
# ============================================================
@tool("Perform Time Series Forecasting")
def time_series_forecast(
    table_name: str,
    date_column: str,
    value_column: str,
    forecast_periods: int = 30,
) -> str:
    """
    Forecast future values using Holt-Winters Exponential Smoothing.

    Args:
        table_name:       PostgreSQL table with time-series data.
        date_column:      Name of the date/datetime column.
        value_column:     Numeric column to forecast (e.g. 'Close', 'Revenue').
        forecast_periods: Number of future time steps to forecast (default 30).

    Returns:
        JSON with forecast range, trend direction, and NVIDIA financial interpretation.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        engine = create_engine(POSTGRES_URI)
        df = pd.read_sql(
            f"SELECT {date_column}, {value_column} FROM {table_name} ORDER BY {date_column}",
            engine,
        )

        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna().set_index(date_column)[value_column]

        if len(df) < 10:
            return json.dumps({"status": "error", "message": "Not enough data points for forecasting (need ≥ 10)."})

        seasonal_periods = min(30, len(df) // 3)
        model = ExponentialSmoothing(
            df,
            trend="add",
            seasonal="add" if len(df) >= seasonal_periods * 2 else None,
            seasonal_periods=seasonal_periods if len(df) >= seasonal_periods * 2 else None,
        ).fit(optimized=True)
        forecast = model.forecast(forecast_periods)

        # Save forecasts to DB
        fcast_df = pd.DataFrame({
            "forecast_date":    forecast.index.astype(str),
            "forecasted_value": forecast.values,
            "source_table":     table_name,
            "source_column":    value_column,
            "created_at":       datetime.now().isoformat(),
        })
        fcast_df.to_sql("forecasts", create_engine(POSTGRES_URI), if_exists="append", index=False)

        trend_dir = "upward" if float(forecast.iloc[-1]) > float(forecast.iloc[0]) else "downward"

        analysis = query_nvidia([
            {"role": "system", "content": "You are a banking financial analyst."},
            {"role": "user",   "content": (
                f"30-day forecast for '{value_column}' (table: {table_name}). "
                f"Last actual: {float(df.iloc[-1]):.2f}. "
                f"Forecast range: {float(forecast.min()):.2f}–{float(forecast.max()):.2f}. "
                f"Trend: {trend_dir}. "
                "Provide risk-adjusted financial interpretation and trading/investment recommendations."
            )},
        ])

        return json.dumps({
            "status":           "success",
            "source":           f"{table_name}.{value_column}",
            "historical_points": int(len(df)),
            "forecast_periods": forecast_periods,
            "trend":            trend_dir,
            "forecast_summary": {
                "min":  round(float(forecast.min()), 2),
                "max":  round(float(forecast.max()), 2),
                "mean": round(float(forecast.mean()), 2),
                "last": round(float(forecast.iloc[-1]), 2),
            },
            "forecast_data": {str(k): round(float(v), 2) for k, v in forecast.items()},
            "ai_analysis":   analysis,
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ============================================================
# Tool 5: Customer Segmentation (K-Means)
# ============================================================
@tool("Perform Customer Segmentation")
def customer_segmentation(table_name: str, n_clusters: int = 4) -> str:
    """
    Segment banking customers (or data rows) using K-Means clustering.

    Args:
        table_name: PostgreSQL table with numeric features.
        n_clusters: Number of segments to create (default 4).

    Returns:
        JSON with cluster sizes, mean profiles per segment, and NVIDIA segment descriptions.
    """
    try:
        engine = create_engine(POSTGRES_URI)
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

        # Drop identifier columns
        id_cols = [c for c in df.columns if c.lower() in ("customerid", "id", "customer_id")]
        df.drop(columns=id_cols, errors="ignore", inplace=True)

        # Coerce numeric-strings (e.g. TotalCharges stored as string)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Label-encode all remaining categorical columns so every feature is used
        for col in df.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        numeric_df = df.select_dtypes(include=np.number).fillna(0)
        if numeric_df.shape[1] < 2:
            return json.dumps({"status": "error", "message": "Need at least 2 numeric columns for segmentation."})

        scaler = StandardScaler()
        X_s = scaler.fit_transform(numeric_df)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["segment"] = kmeans.fit_predict(X_s)

        segment_counts   = df["segment"].value_counts().sort_index().to_dict()
        segment_profiles = (
            df.groupby("segment")[numeric_df.columns.tolist()]
            .mean()
            .round(3)
            .to_dict()
        )

        # Persist segmented table
        seg_table = f"{table_name}_segmented"
        df.to_sql(seg_table, create_engine(POSTGRES_URI), if_exists="replace", index=False)

        analysis = query_nvidia([
            {"role": "system", "content": "You are a banking customer analytics expert."},
            {"role": "user",   "content": (
                f"K-Means ({n_clusters} clusters) on '{table_name}'. "
                f"Cluster sizes: {segment_counts}. "
                f"Mean profiles: {json.dumps({str(k): {str(kk): float(vv) for kk, vv in v.items()} for k, v in segment_profiles.items()})}. "
                "Name each cluster (e.g. 'High-Value Loyalist', 'At-Risk Saver'), "
                "and recommend a tailored product/marketing strategy per segment."
            )},
        ])

        return json.dumps({
            "status":            "success",
            "n_clusters":        n_clusters,
            "segment_counts":    {str(k): int(v) for k, v in segment_counts.items()},
            "segment_profiles":  {str(k): {str(kk): float(vv) for kk, vv in v.items()} for k, v in segment_profiles.items()},
            "segmented_table":   seg_table,
            "ai_analysis":       analysis,
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
