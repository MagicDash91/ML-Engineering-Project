"""
tools/bank_scientist.py – Banking ML model training tools.
Mirrors Bank_Agent/tools/scientist_tools.py with paths adjusted to this system.
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
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from config import POSTGRES_URI, query_nvidia


def _get_db_uri() -> str:
    """Return the active database URI (set per-run via ACTIVE_DB_URI env var)."""
    return os.environ.get("ACTIVE_DB_URI") or POSTGRES_URI

_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_BASE_DIR, "outputs", "models")
_CHARTS_DIR = os.path.join(_BASE_DIR, "outputs", "charts")
os.makedirs(_MODELS_DIR, exist_ok=True)
os.makedirs(_CHARTS_DIR, exist_ok=True)


def _save_model(artifact: dict, name: str) -> str:
    path = os.path.join(_MODELS_DIR, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
    joblib.dump(artifact, path)
    return path


def _plot_feature_importance(feature_names, importances, model_name, top_n=10) -> str:
    pairs  = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_n]
    names  = [p[0] for p in reversed(pairs)]
    values = [p[1] for p in reversed(pairs)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, values, color=sns.color_palette("husl", len(names)), edgecolor="white", height=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_CHARTS_DIR, f"feature_importance_{model_name.lower().replace(' ', '_')}_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def _preprocess(df: pd.DataFrame, target_col: str):
    df = df.copy()
    id_cols = [c for c in df.columns if c.lower() in ("customerid", "id", "customer_id")]
    df.drop(columns=id_cols, errors="ignore", inplace=True)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    if y.dtype == object or str(y.dtype) == "category":
        le_y = LabelEncoder()
        y = pd.Series(le_y.fit_transform(y.astype(str)), index=y.index, name=target_col)

    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.select_dtypes(include=np.number).fillna(X.mean(numeric_only=True))
    return X, y


@tool("Train Fraud Detection Model")
def train_fraud_detection_model(table_name: str, target_column: str = "is_fraud") -> str:
    """
    Train Random Forest + Gradient Boosting fraud detection models.

    Args:
        table_name:    PostgreSQL table containing transaction records.
        target_column: Binary label column (0 = legitimate, 1 = fraud).

    Returns:
        JSON with per-model metrics, best model path, and analysis.
    """
    try:
        engine = create_engine(_get_db_uri())
        df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", engine)

        if target_column not in df.columns:
            return json.dumps({"status": "error",
                               "message": f"Column '{target_column}' not found. Available: {list(df.columns)}"})

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
            results[name] = {"accuracy": round(float(accuracy_score(y_te, y_pred)), 4), "roc_auc": round(float(auc), 4)}
            if auc > best_auc:
                best_auc, best_name, best_model = auc, name, clf

        model_path  = _save_model({"model": best_model, "scaler": scaler, "features": list(X.columns)}, "fraud_detection")
        chart_path  = _plot_feature_importance(list(X.columns), best_model.feature_importances_.tolist(), "Fraud Detection")
        try:
            analysis = query_nvidia([
                {"role": "system", "content": "You are a banking fraud analytics expert."},
                {"role": "user",   "content": f"Fraud detection results:\n{json.dumps(results, indent=2)}\nProvide analysis."},
            ])
        except Exception as _nvidia_err:
            print(f"[Scientist] NVIDIA analysis skipped: {_nvidia_err}", flush=True)
            analysis = f"AI analysis unavailable (NVIDIA timeout). Best model: {best_name}, ROC-AUC: {round(best_auc, 4)}"

        return json.dumps({"status": "success", "best_model": best_name, "best_roc_auc": round(best_auc, 4),
                           "model_path": model_path, "feature_importance_chart": chart_path,
                           "all_results": results, "ai_analysis": analysis}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Train Credit Risk Model")
def train_credit_risk_model(table_name: str, target_column: str = "default") -> str:
    """
    Train a Logistic Regression credit-risk model.

    Args:
        table_name:    PostgreSQL table with loan/applicant data.
        target_column: Binary column – 1 = default, 0 = no default.

    Returns:
        JSON with AUC, risk segment distribution, model path, and analysis.
    """
    try:
        engine = create_engine(_get_db_uri())
        df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", engine)

        if target_column not in df.columns:
            return json.dumps({"status": "error",
                               "message": f"Column '{target_column}' not found. Available: {list(df.columns)}"})

        X, y = _preprocess(df, target_column)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(scaler.fit_transform(X_tr), y_tr)

        y_pred = clf.predict(scaler.transform(X_te))
        y_prob = clf.predict_proba(scaler.transform(X_te))[:, 1]
        auc    = roc_auc_score(y_te, y_prob)

        credit_scores = (1 - y_prob) * 1000
        segments = pd.cut(credit_scores, bins=[0, 300, 500, 700, 1000],
                          labels=["Very High Risk", "High Risk", "Medium Risk", "Low Risk"])
        segment_dist = segments.value_counts().to_dict()

        model_path = _save_model({"model": clf, "scaler": scaler, "features": list(X.columns)}, "credit_risk")
        chart_path = _plot_feature_importance(list(X.columns), np.abs(clf.coef_[0]).tolist(), "Credit Risk")
        try:
            analysis = query_nvidia([
                {"role": "system", "content": "You are a banking credit risk officer."},
                {"role": "user",   "content": f"Credit risk AUC: {auc:.4f}. Segments: {str(segment_dist)}. Provide Basel III analysis."},
            ])
        except Exception as _nvidia_err:
            print(f"[Scientist] NVIDIA analysis skipped: {_nvidia_err}", flush=True)
            analysis = f"AI analysis unavailable (NVIDIA timeout). ROC-AUC: {round(float(auc), 4)}"

        return json.dumps({"status": "success", "roc_auc": round(float(auc), 4),
                           "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
                           "risk_segments": {str(k): int(v) for k, v in segment_dist.items()},
                           "model_path": model_path, "feature_importance_chart": chart_path,
                           "ai_analysis": analysis}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Train Customer Churn Model")
def train_churn_model(table_name: str, target_column: str = "churn") -> str:
    """
    Train a Gradient Boosting churn prediction model.

    Args:
        table_name:    PostgreSQL table with customer records.
        target_column: Binary column – 1 = churned, 0 = retained.

    Returns:
        JSON with AUC, top churn predictors, model path, and retention strategy.
    """
    try:
        engine = create_engine(_get_db_uri())
        df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", engine)

        if target_column not in df.columns:
            return json.dumps({"status": "error",
                               "message": f"Column '{target_column}' not found. Available: {list(df.columns)}"})

        X, y = _preprocess(df, target_column)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
        clf.fit(scaler.fit_transform(X_tr), y_tr)

        y_pred = clf.predict(scaler.transform(X_te))
        y_prob = clf.predict_proba(scaler.transform(X_te))[:, 1]
        auc    = roc_auc_score(y_te, y_prob)

        feature_importance = dict(zip(list(X.columns), clf.feature_importances_.tolist()))
        top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])

        model_path = _save_model({"model": clf, "scaler": scaler, "features": list(X.columns)}, "churn_model")
        chart_path = _plot_feature_importance(list(X.columns), clf.feature_importances_.tolist(), "Customer Churn")
        try:
            analysis = query_nvidia([
                {"role": "system", "content": "You are a banking customer retention strategist."},
                {"role": "user",   "content": f"Churn AUC: {auc:.4f}. Top predictors: {json.dumps(top_features)}. Design a 90-day retention program."},
            ])
        except Exception as _nvidia_err:
            print(f"[Scientist] NVIDIA analysis skipped: {_nvidia_err}", flush=True)
            analysis = f"AI analysis unavailable (NVIDIA timeout). ROC-AUC: {round(float(auc), 4)}"

        return json.dumps({"status": "success", "roc_auc": round(float(auc), 4),
                           "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
                           "top_churn_predictors": {k: round(v, 4) for k, v in top_features.items()},
                           "model_path": model_path, "feature_importance_chart": chart_path,
                           "ai_analysis": analysis}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Perform Time Series Forecasting")
def time_series_forecast(table_name: str, date_column: str, value_column: str, forecast_periods: int = 30) -> str:
    """
    Forecast future values using Holt-Winters Exponential Smoothing.

    Args:
        table_name:       PostgreSQL table with time-series data.
        date_column:      Name of the date/datetime column.
        value_column:     Numeric column to forecast.
        forecast_periods: Number of future time steps to forecast.

    Returns:
        JSON with forecast range, trend direction, and financial interpretation.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        engine = create_engine(_get_db_uri())
        df = pd.read_sql(f"SELECT {date_column}, {value_column} FROM {table_name} ORDER BY {date_column} LIMIT 10000", engine)
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna().set_index(date_column)[value_column]

        if len(df) < 10:
            return json.dumps({"status": "error", "message": "Not enough data points (need ≥ 10)."})

        seasonal_periods = min(30, len(df) // 3)
        model = ExponentialSmoothing(
            df, trend="add",
            seasonal="add" if len(df) >= seasonal_periods * 2 else None,
            seasonal_periods=seasonal_periods if len(df) >= seasonal_periods * 2 else None,
        ).fit(optimized=True)
        forecast = model.forecast(forecast_periods)

        trend_dir = "upward" if float(forecast.iloc[-1]) > float(forecast.iloc[0]) else "downward"
        try:
            analysis = query_nvidia([
                {"role": "system", "content": "You are a banking financial analyst."},
                {"role": "user",   "content": f"30-day forecast for '{value_column}'. Trend: {trend_dir}. Provide investment analysis."},
            ])
        except Exception as _nvidia_err:
            print(f"[Scientist] NVIDIA analysis skipped: {_nvidia_err}", flush=True)
            analysis = f"AI analysis unavailable (NVIDIA timeout). Trend: {trend_dir}"

        return json.dumps({"status": "success", "source": f"{table_name}.{value_column}",
                           "trend": trend_dir, "forecast_summary": {
                               "min": round(float(forecast.min()), 2),
                               "max": round(float(forecast.max()), 2),
                               "mean": round(float(forecast.mean()), 2),
                           }, "ai_analysis": analysis}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Perform Customer Segmentation")
def customer_segmentation(table_name: str, n_clusters: int = 4) -> str:
    """
    Segment banking customers using K-Means clustering.

    Args:
        table_name: PostgreSQL table with numeric features.
        n_clusters: Number of segments to create (default 4).

    Returns:
        JSON with cluster sizes, mean profiles per segment, and segment descriptions.
    """
    try:
        engine = create_engine(_get_db_uri())
        df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", engine)

        id_cols = [c for c in df.columns if c.lower() in ("customerid", "id", "customer_id")]
        df.drop(columns=id_cols, errors="ignore", inplace=True)

        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        for col in df.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        numeric_df = df.select_dtypes(include=np.number).fillna(0)
        if numeric_df.shape[1] < 2:
            return json.dumps({"status": "error", "message": "Need at least 2 numeric columns."})

        scaler = StandardScaler()
        X_s = scaler.fit_transform(numeric_df)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["segment"] = kmeans.fit_predict(X_s)

        segment_counts = df["segment"].value_counts().sort_index().to_dict()

        seg_table = f"{table_name}_segmented"
        df.to_sql(seg_table, create_engine(_get_db_uri()), if_exists="replace", index=False)

        try:
            analysis = query_nvidia([
                {"role": "system", "content": "You are a banking customer analytics expert."},
                {"role": "user",   "content": f"K-Means ({n_clusters} clusters) on '{table_name}'. Cluster sizes: {segment_counts}. Name each cluster and recommend strategy."},
            ])
        except Exception as _nvidia_err:
            print(f"[Scientist] NVIDIA analysis skipped: {_nvidia_err}", flush=True)
            analysis = f"AI analysis unavailable (NVIDIA timeout). Cluster sizes: {segment_counts}"

        return json.dumps({"status": "success", "n_clusters": n_clusters,
                           "segment_counts": {str(k): int(v) for k, v in segment_counts.items()},
                           "segmented_table": seg_table, "ai_analysis": analysis}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
