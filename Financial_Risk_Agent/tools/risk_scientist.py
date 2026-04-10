"""
tools/risk_scientist.py – Quantitative Risk Modelling tools.

Provides: credit risk scoring, VaR/CVaR, Monte Carlo simulation,
stress testing, risk segmentation (4 tiers), and portfolio risk metrics.
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
from sklearn.metrics import accuracy_score, roc_auc_score

from config import query_nvidia


def _get_db_uri() -> str:
    uri = os.environ.get("ACTIVE_DB_URI", "")
    if not uri:
        raise RuntimeError("No active database URI.")
    return uri


_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_BASE_DIR, "outputs", "models")
_CHARTS_DIR = os.path.join(_BASE_DIR, "outputs", "charts")
os.makedirs(_MODELS_DIR, exist_ok=True)
os.makedirs(_CHARTS_DIR, exist_ok=True)


def _save_model(artifact: dict, name: str) -> str:
    path = os.path.join(_MODELS_DIR, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
    joblib.dump(artifact, path)
    return path


def _plot_feature_importance(feature_names, importances, model_name: str, top_n: int = 10) -> str:
    pairs  = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_n]
    names  = [p[0] for p in reversed(pairs)]
    values = [p[1] for p in reversed(pairs)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, values, color=sns.color_palette("husl", len(names)),
                   edgecolor="white", height=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Risk Factors — {model_name}", fontsize=14,
                 fontweight="bold", pad=15)
    plt.tight_layout()

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_CHARTS_DIR,
                        f"feature_importance_{model_name.lower().replace(' ', '_')}_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def _preprocess(df: pd.DataFrame, target_col: str):
    df = df.copy()
    id_patterns = {"id", "row_id", "rowid", "index", "uuid",
                   "customerid", "customer_id", "loanid", "loan_id",
                   "accountid", "account_id", "transactionid", "transaction_id"}
    id_cols = [c for c in df.columns if c.lower() in id_patterns]
    df.drop(columns=id_cols, errors="ignore", inplace=True)

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    if y.dtype == object or str(y.dtype) == "category":
        le_y = LabelEncoder()
        y    = pd.Series(le_y.fit_transform(y.astype(str)), index=y.index, name=target_col)

    for col in X.select_dtypes(include="object").columns:
        le    = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.select_dtypes(include=np.number).fillna(X.mean(numeric_only=True))
    return X, y


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool("Train Credit Risk Model")
def train_credit_risk_model(table_name: str, target_column: str = "default") -> str:
    """
    Train a credit risk model (Logistic Regression + Gradient Boosting) to predict
    probability of default (PD) and assign credit risk scores (0–1000).

    Args:
        table_name:    Table with loan / applicant / credit data.
        target_column: Binary default column (1=default, 0=no default).

    Returns:
        JSON with AUC-ROC, credit score distribution, risk segments,
        top risk factors, model path, and AI analysis.
    """
    try:
        engine = create_engine(_get_db_uri())
        df     = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", engine)

        if target_column not in df.columns:
            return json.dumps({"status": "error",
                               "message": f"Column '{target_column}' not found. "
                                          f"Available: {list(df.columns)}"})

        X, y   = _preprocess(df, target_column)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()

        # Logistic Regression (baseline)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        lr.fit(scaler.fit_transform(X_tr), y_tr)
        lr_prob = lr.predict_proba(scaler.transform(X_te))[:, 1]
        lr_auc  = roc_auc_score(y_te, lr_prob)

        # Gradient Boosting (primary)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(scaler.fit_transform(X_tr), y_tr)
        gb_prob = gb.predict_proba(scaler.transform(X_te))[:, 1]
        gb_auc  = roc_auc_score(y_te, gb_prob)

        best_model = gb if gb_auc >= lr_auc else lr
        best_prob  = gb_prob if gb_auc >= lr_auc else lr_prob
        best_auc   = max(gb_auc, lr_auc)
        best_name  = "GradientBoosting" if gb_auc >= lr_auc else "LogisticRegression"

        # Credit score (0–1000, higher = lower risk)
        credit_scores = (1 - best_prob) * 1000
        segments      = pd.cut(credit_scores,
                               bins=[0, 300, 500, 700, 1000],
                               labels=["Critical Risk", "High Risk", "Medium Risk", "Low Risk"])
        segment_dist  = segments.value_counts().to_dict()

        model_path  = _save_model(
            {"model": best_model, "scaler": scaler, "features": list(X.columns)}, "credit_risk"
        )
        chart_path = _plot_feature_importance(
            list(X.columns),
            (best_model.feature_importances_.tolist()
             if hasattr(best_model, "feature_importances_")
             else np.abs(best_model.coef_[0]).tolist()),
            "Credit Risk"
        )

        top_features = dict(
            sorted(
                zip(list(X.columns),
                    (best_model.feature_importances_.tolist()
                     if hasattr(best_model, "feature_importances_")
                     else np.abs(best_model.coef_[0]).tolist())),
                key=lambda x: x[1], reverse=True
            )[:5]
        )

        try:
            analysis = query_nvidia([
                {"role": "system",
                 "content": "You are a senior credit risk officer. Reference Basel III / IFRS 9."},
                {"role": "user",
                 "content": (f"Credit risk model results:\n"
                             f"Best model: {best_name}, AUC-ROC: {best_auc:.4f}\n"
                             f"Risk segments: {str(segment_dist)}\n"
                             f"Top risk factors: {json.dumps(top_features)}\n"
                             "Provide risk interpretation and Basel III / IFRS 9 commentary.")},
            ])
        except Exception as _e:
            analysis = (f"AI analysis unavailable ({_e}). "
                        f"Best model: {best_name}, AUC-ROC: {best_auc:.4f}")

        return json.dumps({
            "status":         "success",
            "best_model":     best_name,
            "roc_auc":        round(float(best_auc), 4),
            "accuracy":       round(float(accuracy_score(y_te, best_model.predict(
                                  scaler.transform(X_te)))), 4),
            "risk_segments":  {str(k): int(v) for k, v in segment_dist.items()},
            "top_risk_factors": {k: round(v, 4) for k, v in top_features.items()},
            "model_path":     model_path,
            "feature_importance_chart": chart_path,
            "ai_analysis":    analysis,
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Compute Value at Risk (VaR and CVaR)")
def compute_var_cvar(table_name: str, value_column: str,
                     confidence_levels: str = "0.95,0.99") -> str:
    """
    Compute Historical VaR and CVaR (Expected Shortfall) for a loss/return column.

    Args:
        table_name:         Table containing loss or return data.
        value_column:       Numeric column with losses (positive = loss) or returns.
        confidence_levels:  Comma-separated confidence levels, e.g. '0.95,0.99'.

    Returns:
        JSON with VaR, CVaR, distribution stats, and risk interpretation.
    """
    try:
        engine = create_engine(_get_db_uri())
        df     = pd.read_sql(f"SELECT {value_column} FROM {table_name} LIMIT 10000", engine)
        df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
        series = df[value_column].dropna()

        if len(series) < 30:
            return json.dumps({"status": "error",
                               "message": "Need ≥ 30 data points for reliable VaR."})

        levels  = [float(c.strip()) for c in confidence_levels.split(",") if c.strip()]
        results = {}
        for cl in levels:
            var  = float(np.percentile(series, (1 - cl) * 100))
            cvar = float(series[series <= var].mean())
            results[f"{int(cl*100)}%"] = {
                "VaR":   round(var,  4),
                "CVaR":  round(cvar, 4),
                "label": f"VaR {int(cl*100)}% = {var:.4f}; CVaR = {cvar:.4f}",
            }

        dist_stats = {
            "mean":     round(float(series.mean()),  4),
            "std":      round(float(series.std()),   4),
            "skewness": round(float(series.skew()),  4),
            "kurtosis": round(float(series.kurt()),  4),
            "min":      round(float(series.min()),   4),
            "max":      round(float(series.max()),   4),
            "n":        int(len(series)),
        }

        try:
            analysis = query_nvidia([
                {"role": "system",
                 "content": "You are a market risk officer specialising in VaR methodologies."},
                {"role": "user",
                 "content": (f"VaR/CVaR results for column '{value_column}':\n"
                             f"{json.dumps(results, indent=2)}\n"
                             f"Distribution: {json.dumps(dist_stats)}\n"
                             "Interpret these risk metrics and provide FRTB / Basel III commentary.")},
            ])
        except Exception as _e:
            analysis = f"AI analysis unavailable ({_e})."

        return json.dumps({
            "status":     "success",
            "column":     value_column,
            "table":      table_name,
            "var_cvar":   results,
            "distribution_stats": dist_stats,
            "ai_analysis": analysis,
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Run Monte Carlo Stress Test")
def monte_carlo_stress_test(table_name: str, value_column: str,
                            simulations: int = 1000, horizon_days: int = 30) -> str:
    """
    Run a Monte Carlo simulation to stress-test a portfolio or position.
    Simulates future loss/return distributions under shock scenarios.

    Args:
        table_name:     Table with historical financial data.
        value_column:   Numeric column (returns, P&L, or exposure amounts).
        simulations:    Number of Monte Carlo paths (default 1000).
        horizon_days:   Forecast horizon in days (default 30).

    Returns:
        JSON with simulated VaR, CVaR, worst-case scenario, and stress test summary.
    """
    try:
        engine = create_engine(_get_db_uri())
        df     = pd.read_sql(f"SELECT {value_column} FROM {table_name} LIMIT 10000", engine)
        df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
        returns = df[value_column].dropna()

        if len(returns) < 20:
            return json.dumps({"status": "error", "message": "Need ≥ 20 data points."})

        mu    = float(returns.mean())
        sigma = float(returns.std())

        np.random.seed(42)
        paths  = np.random.normal(mu, sigma, size=(simulations, horizon_days))
        total  = paths.sum(axis=1)

        var_95  = float(np.percentile(total, 5))
        var_99  = float(np.percentile(total, 1))
        cvar_95 = float(total[total <= var_95].mean())
        cvar_99 = float(total[total <= var_99].mean())

        # Shock scenarios
        shock_factor = 2.5
        shock_paths  = np.random.normal(mu, sigma * shock_factor,
                                        size=(simulations, horizon_days))
        shock_total  = shock_paths.sum(axis=1)
        shock_var_99 = float(np.percentile(shock_total, 1))

        # Save chart
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(_CHARTS_DIR, f"montecarlo_{table_name}_{ts}.png")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(total, bins=50, color="#3b82f6", alpha=0.75, edgecolor="white")
        axes[0].axvline(var_95, color="#f59e0b", linestyle="--", linewidth=2,
                        label=f"VaR 95%: {var_95:.2f}")
        axes[0].axvline(var_99, color="#ef4444", linestyle="--", linewidth=2,
                        label=f"VaR 99%: {var_99:.2f}")
        axes[0].legend()
        axes[0].set_title(f"Monte Carlo Loss Distribution ({simulations:,} paths)",
                          fontweight="bold")
        axes[0].set_xlabel("Simulated P&L")
        axes[0].set_ylabel("Frequency")

        axes[1].hist(shock_total, bins=50, color="#ef4444", alpha=0.7, edgecolor="white")
        axes[1].axvline(shock_var_99, color="#7c3aed", linestyle="--", linewidth=2,
                        label=f"Shock VaR 99%: {shock_var_99:.2f}")
        axes[1].legend()
        axes[1].set_title(f"Stress Scenario (σ × {shock_factor})", fontweight="bold")
        axes[1].set_xlabel("Simulated P&L under Stress")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        # Persist chart metadata
        _meta_path = os.path.join(_CHARTS_DIR, "session_charts.json")
        try:
            _existing: list = []
            if os.path.exists(_meta_path):
                import json as _json
                with open(_meta_path, "r", encoding="utf-8") as _fh:
                    _existing = _json.load(_fh)
            _existing.append({
                "title":           f"Monte Carlo Stress Test — {value_column}",
                "chart_type":      "montecarlo",
                "chart_url":       "/outputs/charts/" + os.path.basename(path),
                "gemini_analysis": (f"VaR 95%: {var_95:.4f} | VaR 99%: {var_99:.4f} | "
                                    f"CVaR 95%: {cvar_95:.4f} | Shock VaR 99%: {shock_var_99:.4f}"),
                "nvidia_recommendations": "",
            })
            with open(_meta_path, "w", encoding="utf-8") as _fh:
                import json as _json2
                _json2.dump(_existing, _fh, indent=2, ensure_ascii=False)
        except Exception:
            pass

        try:
            analysis = query_nvidia([
                {"role": "system",
                 "content": "You are a quantitative risk analyst specialising in stress testing."},
                {"role": "user",
                 "content": (f"Monte Carlo stress test on '{value_column}' ({simulations} paths, "
                             f"{horizon_days}-day horizon):\n"
                             f"VaR 95%: {var_95:.4f}\nVaR 99%: {var_99:.4f}\n"
                             f"CVaR 95%: {cvar_95:.4f}\nCVaR 99%: {cvar_99:.4f}\n"
                             f"Shock VaR 99% (σ×{shock_factor}): {shock_var_99:.4f}\n"
                             "Interpret these stress test results under Basel III / FRTB framework.")},
            ])
        except Exception as _e:
            analysis = (f"AI analysis unavailable ({_e}). "
                        f"VaR 99%: {var_99:.4f}, Shock VaR 99%: {shock_var_99:.4f}")

        return json.dumps({
            "status":       "success",
            "table":        table_name,
            "column":       value_column,
            "simulations":  simulations,
            "horizon_days": horizon_days,
            "base_scenario": {
                "var_95":   round(float(var_95),  4),
                "var_99":   round(float(var_99),  4),
                "cvar_95":  round(float(cvar_95), 4),
                "cvar_99":  round(float(cvar_99), 4),
            },
            "stress_scenario": {
                "shock_multiplier": shock_factor,
                "shock_var_99":     round(float(shock_var_99), 4),
            },
            "chart_path":  path,
            "ai_analysis": analysis,
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Perform Risk Segmentation")
def risk_segmentation(table_name: str, n_clusters: int = 4) -> str:
    """
    Segment entities (loans, customers, counterparties) into risk tiers
    using K-Means clustering: Critical / High / Medium / Low risk.

    Args:
        table_name: Table with numeric risk features.
        n_clusters: Number of risk tiers (default 4).

    Returns:
        JSON with tier sizes, mean profiles per tier, and segment descriptions.
    """
    try:
        engine = create_engine(_get_db_uri())
        df     = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", engine)

        id_patterns = {"id", "row_id", "rowid", "index", "uuid",
                       "customerid", "customer_id", "loanid", "loan_id",
                       "accountid", "account_id"}
        df.drop(columns=[c for c in df.columns if c.lower() in id_patterns],
                errors="ignore", inplace=True)

        for col in df.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        numeric_df = df.select_dtypes(include=np.number).fillna(0)
        if numeric_df.shape[1] < 2:
            return json.dumps({"status": "error",
                               "message": "Need at least 2 numeric columns."})

        scaler = StandardScaler()
        X_s    = scaler.fit_transform(numeric_df)

        kmeans    = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["risk_tier"] = kmeans.fit_predict(X_s)

        tier_counts = df["risk_tier"].value_counts().sort_index().to_dict()
        seg_table   = f"{table_name}_risk_tiers"
        df.to_sql(seg_table, create_engine(_get_db_uri()), if_exists="replace", index=False)

        try:
            analysis = query_nvidia([
                {"role": "system",
                 "content": "You are a risk analyst specialising in portfolio segmentation."},
                {"role": "user",
                 "content": (f"K-Means risk segmentation ({n_clusters} tiers) on '{table_name}'.\n"
                             f"Tier sizes: {tier_counts}\n"
                             "Name each tier (Critical/High/Medium/Low Risk), describe their "
                             "key characteristics, and recommend risk treatment strategies.")},
            ])
        except Exception as _e:
            analysis = f"AI analysis unavailable ({_e}). Tier sizes: {tier_counts}"

        return json.dumps({
            "status":        "success",
            "n_tiers":       n_clusters,
            "tier_counts":   {str(k): int(v) for k, v in tier_counts.items()},
            "segmented_table": seg_table,
            "ai_analysis":   analysis,
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@tool("Train Fraud Detection Model")
def train_fraud_model(table_name: str, target_column: str = "is_fraud") -> str:
    """
    Train a Random Forest + Gradient Boosting fraud detection model.

    Args:
        table_name:    Table containing transaction records.
        target_column: Binary fraud label (0=legitimate, 1=fraud).

    Returns:
        JSON with AUC-ROC, precision/recall, top fraud indicators, and model path.
    """
    try:
        engine = create_engine(_get_db_uri())
        df     = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", engine)

        if target_column not in df.columns:
            return json.dumps({"status": "error",
                               "message": f"Column '{target_column}' not found. "
                                          f"Available: {list(df.columns)}"})

        X, y   = _preprocess(df, target_column)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        candidates = {
            "RandomForest":     RandomForestClassifier(n_estimators=100, random_state=42,
                                                        class_weight="balanced"),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        results    = {}
        best_name, best_model, best_auc = None, None, 0.0

        for name, clf in candidates.items():
            clf.fit(X_tr_s, y_tr)
            y_prob = clf.predict_proba(X_te_s)[:, 1]
            auc    = roc_auc_score(y_te, y_prob)
            results[name] = {
                "accuracy": round(float(accuracy_score(y_te, clf.predict(X_te_s))), 4),
                "roc_auc":  round(float(auc), 4),
            }
            if auc > best_auc:
                best_auc, best_name, best_model = auc, name, clf

        model_path = _save_model(
            {"model": best_model, "scaler": scaler, "features": list(X.columns)}, "fraud_detection"
        )
        chart_path = _plot_feature_importance(
            list(X.columns), best_model.feature_importances_.tolist(), "Fraud Detection"
        )

        try:
            analysis = query_nvidia([
                {"role": "system",
                 "content": "You are a financial crime and fraud risk expert."},
                {"role": "user",
                 "content": (f"Fraud detection results:\n{json.dumps(results, indent=2)}\n"
                             "Provide fraud pattern analysis and AML compliance commentary.")},
            ])
        except Exception as _e:
            analysis = (f"AI analysis unavailable ({_e}). "
                        f"Best model: {best_name}, AUC: {best_auc:.4f}")

        return json.dumps({
            "status":        "success",
            "best_model":    best_name,
            "best_roc_auc":  round(float(best_auc), 4),
            "all_results":   results,
            "model_path":    model_path,
            "feature_importance_chart": chart_path,
            "ai_analysis":   analysis,
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
