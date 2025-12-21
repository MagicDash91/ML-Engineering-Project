"""
MediWatch: AI-Powered Healthcare Analytics Platform
FastAPI Backend with Integrated Web Frontend
"""

import os
import io
import base64
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import snowflake.connector
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import google.generativeai as genai
from PIL import Image

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="MediWatch API",
    description="AI-Powered Healthcare Analytics Platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
gemini_vision_model = genai.GenerativeModel('gemini-2.5-flash')

# Create static directory for visualizations
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Snowflake configuration
SNOWFLAKE_CONFIG = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "role": os.getenv("SNOWFLAKE_ROLE")
}

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ==================== DATA MODELS ====================

class RiskLevel(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class AnomalySeverity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QueryLanguage(str, Enum):
    EN = "en"
    ID = "id"


class PatientRisk(BaseModel):
    patient_id: str
    name: str
    age: int
    condition: str
    risk_score: float
    risk_level: RiskLevel
    last_admission: str
    contributing_factors: List[str]
    recommendations: List[str]


class BillingAnomaly(BaseModel):
    patient_id: str
    name: str
    condition: str
    actual_amount: float
    expected_amount: float
    deviation_pct: float
    severity: AnomalySeverity
    reason: str
    recommendation: str


class NaturalLanguageQuery(BaseModel):
    question: str = Field(..., min_length=3)
    language: QueryLanguage = QueryLanguage.EN
    conversation_id: Optional[str] = None


class ErrorResponse(BaseModel):
    error: Dict[str, Any]


# ==================== DATABASE CONNECTION ====================

def get_snowflake_connection():
    """Create and return a Snowflake connection"""
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        return conn
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "DATABASE_CONNECTION_ERROR",
                "message": f"Failed to connect to Snowflake: {str(e)}"
            }
        )


def execute_query(query: str, params: Optional[Dict] = None) -> pd.DataFrame:
    """Execute a Snowflake query and return results as DataFrame"""
    conn = get_snowflake_connection()
    try:
        df = pd.read_sql(query, conn, params=params)
        return df
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "QUERY_EXECUTION_ERROR",
                "message": f"Failed to execute query: {str(e)}"
            }
        )
    finally:
        conn.close()


# ==================== RISK SCORING FUNCTIONS ====================

def calculate_risk_score(row: pd.Series) -> float:
    """Calculate readmission risk score based on patient attributes"""
    age_risk = min(row['age'] / 100, 1.0) if 'age' in row else 0

    high_risk_conditions = ['Cancer', 'Heart Disease', 'Diabetes', 'Stroke']
    condition_severity = 0.8 if row.get('medicalcondition') in high_risk_conditions else 0.4

    prior_admissions = row.get('admission_count', 1) / 5.0
    prior_admissions = min(prior_admissions, 1.0)

    if 'dateofadmission' in row and 'dischargedate' in row and pd.notna(row['dischargedate']):
        try:
            los = (pd.to_datetime(row['dischargedate']) - pd.to_datetime(row['dateofadmission'])).days
            length_of_stay = min(los / 30, 1.0)
        except:
            length_of_stay = 0.5
    else:
        length_of_stay = 0.5

    test_abnormality = 0.8 if row.get('testresults') == 'Abnormal' else 0.2

    risk_score = (
        age_risk * 0.20 +
        condition_severity * 0.30 +
        prior_admissions * 0.25 +
        length_of_stay * 0.15 +
        test_abnormality * 0.10
    )

    return round(risk_score * 100, 2)


def get_risk_level(score: float) -> RiskLevel:
    """Determine risk level based on score"""
    if score >= 80:
        return RiskLevel.CRITICAL
    elif score >= 60:
        return RiskLevel.HIGH
    elif score >= 40:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


def get_contributing_factors(row: pd.Series, score: float) -> List[str]:
    """Generate list of contributing factors for high risk"""
    factors = []

    if row.get('age', 0) > 65:
        factors.append(f"Advanced age ({row['age']} years)")

    admission_count = row.get('admission_count', 1)
    if admission_count > 2:
        factors.append(f"Multiple prior admissions ({admission_count})")

    if row.get('testresults') == 'Abnormal':
        factors.append("Abnormal test results")

    high_risk_conditions = ['Cancer', 'Heart Disease', 'Diabetes', 'Stroke']
    if row.get('medicalcondition') in high_risk_conditions:
        factors.append(f"High-complexity condition ({row['medicalcondition']})")

    return factors if factors else ["General risk factors"]


def get_recommendations(row: pd.Series, risk_level: RiskLevel) -> List[str]:
    """Generate recommendations based on risk level"""
    recommendations = []

    if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
        recommendations.append("Schedule follow-up within 7 days")
        recommendations.append("Assign dedicated care coordinator")
        recommendations.append("Monitor medication adherence closely")

        if row.get('testresults') == 'Abnormal':
            recommendations.append("Repeat laboratory tests within 48 hours")
    elif risk_level == RiskLevel.MEDIUM:
        recommendations.append("Schedule follow-up within 14 days")
        recommendations.append("Provide patient education materials")
    else:
        recommendations.append("Standard follow-up protocol")

    return recommendations


# ==================== VISUALIZATION FUNCTIONS ====================

def create_chart_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def save_and_analyze_chart(fig, filename: str, analysis_context: str, language: str = "English") -> Dict[str, str]:
    """Save chart to static folder and analyze with Gemini Vision"""
    try:
        # Save the figure to static folder
        filepath = STATIC_DIR / filename
        fig.savefig(filepath, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Open image for Gemini analysis
        img = Image.open(filepath)

        # Create comprehensive analysis prompt
        prompt = f"""
Please respond in {language}.

You are acting as a senior healthcare data analyst responsible for analyzing visualization charts derived from medical and billing data. Your task is to extract strategic insights from the visual patterns, trends, and distributions shown in this healthcare analytics chart.

Context: {analysis_context}

Please provide a comprehensive, structured analysis:

1. **Visual Pattern Analysis** (2-3 sentences)
   - Describe the key visual patterns, distributions, and trends you observe
   - Identify any notable outliers, clusters, or anomalies in the data
   - Highlight the most significant visual findings

2. **Healthcare Implications** (3-4 bullet points)
   - Interpret what these patterns mean for patient care and outcomes
   - Identify potential risk areas or opportunities for intervention
   - Note any correlations or relationships between variables
   - Assess whether the patterns indicate normal operations or areas of concern

3. **Strategic Insights** (2-3 bullet points)
   - What do these visual patterns tell us about operational efficiency?
   - Are there resource allocation implications?
   - What trends should healthcare administrators monitor?

4. **Actionable Recommendations** (3-4 specific suggestions)
   - Provide practical, measurable recommendations based on visual insights
   - Each recommendation should address a specific pattern or trend observed
   - Focus on improving patient care, reducing costs, or enhancing operational efficiency

Additional Instructions:
- Do NOT mention the visualization tool, chart type, or analysis method
- Use clear, professional language suitable for healthcare executives
- Include specific observations from the visual data
- Focus on healthcare business value and clinical implications
"""

        # Analyze with Gemini Vision
        response = gemini_vision_model.generate_content([prompt, img])
        analysis = response.text.strip()

        return {
            "image_path": f"/static/{filename}",
            "analysis": analysis
        }

    except Exception as e:
        print(f"Error saving/analyzing chart: {e}")
        return {
            "image_path": "",
            "analysis": f"Error: Failed to analyze visualization - {str(e)}"
        }


def create_readmission_chart(df: pd.DataFrame) -> str:
    """Create readmission risk visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    risk_counts = df['risk_level'].value_counts()
    axes[0, 0].bar(risk_counts.index, risk_counts.values, color=['#d73027', '#fc8d59', '#fee08b', '#91cf60'])
    axes[0, 0].set_title('Patient Distribution by Risk Level', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Risk Level')
    axes[0, 0].set_ylabel('Number of Patients')

    condition_risk = df.groupby('medicalcondition')['risk_score'].mean().sort_values(ascending=False).head(10)
    axes[0, 1].barh(condition_risk.index, condition_risk.values, color='steelblue')
    axes[0, 1].set_title('Average Risk Score by Medical Condition', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Average Risk Score')

    axes[1, 0].scatter(df['age'], df['risk_score'], alpha=0.5, c=df['risk_score'], cmap='RdYlGn_r')
    axes[1, 0].set_title('Age vs Risk Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Risk Score')

    axes[1, 1].hist(df['risk_score'], bins=20, color='coral', edgecolor='black')
    axes[1, 1].set_title('Risk Score Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Risk Score')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    return create_chart_base64(fig)


def create_billing_anomaly_chart(df: pd.DataFrame) -> str:
    """Create billing anomaly visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    severity_counts = df['severity'].value_counts()
    axes[0, 0].pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%',
                   colors=['#d73027', '#fee08b', '#91cf60'])
    axes[0, 0].set_title('Anomalies by Severity', fontsize=12, fontweight='bold')

    condition_dev = df.groupby('medicalcondition')['deviation_pct'].mean().sort_values(ascending=False).head(10)
    axes[0, 1].barh(condition_dev.index, condition_dev.values, color='indianred')
    axes[0, 1].set_title('Average Billing Deviation by Condition', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Deviation %')

    # Use billingamount instead of actual_amount for the chart
    axes[1, 0].scatter(df['expected_amount'], df['billingamount'], alpha=0.6, c=df['deviation_pct'], cmap='RdYlGn_r')
    axes[1, 0].plot([df['expected_amount'].min(), df['expected_amount'].max()],
                    [df['expected_amount'].min(), df['expected_amount'].max()],
                    'k--', label='Expected = Actual')
    axes[1, 0].set_title('Actual vs Expected Billing', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Expected Amount ($)')
    axes[1, 0].set_ylabel('Actual Amount ($)')
    axes[1, 0].legend()

    if 'dateofadmission' in df.columns:
        df_timeline = df.copy()
        df_timeline['dateofadmission'] = pd.to_datetime(df_timeline['dateofadmission'])
        df_timeline = df_timeline.sort_values('dateofadmission')
        axes[1, 1].scatter(df_timeline['dateofadmission'], df_timeline['deviation_pct'],
                          alpha=0.6, c=df_timeline['deviation_pct'], cmap='RdYlGn_r')
        axes[1, 1].set_title('Anomalies Over Time', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Deviation %')
        axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return create_chart_base64(fig)


def create_hospital_comparison_chart(df: pd.DataFrame) -> str:
    """Create hospital performance comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top hospitals by admissions
    top_hospitals = df.nsmallest(10, 'rank')
    axes[0, 0].barh(top_hospitals['hospital'], top_hospitals['total_admissions'], color='steelblue')
    axes[0, 0].set_title('Top 10 Hospitals by Total Admissions', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Total Admissions')
    axes[0, 0].invert_yaxis()

    # Average billing comparison
    axes[0, 1].barh(top_hospitals['hospital'], top_hospitals['avg_billing'], color='coral')
    axes[0, 1].set_title('Average Billing Amount by Hospital', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Average Billing ($)')
    axes[0, 1].invert_yaxis()

    # Readmission rate comparison
    axes[1, 0].barh(top_hospitals['hospital'], top_hospitals['readmission_rate'] * 100, color='indianred')
    axes[1, 0].set_title('Readmission Rate by Hospital', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Readmission Rate (%)')
    axes[1, 0].invert_yaxis()

    # Efficiency score
    axes[1, 1].barh(top_hospitals['hospital'], top_hospitals['efficiency_score'], color='mediumseagreen')
    axes[1, 1].set_title('Efficiency Score by Hospital', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Efficiency Score')
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    return create_chart_base64(fig)


def create_demographics_chart(age_dist: Dict, gender_split: Dict, blood_dist: Dict) -> str:
    """Create demographics visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Age distribution
    ages = list(age_dist.keys())
    counts = list(age_dist.values())
    axes[0, 0].bar(ages, counts, color='skyblue', edgecolor='navy')
    axes[0, 0].set_title('Patient Age Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Age Group')
    axes[0, 0].set_ylabel('Number of Patients')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Gender split pie chart
    genders = list(gender_split.keys())
    percentages = [gender_split[g] * 100 for g in genders]
    colors = ['#3498db', '#e74c3c']
    axes[0, 1].pie(percentages, labels=genders, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0, 1].set_title('Gender Distribution', fontsize=12, fontweight='bold')

    # Blood type distribution
    blood_types = list(blood_dist.keys())
    blood_counts = [blood_dist[bt] for bt in blood_types]
    axes[1, 0].bar(blood_types, blood_counts, color='lightcoral', edgecolor='darkred')
    axes[1, 0].set_title('Blood Type Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Blood Type')
    axes[1, 0].set_ylabel('Proportion')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Age group percentages
    total = sum(counts)
    age_pcts = [(c/total)*100 if total > 0 else 0 for c in counts]
    axes[1, 1].barh(ages, age_pcts, color='mediumseagreen')
    axes[1, 1].set_title('Age Distribution Percentages', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Percentage (%)')
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    return create_chart_base64(fig)


# ==================== FRONTEND HTML ====================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MediWatch - Healthcare Analytics Platform</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --primary: #0ea5e9;
            --primary-dark: #0284c7;
            --secondary: #06b6d4;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #0f172a;
            --light: #f8fafc;
            --gray-50: #f9fafb;
            --gray-100: #f1f5f9;
            --gray-200: #e2e8f0;
            --gray-400: #94a3b8;
            --gray-500: #64748b;
            --gray-700: #334155;
            --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }

        * { font-family: 'Inter', sans-serif; margin: 0; padding: 0; box-sizing: border-box; }
        body { background: var(--gray-100); min-height: 100vh; }

        .sidebar {
            position: fixed; left: 0; top: 0; bottom: 0; width: 260px;
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            padding: 1.5rem; z-index: 1000; overflow-y: auto;
        }
        .sidebar-brand {
            color: white; font-size: 1.3rem; font-weight: 700;
            padding-bottom: 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.5rem;
        }
        .sidebar-brand i { color: var(--primary); }
        .sidebar-nav { list-style: none; padding: 0; margin: 0; }
        .sidebar-nav a {
            display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1rem;
            color: rgba(255,255,255,0.7); text-decoration: none; border-radius: 8px;
            font-size: 0.9rem; margin-bottom: 0.25rem; transition: all 0.2s;
        }
        .sidebar-nav a:hover, .sidebar-nav a.active {
            background: rgba(255,255,255,0.1); color: white;
        }
        .sidebar-nav a.active { background: var(--primary); color: white; }

        .main-content { margin-left: 260px; padding: 2rem; min-height: 100vh; }
        .page-header { margin-bottom: 2rem; }
        .page-title {
            font-size: 1.75rem; font-weight: 700; color: var(--dark); margin: 0 0 0.25rem 0;
        }
        .page-subtitle { color: var(--gray-500); font-size: 0.9rem; }

        .stat-card {
            background: white; border-radius: 16px; padding: 1.5rem;
            box-shadow: var(--shadow); border: 1px solid var(--gray-200);
            transition: transform 0.2s, box-shadow 0.2s; height: 100%;
        }
        .stat-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-lg); }
        .stat-card-header {
            display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;
        }
        .stat-icon {
            width: 48px; height: 48px; border-radius: 12px;
            display: flex; align-items: center; justify-content: center; font-size: 1.2rem;
        }
        .stat-icon.primary { background: rgba(14, 165, 233, 0.1); color: var(--primary); }
        .stat-icon.success { background: rgba(16, 185, 129, 0.1); color: var(--success); }
        .stat-icon.warning { background: rgba(245, 158, 11, 0.1); color: var(--warning); }
        .stat-icon.danger { background: rgba(239, 68, 68, 0.1); color: var(--danger); }
        .stat-label {
            color: var(--gray-500); font-size: 0.8rem; font-weight: 500;
            text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;
        }
        .stat-value { font-size: 1.75rem; font-weight: 700; color: var(--dark); margin: 0; }
        .stat-change {
            font-size: 0.75rem; margin-top: 0.5rem;
            display: inline-flex; align-items: center; gap: 0.25rem;
        }
        .stat-change.positive { color: var(--success); }
        .stat-change.negative { color: var(--danger); }

        .chart-card {
            background: white; border-radius: 16px; box-shadow: var(--shadow);
            border: 1px solid var(--gray-200); overflow: hidden; margin-bottom: 1.5rem;
        }
        .chart-header {
            padding: 1.25rem 1.5rem; border-bottom: 1px solid var(--gray-200);
            display: flex; justify-content: space-between; align-items: center;
            background: linear-gradient(to right, #f8fafc, #ffffff);
        }
        .chart-title {
            font-weight: 600; color: var(--dark); margin: 0; font-size: 1rem;
            display: flex; align-items: center; gap: 0.5rem;
        }
        .chart-body { padding: 1.5rem; }
        .chart-img { width: 100%; height: auto; border-radius: 8px; background: var(--gray-50); }

        .ai-insight {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border: 1px solid #bae6fd; border-radius: 12px; padding: 1.25rem; margin-top: 1rem;
        }
        .ai-insight-header {
            display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;
            font-weight: 600; color: var(--primary); font-size: 0.85rem;
        }
        .ai-insight-content { font-size: 0.875rem; line-height: 1.7; color: var(--gray-700); }
        .ai-insight-content h1, .ai-insight-content h2, .ai-insight-content h3 {
            font-size: 1rem; font-weight: 600; color: var(--dark); margin: 1rem 0 0.5rem;
        }
        .ai-insight-content h1:first-child,
        .ai-insight-content h2:first-child,
        .ai-insight-content h3:first-child { margin-top: 0; }
        .ai-insight-content p { margin: 0.5rem 0; }
        .ai-insight-content ul, .ai-insight-content ol { margin: 0.5rem 0; padding-left: 1.5rem; }
        .ai-insight-content li { margin: 0.25rem 0; }
        .ai-insight-content strong { color: var(--dark); font-weight: 600; }
        .ai-insight-loading { display: flex; align-items: center; gap: 0.5rem; color: var(--gray-400); }

        .btn-analyze {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            border: none; color: white; padding: 0.5rem 1rem; border-radius: 8px;
            font-size: 0.8rem; font-weight: 500; cursor: pointer;
            transition: opacity 0.2s, transform 0.2s;
            display: inline-flex; align-items: center; gap: 0.5rem;
        }
        .btn-analyze:hover { opacity: 0.9; transform: translateY(-1px); }
        .btn-analyze:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

        .btn-refresh {
            background: var(--gray-100); border: 1px solid var(--gray-200);
            color: var(--gray-700); padding: 0.4rem 0.75rem; border-radius: 6px;
            font-size: 0.75rem; cursor: pointer; transition: all 0.2s;
        }
        .btn-refresh:hover { background: var(--gray-200); }

        .form-control, .form-select {
            border: 1px solid var(--gray-200); border-radius: 8px;
            padding: 0.6rem 1rem; font-size: 0.9rem; transition: border-color 0.2s;
        }
        .form-control:focus, .form-select:focus {
            border-color: var(--primary); box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1); outline: none;
        }

        .spinner-border { width: 1rem; height: 1rem; border-width: 2px; }

        .nav-tabs { border: none; gap: 0.5rem; margin-bottom: 1.5rem; }
        .nav-tabs .nav-link {
            border: 1px solid var(--gray-200); background: white; color: var(--gray-700);
            border-radius: 8px; padding: 0.6rem 1.25rem; font-weight: 500; transition: all 0.2s;
        }
        .nav-tabs .nav-link:hover { background: var(--gray-50); border-color: var(--gray-300); }
        .nav-tabs .nav-link.active {
            background: var(--primary); color: white; border-color: var(--primary);
        }

        @media (max-width: 768px) {
            .sidebar { width: 0; padding: 0; overflow: hidden; }
            .main-content { margin-left: 0; padding: 1rem; }
            .stat-value { font-size: 1.5rem; }
        }

        .alert {
            border-radius: 12px; border: none; padding: 1rem 1.25rem;
        }
        .alert-info {
            background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); color: #1e40af;
        }
        .alert-success {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); color: #065f46;
        }
        .alert-danger {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); color: #991b1b;
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="sidebar-brand">
            <i class="bi bi-heart-pulse-fill"></i> MediWatch
        </div>
        <ul class="sidebar-nav">
            <li><a href="#" class="active" data-section="dashboard"><i class="bi bi-house-door"></i> Dashboard</a></li>
            <li><a href="#" data-section="risk-prediction"><i class="bi bi-exclamation-triangle"></i> Risk Prediction</a></li>
            <li><a href="#" data-section="billing"><i class="bi bi-receipt"></i> Billing Analysis</a></li>
            <li><a href="#" data-section="analytics"><i class="bi bi-bar-chart"></i> Analytics</a></li>
            <li><a href="#" data-section="ai-assistant"><i class="bi bi-robot"></i> AI Assistant</a></li>
            <li><a href="#" data-section="api-docs"><i class="bi bi-file-code"></i> API Docs</a></li>
        </ul>
    </div>

    <div class="main-content">
        <div id="dashboard-section" class="content-section">
            <div class="page-header">
                <h1 class="page-title">Healthcare Analytics Dashboard</h1>
                <p class="page-subtitle">Real-time insights powered by AI</p>
            </div>

            <div class="row g-4 mb-4">
                <div class="col-12 col-md-6 col-lg-3">
                    <div class="stat-card">
                        <div class="stat-card-header">
                            <div class="stat-icon primary"><i class="bi bi-people"></i></div>
                        </div>
                        <div class="stat-label">Total Patients</div>
                        <div class="stat-value" id="total-patients">Loading...</div>
                        <div class="stat-change positive"><i class="bi bi-arrow-up"></i> 12% from last month</div>
                    </div>
                </div>
                <div class="col-12 col-md-6 col-lg-3">
                    <div class="stat-card">
                        <div class="stat-card-header">
                            <div class="stat-icon danger"><i class="bi bi-exclamation-circle"></i></div>
                        </div>
                        <div class="stat-label">High Risk Patients</div>
                        <div class="stat-value" id="high-risk">0</div>
                        <div class="stat-change negative"><i class="bi bi-arrow-down"></i> Needs attention</div>
                    </div>
                </div>
                <div class="col-12 col-md-6 col-lg-3">
                    <div class="stat-card">
                        <div class="stat-card-header">
                            <div class="stat-icon warning"><i class="bi bi-cash-coin"></i></div>
                        </div>
                        <div class="stat-label">Billing Anomalies</div>
                        <div class="stat-value" id="anomalies">0</div>
                        <div class="stat-change positive"><i class="bi bi-check-circle"></i> Under review</div>
                    </div>
                </div>
                <div class="col-12 col-md-6 col-lg-3">
                    <div class="stat-card">
                        <div class="stat-card-header">
                            <div class="stat-icon success"><i class="bi bi-hospital"></i></div>
                        </div>
                        <div class="stat-label">Active Hospitals</div>
                        <div class="stat-value" id="total-hospitals">Loading...</div>
                        <div class="stat-change positive"><i class="bi bi-arrow-up"></i> Expanding network</div>
                    </div>
                </div>
            </div>

            <div class="alert alert-info mb-4">
                <i class="bi bi-info-circle"></i>
                <strong>Welcome to MediWatch!</strong> Your AI-powered healthcare analytics platform. Navigate through different sections using the sidebar.
            </div>
        </div>

        <div id="risk-prediction-section" class="content-section" style="display:none;">
            <div class="page-header">
                <h1 class="page-title">Patient Readmission Risk Prediction</h1>
                <p class="page-subtitle">AI-powered risk assessment and recommendations</p>
            </div>

            <div class="row mb-4">
                <div class="col-md-8">
                    <div class="chart-card">
                        <div class="chart-header">
                            <h3 class="chart-title"><i class="bi bi-activity"></i> Risk Assessment</h3>
                            <button class="btn-analyze" onclick="fetchRiskPrediction()">
                                <i class="bi bi-arrow-clockwise"></i> Refresh Data
                            </button>
                        </div>
                        <div class="chart-body">
                            <div class="mb-3">
                                <label class="form-label">Filter by Medical Condition</label>
                                <select class="form-select" id="risk-condition">
                                    <option value="">All Conditions</option>
                                    <option value="Cancer">Cancer</option>
                                    <option value="Diabetes">Diabetes</option>
                                    <option value="Heart Disease">Heart Disease</option>
                                    <option value="Obesity">Obesity</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Risk Threshold</label>
                                <input type="range" class="form-range" min="0" max="100" value="50" id="risk-threshold">
                                <div class="text-muted small">Current: <span id="threshold-value">50</span>%</div>
                            </div>
                            <div id="risk-results">
                                <p class="text-muted">Click "Refresh Data" to load risk predictions</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div id="billing-section" class="content-section" style="display:none;">
            <div class="page-header">
                <h1 class="page-title">Billing Anomaly Detection</h1>
                <p class="page-subtitle">Identify unusual billing patterns and potential fraud</p>
            </div>

            <div class="chart-card">
                <div class="chart-header">
                    <h3 class="chart-title"><i class="bi bi-receipt"></i> Billing Analysis</h3>
                    <button class="btn-analyze" onclick="fetchBillingAnomalies()">
                        <i class="bi bi-search"></i> Detect Anomalies
                    </button>
                </div>
                <div class="chart-body">
                    <div id="billing-results">
                        <p class="text-muted">Click "Detect Anomalies" to analyze billing data</p>
                    </div>
                </div>
            </div>
        </div>

        <div id="analytics-section" class="content-section" style="display:none;">
            <div class="page-header">
                <h1 class="page-title">Healthcare Analytics</h1>
                <p class="page-subtitle">Comprehensive insights with AI-powered analysis</p>
            </div>

            <div class="chart-card">
                <div class="chart-header">
                    <h3 class="chart-title"><i class="bi bi-building"></i> Hospital Performance Comparison</h3>
                    <button class="btn-analyze" onclick="loadHospitalComparison()">
                        <i class="bi bi-robot"></i> Analyze with AI
                    </button>
                </div>
                <div class="chart-body">
                    <div id="hospital-viz"></div>
                    <div id="hospital-ai-insight"></div>
                </div>
            </div>

            <div class="chart-card">
                <div class="chart-header">
                    <h3 class="chart-title"><i class="bi bi-person-badge"></i> Patient Demographics</h3>
                    <button class="btn-analyze" onclick="loadDemographics()">
                        <i class="bi bi-robot"></i> Analyze with AI
                    </button>
                </div>
                <div class="chart-body">
                    <div id="demographics-viz"></div>
                    <div id="demographics-ai-insight"></div>
                </div>
            </div>
        </div>

        <div id="ai-assistant-section" class="content-section" style="display:none;">
            <div class="page-header">
                <h1 class="page-title">AI Assistant</h1>
                <p class="page-subtitle">Ask questions about your healthcare data</p>
            </div>

            <div class="chart-card">
                <div class="chart-header">
                    <h3 class="chart-title"><i class="bi bi-chat-dots"></i> Natural Language Query</h3>
                </div>
                <div class="chart-body">
                    <div class="mb-3">
                        <label class="form-label">Ask a question about your data</label>
                        <textarea class="form-control" id="ai-question" rows="3" placeholder="e.g., What is the average billing for diabetes patients?"></textarea>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Language</label>
                        <select class="form-select" id="ai-language">
                            <option value="en">English</option>
                            <option value="id">Indonesian</option>
                        </select>
                    </div>
                    <button class="btn-analyze" onclick="askAI()">
                        <i class="bi bi-send"></i> Ask AI
                    </button>
                    <div id="ai-answer" class="mt-4"></div>
                </div>
            </div>
        </div>

        <div id="api-docs-section" class="content-section" style="display:none;">
            <div class="page-header">
                <h1 class="page-title">API Documentation</h1>
                <p class="page-subtitle">Explore our REST API endpoints</p>
            </div>

            <div class="alert alert-info">
                <i class="bi bi-info-circle"></i>
                <strong>Interactive API Documentation:</strong> Visit <a href="/api/docs" target="_blank">/api/docs</a> for Swagger UI
            </div>

            <div class="chart-card">
                <div class="chart-header">
                    <h3 class="chart-title"><i class="bi bi-code-square"></i> Available Endpoints</h3>
                </div>
                <div class="chart-body">
                    <h5>Core Features</h5>
                    <ul>
                        <li><code>GET /api/v1/readmission-risk</code> - Patient readmission risk prediction</li>
                        <li><code>GET /api/v1/billing-anomalies</code> - Billing anomaly detection</li>
                        <li><code>POST /api/v1/ask</code> - Natural language query interface</li>
                        <li><code>GET /api/v1/hospitals/compare</code> - Hospital performance comparison</li>
                        <li><code>GET /api/v1/demographics/summary</code> - Patient demographics</li>
                        <li><code>GET /api/v1/conditions/{condition}/analytics</code> - Condition-based analytics</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.querySelectorAll('.sidebar-nav a').forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                document.querySelectorAll('.sidebar-nav a').forEach(l => l.classList.remove('active'));
                this.classList.add('active');
                const section = this.dataset.section;
                document.querySelectorAll('.content-section').forEach(s => s.style.display = 'none');
                document.getElementById(section + '-section').style.display = 'block';
            });
        });

        document.getElementById('risk-threshold')?.addEventListener('input', function() {
            document.getElementById('threshold-value').textContent = this.value;
        });

        async function loadDashboardStats() {
            try {
                const response = await fetch('/api/v1/demographics/summary');
                const data = await response.json();
                document.getElementById('total-patients').textContent = data.total_patients.toLocaleString();
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        }

        async function fetchRiskPrediction() {
            const condition = document.getElementById('risk-condition').value;
            const threshold = document.getElementById('risk-threshold').value;
            const resultsDiv = document.getElementById('risk-results');
            resultsDiv.innerHTML = '<div class="ai-insight-loading"><div class="spinner-border spinner-sm"></div> Loading risk predictions...</div>';
            try {
                let url = `/api/v1/readmission-risk?risk_threshold=${threshold}&limit=20`;
                if (condition) url += `&condition=${condition}`;
                const response = await fetch(url);
                const data = await response.json();
                document.getElementById('high-risk').textContent = data.high_risk_count;
                let html = `<h6>Found ${data.patients.length} high-risk patients</h6>`;
                html += '<div class="table-responsive"><table class="table table-hover">';
                html += '<thead><tr><th>Patient</th><th>Condition</th><th>Risk Score</th><th>Risk Level</th></tr></thead><tbody>';
                data.patients.slice(0, 10).forEach(p => {
                    const badgeColor = p.risk_level === 'CRITICAL' ? 'danger' : p.risk_level === 'HIGH' ? 'warning' : 'info';
                    html += `<tr><td>${p.name}</td><td>${p.condition}</td><td>${p.risk_score.toFixed(1)}%</td><td><span class="badge bg-${badgeColor}">${p.risk_level}</span></td></tr>`;
                });
                html += '</tbody></table></div>';
                resultsDiv.innerHTML = html;
            } catch (error) {
                resultsDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            }
        }

        async function fetchBillingAnomalies() {
            const resultsDiv = document.getElementById('billing-results');
            resultsDiv.innerHTML = '<div class="ai-insight-loading"><div class="spinner-border spinner-sm"></div> Analyzing billing data...</div>';
            try {
                const response = await fetch('/api/v1/billing-anomalies');
                const data = await response.json();
                document.getElementById('anomalies').textContent = data.total_anomalies;
                let html = `<div class="alert alert-warning">Found ${data.total_anomalies} billing anomalies. Potential fraud amount: $${data.potential_fraud_amount.toLocaleString()}</div>`;
                if (data.anomalies.length > 0) {
                    html += '<div class="table-responsive"><table class="table table-hover">';
                    html += '<thead><tr><th>Patient</th><th>Condition</th><th>Actual</th><th>Expected</th><th>Deviation</th><th>Severity</th></tr></thead><tbody>';
                    data.anomalies.slice(0, 10).forEach(a => {
                        const badgeColor = a.severity === 'HIGH' ? 'danger' : a.severity === 'MEDIUM' ? 'warning' : 'info';
                        html += `<tr><td>${a.name}</td><td>${a.condition}</td><td>$${a.actual_amount.toLocaleString()}</td><td>$${a.expected_amount.toLocaleString()}</td><td>${a.deviation_pct.toFixed(1)}%</td><td><span class="badge bg-${badgeColor}">${a.severity}</span></td></tr>`;
                    });
                    html += '</tbody></table></div>';
                }
                resultsDiv.innerHTML = html;
            } catch (error) {
                resultsDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            }
        }

        async function loadHospitalComparison() {
            const vizDiv = document.getElementById('hospital-viz');
            const insightDiv = document.getElementById('hospital-ai-insight');
            vizDiv.innerHTML = '<div class="ai-insight-loading"><div class="spinner-border spinner-sm"></div> Loading visualization...</div>';
            insightDiv.innerHTML = '';
            try {
                const response = await fetch('/api/v1/hospitals/compare');
                const data = await response.json();
                if (data.visualization) {
                    vizDiv.innerHTML = `<img src="${data.visualization.image_path}" class="chart-img" alt="Hospital Comparison">`;
                    if (data.visualization.analysis) {
                        insightDiv.innerHTML = `
                            <div class="ai-insight">
                                <div class="ai-insight-header"><i class="bi bi-robot"></i> AI Analysis</div>
                                <div class="ai-insight-content">${marked.parse(data.visualization.analysis)}</div>
                            </div>
                        `;
                    }
                } else {
                    vizDiv.innerHTML = '<div class="alert alert-info">No visualization data available</div>';
                }
            } catch (error) {
                vizDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            }
        }

        async function loadDemographics() {
            const vizDiv = document.getElementById('demographics-viz');
            const insightDiv = document.getElementById('demographics-ai-insight');
            vizDiv.innerHTML = '<div class="ai-insight-loading"><div class="spinner-border spinner-sm"></div> Loading visualization...</div>';
            insightDiv.innerHTML = '';
            try {
                const response = await fetch('/api/v1/demographics/summary');
                const data = await response.json();
                if (data.visualization) {
                    vizDiv.innerHTML = `<img src="${data.visualization.image_path}" class="chart-img" alt="Demographics">`;
                    if (data.visualization.analysis) {
                        insightDiv.innerHTML = `
                            <div class="ai-insight">
                                <div class="ai-insight-header"><i class="bi bi-robot"></i> AI Analysis</div>
                                <div class="ai-insight-content">${marked.parse(data.visualization.analysis)}</div>
                            </div>
                        `;
                    }
                } else {
                    vizDiv.innerHTML = '<div class="alert alert-info">No visualization data available</div>';
                }
            } catch (error) {
                vizDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            }
        }

        async function askAI() {
            const question = document.getElementById('ai-question').value;
            const language = document.getElementById('ai-language').value;
            const answerDiv = document.getElementById('ai-answer');
            if (!question.trim()) {
                answerDiv.innerHTML = '<div class="alert alert-warning">Please enter a question</div>';
                return;
            }
            answerDiv.innerHTML = '<div class="ai-insight-loading"><div class="spinner-border spinner-sm"></div> AI is thinking...</div>';
            try {
                const response = await fetch('/api/v1/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, language })
                });
                const data = await response.json();
                answerDiv.innerHTML = `
                    <div class="ai-insight">
                        <div class="ai-insight-header"><i class="bi bi-robot"></i> AI Answer</div>
                        <div class="ai-insight-content">${marked.parse(data.answer)}</div>
                    </div>
                `;
            } catch (error) {
                answerDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            }
        }

        window.addEventListener('load', () => {
            loadDashboardStats();
        });
    </script>
</body>
</html>
"""


# ==================== API ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the frontend HTML"""
    return HTML_TEMPLATE


@app.get("/api/v1/readmission-risk")
def get_readmission_risk(
    condition: Optional[str] = Query(None, description="Filter by medical condition"),
    hospital: Optional[str] = Query(None, description="Filter by hospital"),
    risk_threshold: int = Query(50, ge=0, le=100, description="Minimum risk score"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    include_chart: bool = Query(False, description="Include visualization")
):
    """F-001: Patient Readmission Risk Predictor"""
    try:
        params = {}

        # Build filter conditions
        condition_filter = ""
        hospital_filter = ""

        if condition:
            condition_filter = " AND UPPER(p.MEDICALCONDITION) = UPPER(%(condition)s)"
            params['condition'] = condition

        if hospital:
            hospital_filter = " AND UPPER(p.HOSPITAL) = UPPER(%(hospital)s)"
            params['hospital'] = hospital

        query = f"""
            SELECT
                p.NAME,
                p.AGE,
                p.GENDER,
                p.MEDICALCONDITION,
                p.DATEOFADMISSION,
                p.DISCHARGEDATE,
                p.HOSPITAL,
                p.TESTRESULTS,
                p.MEDICATION,
                COUNT(*) OVER (PARTITION BY p.NAME) as ADMISSION_COUNT
            FROM HEALTHCARE p
            WHERE 1=1
            {condition_filter}
            {hospital_filter}
            ORDER BY p.DATEOFADMISSION DESC
        """

        df = execute_query(query, params)

        if df.empty:
            return {
                "total_patients": 0,
                "high_risk_count": 0,
                "patients": [],
                "message": "No patients found matching the criteria"
            }

        # Convert column names to lowercase
        df.columns = df.columns.str.lower()

        df['risk_score'] = df.apply(calculate_risk_score, axis=1)
        df['risk_level'] = df['risk_score'].apply(get_risk_level)
        df = df[df['risk_score'] >= risk_threshold]
        df = df.sort_values('risk_score', ascending=False).head(limit)

        patients = []
        for _, row in df.iterrows():
            risk_level = get_risk_level(row['risk_score'])
            patients.append({
                "patient_id": row['name'],  # Using NAME as patient identifier
                "name": row['name'],
                "age": int(row['age']),
                "condition": row['medicalcondition'],
                "risk_score": float(row['risk_score']),
                "risk_level": risk_level.value,
                "last_admission": str(row['dateofadmission']),
                "hospital": row.get('hospital', 'Unknown'),
                "contributing_factors": get_contributing_factors(row, row['risk_score']),
                "recommendations": get_recommendations(row, risk_level)
            })

        high_risk_count = len(df[df['risk_score'] >= 60])

        response = {
            "total_patients": len(df),
            "high_risk_count": high_risk_count,
            "patients": patients
        }

        if include_chart and not df.empty:
            chart_base64 = create_readmission_chart(df)
            response["visualization"] = {
                "chart_base64": chart_base64,
                "format": "png"
            }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": f"An error occurred: {str(e)}"
            }
        )


@app.get("/api/v1/billing-anomalies")
def get_billing_anomalies(
    start_date: Optional[date] = Query(None, description="Start date filter"),
    end_date: Optional[date] = Query(None, description="End date filter"),
    severity: Optional[AnomalySeverity] = Query(None, description="Filter by severity"),
    include_chart: bool = Query(False, description="Include visualization")
):
    """F-002: Billing Anomaly Detection"""
    try:
        # Build the base query with optional date filters
        date_filter_cte = ""
        date_filter_main = ""
        params = {}

        if start_date:
            date_filter_cte += " AND DATEOFADMISSION >= %(start_date)s"
            date_filter_main += " AND p.DATEOFADMISSION >= %(start_date)s"
            params['start_date'] = start_date

        if end_date:
            date_filter_cte += " AND DATEOFADMISSION <= %(end_date)s"
            date_filter_main += " AND p.DATEOFADMISSION <= %(end_date)s"
            params['end_date'] = end_date

        query = f"""
            WITH billing_stats AS (
                SELECT
                    MEDICALCONDITION,
                    AVG(BILLINGAMOUNT) as avg_billing,
                    STDDEV_POP(BILLINGAMOUNT) as stddev_billing,
                    COUNT(*) as sample_count
                FROM HEALTHCARE
                WHERE 1=1 {date_filter_cte}
                GROUP BY MEDICALCONDITION
                HAVING COUNT(*) >= 3 AND STDDEV_POP(BILLINGAMOUNT) > 0
            )
            SELECT
                p.NAME,
                p.MEDICALCONDITION,
                p.BILLINGAMOUNT,
                p.DATEOFADMISSION,
                p.HOSPITAL,
                s.avg_billing,
                s.stddev_billing,
                ABS(p.BILLINGAMOUNT - s.avg_billing) / NULLIF(s.stddev_billing, 0) as z_score
            FROM HEALTHCARE p
            INNER JOIN billing_stats s ON p.MEDICALCONDITION = s.MEDICALCONDITION
            WHERE ABS(p.BILLINGAMOUNT - s.avg_billing) > (2 * s.stddev_billing)
            {date_filter_main}
            ORDER BY z_score DESC
        """

        df = execute_query(query, params)

        if df.empty:
            return {
                "total_anomalies": 0,
                "potential_fraud_amount": 0.0,
                "anomalies": []
            }

        # Convert column names to lowercase
        df.columns = df.columns.str.lower()

        df['expected_amount'] = df['avg_billing']
        df['deviation_pct'] = ((df['billingamount'] - df['expected_amount']) / df['expected_amount'] * 100).abs()

        def get_severity(deviation):
            if deviation > 150:
                return AnomalySeverity.HIGH
            elif deviation > 75:
                return AnomalySeverity.MEDIUM
            else:
                return AnomalySeverity.LOW

        df['severity'] = df['deviation_pct'].apply(get_severity)

        if severity:
            df = df[df['severity'] == severity]

        anomalies = []
        for _, row in df.iterrows():
            anomalies.append({
                "patient_id": row['name'],  # Using NAME as patient identifier
                "name": row['name'],
                "condition": row['medicalcondition'],
                "actual_amount": round(float(row['billingamount']), 2),
                "expected_amount": round(float(row['expected_amount']), 2),
                "deviation_pct": round(float(row['deviation_pct']), 2),
                "severity": row['severity'].value,
                "hospital": row.get('hospital', 'Unknown'),
                "reason": f"Billing {row['deviation_pct']:.1f}% {('higher' if row['billingamount'] > row['expected_amount'] else 'lower')} than condition average",
                "recommendation": "Urgent audit required" if row['severity'] == AnomalySeverity.HIGH else "Review billing codes"
            })

        potential_fraud = df[df['billingamount'] > df['expected_amount']]['billingamount'].sum() - \
                         df[df['billingamount'] > df['expected_amount']]['expected_amount'].sum()

        response = {
            "total_anomalies": len(df),
            "potential_fraud_amount": round(float(potential_fraud), 2),
            "anomalies": anomalies
        }

        if include_chart and not df.empty:
            chart_base64 = create_billing_anomaly_chart(df)
            response["visualization"] = {
                "chart_base64": chart_base64,
                "format": "png"
            }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": f"An error occurred: {str(e)}"
            }
        )


@app.post("/api/v1/ask")
def natural_language_query(query: NaturalLanguageQuery = Body(...)):
    """F-003: Natural Language Query Interface"""
    try:
        system_prompt = """
You are a healthcare data analyst. Convert user questions to SQL queries for Snowflake.

Database Schema:
- HEALTHCARE table with columns:
  * NAME, AGE, GENDER, BLOODTYPE, MEDICALCONDITION
  * DATEOFADMISSION, DOCTOR, HOSPITAL, INSURANCEPROVIDER
  * BILLINGAMOUNT, ROOMNUMBER, ADMISSIONTYPE, DISCHARGEDATE
  * MEDICATION, TESTRESULTS

Rules:
1. Use ONLY SELECT queries (no INSERT/UPDATE/DELETE)
2. Limit results to 1000 rows maximum
3. Return valid Snowflake SQL syntax
4. Handle null values appropriately
5. Use proper aggregations and GROUP BY when needed
6. Always use UPPER() for case-insensitive string comparisons
7. Use NAME to identify unique patients (there is no PATIENT_ID column)

Example:
Question: "What is the average billing for diabetes patients?"
SQL: SELECT AVG(BILLINGAMOUNT) as avg_billing FROM HEALTHCARE WHERE UPPER(MEDICALCONDITION) = 'DIABETES'

Now convert this question to SQL:
"""

        if query.language == QueryLanguage.ID:
            translation_prompt = f"Translate this Indonesian question to English, keeping medical terms: {query.question}"
            translation_response = gemini_model.generate_content(translation_prompt)
            english_question = translation_response.text.strip()
        else:
            english_question = query.question

        sql_prompt = f"{system_prompt}\nQuestion: {english_question}\nSQL:"
        sql_response = gemini_model.generate_content(sql_prompt)
        sql_query = sql_response.text.strip()

        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()

        forbidden_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
        if any(keyword in sql_query.upper() for keyword in forbidden_keywords):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_QUERY",
                    "message": "Query contains forbidden operations"
                }
            )

        df = execute_query(sql_query)
        df.columns = df.columns.str.lower()  # Convert to lowercase

        result_summary = df.head(20).to_string() if not df.empty else "No results found"

        # Enhanced prompt for insights and conclusions
        explanation_prompt = f"""
You are a senior healthcare data analyst providing insights to hospital executives. Based on the query results below, provide a comprehensive analysis in {'Indonesian' if query.language == QueryLanguage.ID else 'English'}.

Original Question: {query.question}
Data Results: {result_summary}
Total Records: {len(df)}

Please provide:

1. **Executive Summary** (2-3 sentences)
   - Directly answer the question with key findings
   - Highlight the most important numbers and trends

2. **Key Insights** (3-5 bullet points)
   - Identify patterns, trends, or anomalies in the data
   - Compare metrics (e.g., higher/lower than expected)
   - Note any significant correlations

3. **Strategic Recommendations** (2-3 actionable suggestions)
   - Provide specific, practical recommendations based on the data
   - Focus on improving patient care, reducing costs, or operational efficiency

Important:
- Use clear, professional language without technical jargon
- Include specific numbers and percentages from the data
- Do NOT mention SQL, queries, or technical processes
- Focus on healthcare implications and business value
"""

        explanation_response = gemini_model.generate_content(explanation_prompt)
        answer = explanation_response.text.strip()

        # Create data summary for additional context
        data_summary = {}
        if not df.empty:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                for col in numeric_cols[:3]:
                    data_summary[f"avg_{col.lower()}"] = round(float(df[col].mean()), 2)

            data_summary["total_rows"] = len(df)

            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                first_col = non_numeric_cols[0]
                top_values = df[first_col].value_counts().head(3).to_dict()
                data_summary["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        # Return response WITHOUT SQL query exposed to frontend
        return {
            "answer": answer,
            "data_summary": data_summary,
            "row_count": len(df)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": f"An error occurred: {str(e)}"
            }
        )


@app.get("/api/v1/conditions/{condition}/analytics")
def get_condition_analytics(condition: str):
    """F-005: Condition-Based Analytics"""
    try:
        query = """
            SELECT
                MEDICALCONDITION,
                COUNT(*) as total_cases,
                COUNT(DISTINCT NAME) as unique_patients,
                ROUND(AVG(AGE), 2) as avg_age,
                SUM(CASE WHEN GENDER = 'Male' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as male_pct,
                SUM(CASE WHEN GENDER = 'Female' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as female_pct,
                ROUND(AVG(BILLINGAMOUNT), 2) as avg_billing,
                ROUND(AVG(DATEDIFF(day, DATEOFADMISSION, COALESCE(DISCHARGEDATE, CURRENT_DATE()))), 2) as avg_length_of_stay,
                SUM(CASE WHEN TESTRESULTS = 'Normal' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as normal_results_pct,
                SUM(CASE WHEN TESTRESULTS = 'Abnormal' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as abnormal_results_pct
            FROM HEALTHCARE
            WHERE UPPER(MEDICALCONDITION) = UPPER(%(condition)s)
            GROUP BY MEDICALCONDITION
        """

        df = execute_query(query, {'condition': condition})

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "CONDITION_NOT_FOUND",
                    "message": f"No data found for condition: {condition}"
                }
            )

        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        row = df.iloc[0]

        med_query = """
            SELECT MEDICATION, COUNT(*) as frequency
            FROM HEALTHCARE
            WHERE UPPER(MEDICALCONDITION) = UPPER(%(condition)s)
                AND MEDICATION IS NOT NULL
            GROUP BY MEDICATION
            ORDER BY frequency DESC
            LIMIT 5
        """
        med_df = execute_query(med_query, {'condition': condition})
        med_df.columns = med_df.columns.str.lower()  # Convert to lowercase

        total_cases = float(df.iloc[0]['total_cases'])
        medications = [
            {"name": row['medication'], "frequency": float(row['frequency']) / total_cases if total_cases > 0 else 0}
            for _, row in med_df.iterrows()
        ]

        hosp_query = """
            SELECT
                HOSPITAL,
                COUNT(*) as cases,
                ROUND(AVG(BILLINGAMOUNT), 2) as avg_billing,
                COUNT(*) * 1.0 / NULLIF(COUNT(DISTINCT NAME), 0) as readmission_rate
            FROM HEALTHCARE
            WHERE UPPER(MEDICALCONDITION) = UPPER(%(condition)s)
                AND HOSPITAL IS NOT NULL
            GROUP BY HOSPITAL
            ORDER BY cases DESC
            LIMIT 5
        """
        hosp_df = execute_query(hosp_query, {'condition': condition})
        hosp_df.columns = hosp_df.columns.str.lower()  # Convert to lowercase

        hospital_comparison = [
            {
                "hospital": row['hospital'],
                "cases": int(row['cases']),
                "avg_billing": float(row['avg_billing']) if pd.notna(row['avg_billing']) else 0,
                "readmission_rate": round(float(row['readmission_rate']) - 1, 3) if pd.notna(row['readmission_rate']) else 0
            }
            for _, row in hosp_df.iterrows()
        ]

        unique_patients = float(row['unique_patients'])
        readmission_rate = (float(row['total_cases']) / unique_patients - 1) if unique_patients > 0 else 0

        return {
            "condition": condition,
            "statistics": {
                "total_cases": int(row['total_cases']),
                "unique_patients": int(row['unique_patients']),
                "readmission_rate": round(readmission_rate, 3),
                "avg_age": float(row['avg_age']),
                "gender_distribution": {
                    "Male": round(float(row['male_pct']), 2),
                    "Female": round(float(row['female_pct']), 2)
                },
                "avg_billing": float(row['avg_billing']),
                "avg_length_of_stay": float(row['avg_length_of_stay']) if pd.notna(row['avg_length_of_stay']) else 0
            },
            "medications": medications,
            "outcomes": {
                "normal_results": round(float(row['normal_results_pct']), 2),
                "abnormal_results": round(float(row['abnormal_results_pct']), 2),
                "inconclusive_results": round(1.0 - float(row['normal_results_pct']) - float(row['abnormal_results_pct']), 2)
            },
            "hospital_comparison": hospital_comparison
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": f"An error occurred: {str(e)}"
            }
        )


@app.get("/api/v1/hospitals/compare")
def compare_hospitals(
    hospitals: Optional[List[str]] = Query(None, description="List of hospitals to compare"),
    metric: str = Query("admissions", description="Metric to compare"),
    period: str = Query("all", description="Time period")
):
    """F-006: Hospital Performance Comparison"""
    try:
        params = {}

        # Build parameterized query for hospital filtering
        hospital_filter = ""
        if hospitals and len(hospitals) > 0:
            # Create parameterized placeholders for hospitals
            hospital_params = {}
            placeholders = []
            for i, hospital in enumerate(hospitals):
                param_name = f"hospital_{i}"
                hospital_params[param_name] = hospital
                placeholders.append(f"%({param_name})s")
            params.update(hospital_params)
            hospital_filter = f" AND HOSPITAL IN ({','.join(placeholders)})"

        query = f"""
            SELECT
                HOSPITAL,
                COUNT(*) as total_admissions,
                ROUND(AVG(BILLINGAMOUNT), 2) as avg_billing,
                COUNT(*) * 1.0 / NULLIF(COUNT(DISTINCT NAME), 0) - 1 as readmission_rate,
                ROUND(AVG(DATEDIFF(day, DATEOFADMISSION, COALESCE(DISCHARGEDATE, CURRENT_DATE()))), 2) as avg_los,
                COUNT(DISTINCT MEDICALCONDITION) as conditions_treated
            FROM HEALTHCARE
            WHERE HOSPITAL IS NOT NULL
            {hospital_filter}
            GROUP BY HOSPITAL
            ORDER BY total_admissions DESC
        """

        df = execute_query(query, params)

        if df.empty:
            return {
                "comparison": [],
                "insights": ["No hospital data found"]
            }

        # Convert column names to lowercase for consistency
        df.columns = df.columns.str.lower()

        # Handle cases where all values are the same (division by zero)
        def safe_normalize(series):
            min_val = series.min()
            max_val = series.max()
            if max_val == min_val:
                return pd.Series([0.5] * len(series), index=series.index)
            return 1 - (series - min_val) / (max_val - min_val)

        df['norm_billing'] = safe_normalize(df['avg_billing'])
        df['norm_readmission'] = safe_normalize(df['readmission_rate'])
        df['norm_los'] = safe_normalize(df['avg_los'])

        df['efficiency_score'] = (df['norm_billing'] * 0.4 + df['norm_readmission'] * 0.4 + df['norm_los'] * 0.2) * 100
        df['rank'] = df['efficiency_score'].rank(ascending=False).astype(int)

        comparison = []
        for _, row in df.iterrows():
            comparison.append({
                "hospital": row['hospital'],
                "rank": int(row['rank']),
                "total_admissions": int(row['total_admissions']),
                "avg_billing": float(row['avg_billing']) if pd.notna(row['avg_billing']) else 0,
                "readmission_rate": round(float(row['readmission_rate']), 3) if pd.notna(row['readmission_rate']) else 0,
                "avg_los": float(row['avg_los']) if pd.notna(row['avg_los']) else 0,
                "efficiency_score": round(float(row['efficiency_score']), 2),
                "conditions_treated": int(row['conditions_treated'])
            })

        insights = []
        top_hospital = df.iloc[0]
        avg_readmission = df['readmission_rate'].mean()

        if pd.notna(top_hospital['readmission_rate']) and pd.notna(avg_readmission) and avg_readmission > 0:
            if top_hospital['readmission_rate'] < avg_readmission:
                pct_diff = ((avg_readmission - top_hospital['readmission_rate']) / avg_readmission * 100)
                insights.append(f"{top_hospital['hospital']} has {pct_diff:.0f}% lower readmission rate than average")

        if len(df) > 0:
            best_value = df.nsmallest(1, 'avg_billing').iloc[0]
            if pd.notna(best_value['avg_billing']):
                insights.append(f"{best_value['hospital']} offers most cost-effective care with ${best_value['avg_billing']:.0f} average billing")

        # Generate visualization with Gemini analysis
        visualization_result = None
        if len(df) > 0:
            fig = plt.figure(figsize=(14, 10))
            # Create the hospital comparison visualization
            top_hospitals = df.nsmallest(10, 'rank')

            ax1 = plt.subplot(2, 2, 1)
            ax1.barh(top_hospitals['hospital'], top_hospitals['total_admissions'], color='steelblue')
            ax1.set_title('Top 10 Hospitals by Total Admissions', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Total Admissions')
            ax1.invert_yaxis()

            ax2 = plt.subplot(2, 2, 2)
            ax2.barh(top_hospitals['hospital'], top_hospitals['avg_billing'], color='coral')
            ax2.set_title('Average Billing Amount by Hospital', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Average Billing ($)')
            ax2.invert_yaxis()

            ax3 = plt.subplot(2, 2, 3)
            ax3.barh(top_hospitals['hospital'], top_hospitals['readmission_rate'] * 100, color='indianred')
            ax3.set_title('Readmission Rate by Hospital', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Readmission Rate (%)')
            ax3.invert_yaxis()

            ax4 = plt.subplot(2, 2, 4)
            ax4.barh(top_hospitals['hospital'], top_hospitals['efficiency_score'], color='mediumseagreen')
            ax4.set_title('Efficiency Score by Hospital', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Efficiency Score')
            ax4.invert_yaxis()

            plt.tight_layout()

            # Save and analyze the visualization
            context = f"Hospital performance comparison showing total admissions, average billing amounts, readmission rates, and efficiency scores for {len(df)} hospitals. The data reveals patterns in hospital operations and patient care quality."
            visualization_result = save_and_analyze_chart(fig, "hospital_comparison.png", context)

        return {
            "comparison": comparison,
            "insights": insights,
            "visualization": visualization_result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": f"An error occurred: {str(e)}"
            }
        )


@app.get("/api/v1/demographics/summary")
def get_demographics_summary():
    """F-007: Demographic Insights"""
    try:
        query = """
            SELECT
                COUNT(DISTINCT NAME) as total_patients,
                SUM(CASE WHEN AGE BETWEEN 0 AND 17 THEN 1 ELSE 0 END) as age_0_17,
                SUM(CASE WHEN AGE BETWEEN 18 AND 35 THEN 1 ELSE 0 END) as age_18_35,
                SUM(CASE WHEN AGE BETWEEN 36 AND 55 THEN 1 ELSE 0 END) as age_36_55,
                SUM(CASE WHEN AGE BETWEEN 56 AND 70 THEN 1 ELSE 0 END) as age_56_70,
                SUM(CASE WHEN AGE > 70 THEN 1 ELSE 0 END) as age_70_plus,
                SUM(CASE WHEN GENDER = 'Male' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as male_pct,
                SUM(CASE WHEN GENDER = 'Female' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) as female_pct
            FROM HEALTHCARE
        """

        df = execute_query(query)

        if df.empty:
            return {
                "total_patients": 0,
                "age_distribution": {"0-17": 0, "18-35": 0, "36-55": 0, "56-70": 0, "70+": 0},
                "gender_split": {"Male": 0, "Female": 0},
                "blood_type_distribution": {},
                "insurance_coverage": {}
            }

        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        row = df.iloc[0]

        blood_query = """
            SELECT BLOODTYPE, COUNT(*) as count
            FROM HEALTHCARE
            WHERE BLOODTYPE IS NOT NULL
            GROUP BY BLOODTYPE
        """
        blood_df = execute_query(blood_query)
        blood_df.columns = blood_df.columns.str.lower()  # Convert to lowercase

        total_patients = int(row['total_patients']) if pd.notna(row['total_patients']) else 0
        blood_distribution = {
            row['bloodtype']: round(float(row['count']) / total_patients, 3) if total_patients > 0 else 0
            for _, row in blood_df.iterrows()
        }

        insurance_query = """
            SELECT INSURANCEPROVIDER, COUNT(*) as count
            FROM HEALTHCARE
            WHERE INSURANCEPROVIDER IS NOT NULL
            GROUP BY INSURANCEPROVIDER
            ORDER BY count DESC
            LIMIT 10
        """
        insurance_df = execute_query(insurance_query)
        insurance_df.columns = insurance_df.columns.str.lower()  # Convert to lowercase

        insurance_coverage = {
            row['insuranceprovider']: round(float(row['count']) / total_patients, 3) if total_patients > 0 else 0
            for _, row in insurance_df.iterrows()
        }

        age_dist = {
            "0-17": int(row['age_0_17']) if pd.notna(row['age_0_17']) else 0,
            "18-35": int(row['age_18_35']) if pd.notna(row['age_18_35']) else 0,
            "36-55": int(row['age_36_55']) if pd.notna(row['age_36_55']) else 0,
            "56-70": int(row['age_56_70']) if pd.notna(row['age_56_70']) else 0,
            "70+": int(row['age_70_plus']) if pd.notna(row['age_70_plus']) else 0
        }

        gender_split = {
            "Male": round(float(row['male_pct']), 2) if pd.notna(row['male_pct']) else 0,
            "Female": round(float(row['female_pct']), 2) if pd.notna(row['female_pct']) else 0
        }

        # Generate visualization with Gemini analysis
        visualization_result = None
        if total_patients > 0:
            fig = plt.figure(figsize=(14, 10))

            # Age distribution
            ages = list(age_dist.keys())
            counts = list(age_dist.values())
            ax1 = plt.subplot(2, 2, 1)
            ax1.bar(ages, counts, color='skyblue', edgecolor='navy')
            ax1.set_title('Patient Age Distribution', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Age Group')
            ax1.set_ylabel('Number of Patients')
            ax1.tick_params(axis='x', rotation=45)

            # Gender split pie chart
            ax2 = plt.subplot(2, 2, 2)
            genders = list(gender_split.keys())
            percentages = [gender_split[g] * 100 for g in genders]
            colors = ['#3498db', '#e74c3c']
            ax2.pie(percentages, labels=genders, autopct='%1.1f%%', colors=colors, startangle=90)
            ax2.set_title('Gender Distribution', fontsize=12, fontweight='bold')

            # Blood type distribution
            ax3 = plt.subplot(2, 2, 3)
            blood_types = list(blood_distribution.keys())
            blood_counts = [blood_distribution[bt] for bt in blood_types]
            ax3.bar(blood_types, blood_counts, color='lightcoral', edgecolor='darkred')
            ax3.set_title('Blood Type Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Blood Type')
            ax3.set_ylabel('Proportion')
            ax3.tick_params(axis='x', rotation=45)

            # Age group percentages
            ax4 = plt.subplot(2, 2, 4)
            total = sum(counts)
            age_pcts = [(c/total)*100 if total > 0 else 0 for c in counts]
            ax4.barh(ages, age_pcts, color='mediumseagreen')
            ax4.set_title('Age Distribution Percentages', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Percentage (%)')
            ax4.invert_yaxis()

            plt.tight_layout()

            # Save and analyze the visualization
            context = f"Patient demographics showing age distribution, gender split, and blood type distribution for {total_patients} patients. This data reveals patterns in patient population characteristics."
            visualization_result = save_and_analyze_chart(fig, "demographics.png", context)

        return {
            "total_patients": total_patients,
            "age_distribution": age_dist,
            "gender_split": gender_split,
            "blood_type_distribution": blood_distribution,
            "insurance_coverage": insurance_coverage,
            "visualization": visualization_result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": f"An error occurred: {str(e)}"
            }
        )


@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        conn = get_snowflake_connection()
        conn.close()

        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "database": "disconnected",
                "timestamp": datetime.now().isoformat()
            }
        )


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))

    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║         MediWatch Healthcare Analytics Platform          ║
    ║                  Version 1.0.0                          ║
    ╚══════════════════════════════════════════════════════════╝

    🏥 API Server starting...
    📍 Host: {host}
    🔌 Port: {port}
    🌐 Web Interface: http://{host}:{port}/
    📚 API Documentation: http://{host}:{port}/api/docs
    🔍 Health Check: http://{host}:{port}/health

    Press CTRL+C to quit
    """)

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "False").lower() == "true"
    )
