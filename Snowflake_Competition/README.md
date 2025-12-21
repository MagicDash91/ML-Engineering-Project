# MediWatch Healthcare Analytics Platform

AI-Powered Healthcare Analytics & Predictive Insights

A modern web application for healthcare professionals to analyze patient data, predict readmission risks, detect billing anomalies, and gain AI-powered insights.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Architecture](#architecture)

---

## Features

### Core Analytics
- **Readmission Risk Prediction** - Calculate patient readmission risk scores based on medical history, conditions, and demographics
- **Billing Anomaly Detection** - Identify unusual billing patterns using statistical z-score analysis
- **Hospital Performance Comparison** - Compare hospitals across multiple metrics (admissions, billing, readmission rates)
- **Patient Demographics Analysis** - Visualize patient distribution by age, gender, and medical conditions

### AI-Powered Insights
- **Natural Language Query Interface** - Ask questions in plain language using Google Gemini AI
- **Automated Chart Analysis** - AI-powered visualization analysis with actionable recommendations
- **Executive Summaries** - Structured insights with key findings and strategic recommendations
- **Multi-language Support** - Get AI responses in your preferred language

### Modern User Interface
- **Responsive Design** - Bootstrap 5 with modern gradient sidebar
- **Real-time Statistics** - Dashboard with live patient, admission, and billing metrics
- **Interactive Visualizations** - Matplotlib and Seaborn charts with AI analysis
- **Professional Typography** - Inter font family for clean, readable interface

---

## Tech Stack

### Backend
- **FastAPI** (0.109.0) - Modern Python web framework
- **Uvicorn** (0.27.0) - ASGI server
- **Pydantic** (2.5.3) - Data validation
- **Python-dotenv** (1.0.0) - Environment configuration

### Database
- **Snowflake** - Cloud data warehouse
- **Snowflake Connector** (3.6.0) - Python database driver
- **Snowflake SQLAlchemy** (1.5.1) - ORM support

### AI & Machine Learning
- **Google Generative AI** (0.3.2) - Gemini 2.5 Flash model
- **Gemini Vision** - Chart and visualization analysis

### Data Processing & Visualization
- **Pandas** (2.1.4) - Data manipulation
- **NumPy** (1.26.3) - Numerical computing
- **Matplotlib** (3.8.2) - Chart generation
- **Seaborn** (0.13.1) - Statistical visualizations
- **Plotly** (5.18.0) - Interactive plots
- **Pillow** - Image processing

### Frontend
- **Bootstrap 5.3.2** - UI framework
- **Bootstrap Icons 1.11.1** - Icon library
- **Inter Font** - Typography
- **Marked.js** - Markdown rendering

### Deployment
- **AWS EC2** - Ubuntu server (t2.medium)
- **AWS VPC** - Network isolation (172.31.0.0/16)

---

## Prerequisites

- **Python** 3.12 or higher
- **Snowflake Account** with HEALTHCARE table
- **Google AI API Key** (for Gemini models)
- **pip** package manager

### System Dependencies (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y build-essential gcc g++ python3-dev libssl-dev libffi-dev
```
---

## Project Screenshots :

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Snowflake_Competition/static/m1.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Snowflake_Competition/static/m2.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Snowflake_Competition/static/m3.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Snowflake_Competition/static/m4.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Snowflake_Competition/static/m5.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Snowflake_Competition/static/m6.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Snowflake_Competition/static/m7.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Snowflake_Competition/static/m8.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Snowflake_Competition/static/m9.JPG)

---

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Langsmith-main/snowflake
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Upgrade pip and Install Dependencies
```bash
python3 -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Create Static Directory
```bash
mkdir -p static
```

---

## Configuration

### 1. Create `.env` File
Create a `.env` file in the `snowflake/` directory:

```env
# Snowflake Configuration
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
SNOWFLAKE_WAREHOUSE=your_warehouse

# Google AI Configuration
GOOGLE_AI_API_KEY=your_gemini_api_key

# Application Configuration
APP_PORT=8000
APP_HOST=0.0.0.0
```

### 2. Database Schema
Ensure your Snowflake database has a `HEALTHCARE` table with the following columns:

```sql
CREATE TABLE HEALTHCARE (
    NAME VARCHAR,
    AGE INTEGER,
    GENDER VARCHAR,
    BLOODTYPE VARCHAR,
    MEDICALCONDITION VARCHAR,
    DATEOFADMISSION DATE,
    DOCTOR VARCHAR,
    HOSPITAL VARCHAR,
    INSURANCEPROVIDER VARCHAR,
    BILLINGAMOUNT DECIMAL(10,2),
    ROOMNUMBER INTEGER,
    ADMISSIONTYPE VARCHAR,
    DISCHARGEDATE DATE,
    MEDICATION VARCHAR,
    TESTRESULTS VARCHAR
);
```

---

## Running the Application

### Development Mode
```bash
cd snowflake
python main.py
```

### Production Mode (with Uvicorn)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Access the Application
Open your browser and navigate to:
- **Main App**: http://localhost:8000
- **API Docs**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/health

---

## API Endpoints

### Core Endpoints

#### `GET /`
Main web application interface with dashboard

#### `GET /health`
Health check endpoint
```json
{"status": "healthy"}
```

#### `GET /api/docs`
Interactive Swagger API documentation

---

### Analytics Endpoints

#### `POST /api/v1/readmission-risk`
Calculate patient readmission risk scores

**Request Body**:
```json
{
  "age_min": 18,
  "age_max": 65,
  "gender": "Male",
  "condition": "Diabetes"
}
```

**Response**:
```json
{
  "risk_scores": [...],
  "statistics": {...},
  "insights": "AI-generated insights"
}
```

---

#### `POST /api/v1/billing-anomalies`
Detect unusual billing patterns

**Request Body**:
```json
{
  "threshold": 2.0,
  "start_date": "2024-01-01",
  "end_date": "2024-12-31"
}
```

**Response**:
```json
{
  "anomalies": [...],
  "total_anomalies": 42,
  "insights": "AI analysis of billing patterns"
}
```

---

#### `POST /api/v1/hospitals/compare`
Compare hospital performance metrics

**Request Body**:
```json
{
  "hospitals": ["City Hospital", "County Medical Center"],
  "metrics": ["total_admissions", "avg_billing", "readmission_rate"]
}
```

**Response**:
```json
{
  "comparison": [...],
  "insights": "AI-powered hospital comparison analysis",
  "visualization": {
    "image_path": "/static/hospital_comparison.png",
    "analysis": "Gemini Vision chart analysis"
  }
}
```

---

#### `POST /api/v1/demographics`
Get patient demographics with visualizations

**Request Body**:
```json
{
  "group_by": "age_group",
  "condition": "Diabetes"
}
```

**Response**:
```json
{
  "demographics": [...],
  "summary": {...},
  "visualization": {
    "image_path": "/static/demographics.png",
    "analysis": "AI insights on patient distribution"
  }
}
```

---

#### `POST /api/v1/ask`
Natural language query interface

**Request Body**:
```json
{
  "question": "What is the average age of patients with diabetes?",
  "language": "English"
}
```

**Response**:
```json
{
  "answer": "Based on the data analysis...",
  "data_summary": "15 patients found, average age 54.2 years",
  "row_count": 15
}
```

---

## Project Structure

```
Langsmith-main/
├── snowflake/
│   ├── main.py                 # FastAPI application
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # Environment configuration
│   ├── static/                 # Generated visualizations
│   │   ├── hospital_comparison.png
│   │   └── demographics.png
│   ├── venv/                   # Virtual environment
│   └── README.md              # This file
├── diagram/
│   ├── mediwatch_architecture.py  # Architecture diagram generator
│   ├── aws_ml_pipeline.py      # Reference diagram
│   ├── gemini.png              # Gemini AI icon
│   └── snowflake.png           # Snowflake icon
└── dashboard-ecommerce/        # UI reference templates
```

---

## Deployment

### AWS EC2 Deployment

#### 1. Launch EC2 Instance
- **AMI**: Ubuntu Server 22.04 LTS
- **Instance Type**: t2.medium (recommended)
- **VPC**: 172.31.0.0/16 (default)
- **Security Group**: Allow inbound on port 8000

#### 2. Configure Security Group
```bash
# Add inbound rule
Type: Custom TCP
Port: 8000
Source: 0.0.0.0/0 (or restrict to your IP)
```

#### 3. Connect and Install
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install system dependencies
sudo apt-get update
sudo apt-get install -y build-essential gcc g++ python3-dev libssl-dev libffi-dev python3-pip python3-venv

# Clone and setup
git clone <repository-url>
cd Langsmith-main/snowflake
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Configure environment
nano .env  # Add your credentials

# Create static directory
mkdir static

# Run application
python main.py
```

#### 4. Access Application
```
http://YOUR_EC2_PUBLIC_IP:8000
```

### Production Considerations
- Use **systemd** service for auto-restart
- Configure **Nginx** as reverse proxy
- Enable **HTTPS** with Let's Encrypt
- Set up **CloudWatch** for monitoring
- Use proper **secrets management** for credentials

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        End Users                             │
│  Healthcare Professionals | Data Analysts | Administrators   │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTPS (Port 8000)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    AWS Infrastructure                        │
│  ┌──────────────┐              ┌──────────────┐            │
│  │   AWS VPC    │─────────────▶│   AWS EC2    │            │
│  │ 172.31.0.0/16│              │Ubuntu t2.medium            │
│  └──────────────┘              └──────┬───────┘            │
└────────────────────────────────────────┼────────────────────┘
                                         ▼
┌─────────────────────────────────────────────────────────────┐
│              MediWatch Application (FastAPI)                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  Web UI    │  │  API Layer │  │Data Process│            │
│  │ Bootstrap 5│◀─│  7 Endpoints│─▶│Pandas/NumPy│            │
│  └────────────┘  └────────────┘  └────────────┘            │
│                         │                 │                  │
│                         ▼                 ▼                  │
│                  ┌────────────┐    ┌────────────┐           │
│                  │Visualization│   │  AI/ML     │           │
│                  │Matplotlib   │   │  Gemini    │           │
│                  │Seaborn      │   │  Vision    │           │
│                  └────────────┘    └────────────┘           │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Sources                               │
│  ┌──────────────────────────────────────────────┐           │
│  │  Snowflake HEALTHCARE Table                  │           │
│  │  Patient Records | Medical Data | Billing    │           │
│  └──────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow
1. **User Request** → Web UI (Bootstrap 5)
2. **Web UI** → FastAPI Backend (7 API endpoints)
3. **API** → Snowflake Database (SQL queries)
4. **Data Processing** → Pandas/NumPy (calculations)
5. **Visualization** → Matplotlib/Seaborn (chart generation)
6. **AI Analysis** → Gemini Vision (chart insights)
7. **AI Assistant** → Gemini 2.5 Flash (natural language)
8. **Response** → User with insights + visualizations

---

## Key Features Explained

### 1. Risk Prediction Algorithm
Uses multiple factors to calculate readmission risk:
- Patient age and demographics
- Medical condition complexity
- Previous admission history
- Billing patterns
- Medication adherence

### 2. Billing Anomaly Detection
Statistical z-score analysis:
- Calculates mean and standard deviation per medical condition
- Identifies bills beyond 2σ threshold
- Flags potential fraud or coding errors

### 3. AI-Powered Insights
Gemini AI provides:
- **Executive Summaries** - 2-3 sentence overviews
- **Key Insights** - 3-5 actionable bullet points
- **Strategic Recommendations** - Specific improvement suggestions
- **Chart Analysis** - Visual pattern recognition and healthcare implications

### 4. Visualization Engine
Generates professional charts:
- Hospital performance comparisons (4-panel dashboard)
- Patient demographics (age distribution, condition breakdown)
- Billing trends and anomalies
- Readmission risk distributions

---

## Security Features

- **Environment Variables** - Sensitive credentials stored in .env
- **Parameterized Queries** - SQL injection prevention
- **AWS VPC Isolation** - Network security
- **HTTPS Ready** - TLS/SSL support
- **Input Validation** - Pydantic models for request validation

---

## License

This project is proprietary software. All rights reserved.

---

## Acknowledgments

- **Snowflake** - Cloud data warehouse platform
- **Google AI** - Gemini 2.5 Flash and Vision models
- **FastAPI** - Modern Python web framework
- **Bootstrap** - Responsive UI components

---

<div align="center">
  <p><strong>MediWatch</strong> - Transforming Healthcare Data into Actionable Insights</p>
</div>
