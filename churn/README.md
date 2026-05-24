# Telco Customer Churn Prediction Pipeline

A comprehensive machine learning pipeline for predicting customer churn in telecommunications using **Apache Airflow**, **DBT**, **Snowflake**, **AWS S3**, and **scikit-learn**.

## ✅ Pipeline Status: **FULLY OPERATIONAL**

**Successfully Deployed Components:**
- ✅ Airflow DAGs (both versions working)
- ✅ AWS S3 Infrastructure (3 buckets deployed via Terraform)  
- ✅ Snowflake Data Source Connection
- ✅ DBT Transformations
- ✅ ML Model Training & Prediction
- ✅ Automatic S3 Upload Integration

## Project Screenshots :

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/churn/static/a1.jpg)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/churn/static/a2.jpg)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/churn/static/a3.jpg)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/churn/static/a4.jpg)

## 🏗️ Project Structure

```
churn/
├── config/                     # Configuration files
│   ├── snowflake_config.py    # Snowflake connection settings
│   └── aws_config.py          # AWS S3 configuration and utilities
├── dags/                      # Airflow DAG definitions (LOCAL)
│   ├── telco_churn_pipeline.py           # Basic pipeline (6 tasks)
│   ├── telco_churn_pipeline_with_s3.py   # Complete pipeline with S3 (7 tasks) ⭐
│   └── telco_churn_pipeline_with_dbt.py  # DBT-enhanced pipeline
├── dbt_churn/                 # DBT transformation project
│   ├── models/                # DBT models
│   │   ├── staging/           # Data cleaning and standardization
│   │   ├── intermediate/      # Feature engineering
│   │   └── marts/            # ML-ready and analytics tables
│   ├── tests/                # Data quality tests
│   ├── dbt_project.yml       # DBT configuration
│   └── README.md             # DBT documentation
├── scripts/                   # Utility scripts
│   ├── test_snowflake_connection.py # Connection testing
│   ├── s3_uploader.py         # S3 upload functions for ML results
│   ├── s3_monitor.py          # S3 monitoring and management utilities
│   └── churn_monitoring.py    # Pipeline monitoring & reporting
├── terraform/                 # Infrastructure as Code ⭐
│   ├── main.tf               # AWS provider configuration
│   ├── s3.tf                 # S3 bucket definitions (DEPLOYED)
│   ├── variables.tf          # Terraform variables
│   └── outputs.tf            # S3 bucket URLs and names
├── data/                      # Data storage (LOCAL)
│   ├── raw/                   # Raw data from Snowflake
│   ├── processed/             # Preprocessed data
│   └── predictions/           # Model predictions output ⭐
├── models/                    # Trained ML models and artifacts ⭐
├── requirements.txt           # Python dependencies
├── setup.sh                  # Setup script
└── README.md                 # This file
```

## 🎯 Features

### ELT Pipeline with DBT
- **Automated Data Extraction**: Connects to Snowflake `DATABASE.PUBLIC.CHURN` table
- **DBT Transformations**: SQL-based data transformations in Snowflake
- **Staging Layer**: Data cleaning and standardization
- **Intermediate Layer**: Advanced feature engineering
- **Marts Layer**: ML-ready and analytics-ready datasets
- **Data Quality Checks**: Comprehensive DBT tests for data validation

### Machine Learning
- **Model Training**: Random Forest classifier with hyperparameter optimization
- **Model Validation**: Automated accuracy checks and performance metrics
- **Feature Importance**: Identifies key churn prediction factors
- **Prediction Generation**: Scores new customers with churn probabilities

### AWS S3 Integration
- **Result Storage**: Automated upload of predictions, models, and metadata to S3
- **Organized Structure**: Date-based partitioning with run IDs for traceability
- **Multi-Bucket Strategy**: Separate buckets for raw data, results, and ML models
- **Monitoring Tools**: S3 usage reporting and pipeline run tracking

### Monitoring & Reporting
- **Pipeline Health Checks**: Monitors component status
- **Performance Reports**: Model accuracy and feature importance analysis
- **Data Freshness**: Tracks data age and update frequency
- **Risk Segmentation**: Categorizes customers by churn risk level
- **S3 Activity Monitoring**: Tracks uploads, storage usage, and pipeline runs

## 🔧 Setup & Installation

### Prerequisites
- Python 3.8+
- Apache Airflow 2.7+
- Access to Snowflake account: `ORGKMBU-HU54176`

### Quick Start

1. **Run setup script**:
   ```bash
   cd /home/magicdash/airflow/churn
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Test Snowflake connection**:
   ```bash
   python3 scripts/test_snowflake_connection.py
   ```

3. **Start Airflow** (if not running):
   ```bash
   airflow webserver -p 8080 &
   airflow scheduler &
   ```

4. **Enable DAG in Airflow UI**:
   - Open http://localhost:8080
   - Find `telco_churn_pipeline_with_s3` DAG
   - Toggle it to enabled

### Manual Installation

1. **Install dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Create directories**:
   ```bash
   mkdir -p {data/{raw,processed,predictions},models,logs}
   ```

3. **Copy DAG to Airflow**:
   ```bash
   cp dags/telco_churn_pipeline.py /path/to/airflow/dags/
   ```

## 📊 Configuration

### Snowflake Connection
The pipeline connects to your Snowflake account with these settings:

```python
SNOWFLAKE_CONFIG = {
    'account': 'ORGKMBU-HU54176',
    'user': 'MAGICDASH', 
    'database': 'DATABASE',
    'schema': 'PUBLIC',
    'warehouse': 'COMPUTE_WH',
    'role': 'ACCOUNTADMIN'
}
```

**Security Note**: Set environment variable to avoid hardcoding password:
```bash
export SNOWFLAKE_PASSWORD='your_password'
```

### AWS S3 Configuration
The pipeline uses three S3 buckets defined in `config/aws_config.py`:

```python
AWS_CONFIG = {
    'region': 'ap-southeast-1',
    'buckets': {
        'raw_data': 'magicdash-data-pipeline-raw-data',
        'churn_results': 'magicdash-data-pipeline-churn-results', 
        'ml_models': 'magicdash-data-pipeline-ml-models'
    }
}
```

**AWS Credentials**: Configure via environment variables or IAM roles:
```bash
export AWS_ACCESS_KEY_ID='your_key'
export AWS_SECRET_ACCESS_KEY='your_secret'
# or use IAM roles for EC2/container deployment
```

## 🔄 Pipeline Workflow

### Available DAG Options

**1. Basic Pipeline:** `telco_churn_pipeline` (6 tasks)
```
extract_data_from_snowflake → data_quality_check → preprocess_data → 
train_churn_model → validate_model → create_predictions
```

**2. S3-Integrated Pipeline:** `telco_churn_pipeline_with_s3` (7 tasks) ⭐ **RECOMMENDED**
```
extract_data_from_snowflake → data_quality_check → preprocess_data → 
train_churn_model → validate_model → create_predictions → upload_results_to_s3
```

### Task Details

1. **Extract Data** → Fetch customer data from Snowflake `DATABASE.PUBLIC.CHURN`
2. **Data Quality Check** → Validate data integrity and completeness  
3. **Preprocess** → Clean data, encode features, scale variables
4. **Train Model** → Train Random Forest classifier with hyperparameters
5. **Validate** → Check model performance against accuracy thresholds (70%)
6. **Generate Predictions** → Score new customers for churn risk with probabilities
7. **Upload to S3** → ⭐ **NEW**: Automatically store all outputs in AWS S3

### S3 Output Structure ⭐ **DEPLOYED & WORKING**
```
# Churn Results Bucket
s3://magicdash-data-pipeline-churn-results/
├── predictions/YYYYMMDD/
│   └── churn_predictions_YYYYMMDD_HHMMSS.csv  # Customer churn predictions
└── processed/YYYYMMDD/
    ├── raw_churn_data.csv                     # Extracted raw data
    └── preprocessed_churn_data.csv            # Cleaned & encoded data

# ML Models Bucket  
s3://magicdash-data-pipeline-ml-models/
└── models/YYYYMMDD/
    ├── churn_model.pkl          # Trained RandomForest model
    ├── model_results.pkl        # Training results & feature importance
    ├── validation_results.pkl   # Validation metrics
    ├── metadata.pkl            # Model metadata & preprocessing info
    ├── label_encoders.pkl      # Categorical encoders
    └── scaler.pkl             # Numerical feature scaler

# Raw Data Bucket (Reserved for future use)
s3://magicdash-data-pipeline-raw-data/
```

## 📈 Monitoring & Maintenance

### Pipeline Health Checks
```bash
python3 scripts/churn_monitoring.py --health
```

### Performance Reports  
```bash
python3 scripts/churn_monitoring.py --report
```

### S3 Monitoring
```bash
# Monitor recent S3 activity (last 7 days)
python3 scripts/s3_monitor.py monitor 7

# List recent pipeline runs (last 30 days)
python3 scripts/s3_monitor.py list-runs 30

# Get details for specific run
python3 scripts/s3_monitor.py run-details 20241201 run_123456

# Generate S3 usage report
python3 scripts/s3_monitor.py usage-report

# Analyze old files for cleanup (90+ days)
python3 scripts/s3_monitor.py cleanup-analysis 90
```

### Combined Monitoring
```bash
python3 scripts/churn_monitoring.py
```

## 🚀 Deployment

### Local Development
- Use the setup script for local development
- DAG runs daily by default (`schedule_interval='@daily'`)

### Infrastructure Deployment
```bash
cd terraform/
terraform init
terraform plan  
terraform apply
```

## 📋 Model Details

### Algorithm
- **Random Forest Classifier**
- 100 estimators, max_depth=10
- Handles both categorical and numerical features
- Built-in feature importance ranking

### Performance Metrics
- **Accuracy**: Primary validation metric (threshold: 70%)
- **Classification Report**: Precision, recall, F1-score
- **Feature Importance**: Identifies top churn indicators

### Risk Segmentation
- **High Risk**: >70% churn probability
- **Medium Risk**: 50-70% churn probability  
- **Low Risk**: ≤50% churn probability

## 🔐 Security

- Snowflake credentials configurable via environment variables
- No hardcoded passwords in source code
- Secure connection handling with proper cleanup

## 🤝 Usage Examples

### Test Connection
```bash
cd /home/magicdash/airflow/churn
python3 scripts/test_snowflake_connection.py
```

### Manual Pipeline Run
Trigger via Airflow UI or CLI:
```bash
# S3-integrated pipeline (RECOMMENDED)
airflow dags trigger telco_churn_pipeline_with_s3

# Basic pipeline (no S3 upload)
airflow dags trigger telco_churn_pipeline
```

### Monitor S3 Uploads
```bash
# Check S3 bucket contents
aws s3 ls s3://magicdash-data-pipeline-churn-results --recursive
aws s3 ls s3://magicdash-data-pipeline-ml-models --recursive

# Download latest predictions
aws s3 sync s3://magicdash-data-pipeline-churn-results/predictions/ ./downloaded_predictions/
```

### View Predictions
```bash
ls -la data/predictions/
head data/predictions/churn_predictions_*.csv
```

## 📞 Support

For issues or questions:
1. Check pipeline health: `python3 scripts/churn_monitoring.py --health`
2. Review Airflow logs in the UI
3. Test Snowflake connectivity: `python3 scripts/test_snowflake_connection.py`

## 🎉 Deployment Success Log

### Infrastructure Deployment (Terraform)
```bash
✅ terraform apply completed successfully
✅ Created 8 AWS resources:
   - 3 S3 buckets with versioning and encryption
   - 3 versioning configurations
   - 2 server-side encryption configurations

Deployed S3 Buckets:
✅ s3://magicdash-data-pipeline-raw-data
✅ s3://magicdash-data-pipeline-churn-results  
✅ s3://magicdash-data-pipeline-ml-models
```

### DAG Deployment
```bash
✅ telco_churn_pipeline.py - Basic 6-task pipeline (working)
✅ telco_churn_pipeline_with_s3.py - S3-integrated 7-task pipeline (working)  
✅ DAGs visible in Airflow UI
✅ Successful test runs completed
✅ S3 automatic upload confirmed working
```

### Verification Tests
```bash
✅ Snowflake connection: PASSED
✅ AWS S3 access: PASSED  
✅ DAG syntax validation: PASSED
✅ End-to-end pipeline execution: PASSED
✅ S3 file uploads: CONFIRMED WORKING
✅ ML model training: PASSED (>70% accuracy)
✅ Predictions generation: PASSED
```

## 📜 License

Internal project for ORGKMBU organization.

---

**Last Updated**: 2026-05-24  
**Version**: 2.0 ⭐ **S3-INTEGRATED**  
**Maintainer**: magicdash  
**Status**: ✅ **PRODUCTION READY**
