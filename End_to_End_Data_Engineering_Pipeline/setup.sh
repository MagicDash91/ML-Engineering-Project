#!/bin/bash

echo "🚀 Setting up Telco Churn Prediction Pipeline..."
echo "================================================"

PROJECT_ROOT="/home/magicdash/airflow/churn"

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p "$PROJECT_ROOT/data/raw"
mkdir -p "$PROJECT_ROOT/data/processed" 
mkdir -p "$PROJECT_ROOT/data/predictions"
mkdir -p "$PROJECT_ROOT/models"
mkdir -p "$PROJECT_ROOT/logs"

# Make scripts executable
echo "🔧 Setting script permissions..."
chmod +x "$PROJECT_ROOT/scripts/test_snowflake_connection.py"
chmod +x "$PROJECT_ROOT/scripts/churn_monitoring.py"
chmod +x "$PROJECT_ROOT/setup.sh"

# Copy DAG to Airflow DAGs directory
echo "📂 Setting up Airflow DAG..."
AIRFLOW_DAG_DIR="/home/magicdash/airflow/dags"
if [ -d "$AIRFLOW_DAG_DIR" ]; then
    cp "$PROJECT_ROOT/dags/telco_churn_pipeline.py" "$AIRFLOW_DAG_DIR/"
    echo "✅ DAG copied to Airflow DAGs directory"
else
    echo "⚠️  Airflow DAGs directory not found. Please copy DAG manually:"
    echo "   cp $PROJECT_ROOT/dags/telco_churn_pipeline.py [your-airflow-dags-dir]/"
fi

# Install requirements if pip is available
if command -v pip3 &> /dev/null; then
    echo "📦 Installing Python requirements..."
    pip3 install -r "$PROJECT_ROOT/requirements.txt"
    echo "✅ Requirements installed"
else
    echo "⚠️  pip3 not found. Please install requirements manually:"
    echo "   pip3 install -r $PROJECT_ROOT/requirements.txt"
fi

# Set environment variable for Snowflake password (optional)
echo ""
echo "🔐 Environment Setup:"
echo "To avoid hardcoding password, set environment variable:"
echo "export SNOWFLAKE_PASSWORD='Permataputihg101'"

echo ""
echo "✅ Churn pipeline setup completed!"
echo ""
echo "📋 Project Structure:"
echo "$PROJECT_ROOT/"
echo "├── config/           # Snowflake configuration"
echo "├── dags/             # Airflow DAG files"  
echo "├── scripts/          # Utility scripts"
echo "├── terraform/        # Infrastructure as Code"
echo "├── data/             # Data storage"
echo "│   ├── raw/          # Raw data from Snowflake"
echo "│   ├── processed/    # Preprocessed data"
echo "│   └── predictions/  # Model predictions"
echo "├── models/           # Trained ML models"
echo "└── requirements.txt  # Python dependencies"
echo ""
echo "🔧 Next Steps:"
echo "1. Test Snowflake connection:"
echo "   cd $PROJECT_ROOT && python3 scripts/test_snowflake_connection.py"
echo ""
echo "2. Start Airflow (if not already running):"
echo "   airflow webserver -p 8080 &"
echo "   airflow scheduler &"
echo ""
echo "3. Enable the 'telco_churn_pipeline' DAG in Airflow UI"
echo ""
echo "4. Monitor pipeline:"
echo "   cd $PROJECT_ROOT && python3 scripts/churn_monitoring.py"
echo ""
echo "5. Deploy infrastructure (optional):"
echo "   cd $PROJECT_ROOT/terraform && terraform init && terraform plan"