from datetime import datetime, timedelta
import sys
import os

# Add churn project to path
sys.path.append('/home/magicdash/airflow/churn')

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from cosmos import DbtDag, ProfileConfig, ProjectConfig
from cosmos.profiles import SnowflakeUserPasswordProfileMapping
import pandas as pd
import snowflake.connector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import logging
from config.snowflake_config import get_snowflake_connection_params

# Project paths
PROJECT_ROOT = '/home/magicdash/airflow/churn'
DATA_DIR = f'{PROJECT_ROOT}/data'
MODELS_DIR = f'{PROJECT_ROOT}/models'
DBT_PROJECT_DIR = f'{PROJECT_ROOT}/dbt_churn'

default_args = {
    'owner': 'magicdash',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Create DBT DAG for data transformation
profile_config = ProfileConfig(
    profile_name="dbt_churn",
    target_name="dev",
    profile_mapping=SnowflakeUserPasswordProfileMapping(
        conn_id="snowflake_default",
        profile_args={
            "account": "ORGKMBU-HU54176",
            "user": "MAGICDASH",
            "password": "Permataputihg101",
            "role": "ACCOUNTADMIN",
            "database": "DATABASE",
            "warehouse": "COMPUTE_WH",
            "schema": "dbt_staging"
        }
    )
)

project_config = ProjectConfig(dbt_project_path=DBT_PROJECT_DIR)

# DBT transformation DAG
dbt_dag = DbtDag(
    project_config=project_config,
    profile_config=profile_config,
    schedule_interval=None,  # Will be triggered by main DAG
    start_date=datetime(2024, 1, 1),
    catchup=False,
    dag_id="dbt_churn_transformation",
    default_args=default_args
)

# Main ML Pipeline DAG
dag = DAG(
    'telco_churn_pipeline_with_dbt',
    default_args=default_args,
    description='Telco Customer Churn Prediction Pipeline with DBT',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1
)

def extract_data_from_snowflake(**context):
    """Extract telco churn data from Snowflake (now just for validation)."""
    try:
        conn_params = get_snowflake_connection_params()
        
        conn = snowflake.connector.connect(**conn_params)
        cursor = conn.cursor()
        
        # Just validate that source data exists
        query = "SELECT COUNT(*) FROM DATABASE.PUBLIC.CHURN"
        
        logging.info(f"Executing validation query: {query}")
        cursor.execute(query)
        
        count = cursor.fetchone()[0]
        logging.info(f"Source table has {count} records")
        
        cursor.close()
        conn.close()
        
        return f"Source validation completed: {count} records available"
        
    except Exception as e:
        logging.error(f"Error validating source data: {str(e)}")
        raise

def load_ml_features(**context):
    """Load ML-ready features from DBT mart tables."""
    try:
        conn_params = get_snowflake_connection_params()
        
        conn = snowflake.connector.connect(**conn_params)
        cursor = conn.cursor()
        
        # Load from DBT mart table instead of raw data
        query = """
        SELECT * FROM DATABASE.dbt_staging.mart_churn_prediction
        WHERE feature_completeness_score > 0.8
        """
        
        logging.info(f"Loading ML features from DBT mart")
        cursor.execute(query)
        
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        
        df = pd.DataFrame(data, columns=columns)
        
        # Save processed features
        os.makedirs(f'{DATA_DIR}/processed', exist_ok=True)
        df.to_csv(f'{DATA_DIR}/processed/dbt_ml_features.csv', index=False)
        
        logging.info(f"Loaded {len(df)} records with {len(df.columns)} features from DBT mart")
        
        cursor.close()
        conn.close()
        
        return f"Successfully loaded {len(df)} ML-ready records from DBT"
        
    except Exception as e:
        logging.error(f"Error loading ML features: {str(e)}")
        raise

def train_churn_model_dbt(**context):
    """Train machine learning model using DBT-processed features."""
    try:
        # Load DBT-processed features
        df = pd.read_csv(f'{DATA_DIR}/processed/dbt_ml_features.csv')
        
        logging.info(f"Training model with {len(df)} DBT-processed records")
        
        # Features are already encoded by DBT
        feature_columns = [col for col in df.columns 
                          if col not in ['CUSTOMER_ID', 'TARGET_CHURNED', 'FEATURE_CREATED_AT', 'MART_CREATED_AT', 'HAS_DATA_QUALITY_ISSUE']]
        
        X = df[feature_columns].fillna(0)  # Handle any remaining nulls
        y = df['TARGET_CHURNED']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        logging.info("Training model on DBT-engineered features...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"DBT-enhanced model training completed")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Training set size: {len(X_train)}")
        logging.info(f"Test set size: {len(X_test)}")
        logging.info(f"Features used: {len(feature_columns)}")
        
        # Save model and metadata
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(f'{MODELS_DIR}/dbt_churn_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Save model metadata
        model_metadata = {
            'accuracy': accuracy,
            'feature_columns': feature_columns,
            'model_type': 'RandomForest_DBT_Enhanced',
            'n_features': len(feature_columns),
            'training_records': len(X_train),
            'test_records': len(X_test),
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': dict(zip(feature_columns, model.feature_importances_))
        }
        
        with open(f'{MODELS_DIR}/dbt_model_metadata.pkl', 'wb') as f:
            pickle.dump(model_metadata, f)
        
        return f"DBT-enhanced model trained with accuracy: {accuracy:.4f}"
        
    except Exception as e:
        logging.error(f"Error in DBT model training: {str(e)}")
        raise

def validate_dbt_model(**context):
    """Validate the DBT-enhanced model."""
    try:
        # Load model results
        with open(f'{MODELS_DIR}/dbt_model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        accuracy = metadata['accuracy']
        min_accuracy = 0.75  # Higher threshold for DBT-enhanced model
        
        if accuracy >= min_accuracy:
            logging.info(f"DBT Model validation PASSED. Accuracy: {accuracy:.4f} >= {min_accuracy}")
            validation_status = "PASSED"
        else:
            logging.warning(f"DBT Model validation FAILED. Accuracy: {accuracy:.4f} < {min_accuracy}")
            validation_status = "FAILED"
        
        # Log top features
        logging.info("Top 10 most important features from DBT processing:")
        sorted_features = sorted(metadata['feature_importance'].items(), 
                               key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:10]:
            logging.info(f"  {feature}: {importance:.4f}")
        
        if validation_status == "FAILED":
            raise ValueError(f"DBT Model validation failed. Accuracy {accuracy:.4f} below threshold {min_accuracy}")
        
        return f"DBT Model validation {validation_status} with accuracy {accuracy:.4f}"
        
    except Exception as e:
        logging.error(f"Error in DBT model validation: {str(e)}")
        raise

def create_dbt_predictions(**context):
    """Generate predictions using DBT-enhanced model."""
    try:
        # Load model
        with open(f'{MODELS_DIR}/dbt_churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(f'{MODELS_DIR}/dbt_model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load latest DBT features for prediction
        df = pd.read_csv(f'{DATA_DIR}/processed/dbt_ml_features.csv')
        
        # Take a sample for prediction
        new_data = df.sample(n=min(200, len(df)), random_state=42)
        
        feature_columns = metadata['feature_columns']
        X_new = new_data[feature_columns].fillna(0)
        
        # Generate predictions
        predictions = model.predict(X_new)
        prediction_proba = model.predict_proba(X_new)
        
        # Create results with original customer data
        results_df = pd.DataFrame({
            'customer_id': new_data['CUSTOMER_ID'],
            'predicted_churn': predictions,
            'churn_probability': prediction_proba[:, 1],
            'model_type': 'dbt_enhanced',
            'prediction_date': datetime.now()
        })
        
        # Add risk categories based on DBT features
        results_df['risk_category'] = pd.cut(
            results_df['churn_probability'], 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['low_risk', 'medium_risk', 'high_risk']
        )
        
        # Save predictions
        os.makedirs(f'{DATA_DIR}/predictions', exist_ok=True)
        predictions_file = f'{DATA_DIR}/predictions/dbt_churn_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(predictions_file, index=False)
        
        # Calculate summary statistics
        churn_rate = predictions.mean()
        high_risk_customers = (prediction_proba[:, 1] > 0.7).sum()
        medium_risk_customers = ((prediction_proba[:, 1] > 0.3) & (prediction_proba[:, 1] <= 0.7)).sum()
        low_risk_customers = (prediction_proba[:, 1] <= 0.3).sum()
        
        logging.info(f"DBT-enhanced predictions generated for {len(new_data)} customers")
        logging.info(f"Predicted churn rate: {churn_rate:.2%}")
        logging.info(f"High-risk customers: {high_risk_customers}")
        logging.info(f"Medium-risk customers: {medium_risk_customers}")
        logging.info(f"Low-risk customers: {low_risk_customers}")
        
        return f"DBT predictions: {len(new_data)} customers, {churn_rate:.2%} churn rate"
        
    except Exception as e:
        logging.error(f"Error generating DBT predictions: {str(e)}")
        raise

# Define main pipeline tasks
extract_task = PythonOperator(
    task_id='validate_source_data',
    python_callable=extract_data_from_snowflake,
    dag=dag
)

# DBT transformation task
dbt_transform_task = BashOperator(
    task_id='run_dbt_transformations',
    bash_command=f'cd {DBT_PROJECT_DIR} && /home/magicdash/airflow-project/airflow-env/bin/dbt run --profiles-dir ~/.dbt',
    dag=dag
)

dbt_test_task = BashOperator(
    task_id='run_dbt_tests',
    bash_command=f'cd {DBT_PROJECT_DIR} && /home/magicdash/airflow-project/airflow-env/bin/dbt test --profiles-dir ~/.dbt',
    dag=dag
)

load_features_task = PythonOperator(
    task_id='load_ml_features',
    python_callable=load_ml_features,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_dbt_enhanced_model',
    python_callable=train_churn_model_dbt,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate_dbt_model',
    python_callable=validate_dbt_model,
    dag=dag
)

predict_task = PythonOperator(
    task_id='create_dbt_predictions',
    python_callable=create_dbt_predictions,
    dag=dag
)

# Set task dependencies
extract_task >> dbt_transform_task >> dbt_test_task >> load_features_task >> train_task >> validate_task >> predict_task