from datetime import datetime, timedelta
import sys
import os

# Add churn project to path
sys.path.append('/home/magicdash/airflow/churn')

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
import pandas as pd
import snowflake.connector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import logging
import json
from config.snowflake_config import get_snowflake_connection_params
# Removed problematic import - using inline boto3 instead

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

dag = DAG(
    'telco_churn_pipeline_with_s3',
    default_args=default_args,
    description='Telco Customer Churn Prediction Pipeline with DBT and S3',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1
)

def validate_source_data(**context):
    """Validate that source data exists in Snowflake."""
    try:
        conn_params = get_snowflake_connection_params()
        
        conn = snowflake.connector.connect(**conn_params)
        cursor = conn.cursor()
        
        query = "SELECT COUNT(*) FROM DATABASE.PUBLIC.CHURN"
        
        logging.info(f"Executing validation query: {query}")
        cursor.execute(query)
        
        count = cursor.fetchone()[0]
        logging.info(f"Source table has {count:,} records")
        
        cursor.close()
        conn.close()
        
        if count == 0:
            raise ValueError("Source table is empty")
        
        return f"Source validation completed: {count:,} records available"
        
    except Exception as e:
        logging.error(f"Error validating source data: {str(e)}")
        raise

def load_ml_features_from_dbt(**context):
    """Load ML-ready features from DBT mart tables."""
    try:
        conn_params = get_snowflake_connection_params()
        
        conn = snowflake.connector.connect(**conn_params)
        cursor = conn.cursor()
        
        # Load from DBT mart table
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
        # If DBT tables don't exist, fall back to direct raw data processing
        logging.warning("Falling back to direct raw data processing")
        return load_raw_data_fallback()

def load_raw_data_fallback(**context):
    """Fallback: Load and process raw data directly."""
    try:
        conn_params = get_snowflake_connection_params()
        
        conn = snowflake.connector.connect(**conn_params)
        cursor = conn.cursor()
        
        query = "SELECT * FROM DATABASE.PUBLIC.CHURN"
        
        logging.info(f"Loading raw data as fallback")
        cursor.execute(query)
        
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        
        df = pd.DataFrame(data, columns=columns)
        
        # Basic preprocessing
        df = df.dropna()
        
        # Save processed data
        os.makedirs(f'{DATA_DIR}/processed', exist_ok=True)
        df.to_csv(f'{DATA_DIR}/processed/raw_fallback_data.csv', index=False)
        
        logging.info(f"Loaded {len(df)} records in fallback mode")
        
        cursor.close()
        conn.close()
        
        return f"Fallback: loaded {len(df)} records from raw data"
        
    except Exception as e:
        logging.error(f"Error in fallback data loading: {str(e)}")
        raise

def train_enhanced_churn_model(**context):
    """Train machine learning model using processed features."""
    try:
        # Try to load DBT-processed features first
        dbt_file = f'{DATA_DIR}/processed/dbt_ml_features.csv'
        fallback_file = f'{DATA_DIR}/processed/raw_fallback_data.csv'
        
        if os.path.exists(dbt_file):
            df = pd.read_csv(dbt_file)
            model_type = "DBT_Enhanced"
            logging.info(f"Training with DBT-enhanced features: {len(df)} records")
            
            # Features are already encoded by DBT
            feature_columns = [col for col in df.columns 
                              if col not in ['CUSTOMER_ID', 'TARGET_CHURNED', 'FEATURE_CREATED_AT', 'MART_CREATED_AT', 'HAS_DATA_QUALITY_ISSUE']]
            
            X = df[feature_columns].fillna(0)
            y = df['TARGET_CHURNED']
            
        else:
            # Fallback to raw data processing
            df = pd.read_csv(fallback_file)
            model_type = "Fallback_Basic"
            logging.info(f"Training with fallback features: {len(df)} records")
            
            # Basic feature engineering for fallback
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Find target column
            target_col = 'CHURN'
            for col in ['Churn', 'CHURN', 'churn']:
                if col in df.columns:
                    target_col = col
                    break
            
            if target_col in categorical_columns:
                categorical_columns.remove(target_col)
            if target_col in numerical_columns:
                numerical_columns.remove(target_col)
            
            # Encode categorical variables
            for col in categorical_columns:
                df[col] = pd.Categorical(df[col]).codes
            
            # Scale numerical features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            if numerical_columns:
                df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            
            # Prepare features
            feature_columns = [col for col in df.columns if col != target_col and col != 'CUSTOMERID']
            X = df[feature_columns].fillna(0)
            
            # Encode target
            if df[target_col].dtype == 'object':
                y = pd.Categorical(df[target_col]).codes
            else:
                y = df[target_col]
        
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
        
        logging.info(f"Training {model_type} model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"{model_type} model training completed")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Training set size: {len(X_train)}")
        logging.info(f"Test set size: {len(X_test)}")
        logging.info(f"Features used: {len(feature_columns)}")
        
        # Save model and metadata
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(f'{MODELS_DIR}/enhanced_churn_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Save model metadata
        model_metadata = {
            'accuracy': accuracy,
            'feature_columns': feature_columns,
            'model_type': model_type,
            'n_features': len(feature_columns),
            'training_records': len(X_train),
            'test_records': len(X_test),
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': dict(zip(feature_columns, model.feature_importances_)),
            'training_timestamp': datetime.now().isoformat()
        }
        
        with open(f'{MODELS_DIR}/enhanced_model_metadata.pkl', 'wb') as f:
            pickle.dump(model_metadata, f)
        
        return f"{model_type} model trained with accuracy: {accuracy:.4f}"
        
    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        raise

def validate_enhanced_model(**context):
    """Validate the trained model."""
    try:
        # Load model results
        with open(f'{MODELS_DIR}/enhanced_model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        accuracy = metadata['accuracy']
        model_type = metadata['model_type']
        
        # Set validation threshold based on model type
        min_accuracy = 0.75 if model_type == "DBT_Enhanced" else 0.70
        
        if accuracy >= min_accuracy:
            logging.info(f"Model validation PASSED. Accuracy: {accuracy:.4f} >= {min_accuracy}")
            validation_status = "PASSED"
        else:
            logging.warning(f"Model validation FAILED. Accuracy: {accuracy:.4f} < {min_accuracy}")
            validation_status = "FAILED"
        
        # Log top features
        logging.info("Top 10 most important features:")
        sorted_features = sorted(metadata['feature_importance'].items(), 
                               key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:10]:
            logging.info(f"  {feature}: {importance:.4f}")
        
        # Save validation results
        validation_results = {
            'validation_status': validation_status,
            'accuracy': accuracy,
            'min_accuracy_threshold': min_accuracy,
            'model_type': model_type,
            'top_features': sorted_features[:10],
            'validation_timestamp': datetime.now().isoformat()
        }
        
        with open(f'{MODELS_DIR}/enhanced_validation_results.pkl', 'wb') as f:
            pickle.dump(validation_results, f)
        
        if validation_status == "FAILED":
            raise ValueError(f"Model validation failed. Accuracy {accuracy:.4f} below threshold {min_accuracy}")
        
        return f"Model validation {validation_status} with accuracy {accuracy:.4f}"
        
    except Exception as e:
        logging.error(f"Error in model validation: {str(e)}")
        raise

def create_enhanced_predictions(**context):
    """Generate predictions using the trained model."""
    try:
        # Load model
        with open(f'{MODELS_DIR}/enhanced_churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(f'{MODELS_DIR}/enhanced_model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load features for prediction
        dbt_file = f'{DATA_DIR}/processed/dbt_ml_features.csv'
        fallback_file = f'{DATA_DIR}/processed/raw_fallback_data.csv'
        
        if os.path.exists(dbt_file):
            df = pd.read_csv(dbt_file)
            data_source = "DBT"
        else:
            df = pd.read_csv(fallback_file)
            data_source = "Fallback"
        
        # Take a sample for prediction
        new_data = df.sample(n=min(200, len(df)), random_state=42)
        
        feature_columns = metadata['feature_columns']
        X_new = new_data[feature_columns].fillna(0)
        
        # Generate predictions
        predictions = model.predict(X_new)
        prediction_proba = model.predict_proba(X_new)
        
        # Create results DataFrame
        if data_source == "DBT" and 'CUSTOMER_ID' in new_data.columns:
            customer_ids = new_data['CUSTOMER_ID']
        elif 'CUSTOMERID' in new_data.columns:
            customer_ids = new_data['CUSTOMERID']
        else:
            customer_ids = range(len(new_data))
        
        results_df = pd.DataFrame({
            'customer_id': customer_ids,
            'predicted_churn': predictions,
            'churn_probability': prediction_proba[:, 1],
            'model_type': metadata['model_type'],
            'data_source': data_source,
            'prediction_date': datetime.now(),
            'risk_category': pd.cut(
                prediction_proba[:, 1], 
                bins=[0, 0.3, 0.7, 1.0], 
                labels=['low_risk', 'medium_risk', 'high_risk']
            )
        })
        
        # Save predictions
        os.makedirs(f'{DATA_DIR}/predictions', exist_ok=True)
        predictions_file = f'{DATA_DIR}/predictions/enhanced_churn_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(predictions_file, index=False)
        
        # Calculate summary statistics
        churn_rate = predictions.mean()
        high_risk_customers = (prediction_proba[:, 1] > 0.7).sum()
        medium_risk_customers = ((prediction_proba[:, 1] > 0.3) & (prediction_proba[:, 1] <= 0.7)).sum()
        low_risk_customers = (prediction_proba[:, 1] <= 0.3).sum()
        
        logging.info(f"Enhanced predictions generated for {len(new_data)} customers")
        logging.info(f"Predicted churn rate: {churn_rate:.2%}")
        logging.info(f"High-risk customers: {high_risk_customers}")
        logging.info(f"Medium-risk customers: {medium_risk_customers}")
        logging.info(f"Low-risk customers: {low_risk_customers}")
        
        return f"Enhanced predictions: {len(new_data)} customers, {churn_rate:.2%} churn rate"
        
    except Exception as e:
        logging.error(f"Error generating predictions: {str(e)}")
        raise

def upload_results_to_s3(**context):
    """Upload all results to S3 using simple boto3."""
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError
        import os
        import glob
        
        # Get run information
        run_date = context['ds_nodash']  # YYYYMMDD format
        run_id = context['run_id']
        
        logging.info(f"Starting S3 upload for run {run_id} on {run_date}")
        
        # Initialize S3 client
        s3_client = boto3.client('s3', region_name='ap-southeast-1')
        
        upload_count = 0
        total_files = 0
        
        # S3 bucket names
        churn_bucket = 'magicdash-data-pipeline-churn-results'
        models_bucket = 'magicdash-data-pipeline-ml-models'
        
        # Upload prediction files
        pred_files = glob.glob(f'{DATA_DIR}/predictions/*.csv')
        for file_path in pred_files:
            try:
                filename = os.path.basename(file_path)
                s3_key = f'predictions/{run_date}/{filename}'
                s3_client.upload_file(file_path, churn_bucket, s3_key)
                logging.info(f"Uploaded: s3://{churn_bucket}/{s3_key}")
                upload_count += 1
            except Exception as e:
                logging.error(f"Failed to upload {file_path}: {str(e)}")
            total_files += 1
        
        # Upload processed data files
        proc_files = glob.glob(f'{DATA_DIR}/processed/*.csv')
        for file_path in proc_files:
            try:
                filename = os.path.basename(file_path)
                s3_key = f'processed/{run_date}/{filename}'
                s3_client.upload_file(file_path, churn_bucket, s3_key)
                logging.info(f"Uploaded: s3://{churn_bucket}/{s3_key}")
                upload_count += 1
            except Exception as e:
                logging.error(f"Failed to upload {file_path}: {str(e)}")
            total_files += 1
        
        # Upload model files
        model_files = glob.glob(f'{MODELS_DIR}/*.pkl')
        for file_path in model_files:
            try:
                filename = os.path.basename(file_path)
                s3_key = f'models/{run_date}/{filename}'
                s3_client.upload_file(file_path, models_bucket, s3_key)
                logging.info(f"Uploaded: s3://{models_bucket}/{s3_key}")
                upload_count += 1
            except Exception as e:
                logging.error(f"Failed to upload {file_path}: {str(e)}")
            total_files += 1
        
        # Upload dbt outputs if they exist
        dbt_target_dir = f'{DBT_PROJECT_DIR}/target'
        if os.path.exists(dbt_target_dir):
            for root, dirs, files in os.walk(dbt_target_dir):
                for file in files:
                    if file.endswith(('.sql', '.json')):
                        try:
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, dbt_target_dir)
                            s3_key = f'dbt-outputs/{run_date}/{rel_path}'
                            s3_client.upload_file(file_path, churn_bucket, s3_key)
                            logging.info(f"Uploaded: s3://{churn_bucket}/{s3_key}")
                            upload_count += 1
                        except Exception as e:
                            logging.error(f"Failed to upload {file_path}: {str(e)}")
                        total_files += 1
        
        logging.info(f"✅ S3 upload completed: {upload_count}/{total_files} files uploaded successfully")
        return f"S3 upload completed: {upload_count}/{total_files} files uploaded"
        
    except Exception as e:
        logging.error(f"Error uploading to S3: {str(e)}")
        # Don't fail the entire pipeline if S3 upload fails
        return f"S3 upload failed: {str(e)}"

# Define pipeline tasks
validate_task = PythonOperator(
    task_id='validate_source_data',
    python_callable=validate_source_data,
    dag=dag
)

# DBT transformation task
dbt_transform_task = BashOperator(
    task_id='run_dbt_transformations',
    bash_command=f'cd {DBT_PROJECT_DIR} && /home/magicdash/airflow-project/airflow-env/bin/dbt run --profiles-dir ~/.dbt || echo "DBT failed, will use fallback"',
    dag=dag
)

dbt_test_task = BashOperator(
    task_id='run_dbt_tests',
    bash_command=f'cd {DBT_PROJECT_DIR} && /home/magicdash/airflow-project/airflow-env/bin/dbt test --profiles-dir ~/.dbt || echo "DBT tests failed, continuing"',
    dag=dag
)

load_features_task = PythonOperator(
    task_id='load_ml_features',
    python_callable=load_ml_features_from_dbt,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_enhanced_model',
    python_callable=train_enhanced_churn_model,
    dag=dag
)

validate_model_task = PythonOperator(
    task_id='validate_enhanced_model',
    python_callable=validate_enhanced_model,
    dag=dag
)

predict_task = PythonOperator(
    task_id='create_enhanced_predictions',
    python_callable=create_enhanced_predictions,
    dag=dag
)

s3_upload_task = PythonOperator(
    task_id='upload_results_to_s3',
    python_callable=upload_results_to_s3,
    dag=dag
)

# Set task dependencies
validate_task >> dbt_transform_task >> dbt_test_task >> load_features_task >> train_task >> validate_model_task >> predict_task >> s3_upload_task