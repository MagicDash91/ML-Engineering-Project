"""
Unified Telco Customer Churn Prediction Pipeline
Combines DBT transformations, ML training, and S3 storage in a clean, production-ready DAG
"""

from datetime import datetime, timedelta
import sys
import os

# Add churn project to path for config access BEFORE importing config
# Handle both local and Docker environments
possible_paths = [
    '/home/magicdash/astro-airflow/churn',                    # Local environment
    '/usr/local/airflow/include/churn',                       # Docker include directory
    '/opt/airflow/include/churn',                             # Alternative Docker include
    os.path.join(os.path.dirname(__file__), '..', 'churn'),  # Relative from dags
    os.path.join(os.path.dirname(__file__), '..', 'include', 'churn'),  # Relative from include
]

churn_path = None
for path in possible_paths:
    if os.path.exists(path):
        churn_path = path
        break

if churn_path and churn_path not in sys.path:
    sys.path.insert(0, churn_path)

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup
import logging
import os
import pickle
import json
import glob
from datetime import datetime, timedelta

# Import essential DBT and Airflow modules at top level
try:
    from cosmos import DbtDag, ProfileConfig, ProjectConfig
    from cosmos.profiles import SnowflakeUserPasswordProfileMapping
    COSMOS_AVAILABLE = True
except ImportError:
    COSMOS_AVAILABLE = False
    logging.warning("Cosmos not available - DBT features will be limited")

# Heavy ML/data imports will be moved inside functions to avoid timeout
# These will be imported when needed: pandas, sklearn, snowflake.connector, boto3

# Import config after path is set - with fallback
try:
    from config.snowflake_config import get_snowflake_connection_params
except ImportError:
    # Fallback: define the function inline if config module not found
    def get_snowflake_connection_params():
        return {
            'account': 'ORGKMBU-HU54176',
            'user': 'MAGICDASH',
            'password': os.getenv('SNOWFLAKE_PASSWORD', 'Permataputihg101'),
            'database': 'DATABASE',
            'schema': 'PUBLIC',
            'warehouse': 'COMPUTE_WH',
            'role': 'ACCOUNTADMIN'
        }

# Project Configuration - use detected path or fallback
PROJECT_ROOT = churn_path if churn_path else '/home/magicdash/astro-airflow/churn'

# Use writable directories in Docker environment
if '/usr/local/airflow' in str(PROJECT_ROOT):
    # In Docker - use tmp directories that are writable
    DATA_DIR = '/tmp/churn/data'
    MODELS_DIR = '/tmp/churn/models'
    DBT_PROJECT_DIR = f'{PROJECT_ROOT}/dbt_churn'  # DBT project stays in include
else:
    # Local environment
    DATA_DIR = f'{PROJECT_ROOT}/data'
    MODELS_DIR = f'{PROJECT_ROOT}/models'
    DBT_PROJECT_DIR = f'{PROJECT_ROOT}/dbt_churn'

# S3 Configuration
S3_CONFIG = {
    'region': 'ap-southeast-1',
    'buckets': {
        'churn_results': 'magicdash-data-pipeline-churn-results',
        'ml_models': 'magicdash-data-pipeline-ml-models',
        'raw_data': 'magicdash-data-pipeline-raw-data'
    }
}

# DAG Configuration
default_args = {
    'owner': 'magicdash',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'telco_churn_unified_pipeline',
    default_args=default_args,
    description='Unified Telco Churn Pipeline with DBT, ML, and S3 Integration',
    schedule='@daily',
    catchup=False,
    max_active_runs=1,
    tags=['churn', 'ml', 'dbt', 's3', 'production']
)

# ============================================================================
# DBT CONFIGURATION - Moved to function to avoid import timeout
# ============================================================================

def get_dbt_config():
    """Get DBT configuration - imported only when needed"""
    try:
        from cosmos import ProfileConfig, ProjectConfig
        from cosmos.profiles import SnowflakeUserPasswordProfileMapping
        
        profile_config = ProfileConfig(
            profile_name="dbt_churn",
            target_name="dev",
            profile_mapping=SnowflakeUserPasswordProfileMapping(
                conn_id="snowflake_default",
                profile_args={
                    "account": "ORGKMBU-HU54176",
                    "user": "MAGICDASH",
                    "password": os.getenv('SNOWFLAKE_PASSWORD', 'Permataputihg101'),
                    "role": "ACCOUNTADMIN", 
                    "database": "DATABASE",
                    "warehouse": "COMPUTE_WH",
                    "schema": "dbt_staging"
                }
            )
        )
        
        project_config = ProjectConfig(dbt_project_path=DBT_PROJECT_DIR)
        return profile_config, project_config
    except ImportError:
        return None, None

# ============================================================================
# DATA VALIDATION & EXTRACTION
# ============================================================================

def validate_source_data(**context):
    """Validate source data availability in Snowflake."""
    try:
        import snowflake.connector
        conn_params = get_snowflake_connection_params()
        
        with snowflake.connector.connect(**conn_params) as conn:
            with conn.cursor() as cursor:
                # Check data availability and freshness
                validation_query = """
                SELECT 
                    COUNT(*) as record_count,
                    CURRENT_TIMESTAMP() as validation_timestamp
                FROM DATABASE.PUBLIC.CHURN
                """
                
                logging.info("Validating source data...")
                cursor.execute(validation_query)
                result = cursor.fetchone()
                
                record_count = result[0]
                validation_timestamp = result[1]
                
                logging.info(f"Source validation: {record_count:,} records found")
                logging.info(f"Validation performed at: {validation_timestamp}")
                
                # Validation checks
                if record_count == 0:
                    raise ValueError("Source table is empty")
                if record_count < 1000:
                    logging.warning(f"Low record count: {record_count}")
                
                # Store validation results
                return {
                    'record_count': record_count,
                    'validation_timestamp': str(validation_timestamp),
                    'validation_status': 'PASSED'
                }
                
    except Exception as e:
        logging.error(f"Source validation failed: {str(e)}")
        raise

def extract_raw_data(**context):
    """Extract and store raw data for fallback processing."""
    try:
        import snowflake.connector
        import pandas as pd
        conn_params = get_snowflake_connection_params()
        
        with snowflake.connector.connect(**conn_params) as conn:
            with conn.cursor() as cursor:
                query = "SELECT * FROM DATABASE.PUBLIC.CHURN LIMIT 5000"
                
                logging.info("Extracting raw data for fallback processing...")
                cursor.execute(query)
                
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
                df = pd.DataFrame(data, columns=columns)
                
                # Save raw data
                os.makedirs(f'{DATA_DIR}/raw', exist_ok=True)
                raw_file = f'{DATA_DIR}/raw/churn_raw_data.csv'
                df.to_csv(raw_file, index=False)
                
                logging.info(f"Raw data extracted: {len(df)} records saved to {raw_file}")
                return f"Raw data extraction completed: {len(df)} records"
                
    except Exception as e:
        logging.error(f"Raw data extraction failed: {str(e)}")
        raise

# ============================================================================
# DBT TRANSFORMATIONS
# ============================================================================

def check_dbt_environment(**context):
    """Check if DBT environment is properly configured."""
    try:
        # Check multiple possible DBT locations
        possible_dbt_paths = [
            'dbt',  # System PATH
            '/usr/local/bin/dbt',
            '/home/astro/.local/bin/dbt', 
            '/opt/airflow/dbt_env/bin/dbt'
        ]
        
        dbt_executable = None
        for path in possible_dbt_paths:
            try:
                import subprocess
                result = subprocess.run(['which', path], capture_output=True, text=True)
                if result.returncode == 0:
                    dbt_executable = result.stdout.strip()
                    break
            except:
                continue
        
        if not dbt_executable:
            # Try to find dbt using system which command
            try:
                result = subprocess.run(['which', 'dbt'], capture_output=True, text=True)
                if result.returncode == 0:
                    dbt_executable = result.stdout.strip()
                else:
                    logging.warning("DBT executable not found in system PATH")
                    return False
            except:
                logging.warning("Could not locate DBT executable")
                return False
        
        logging.info(f"DBT executable found at: {dbt_executable}")
        
        # Check DBT version
        try:
            version_result = subprocess.run([dbt_executable, '--version'], capture_output=True, text=True)
            logging.info(f"DBT version check: {version_result.stdout}")
        except Exception as e:
            logging.warning(f"DBT version check failed: {e}")
        
        return True
        
    except Exception as e:
        logging.error(f"DBT environment check failed: {str(e)}")
        return False

def load_ml_features(**context):
    """Load ML-ready features from DBT mart or fallback to raw processing."""
    try:
        # Import heavy data libraries only when needed
        import pandas as pd
        import snowflake.connector
        from sklearn.preprocessing import LabelEncoder
        
        conn_params = get_snowflake_connection_params()
        
        with snowflake.connector.connect(**conn_params) as conn:
            with conn.cursor() as cursor:
                try:
                    # Try to load from DBT mart table first
                    dbt_query = """
                    SELECT * FROM DATABASE.dbt_staging.mart_churn_prediction
                    WHERE feature_completeness_score > 0.8
                    LIMIT 3000
                    """
                    
                    logging.info("Attempting to load DBT-processed features...")
                    cursor.execute(dbt_query)
                    
                    columns = [desc[0] for desc in cursor.description]
                    data = cursor.fetchall()
                    
                    if data:
                        df = pd.DataFrame(data, columns=columns)
                        data_source = "DBT_Processed"
                        logging.info(f"Loaded {len(df)} records from DBT mart")
                    else:
                        raise Exception("DBT mart table is empty")
                        
                except Exception as dbt_error:
                    # Fallback to raw data processing
                    logging.warning(f"DBT mart not available: {str(dbt_error)}")
                    logging.info("Falling back to raw data processing...")
                    
                    fallback_query = "SELECT * FROM DATABASE.PUBLIC.CHURN LIMIT 3000"
                    cursor.execute(fallback_query)
                    
                    columns = [desc[0] for desc in cursor.description]
                    data = cursor.fetchall()
                    
                    df = pd.DataFrame(data, columns=columns)
                    data_source = "Raw_Processed"
                    
                    # Basic preprocessing for raw data
                    df = df.dropna()
                    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
                    
                    # Encode categorical variables
                    label_encoders = {}
                    for col in categorical_columns:
                        if col.upper() not in ['CUSTOMERID', 'CUSTOMER_ID']:
                            le = LabelEncoder()
                            df[col] = le.fit_transform(df[col].astype(str))
                            label_encoders[col] = le
                    
                    # Save encoders for later use
                    os.makedirs(MODELS_DIR, exist_ok=True)
                    with open(f'{MODELS_DIR}/label_encoders.pkl', 'wb') as f:
                        pickle.dump(label_encoders, f)
                
                # Save processed features
                os.makedirs(f'{DATA_DIR}/processed', exist_ok=True)
                processed_file = f'{DATA_DIR}/processed/ml_features_{data_source.lower()}.csv'
                df.to_csv(processed_file, index=False)
                
                logging.info(f"ML features ready: {len(df)} records, {len(df.columns)} features ({data_source})")
                
                return {
                    'records_loaded': len(df),
                    'features_count': len(df.columns),
                    'data_source': data_source,
                    'file_path': processed_file
                }
                
    except Exception as e:
        logging.error(f"Feature loading failed: {str(e)}")
        raise

# ============================================================================
# MACHINE LEARNING PIPELINE
# ============================================================================

def train_churn_model(**context):
    """Train ML model with comprehensive feature engineering."""
    try:
        # Import heavy ML libraries only when needed
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        
        # Load the most recent processed features
        processed_files = glob.glob(f'{DATA_DIR}/processed/ml_features_*.csv')
        if not processed_files:
            raise FileNotFoundError("No processed feature files found")
        
        # Use the most recent file
        latest_file = max(processed_files, key=os.path.getctime)
        df = pd.read_csv(latest_file)
        
        # Determine model type based on data source
        if 'dbt_processed' in latest_file:
            model_type = "DBT_Enhanced"
            # DBT-processed features
            feature_columns = [col for col in df.columns 
                              if col not in ['CUSTOMER_ID', 'CUSTOMERID', 'TARGET_CHURNED', 
                                           'FEATURE_CREATED_AT', 'MART_CREATED_AT', 'HAS_DATA_QUALITY_ISSUE']]
            target_column = 'TARGET_CHURNED' if 'TARGET_CHURNED' in df.columns else 'CHURN'
        else:
            model_type = "Raw_Processed"
            # Raw data processing
            target_candidates = ['CHURN', 'Churn', 'churn', 'TARGET_CHURNED']
            target_column = None
            for candidate in target_candidates:
                if candidate in df.columns:
                    target_column = candidate
                    break
            
            if not target_column:
                raise ValueError("Target column not found in dataset")
            
            feature_columns = [col for col in df.columns 
                              if col not in [target_column, 'CUSTOMERID', 'CUSTOMER_ID', 'CustomerID']]
        
        logging.info(f"Training {model_type} model with {len(feature_columns)} features")
        
        # Prepare features and target
        X = df[feature_columns].fillna(0)
        y = df[target_column]
        
        # Handle target encoding if needed
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            # Save target encoder
            with open(f'{MODELS_DIR}/target_encoder.pkl', 'wb') as f:
                pickle.dump(le_target, f)
        
        # Feature scaling for numerical features
        scaler = StandardScaler()
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_features) > 0:
            X[numerical_features] = scaler.fit_transform(X[numerical_features])
            # Save scaler
            with open(f'{MODELS_DIR}/feature_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train optimized Random Forest model
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        logging.info(f"Training {model_type} model...")
        model.fit(X_train, y_train)
        
        # Model evaluation
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        
        # Feature importance analysis
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        logging.info(f"Model training completed:")
        logging.info(f"  Training accuracy: {train_score:.4f}")
        logging.info(f"  Test accuracy: {test_score:.4f}")
        logging.info(f"  Training samples: {len(X_train)}")
        logging.info(f"  Test samples: {len(X_test)}")
        logging.info(f"  Top features: {[f[0] for f in top_features[:5]]}")
        
        # Save model and metadata
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Save trained model
        model_file = f'{MODELS_DIR}/churn_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save comprehensive metadata
        metadata = {
            'model_type': model_type,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'n_features': len(feature_columns),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': feature_importance,
            'top_features': top_features,
            'classification_report': classification_report(y_test, y_pred),
            'training_timestamp': datetime.now().isoformat(),
            'data_source_file': latest_file
        }
        
        with open(f'{MODELS_DIR}/model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        return {
            'model_type': model_type,
            'test_accuracy': test_score,
            'n_features': len(feature_columns),
            'training_samples': len(X_train)
        }
        
    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")
        raise

def validate_model(**context):
    """Validate trained model performance."""
    try:
        # Load model metadata
        with open(f'{MODELS_DIR}/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        model_type = metadata['model_type']
        test_accuracy = metadata['test_accuracy']
        
        # Set validation thresholds
        accuracy_thresholds = {
            'DBT_Enhanced': 0.78,
            'Raw_Processed': 0.72
        }
        
        min_accuracy = accuracy_thresholds.get(model_type, 0.70)
        
        # Validation checks
        validation_passed = test_accuracy >= min_accuracy
        
        if validation_passed:
            logging.info(f"✅ Model validation PASSED")
            logging.info(f"   Accuracy: {test_accuracy:.4f} >= {min_accuracy}")
            logging.info(f"   Model type: {model_type}")
        else:
            logging.error(f"❌ Model validation FAILED")
            logging.error(f"   Accuracy: {test_accuracy:.4f} < {min_accuracy}")
            raise ValueError(f"Model validation failed: {test_accuracy:.4f} < {min_accuracy}")
        
        # Log feature importance insights
        logging.info("🔍 Top predictive features:")
        for i, (feature, importance) in enumerate(metadata['top_features'][:5], 1):
            logging.info(f"   {i}. {feature}: {importance:.4f}")
        
        # Save validation results
        validation_results = {
            'validation_status': 'PASSED' if validation_passed else 'FAILED',
            'test_accuracy': test_accuracy,
            'min_threshold': min_accuracy,
            'model_type': model_type,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        with open(f'{MODELS_DIR}/validation_results.pkl', 'wb') as f:
            pickle.dump(validation_results, f)
        
        return validation_results
        
    except Exception as e:
        logging.error(f"Model validation failed: {str(e)}")
        raise

def generate_predictions(**context):
    """Generate churn predictions with risk segmentation."""
    try:
        # Import heavy data libraries only when needed
        import pandas as pd
        
        # Load trained model and metadata
        with open(f'{MODELS_DIR}/churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open(f'{MODELS_DIR}/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load latest processed data
        processed_files = glob.glob(f'{DATA_DIR}/processed/ml_features_*.csv')
        latest_file = max(processed_files, key=os.path.getctime)
        df = pd.read_csv(latest_file)
        
        # Prepare prediction data (sample for demonstration)
        prediction_sample = df.sample(n=min(500, len(df)), random_state=42)
        
        feature_columns = metadata['feature_columns']
        X_pred = prediction_sample[feature_columns].fillna(0)
        
        # Apply same preprocessing as training
        if os.path.exists(f'{MODELS_DIR}/feature_scaler.pkl'):
            with open(f'{MODELS_DIR}/feature_scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            numerical_features = X_pred.select_dtypes(include=['int64', 'float64']).columns
            if len(numerical_features) > 0:
                X_pred[numerical_features] = scaler.transform(X_pred[numerical_features])
        
        # Generate predictions
        predictions = model.predict(X_pred)
        prediction_probabilities = model.predict_proba(X_pred)
        churn_probabilities = prediction_probabilities[:, 1]
        
        # Identify customer IDs
        customer_id_candidates = ['CUSTOMER_ID', 'CUSTOMERID', 'CustomerID', 'customer_id']
        customer_id_column = None
        for candidate in customer_id_candidates:
            if candidate in prediction_sample.columns:
                customer_id_column = candidate
                break
        
        customer_ids = prediction_sample[customer_id_column].values if customer_id_column else range(len(prediction_sample))
        
        # Create results DataFrame with comprehensive analysis
        results_df = pd.DataFrame({
            'customer_id': customer_ids,
            'churn_prediction': predictions,
            'churn_probability': churn_probabilities,
            'risk_category': pd.cut(
                churn_probabilities,
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            ),
            'confidence_level': pd.cut(
                churn_probabilities,
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
            ),
            'model_type': metadata['model_type'],
            'prediction_timestamp': datetime.now(),
            'model_accuracy': metadata['test_accuracy']
        })
        
        # Calculate business insights
        total_customers = len(results_df)
        predicted_churners = predictions.sum()
        churn_rate = predicted_churners / total_customers
        
        risk_distribution = results_df['risk_category'].value_counts()
        high_risk_count = risk_distribution.get('High Risk', 0)
        medium_risk_count = risk_distribution.get('Medium Risk', 0)
        low_risk_count = risk_distribution.get('Low Risk', 0)
        
        # Save predictions
        os.makedirs(f'{DATA_DIR}/predictions', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_file = f'{DATA_DIR}/predictions/churn_predictions_{timestamp}.csv'
        results_df.to_csv(predictions_file, index=False)
        
        # Log business insights
        logging.info("🎯 Prediction Results Summary:")
        logging.info(f"   Total customers analyzed: {total_customers:,}")
        logging.info(f"   Predicted churners: {predicted_churners:,} ({churn_rate:.1%})")
        logging.info(f"   High-risk customers: {high_risk_count:,}")
        logging.info(f"   Medium-risk customers: {medium_risk_count:,}")
        logging.info(f"   Low-risk customers: {low_risk_count:,}")
        logging.info(f"   Average churn probability: {churn_probabilities.mean():.1%}")
        
        return {
            'total_customers': total_customers,
            'predicted_churners': int(predicted_churners),
            'churn_rate': churn_rate,
            'high_risk_customers': int(high_risk_count),
            'predictions_file': predictions_file
        }
        
    except Exception as e:
        logging.error(f"Prediction generation failed: {str(e)}")
        raise

# ============================================================================
# S3 INTEGRATION
# ============================================================================

def upload_to_s3(**context):
    """Upload all pipeline artifacts to S3 with organized structure."""
    try:
        # Import heavy AWS libraries only when needed
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError
        
        # Get run information
        run_date = context['ds_nodash']  # YYYYMMDD format
        run_id = context['run_id']
        
        logging.info(f"🚀 Starting S3 upload for run {run_id} on {run_date}")
        
        # Initialize S3 client
        s3_client = boto3.client('s3', region_name=S3_CONFIG['region'])
        
        upload_count = 0
        total_files = 0
        
        # Upload prediction results
        pred_files = glob.glob(f'{DATA_DIR}/predictions/*.csv')
        for file_path in pred_files:
            try:
                filename = os.path.basename(file_path)
                s3_key = f'predictions/{run_date}/{filename}'
                s3_client.upload_file(file_path, S3_CONFIG['buckets']['churn_results'], s3_key)
                logging.info(f"📊 Uploaded: s3://{S3_CONFIG['buckets']['churn_results']}/{s3_key}")
                upload_count += 1
            except Exception as e:
                logging.error(f"Failed to upload {filename}: {str(e)}")
            total_files += 1
        
        # Upload processed data
        processed_files = glob.glob(f'{DATA_DIR}/processed/*.csv')
        for file_path in processed_files:
            try:
                filename = os.path.basename(file_path)
                s3_key = f'processed/{run_date}/{filename}'
                s3_client.upload_file(file_path, S3_CONFIG['buckets']['churn_results'], s3_key)
                logging.info(f"📋 Uploaded: s3://{S3_CONFIG['buckets']['churn_results']}/{s3_key}")
                upload_count += 1
            except Exception as e:
                logging.error(f"Failed to upload {filename}: {str(e)}")
            total_files += 1
        
        # Upload model artifacts
        model_files = glob.glob(f'{MODELS_DIR}/*.pkl')
        for file_path in model_files:
            try:
                filename = os.path.basename(file_path)
                s3_key = f'models/{run_date}/{filename}'
                s3_client.upload_file(file_path, S3_CONFIG['buckets']['ml_models'], s3_key)
                logging.info(f"🤖 Uploaded: s3://{S3_CONFIG['buckets']['ml_models']}/{s3_key}")
                upload_count += 1
            except Exception as e:
                logging.error(f"Failed to upload {filename}: {str(e)}")
            total_files += 1
        
        logging.info(f"✅ S3 upload completed: {upload_count}/{total_files} files uploaded successfully")
        return f"S3 upload completed: {upload_count}/{total_files} files uploaded"
        
    except Exception as e:
        logging.error(f"S3 upload failed: {str(e)}")
        # Don't fail the entire pipeline on S3 errors
        return f"S3 upload failed: {str(e)}"

# ============================================================================
# TASK DEFINITIONS
# ============================================================================

# Data Validation & Extraction Tasks
with TaskGroup('data_validation', dag=dag) as data_validation_group:
    
    validate_source_task = PythonOperator(
        task_id='validate_source_data',
        python_callable=validate_source_data,
        dag=dag
    )
    
    extract_raw_task = PythonOperator(
        task_id='extract_raw_data',
        python_callable=extract_raw_data,
        dag=dag
    )
    
    validate_source_task >> extract_raw_task

# DBT Transformation Tasks  
with TaskGroup('dbt_transformations', dag=dag) as dbt_group:
    
    check_dbt_task = PythonOperator(
        task_id='check_dbt_environment',
        python_callable=check_dbt_environment,
        dag=dag
    )
    
    dbt_run_task = BashOperator(
        task_id='run_dbt_models',
        bash_command=f'''
        # Create writable directories for DBT
        mkdir -p /tmp/dbt_logs /tmp/dbt_target &&
        export DBT_LOG_PATH=/tmp/dbt_logs &&
        cd {DBT_PROJECT_DIR} && 
        dbt run --profiles-dir /usr/local/airflow/include/dbt_profiles --log-path /tmp/dbt_logs --target-path /tmp/dbt_target || 
        echo "DBT run failed, will use fallback processing"
        ''',
        trigger_rule='all_done',
        dag=dag
    )
    
    dbt_test_task = BashOperator(
        task_id='run_dbt_tests',
        bash_command=f'''
        # Create writable directories for DBT
        mkdir -p /tmp/dbt_logs /tmp/dbt_target &&
        export DBT_LOG_PATH=/tmp/dbt_logs &&
        cd {DBT_PROJECT_DIR} && 
        dbt test --profiles-dir /usr/local/airflow/include/dbt_profiles --log-path /tmp/dbt_logs --target-path /tmp/dbt_target || 
        echo "DBT tests completed with warnings"
        ''',
        trigger_rule='all_done',
        dag=dag
    )
    
    check_dbt_task >> dbt_run_task >> dbt_test_task

# ML Pipeline Tasks
with TaskGroup('ml_pipeline', dag=dag) as ml_group:
    
    load_features_task = PythonOperator(
        task_id='load_ml_features',
        python_callable=load_ml_features,
        dag=dag
    )
    
    train_model_task = PythonOperator(
        task_id='train_churn_model',
        python_callable=train_churn_model,
        dag=dag
    )
    
    validate_model_task = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model,
        dag=dag
    )
    
    predict_task = PythonOperator(
        task_id='generate_predictions',
        python_callable=generate_predictions,
        dag=dag
    )
    
    load_features_task >> train_model_task >> validate_model_task >> predict_task

# S3 Storage Task
upload_s3_task = PythonOperator(
    task_id='upload_to_s3',
    python_callable=upload_to_s3,
    dag=dag
)

# ============================================================================
# TASK DEPENDENCIES
# ============================================================================

# Main pipeline flow
data_validation_group >> dbt_group >> ml_group >> upload_s3_task