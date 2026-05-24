from datetime import datetime, timedelta
import sys
import os

# Add churn project to path
sys.path.append('/home/magicdash/airflow/churn')

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.models import Variable
import pandas as pd
import snowflake.connector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import logging
from config.snowflake_config import get_snowflake_connection_params, CHURN_TABLE

# Project paths
PROJECT_ROOT = '/home/magicdash/airflow/churn'
DATA_DIR = f'{PROJECT_ROOT}/data'
MODELS_DIR = f'{PROJECT_ROOT}/models'

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
    'telco_churn_pipeline',
    default_args=default_args,
    description='Telco Customer Churn Prediction Pipeline',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1
)

def extract_data_from_snowflake(**context):
    """Extract telco churn data from Snowflake."""
    try:
        conn_params = get_snowflake_connection_params()
        
        conn = snowflake.connector.connect(**conn_params)
        cursor = conn.cursor()
        
        query = f"""
        SELECT * FROM {CHURN_TABLE}
        WHERE 1=1
        """
        
        logging.info(f"Executing query: {query}")
        cursor.execute(query)
        
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        
        df = pd.DataFrame(data, columns=columns)
        
        os.makedirs(f'{DATA_DIR}/processed', exist_ok=True)
        df.to_csv(f'{DATA_DIR}/processed/raw_churn_data.csv', index=False)
        
        logging.info(f"Extracted {len(df)} records from Snowflake")
        logging.info(f"Data shape: {df.shape}")
        logging.info(f"Columns: {list(df.columns)}")
        
        cursor.close()
        conn.close()
        
        return f"Successfully extracted {len(df)} records"
        
    except Exception as e:
        logging.error(f"Error extracting data: {str(e)}")
        raise

def preprocess_data(**context):
    """Preprocess and engineer features for churn prediction."""
    try:
        df = pd.read_csv(f'{DATA_DIR}/processed/raw_churn_data.csv')
        
        logging.info(f"Starting preprocessing with {len(df)} records")
        
        # Handle missing values
        df = df.dropna()
        
        # Identify categorical and numerical columns
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Assume 'Churn' is the target variable (common in telco datasets)
        target_col = None
        possible_targets = ['Churn', 'CHURN', 'churn', 'Churned', 'CHURNED']
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            target_col = df.columns[-1]  # Use last column as target
            logging.warning(f"No standard churn column found, using {target_col} as target")
        
        # Remove target from categorical columns if present
        if target_col in categorical_columns:
            categorical_columns.remove(target_col)
        if target_col in numerical_columns:
            numerical_columns.remove(target_col)
        
        # Encode categorical variables
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        
        # Scale numerical features
        scaler = StandardScaler()
        if numerical_columns:
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        
        # Encode target variable if it's categorical
        target_encoder = None
        if df[target_col].dtype == 'object':
            target_encoder = LabelEncoder()
            df[target_col] = target_encoder.fit_transform(df[target_col])
        
        # Save processed data
        df.to_csv(f'{DATA_DIR}/processed/preprocessed_churn_data.csv', index=False)
        
        # Save encoders and scaler
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(f'{MODELS_DIR}/label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
        with open(f'{MODELS_DIR}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        if target_encoder:
            with open(f'{MODELS_DIR}/target_encoder.pkl', 'wb') as f:
                pickle.dump(target_encoder, f)
        
        # Save metadata
        metadata = {
            'target_column': target_col,
            'categorical_columns': categorical_columns,
            'numerical_columns': numerical_columns,
            'total_records': len(df),
            'features': [col for col in df.columns if col != target_col]
        }
        with open(f'{MODELS_DIR}/metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        logging.info(f"Preprocessing completed. Final shape: {df.shape}")
        logging.info(f"Target column: {target_col}")
        logging.info(f"Features: {len(metadata['features'])}")
        
        return f"Preprocessing completed with {len(df)} records"
        
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        raise

def train_churn_model(**context):
    """Train machine learning model for churn prediction."""
    try:
        df = pd.read_csv(f'{DATA_DIR}/processed/preprocessed_churn_data.csv')
        
        # Load metadata
        with open(f'{MODELS_DIR}/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        target_col = metadata['target_column']
        features = metadata['features']
        
        X = df[features]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        logging.info("Starting model training...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"Model training completed")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Training set size: {len(X_train)}")
        logging.info(f"Test set size: {len(X_test)}")
        
        # Save model
        with open(f'{MODELS_DIR}/churn_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Save test results
        results = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': dict(zip(features, model.feature_importances_))
        }
        with open(f'{MODELS_DIR}/model_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        return f"Model trained with accuracy: {accuracy:.4f}"
        
    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        raise

def validate_model(**context):
    """Validate the trained model and generate evaluation report."""
    try:
        # Load model and results
        with open(f'{MODELS_DIR}/churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(f'{MODELS_DIR}/model_results.pkl', 'rb') as f:
            results = pickle.load(f)
        
        accuracy = results['accuracy']
        
        # Define validation criteria
        min_accuracy = 0.7  # Minimum acceptable accuracy
        
        if accuracy >= min_accuracy:
            logging.info(f"Model validation PASSED. Accuracy: {accuracy:.4f} >= {min_accuracy}")
            validation_status = "PASSED"
        else:
            logging.warning(f"Model validation FAILED. Accuracy: {accuracy:.4f} < {min_accuracy}")
            validation_status = "FAILED"
        
        # Log feature importance
        logging.info("Top 10 most important features:")
        sorted_features = sorted(results['feature_importance'].items(), 
                               key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:10]:
            logging.info(f"  {feature}: {importance:.4f}")
        
        # Save validation results
        validation_results = {
            'validation_status': validation_status,
            'accuracy': accuracy,
            'min_accuracy_threshold': min_accuracy,
            'top_features': sorted_features[:10]
        }
        with open(f'{MODELS_DIR}/validation_results.pkl', 'wb') as f:
            pickle.dump(validation_results, f)
        
        if validation_status == "FAILED":
            raise ValueError(f"Model validation failed. Accuracy {accuracy:.4f} below threshold {min_accuracy}")
        
        return f"Model validation {validation_status} with accuracy {accuracy:.4f}"
        
    except Exception as e:
        logging.error(f"Error in model validation: {str(e)}")
        raise

def create_predictions(**context):
    """Generate predictions for new data."""
    try:
        # Load model and preprocessors
        with open(f'{MODELS_DIR}/churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(f'{MODELS_DIR}/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # For demo, we'll use a subset of the processed data as "new" data
        df = pd.read_csv(f'{DATA_DIR}/processed/preprocessed_churn_data.csv')
        
        # Take a sample for prediction (simulate new data)
        new_data = df.sample(n=min(100, len(df)), random_state=42)
        
        features = metadata['features']
        X_new = new_data[features]
        
        # Generate predictions
        predictions = model.predict(X_new)
        prediction_proba = model.predict_proba(X_new)
        
        # Create results DataFrame
        results_df = new_data.copy()
        results_df['predicted_churn'] = predictions
        results_df['churn_probability'] = prediction_proba[:, 1]  # Probability of churn
        results_df['prediction_date'] = datetime.now()
        
        # Save predictions
        os.makedirs(f'{DATA_DIR}/predictions', exist_ok=True)
        predictions_file = f'{DATA_DIR}/predictions/churn_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(predictions_file, index=False)
        
        # Calculate summary statistics
        churn_rate = (predictions == 1).mean()
        high_risk_customers = (prediction_proba[:, 1] > 0.7).sum()
        
        logging.info(f"Generated predictions for {len(new_data)} customers")
        logging.info(f"Predicted churn rate: {churn_rate:.2%}")
        logging.info(f"High-risk customers (>70% probability): {high_risk_customers}")
        
        return f"Generated {len(new_data)} predictions with {churn_rate:.2%} churn rate"
        
    except Exception as e:
        logging.error(f"Error generating predictions: {str(e)}")
        raise

# Define tasks
extract_task = PythonOperator(
    task_id='extract_data_from_snowflake',
    python_callable=extract_data_from_snowflake,
    dag=dag
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_churn_model',
    python_callable=train_churn_model,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag
)

predict_task = PythonOperator(
    task_id='create_predictions',
    python_callable=create_predictions,
    dag=dag
)

# Data quality check task
quality_check_task = BashOperator(
    task_id='data_quality_check',
    bash_command=f'''
    /home/magicdash/airflow-project/airflow-env/bin/python -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('{DATA_DIR}/processed/raw_churn_data.csv')
    
    # Basic quality checks
    if len(df) == 0:
        print('ERROR: No data extracted')
        sys.exit(1)
    
    if df.isnull().sum().sum() / (len(df) * len(df.columns)) > 0.5:
        print('ERROR: More than 50% missing values')
        sys.exit(1)
    
    print(f'Data quality check PASSED: {{len(df)}} records, {{len(df.columns)}} columns')
    print(f'Missing values: {{df.isnull().sum().sum()}} ({{df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.1f}}%)')

except Exception as e:
    print(f'ERROR in data quality check: {{e}}')
    sys.exit(1)
"
    ''',
    dag=dag
)

# Set task dependencies
extract_task >> quality_check_task >> preprocess_task >> train_task >> validate_task >> predict_task