#!/usr/bin/env python3

import sys
import os
import json
import pandas as pd
from datetime import datetime
import logging

# Add churn project to path
sys.path.append('/home/magicdash/airflow/churn')

from config.aws_config import upload_file_to_s3, upload_string_to_s3, generate_churn_s3_paths, get_s3_url

def upload_churn_predictions(predictions_file: str, run_date: str, run_id: str) -> dict:
    """Upload churn predictions CSV to S3."""
    try:
        paths = generate_churn_s3_paths(run_date, run_id)
        
        # Upload predictions CSV
        success = upload_file_to_s3(
            local_file_path=predictions_file,
            bucket_type='churn_results',
            s3_key=paths['predictions'],
            metadata={
                'pipeline': 'telco_churn_prediction',
                'run_date': run_date,
                'run_id': run_id,
                'file_type': 'predictions'
            }
        )
        
        if success:
            s3_url = get_s3_url('churn_results', paths['predictions'])
            logging.info(f"Predictions uploaded to: {s3_url}")
            return {'success': True, 'url': s3_url, 'key': paths['predictions']}
        else:
            return {'success': False, 'error': 'Upload failed'}
            
    except Exception as e:
        logging.error(f"Error uploading predictions: {str(e)}")
        return {'success': False, 'error': str(e)}

def upload_model_metadata(metadata_file: str, run_date: str, run_id: str) -> dict:
    """Upload model metadata JSON to S3."""
    try:
        paths = generate_churn_s3_paths(run_date, run_id)
        
        success = upload_file_to_s3(
            local_file_path=metadata_file,
            bucket_type='churn_results',
            s3_key=paths['model_metadata'],
            metadata={
                'pipeline': 'telco_churn_prediction',
                'run_date': run_date,
                'run_id': run_id,
                'file_type': 'model_metadata'
            }
        )
        
        if success:
            s3_url = get_s3_url('churn_results', paths['model_metadata'])
            logging.info(f"Model metadata uploaded to: {s3_url}")
            return {'success': True, 'url': s3_url, 'key': paths['model_metadata']}
        else:
            return {'success': False, 'error': 'Upload failed'}
            
    except Exception as e:
        logging.error(f"Error uploading model metadata: {str(e)}")
        return {'success': False, 'error': str(e)}

def upload_model_artifacts(models_dir: str, run_date: str, run_id: str) -> dict:
    """Upload model artifacts (pickle files) to S3."""
    try:
        results = {}
        base_path = f"models/{run_date}/{run_id}"
        
        # List of model files to upload
        model_files = [
            'dbt_churn_model.pkl',
            'dbt_model_metadata.pkl',
            'churn_model.pkl',
            'model_results.pkl',
            'validation_results.pkl'
        ]
        
        for model_file in model_files:
            local_path = os.path.join(models_dir, model_file)
            
            if os.path.exists(local_path):
                s3_key = f"{base_path}/{model_file}"
                
                success = upload_file_to_s3(
                    local_file_path=local_path,
                    bucket_type='ml_models',
                    s3_key=s3_key,
                    metadata={
                        'pipeline': 'telco_churn_prediction',
                        'run_date': run_date,
                        'run_id': run_id,
                        'file_type': 'model_artifact',
                        'model_name': model_file.replace('.pkl', '')
                    }
                )
                
                if success:
                    s3_url = get_s3_url('ml_models', s3_key)
                    results[model_file] = {'success': True, 'url': s3_url, 'key': s3_key}
                    logging.info(f"Model {model_file} uploaded to: {s3_url}")
                else:
                    results[model_file] = {'success': False, 'error': 'Upload failed'}
            else:
                results[model_file] = {'success': False, 'error': 'File not found'}
        
        return results
        
    except Exception as e:
        logging.error(f"Error uploading model artifacts: {str(e)}")
        return {'error': str(e)}

def create_and_upload_summary_report(predictions_file: str, metadata_file: str, run_date: str, run_id: str) -> dict:
    """Create a summary report and upload to S3."""
    try:
        # Load predictions data
        df = pd.read_csv(predictions_file)
        
        # Load model metadata if available
        model_metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                model_metadata = json.load(f)
        
        # Create summary report
        summary = {
            'pipeline_run': {
                'run_date': run_date,
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'pipeline_type': 'telco_churn_prediction'
            },
            'data_summary': {
                'total_customers': len(df),
                'predicted_churners': int(df['predicted_churn'].sum()) if 'predicted_churn' in df.columns else 0,
                'churn_rate': float(df['predicted_churn'].mean()) if 'predicted_churn' in df.columns else 0.0,
                'high_risk_customers': int((df['churn_probability'] > 0.7).sum()) if 'churn_probability' in df.columns else 0,
                'medium_risk_customers': int(((df['churn_probability'] > 0.3) & (df['churn_probability'] <= 0.7)).sum()) if 'churn_probability' in df.columns else 0,
                'low_risk_customers': int((df['churn_probability'] <= 0.3).sum()) if 'churn_probability' in df.columns else 0
            },
            'model_performance': {
                'accuracy': model_metadata.get('accuracy', 'N/A'),
                'model_type': model_metadata.get('model_type', 'N/A'),
                'features_used': model_metadata.get('n_features', 'N/A'),
                'training_records': model_metadata.get('training_records', 'N/A')
            },
            's3_outputs': {
                'predictions_file': f"s3://magicdash-data-pipeline-churn-results/churn-predictions/{run_date}/{run_id}/predictions.csv",
                'model_metadata': f"s3://magicdash-data-pipeline-churn-results/churn-predictions/{run_date}/{run_id}/model_metadata.json",
                'model_artifacts': f"s3://magicdash-data-pipeline-ml-models/models/{run_date}/{run_id}/"
            }
        }
        
        # Convert to JSON
        report_content = json.dumps(summary, indent=2)
        
        # Upload summary report
        paths = generate_churn_s3_paths(run_date, run_id)
        summary_key = f"churn-predictions/{run_date}/{run_id}/pipeline_summary.json"
        
        success = upload_string_to_s3(
            content=report_content,
            bucket_type='churn_results',
            s3_key=summary_key,
            content_type='application/json',
            metadata={
                'pipeline': 'telco_churn_prediction',
                'run_date': run_date,
                'run_id': run_id,
                'file_type': 'pipeline_summary'
            }
        )
        
        if success:
            s3_url = get_s3_url('churn_results', summary_key)
            logging.info(f"Pipeline summary uploaded to: {s3_url}")
            return {'success': True, 'url': s3_url, 'key': summary_key, 'summary': summary}
        else:
            return {'success': False, 'error': 'Upload failed'}
            
    except Exception as e:
        logging.error(f"Error creating summary report: {str(e)}")
        return {'success': False, 'error': str(e)}

def upload_all_churn_outputs(data_dir: str, models_dir: str, run_date: str, run_id: str) -> dict:
    """Upload all churn pipeline outputs to S3."""
    logging.info(f"Starting S3 upload for run {run_id} on {run_date}")
    
    results = {
        'run_info': {'run_date': run_date, 'run_id': run_id},
        'uploads': {}
    }
    
    try:
        # Find latest prediction file
        import glob
        prediction_files = glob.glob(f"{data_dir}/predictions/*churn_predictions_*.csv")
        if prediction_files:
            latest_predictions = max(prediction_files, key=os.path.getctime)
            results['uploads']['predictions'] = upload_churn_predictions(latest_predictions, run_date, run_id)
        else:
            results['uploads']['predictions'] = {'success': False, 'error': 'No prediction files found'}
        
        # Upload model metadata
        metadata_files = glob.glob(f"{models_dir}/*model_metadata.pkl")
        if metadata_files:
            latest_metadata = max(metadata_files, key=os.path.getctime)
            # Convert pickle to JSON for upload
            import pickle
            with open(latest_metadata, 'rb') as f:
                metadata = pickle.load(f)
            
            temp_json_file = f"{models_dir}/temp_metadata.json"
            with open(temp_json_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            results['uploads']['metadata'] = upload_model_metadata(temp_json_file, run_date, run_id)
            os.remove(temp_json_file)  # Cleanup temp file
        else:
            results['uploads']['metadata'] = {'success': False, 'error': 'No metadata files found'}
        
        # Upload model artifacts
        results['uploads']['models'] = upload_model_artifacts(models_dir, run_date, run_id)
        
        # Create and upload summary report
        if prediction_files:
            temp_metadata_file = f"{models_dir}/temp_metadata.json"
            if metadata_files:
                with open(latest_metadata, 'rb') as f:
                    metadata = pickle.load(f)
                with open(temp_metadata_file, 'w') as f:
                    json.dump(metadata, f, default=str)
            else:
                with open(temp_metadata_file, 'w') as f:
                    json.dump({}, f)
            
            results['uploads']['summary'] = create_and_upload_summary_report(
                latest_predictions, temp_metadata_file, run_date, run_id
            )
            
            if os.path.exists(temp_metadata_file):
                os.remove(temp_metadata_file)
        
        # Calculate overall success
        total_uploads = 0
        successful_uploads = 0
        
        for upload_type, result in results['uploads'].items():
            if isinstance(result, dict) and 'success' in result:
                total_uploads += 1
                if result['success']:
                    successful_uploads += 1
            elif isinstance(result, dict):  # Model artifacts
                for model_name, model_result in result.items():
                    if model_name != 'error':
                        total_uploads += 1
                        if model_result.get('success', False):
                            successful_uploads += 1
        
        results['overall_success'] = successful_uploads == total_uploads
        results['upload_stats'] = {
            'total': total_uploads,
            'successful': successful_uploads,
            'failed': total_uploads - successful_uploads
        }
        
        logging.info(f"S3 upload completed: {successful_uploads}/{total_uploads} successful")
        return results
        
    except Exception as e:
        logging.error(f"Error in upload_all_churn_outputs: {str(e)}")
        results['error'] = str(e)
        return results

if __name__ == "__main__":
    # Example usage
    data_dir = "/home/magicdash/airflow/churn/data"
    models_dir = "/home/magicdash/airflow/churn/models"
    run_date = datetime.now().strftime("%Y%m%d")
    run_id = f"manual_{datetime.now().strftime('%H%M%S')}"
    
    results = upload_all_churn_outputs(data_dir, models_dir, run_date, run_id)
    print(json.dumps(results, indent=2, default=str))