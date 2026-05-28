import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import logging
from typing import Dict, Optional

# AWS Configuration
AWS_CONFIG = {
    'region': 'ap-southeast-1',
    'project_name': 'magicdash-data-pipeline',
    'buckets': {
        'raw_data': 'magicdash-data-pipeline-raw-data',
        'churn_results': 'magicdash-data-pipeline-churn-results', 
        'ml_models': 'magicdash-data-pipeline-ml-models'
    }
}

def get_s3_client():
    """Get configured S3 client."""
    try:
        # Try to use environment variables or IAM role
        session = boto3.Session(
            region_name=AWS_CONFIG['region']
        )
        s3_client = session.client('s3')
        return s3_client
    except Exception as e:
        logging.error(f"Failed to create S3 client: {str(e)}")
        raise

def get_bucket_name(bucket_type: str) -> str:
    """Get bucket name by type."""
    if bucket_type not in AWS_CONFIG['buckets']:
        raise ValueError(f"Unknown bucket type: {bucket_type}")
    return AWS_CONFIG['buckets'][bucket_type]

def upload_file_to_s3(local_file_path: str, bucket_type: str, s3_key: str, metadata: Optional[Dict] = None) -> bool:
    """
    Upload a file to S3.
    
    Args:
        local_file_path: Path to local file
        bucket_type: Type of bucket (raw_data, churn_results, ml_models)
        s3_key: S3 object key (path/filename.ext)
        metadata: Optional metadata dict
        
    Returns:
        bool: Success status
    """
    try:
        s3_client = get_s3_client()
        bucket_name = get_bucket_name(bucket_type)
        
        # Prepare upload parameters
        upload_args = {}
        if metadata:
            upload_args['Metadata'] = metadata
            
        # Upload file
        s3_client.upload_file(
            local_file_path, 
            bucket_name, 
            s3_key,
            ExtraArgs=upload_args
        )
        
        logging.info(f"Successfully uploaded {local_file_path} to s3://{bucket_name}/{s3_key}")
        return True
        
    except FileNotFoundError:
        logging.error(f"File not found: {local_file_path}")
        return False
    except NoCredentialsError:
        logging.error("AWS credentials not found")
        return False
    except ClientError as e:
        logging.error(f"S3 upload failed: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during S3 upload: {str(e)}")
        return False

def upload_string_to_s3(content: str, bucket_type: str, s3_key: str, content_type: str = 'text/plain', metadata: Optional[Dict] = None) -> bool:
    """
    Upload string content directly to S3.
    
    Args:
        content: String content to upload
        bucket_type: Type of bucket (raw_data, churn_results, ml_models)
        s3_key: S3 object key (path/filename.ext)
        content_type: MIME type of content
        metadata: Optional metadata dict
        
    Returns:
        bool: Success status
    """
    try:
        s3_client = get_s3_client()
        bucket_name = get_bucket_name(bucket_type)
        
        # Prepare upload parameters
        upload_args = {'ContentType': content_type}
        if metadata:
            upload_args['Metadata'] = metadata
            
        # Upload content
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=content.encode('utf-8'),
            **upload_args
        )
        
        logging.info(f"Successfully uploaded content to s3://{bucket_name}/{s3_key}")
        return True
        
    except ClientError as e:
        logging.error(f"S3 upload failed: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during S3 upload: {str(e)}")
        return False

def list_s3_objects(bucket_type: str, prefix: str = "") -> list:
    """List objects in S3 bucket with optional prefix."""
    try:
        s3_client = get_s3_client()
        bucket_name = get_bucket_name(bucket_type)
        
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
        
        objects = []
        if 'Contents' in response:
            for obj in response['Contents']:
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag']
                })
                
        return objects
        
    except Exception as e:
        logging.error(f"Error listing S3 objects: {str(e)}")
        return []

def get_s3_url(bucket_type: str, s3_key: str) -> str:
    """Get full S3 URL for an object."""
    bucket_name = get_bucket_name(bucket_type)
    return f"s3://{bucket_name}/{s3_key}"

def generate_churn_s3_paths(run_date: str, run_id: str) -> Dict[str, str]:
    """Generate standardized S3 paths for churn pipeline outputs."""
    base_path = f"churn-predictions/{run_date}/{run_id}"
    
    return {
        'predictions': f"{base_path}/predictions.csv",
        'model_metadata': f"{base_path}/model_metadata.json",
        'feature_importance': f"{base_path}/feature_importance.json",
        'validation_report': f"{base_path}/validation_report.json",
        'data_quality_report': f"{base_path}/data_quality_report.json"
    }