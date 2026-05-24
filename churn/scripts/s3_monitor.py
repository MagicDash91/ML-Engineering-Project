#!/usr/bin/env python3

import sys
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Add churn project to path
sys.path.append('/home/magicdash/airflow/churn')

from config.aws_config import list_s3_objects, get_bucket_name, get_s3_client

def monitor_s3_pipeline_outputs(days_back: int = 7) -> Dict:
    """Monitor S3 bucket for recent pipeline outputs."""
    try:
        results = {
            'monitoring_timestamp': datetime.now().isoformat(),
            'period_days': days_back,
            'buckets': {},
            'summary': {}
        }
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Monitor each bucket type
        bucket_types = ['churn_results', 'ml_models']
        
        for bucket_type in bucket_types:
            logging.info(f"Monitoring bucket: {bucket_type}")
            
            objects = list_s3_objects(bucket_type)
            recent_objects = []
            
            for obj in objects:
                if obj['last_modified'].replace(tzinfo=None) > cutoff_date:
                    recent_objects.append(obj)
            
            results['buckets'][bucket_type] = {
                'bucket_name': get_bucket_name(bucket_type),
                'total_objects': len(objects),
                'recent_objects': len(recent_objects),
                'recent_files': recent_objects
            }
        
        # Generate summary statistics
        total_files = sum(bucket['total_objects'] for bucket in results['buckets'].values())
        recent_files = sum(bucket['recent_objects'] for bucket in results['buckets'].values())
        
        results['summary'] = {
            'total_files_across_buckets': total_files,
            'recent_files_across_buckets': recent_files,
            'monitoring_status': 'healthy' if recent_files > 0 else 'no_recent_activity'
        }
        
        return results
        
    except Exception as e:
        logging.error(f"Error monitoring S3 outputs: {str(e)}")
        return {'error': str(e)}

def list_pipeline_runs(days_back: int = 30) -> Dict:
    """List all pipeline runs by analyzing S3 structure."""
    try:
        runs = []
        
        # List objects in churn results bucket
        objects = list_s3_objects('churn_results', 'churn-predictions/')
        
        # Extract run information from S3 paths
        run_info = {}
        for obj in objects:
            path_parts = obj['key'].split('/')
            if len(path_parts) >= 4:  # churn-predictions/YYYYMMDD/run_id/file
                run_date = path_parts[1]
                run_id = path_parts[2]
                run_key = f"{run_date}_{run_id}"
                
                if run_key not in run_info:
                    run_info[run_key] = {
                        'run_date': run_date,
                        'run_id': run_id,
                        'files': [],
                        'last_modified': obj['last_modified'],
                        'total_size': 0
                    }
                
                run_info[run_key]['files'].append({
                    'filename': path_parts[-1],
                    'size': obj['size'],
                    'last_modified': obj['last_modified']
                })
                run_info[run_key]['total_size'] += obj['size']
                
                # Update last_modified to the most recent file
                if obj['last_modified'] > run_info[run_key]['last_modified']:
                    run_info[run_key]['last_modified'] = obj['last_modified']
        
        # Filter by date range
        cutoff_date = datetime.now() - timedelta(days=days_back)
        for run_key, run_data in run_info.items():
            if run_data['last_modified'].replace(tzinfo=None) > cutoff_date:
                runs.append(run_data)
        
        # Sort by last_modified (most recent first)
        runs.sort(key=lambda x: x['last_modified'], reverse=True)
        
        return {
            'total_runs': len(runs),
            'period_days': days_back,
            'runs': runs
        }
        
    except Exception as e:
        logging.error(f"Error listing pipeline runs: {str(e)}")
        return {'error': str(e)}

def get_pipeline_run_details(run_date: str, run_id: str) -> Dict:
    """Get detailed information about a specific pipeline run."""
    try:
        details = {
            'run_date': run_date,
            'run_id': run_id,
            'files': {},
            'summary': None
        }
        
        # Check churn results bucket
        churn_prefix = f"churn-predictions/{run_date}/{run_id}/"
        churn_objects = list_s3_objects('churn_results', churn_prefix)
        
        details['files']['churn_results'] = churn_objects
        
        # Check ML models bucket
        models_prefix = f"models/{run_date}/{run_id}/"
        model_objects = list_s3_objects('ml_models', models_prefix)
        
        details['files']['ml_models'] = model_objects
        
        # Look for summary file
        for obj in churn_objects:
            if 'pipeline_summary.json' in obj['key']:
                details['has_summary'] = True
                break
        else:
            details['has_summary'] = False
        
        # Calculate total files and size
        total_files = len(churn_objects) + len(model_objects)
        total_size = sum(obj['size'] for obj in churn_objects + model_objects)
        
        details['statistics'] = {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'churn_result_files': len(churn_objects),
            'model_artifact_files': len(model_objects)
        }
        
        return details
        
    except Exception as e:
        logging.error(f"Error getting run details: {str(e)}")
        return {'error': str(e)}

def cleanup_old_runs(days_to_keep: int = 90) -> Dict:
    """Clean up old pipeline runs from S3 (dry run by default)."""
    try:
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleanup_candidates = []
        
        # Find old objects in churn results
        objects = list_s3_objects('churn_results', 'churn-predictions/')
        
        for obj in objects:
            if obj['last_modified'].replace(tzinfo=None) < cutoff_date:
                cleanup_candidates.append({
                    'bucket_type': 'churn_results',
                    'key': obj['key'],
                    'size': obj['size'],
                    'last_modified': obj['last_modified']
                })
        
        # Find old objects in ML models
        model_objects = list_s3_objects('ml_models', 'models/')
        
        for obj in model_objects:
            if obj['last_modified'].replace(tzinfo=None) < cutoff_date:
                cleanup_candidates.append({
                    'bucket_type': 'ml_models',
                    'key': obj['key'],
                    'size': obj['size'],
                    'last_modified': obj['last_modified']
                })
        
        total_size_to_cleanup = sum(obj['size'] for obj in cleanup_candidates)
        
        return {
            'dry_run': True,
            'cutoff_date': cutoff_date.isoformat(),
            'days_to_keep': days_to_keep,
            'files_to_delete': len(cleanup_candidates),
            'total_size_to_free_bytes': total_size_to_cleanup,
            'candidates': cleanup_candidates[:10]  # Show first 10 as sample
        }
        
    except Exception as e:
        logging.error(f"Error in cleanup analysis: {str(e)}")
        return {'error': str(e)}

def generate_s3_usage_report() -> Dict:
    """Generate comprehensive S3 usage report."""
    try:
        report = {
            'report_date': datetime.now().isoformat(),
            'buckets': {},
            'overall_summary': {}
        }
        
        bucket_types = ['raw_data', 'churn_results', 'ml_models']
        total_objects = 0
        total_size = 0
        
        for bucket_type in bucket_types:
            objects = list_s3_objects(bucket_type)
            bucket_size = sum(obj['size'] for obj in objects)
            
            # Analyze file types
            file_types = {}
            for obj in objects:
                ext = obj['key'].split('.')[-1].lower() if '.' in obj['key'] else 'no_extension'
                file_types[ext] = file_types.get(ext, 0) + 1
            
            # Find largest files
            largest_files = sorted(objects, key=lambda x: x['size'], reverse=True)[:5]
            
            report['buckets'][bucket_type] = {
                'bucket_name': get_bucket_name(bucket_type),
                'object_count': len(objects),
                'total_size_bytes': bucket_size,
                'total_size_mb': round(bucket_size / (1024 * 1024), 2),
                'file_types': file_types,
                'largest_files': largest_files
            }
            
            total_objects += len(objects)
            total_size += bucket_size
        
        report['overall_summary'] = {
            'total_objects': total_objects,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'total_size_gb': round(total_size / (1024 * 1024 * 1024), 2)
        }
        
        return report
        
    except Exception as e:
        logging.error(f"Error generating usage report: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python s3_monitor.py <command> [args]")
        print("Commands:")
        print("  monitor [days]        - Monitor recent S3 activity")
        print("  list-runs [days]      - List recent pipeline runs")
        print("  run-details <date> <id> - Get details for specific run")
        print("  cleanup-analysis [days] - Analyze old files for cleanup")
        print("  usage-report          - Generate S3 usage report")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "monitor":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        result = monitor_s3_pipeline_outputs(days)
        print(json.dumps(result, indent=2, default=str))
    
    elif command == "list-runs":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        result = list_pipeline_runs(days)
        print(json.dumps(result, indent=2, default=str))
    
    elif command == "run-details":
        if len(sys.argv) < 4:
            print("Usage: python s3_monitor.py run-details <run_date> <run_id>")
            sys.exit(1)
        run_date = sys.argv[2]
        run_id = sys.argv[3]
        result = get_pipeline_run_details(run_date, run_id)
        print(json.dumps(result, indent=2, default=str))
    
    elif command == "cleanup-analysis":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 90
        result = cleanup_old_runs(days)
        print(json.dumps(result, indent=2, default=str))
    
    elif command == "usage-report":
        result = generate_s3_usage_report()
        print(json.dumps(result, indent=2, default=str))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)