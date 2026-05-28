#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/magicdash/airflow/churn')

import pandas as pd
import pickle
import json
from datetime import datetime, timedelta
import glob

PROJECT_ROOT = '/home/magicdash/airflow/churn'

def generate_churn_report():
    """Generate a comprehensive churn prediction report."""
    try:
        print("📊 Generating Churn Prediction Report")
        print("=" * 50)
        
        # Check if model exists
        model_path = f'{PROJECT_ROOT}/models/churn_model.pkl'
        if not os.path.exists(model_path):
            print("❌ No trained model found. Please run the training pipeline first.")
            return False
        
        # Load model metadata
        with open(f'{PROJECT_ROOT}/models/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        with open(f'{PROJECT_ROOT}/models/model_results.pkl', 'rb') as f:
            results = pickle.load(f)
        
        print(f"📈 Model Performance:")
        print(f"  - Accuracy: {results['accuracy']:.2%}")
        print(f"  - Target Column: {metadata['target_column']}")
        print(f"  - Features Used: {len(metadata['features'])}")
        print(f"  - Training Records: {metadata['total_records']:,}")
        
        # Top features
        print(f"\n🎯 Top 10 Most Important Features:")
        sorted_features = sorted(results['feature_importance'].items(), 
                               key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"  {i:2d}. {feature}: {importance:.4f}")
        
        # Check recent predictions
        predictions_dir = f'{PROJECT_ROOT}/data/predictions'
        if os.path.exists(predictions_dir):
            prediction_files = glob.glob(f"{predictions_dir}/churn_predictions_*.csv")
            if prediction_files:
                latest_file = max(prediction_files, key=os.path.getctime)
                latest_predictions = pd.read_csv(latest_file)
                
                print(f"\n📅 Latest Predictions ({os.path.basename(latest_file)}):")
                print(f"  - Total Customers Scored: {len(latest_predictions):,}")
                
                if 'predicted_churn' in latest_predictions.columns:
                    churn_rate = latest_predictions['predicted_churn'].mean()
                    print(f"  - Predicted Churn Rate: {churn_rate:.2%}")
                
                if 'churn_probability' in latest_predictions.columns:
                    high_risk = (latest_predictions['churn_probability'] > 0.7).sum()
                    medium_risk = ((latest_predictions['churn_probability'] > 0.5) & 
                                 (latest_predictions['churn_probability'] <= 0.7)).sum()
                    low_risk = (latest_predictions['churn_probability'] <= 0.5).sum()
                    
                    print(f"  - High Risk (>70%): {high_risk:,} customers")
                    print(f"  - Medium Risk (50-70%): {medium_risk:,} customers")
                    print(f"  - Low Risk (≤50%): {low_risk:,} customers")
        
        # Data freshness check
        raw_data_path = f'{PROJECT_ROOT}/data/processed/raw_churn_data.csv'
        if os.path.exists(raw_data_path):
            last_modified = datetime.fromtimestamp(os.path.getmtime(raw_data_path))
            hours_old = (datetime.now() - last_modified).total_seconds() / 3600
            print(f"\n🕐 Data Freshness:")
            print(f"  - Last Updated: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  - Age: {hours_old:.1f} hours")
            
            if hours_old > 24:
                print("  ⚠️  Warning: Data is more than 24 hours old")
        
        print("\n✅ Report generation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")
        return False

def check_pipeline_health():
    """Check the health status of the churn prediction pipeline."""
    try:
        print("\n🏥 Pipeline Health Check")
        print("=" * 30)
        
        checks = []
        
        # Check if configuration exists
        config_path = f'{PROJECT_ROOT}/config/snowflake_config.py'
        checks.append(("Snowflake Configuration", os.path.exists(config_path)))
        
        # Check if DAG file exists
        dag_path = f'{PROJECT_ROOT}/dags/telco_churn_pipeline.py'
        checks.append(("DAG File", os.path.exists(dag_path)))
        
        # Check if directories exist
        dirs_to_check = [
            f'{PROJECT_ROOT}/data/processed',
            f'{PROJECT_ROOT}/models',
            f'{PROJECT_ROOT}/data/predictions',
            f'{PROJECT_ROOT}/terraform'
        ]
        for dir_path in dirs_to_check:
            checks.append((f"Directory {os.path.basename(dir_path)}", os.path.exists(dir_path)))
        
        # Check if model files exist
        model_files = [
            f'{PROJECT_ROOT}/models/churn_model.pkl',
            f'{PROJECT_ROOT}/models/metadata.pkl',
            f'{PROJECT_ROOT}/models/scaler.pkl'
        ]
        for model_file in model_files:
            file_name = os.path.basename(model_file)
            checks.append((f"Model {file_name}", os.path.exists(model_file)))
        
        # Print results
        all_passed = True
        for check_name, status in checks:
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {check_name}")
            if not status:
                all_passed = False
        
        if all_passed:
            print("\n🎉 All health checks passed!")
        else:
            print("\n⚠️  Some components need attention.")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error in health check: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Churn Pipeline Monitoring')
    parser.add_argument('--report', action='store_true', help='Generate churn prediction report')
    parser.add_argument('--health', action='store_true', help='Check pipeline health')
    args = parser.parse_args()
    
    success = True
    
    if args.report:
        success = generate_churn_report() and success
    
    if args.health:
        success = check_pipeline_health() and success
    
    if not args.report and not args.health:
        # Run both by default
        success = generate_churn_report() and success
        success = check_pipeline_health() and success
    
    sys.exit(0 if success else 1)