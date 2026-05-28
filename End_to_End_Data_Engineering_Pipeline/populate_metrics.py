#!/usr/bin/env python3
"""
Script to populate metrics for demonstration and testing
"""
import sys
import os
import time
import random
from datetime import datetime

# Add churn path
sys.path.insert(0, '/usr/local/airflow/include/churn')

from monitoring.metrics_collector import get_metrics

def populate_sample_metrics():
    """Populate comprehensive sample metrics for dashboard testing"""
    print("Getting metrics instance...")
    metrics = get_metrics()
    
    print("Populating DAG and Task metrics...")
    # DAG metrics
    for i in range(10):
        metrics.record_dag_run('real_time_churn_streaming', 'success', random.uniform(60, 300))
        metrics.record_dag_run('batch_training_pipeline', 'success', random.uniform(1200, 3600))
    
    for i in range(3):
        metrics.record_dag_run('real_time_churn_streaming', 'failed', random.uniform(30, 120))
    
    # Task metrics
    tasks = ['start_stream_processor', 'generate_events', 'process_features', 'train_model', 'generate_predictions']
    for task in tasks:
        for i in range(20):
            metrics.record_task_run(task, 'success')
        for i in range(2):
            metrics.record_task_run(task, 'failed')
    
    print("Populating Data Processing metrics...")
    # Data processing metrics
    for i in range(100):
        metrics.record_data_processing('ingestion', 'kafka', random.randint(100, 1000))
        metrics.record_data_processing('preprocessing', 'batch', random.randint(50, 500))
        metrics.record_data_processing('feature_engineering', 'streaming', random.randint(200, 800))
    
    print("Populating Model Performance metrics...")
    # Model performance
    model_types = ['random_forest', 'gradient_boosting', 'logistic_regression']
    for model in model_types:
        metrics.record_model_training(model, random.uniform(120, 1800), random.uniform(0.75, 0.95))
        
        # Generate predictions
        for risk in ['high', 'medium', 'low']:
            count = random.randint(20, 200)
            metrics.record_prediction(model, risk, count)
    
    print("Populating Streaming metrics...")
    # Kafka streaming metrics
    topics = ['customer-events', 'churn-predictions', 'feature-vectors', 'alerts']
    for topic in topics:
        for i in range(50):
            metrics.record_kafka_message(topic, produced=True)
        for i in range(45):
            metrics.record_kafka_message(topic, produced=False, consumer_group='churn-pipeline')
    
    print("Populating Infrastructure metrics...")
    # S3 metrics
    buckets = ['raw-data', 'models', 'results']
    file_types = ['csv', 'pkl', 'json']
    for bucket in buckets:
        for file_type in file_types:
            for i in range(20):
                metrics.record_s3_upload(
                    bucket=bucket, 
                    file_type=file_type,
                    duration_seconds=random.uniform(1, 30),
                    file_size_bytes=random.randint(1024, 10485760)
                )
            # Some errors
            for i in range(2):
                metrics.record_s3_upload(
                    bucket=bucket,
                    file_type=file_type, 
                    duration_seconds=0,
                    file_size_bytes=0,
                    error_type='AccessDenied'
                )
    
    print("Setting Business metrics...")
    # Business metrics - Set gauges directly
    metrics.customers_at_risk.labels(risk_level='high').set(78)
    metrics.customers_at_risk.labels(risk_level='medium').set(245)
    metrics.customers_at_risk.labels(risk_level='low').set(1432)
    
    metrics.revenue_at_risk.labels(risk_level='high').set(156000)
    metrics.revenue_at_risk.labels(risk_level='medium').set(98000)
    metrics.revenue_at_risk.labels(risk_level='low').set(45000)
    
    # Model accuracy gauges
    metrics.model_accuracy.labels(model_type='random_forest').set(0.89)
    metrics.model_accuracy.labels(model_type='gradient_boosting').set(0.91)
    metrics.model_accuracy.labels(model_type='logistic_regression').set(0.84)
    
    # Data quality metrics
    datasets = ['customer_data', 'transaction_data', 'interaction_data']
    checks = ['completeness', 'accuracy', 'consistency']
    for dataset in datasets:
        for check in checks:
            metrics.data_quality_score.labels(dataset=dataset, check_type=check).set(random.uniform(0.8, 1.0))
    
    # Data freshness
    sources = ['raw', 'processed', 'predictions']
    for source in sources:
        metrics.data_freshness_hours.labels(source=source).set(random.uniform(0.5, 24))
    
    print("All metrics populated successfully!")
    print("Metrics should now be visible in Prometheus and Grafana")
    
    # Show current metric values
    print("\nCurrent metrics snapshot:")
    print(f"High risk customers: {metrics.customers_at_risk.labels(risk_level='high')._value._value}")
    print(f"Medium risk customers: {metrics.customers_at_risk.labels(risk_level='medium')._value._value}")
    print(f"Low risk customers: {metrics.customers_at_risk.labels(risk_level='low')._value._value}")
    print(f"Random Forest accuracy: {metrics.model_accuracy.labels(model_type='random_forest')._value._value}")

if __name__ == "__main__":
    populate_sample_metrics()