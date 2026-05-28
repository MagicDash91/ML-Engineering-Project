"""
Prometheus Metrics Collector for Churn Prediction Pipeline
Collects and exposes custom metrics for monitoring
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
import os
import pickle
import threading
from dataclasses import dataclass

from prometheus_client import (
    Counter, Gauge, Histogram, Summary, 
    CollectorRegistry, start_http_server, generate_latest
)
import redis
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetricsConfig:
    """Configuration for metrics collector"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    metrics_port: int = 8000
    collection_interval: int = 30
    s3_region: str = "ap-southeast-1"

class ChurnPipelineMetrics:
    """Custom metrics for churn prediction pipeline"""
    
    def __init__(self, config: MetricsConfig = None):
        self.config = config or MetricsConfig()
        
        # Create custom registry
        self.registry = CollectorRegistry()
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=True
            )
            self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Metrics will use in-memory storage.")
            self.redis_client = None
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client('s3', region_name=self.config.s3_region)
        except Exception as e:
            logger.warning(f"S3 client initialization failed: {e}")
            self.s3_client = None
        
        # Define metrics
        self._define_metrics()
        
        # Start background collection
        self._start_background_collection()
    
    def _define_metrics(self):
        """Define all custom metrics"""
        
        # === DATA PIPELINE METRICS ===
        
        self.dag_runs_total = Counter(
            'churn_dag_runs_total',
            'Total number of DAG runs',
            ['dag_id', 'status'],
            registry=self.registry
        )
        
        self.dag_run_duration = Histogram(
            'churn_dag_run_duration_seconds',
            'DAG run duration in seconds',
            ['dag_id'],
            buckets=[60, 300, 600, 1800, 3600, 7200],
            registry=self.registry
        )
        
        self.task_runs_total = Counter(
            'churn_task_runs_total',
            'Total number of task runs',
            ['task_id', 'status'],
            registry=self.registry
        )
        
        self.data_records_processed = Counter(
            'churn_data_records_processed_total',
            'Total number of data records processed',
            ['stage', 'source'],
            registry=self.registry
        )
        
        # === MODEL PERFORMANCE METRICS ===
        
        self.model_accuracy = Gauge(
            'churn_model_accuracy',
            'Current model accuracy',
            ['model_type'],
            registry=self.registry
        )
        
        self.model_training_duration = Histogram(
            'churn_model_training_duration_seconds',
            'Model training duration in seconds',
            ['model_type'],
            buckets=[60, 300, 600, 1800, 3600],
            registry=self.registry
        )
        
        self.predictions_generated = Counter(
            'churn_predictions_generated_total',
            'Total predictions generated',
            ['model_type', 'risk_level'],
            registry=self.registry
        )
        
        self.model_feature_importance = Gauge(
            'churn_model_feature_importance',
            'Feature importance scores',
            ['feature_name', 'model_type'],
            registry=self.registry
        )
        
        # === DATA QUALITY METRICS ===
        
        self.data_quality_score = Gauge(
            'churn_data_quality_score',
            'Data quality score (0-1)',
            ['dataset', 'check_type'],
            registry=self.registry
        )
        
        self.data_freshness_hours = Gauge(
            'churn_data_freshness_hours',
            'Data freshness in hours',
            ['source'],
            registry=self.registry
        )
        
        self.data_validation_failures = Counter(
            'churn_data_validation_failures_total',
            'Data validation failures',
            ['validation_type', 'dataset'],
            registry=self.registry
        )
        
        # === STREAMING METRICS ===
        
        self.kafka_messages_produced = Counter(
            'churn_kafka_messages_produced_total',
            'Total Kafka messages produced',
            ['topic'],
            registry=self.registry
        )
        
        self.kafka_messages_consumed = Counter(
            'churn_kafka_messages_consumed_total',
            'Total Kafka messages consumed',
            ['topic', 'consumer_group'],
            registry=self.registry
        )
        
        self.stream_processing_latency = Histogram(
            'churn_stream_processing_latency_seconds',
            'Stream processing latency',
            ['processor_type'],
            buckets=[0.1, 0.5, 1, 2, 5, 10],
            registry=self.registry
        )
        
        # === INFRASTRUCTURE METRICS ===
        
        self.s3_upload_duration = Histogram(
            'churn_s3_upload_duration_seconds',
            'S3 upload duration',
            ['bucket', 'file_type'],
            buckets=[1, 5, 10, 30, 60, 300],
            registry=self.registry
        )
        
        self.s3_upload_size_bytes = Histogram(
            'churn_s3_upload_size_bytes',
            'S3 upload file size',
            ['bucket', 'file_type'],
            buckets=[1024, 10240, 102400, 1048576, 10485760, 104857600],
            registry=self.registry
        )
        
        self.s3_upload_errors = Counter(
            'churn_s3_upload_errors_total',
            'S3 upload errors',
            ['bucket', 'error_type'],
            registry=self.registry
        )
        
        # === BUSINESS METRICS ===
        
        self.customers_at_risk = Gauge(
            'churn_customers_at_risk',
            'Number of customers at churn risk',
            ['risk_level'],
            registry=self.registry
        )
        
        self.revenue_at_risk = Gauge(
            'churn_revenue_at_risk_dollars',
            'Revenue at risk from potential churn',
            ['risk_level'],
            registry=self.registry
        )
        
        self.intervention_actions = Counter(
            'churn_intervention_actions_total',
            'Intervention actions triggered',
            ['action_type', 'risk_level'],
            registry=self.registry
        )
    
    def _start_background_collection(self):
        """Start background metrics collection"""
        def collect_metrics():
            while True:
                try:
                    self._collect_pipeline_metrics()
                    self._collect_s3_metrics()
                    self._collect_model_metrics()
                    time.sleep(self.config.collection_interval)
                except Exception as e:
                    logger.error(f"Error collecting metrics: {e}")
                    time.sleep(self.config.collection_interval)
        
        thread = threading.Thread(target=collect_metrics, daemon=True)
        thread.start()
        logger.info("Background metrics collection started")
    
    def _collect_pipeline_metrics(self):
        """Collect pipeline-related metrics"""
        try:
            # Check for recent model training results
            models_dir = "/home/magicdash/astro-airflow/churn/models"
            if os.path.exists(f"{models_dir}/model_metadata.pkl"):
                with open(f"{models_dir}/model_metadata.pkl", 'rb') as f:
                    metadata = pickle.load(f)
                
                # Update model accuracy
                self.model_accuracy.labels(
                    model_type=metadata.get('model_type', 'unknown')
                ).set(metadata.get('test_accuracy', 0))
                
                # Update feature importance
                feature_importance = metadata.get('feature_importance', {})
                for feature, importance in feature_importance.items():
                    self.model_feature_importance.labels(
                        feature_name=feature,
                        model_type=metadata.get('model_type', 'unknown')
                    ).set(importance)
            
            # Check data freshness
            data_dir = "/home/magicdash/astro-airflow/churn/data"
            for subdir in ['raw', 'processed', 'predictions']:
                dir_path = f"{data_dir}/{subdir}"
                if os.path.exists(dir_path):
                    files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
                    if files:
                        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(dir_path, x)))
                        file_path = os.path.join(dir_path, latest_file)
                        file_age_hours = (time.time() - os.path.getctime(file_path)) / 3600
                        
                        self.data_freshness_hours.labels(source=subdir).set(file_age_hours)
        
        except Exception as e:
            logger.error(f"Error collecting pipeline metrics: {e}")
    
    def _collect_s3_metrics(self):
        """Collect S3-related metrics"""
        if not self.s3_client:
            return
        
        try:
            buckets = [
                'magicdash-data-pipeline-churn-results',
                'magicdash-data-pipeline-ml-models',
                'magicdash-data-pipeline-raw-data'
            ]
            
            for bucket in buckets:
                try:
                    # Get bucket size (approximate)
                    response = self.s3_client.list_objects_v2(Bucket=bucket, MaxKeys=1000)
                    
                    if 'Contents' in response:
                        total_size = sum(obj['Size'] for obj in response['Contents'])
                        
                        # Update S3 metrics based on recent uploads
                        # This is a simplified example - in production, you'd track actual upload events
                        
                except ClientError as e:
                    if e.response['Error']['Code'] != 'NoSuchBucket':
                        logger.warning(f"Error accessing bucket {bucket}: {e}")
                        self.s3_upload_errors.labels(
                            bucket=bucket.split('-')[-1],
                            error_type=e.response['Error']['Code']
                        ).inc()
        
        except Exception as e:
            logger.error(f"Error collecting S3 metrics: {e}")
    
    def _collect_model_metrics(self):
        """Collect ML model-related metrics"""
        try:
            # Check for recent predictions
            predictions_dir = "/home/magicdash/astro-airflow/churn/data/predictions"
            if os.path.exists(predictions_dir):
                prediction_files = [f for f in os.listdir(predictions_dir) if f.endswith('.csv')]
                
                if prediction_files:
                    # Read latest prediction file to get risk distribution
                    latest_file = max(prediction_files, key=lambda x: os.path.getctime(os.path.join(predictions_dir, x)))
                    
                    # In a real implementation, you'd parse the CSV and count risk levels
                    # For now, set sample values
                    self.customers_at_risk.labels(risk_level='high').set(45)
                    self.customers_at_risk.labels(risk_level='medium').set(123)
                    self.customers_at_risk.labels(risk_level='low').set(332)
        
        except Exception as e:
            logger.error(f"Error collecting model metrics: {e}")
    
    # === PUBLIC METHODS FOR RECORDING METRICS ===
    
    def record_dag_run(self, dag_id: str, status: str, duration_seconds: float = None):
        """Record DAG run metrics"""
        self.dag_runs_total.labels(dag_id=dag_id, status=status).inc()
        
        if duration_seconds:
            self.dag_run_duration.labels(dag_id=dag_id).observe(duration_seconds)
    
    def record_task_run(self, task_id: str, status: str):
        """Record task run metrics"""
        self.task_runs_total.labels(task_id=task_id, status=status).inc()
    
    def record_data_processing(self, stage: str, source: str, record_count: int):
        """Record data processing metrics"""
        self.data_records_processed.labels(stage=stage, source=source).inc(record_count)
    
    def record_model_training(self, model_type: str, duration_seconds: float, accuracy: float):
        """Record model training metrics"""
        self.model_training_duration.labels(model_type=model_type).observe(duration_seconds)
        self.model_accuracy.labels(model_type=model_type).set(accuracy)
    
    def record_prediction(self, model_type: str, risk_level: str, count: int = 1):
        """Record prediction generation"""
        self.predictions_generated.labels(model_type=model_type, risk_level=risk_level).inc(count)
    
    def record_kafka_message(self, topic: str, produced: bool = True, consumer_group: str = None):
        """Record Kafka message metrics"""
        if produced:
            self.kafka_messages_produced.labels(topic=topic).inc()
        else:
            self.kafka_messages_consumed.labels(topic=topic, consumer_group=consumer_group or 'default').inc()
    
    def record_s3_upload(self, bucket: str, file_type: str, duration_seconds: float, 
                        file_size_bytes: int, error_type: str = None):
        """Record S3 upload metrics"""
        if error_type:
            self.s3_upload_errors.labels(bucket=bucket, error_type=error_type).inc()
        else:
            self.s3_upload_duration.labels(bucket=bucket, file_type=file_type).observe(duration_seconds)
            self.s3_upload_size_bytes.labels(bucket=bucket, file_type=file_type).observe(file_size_bytes)
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')
    
    def start_http_server(self):
        """Start HTTP server for metrics endpoint"""
        start_http_server(self.config.metrics_port, registry=self.registry)
        logger.info(f"Metrics server started on port {self.config.metrics_port}")

# Global metrics instance
_metrics_instance: Optional[ChurnPipelineMetrics] = None

def get_metrics() -> ChurnPipelineMetrics:
    """Get global metrics instance"""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = ChurnPipelineMetrics()
    return _metrics_instance

def main():
    """Main function for standalone metrics server"""
    metrics = ChurnPipelineMetrics()
    metrics.start_http_server()
    
    logger.info("Metrics collector started. Access metrics at http://localhost:8000/metrics")
    
    # Keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Metrics collector stopped")

if __name__ == "__main__":
    main()