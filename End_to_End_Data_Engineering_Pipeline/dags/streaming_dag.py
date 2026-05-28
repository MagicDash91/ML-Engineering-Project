"""
Airflow DAG for Real-time Streaming Infrastructure
Manages Kafka producers, consumers, and streaming analytics
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
from airflow.sensors.filesystem import FileSensor
import logging
import subprocess
import signal
import time

# Import monitoring with fallback
try:
    from monitoring.metrics_collector import get_metrics
except ImportError:
    # Fallback: create a dummy metrics function
    def get_metrics():
        class DummyMetrics:
            def record_task_run(self, task_id, status): pass
        return DummyMetrics()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    'real_time_churn_streaming',
    default_args=default_args,
    description='Real-time Customer Churn Streaming Pipeline',
    schedule='@once',  # Run once to start streaming services
    catchup=False,
    max_active_runs=1,
    tags=['streaming', 'kafka', 'real-time', 'churn']
)

def start_metrics_server(**context):
    """Start the Prometheus metrics server"""
    try:
        from monitoring.metrics_collector import ChurnPipelineMetrics
        
        metrics = ChurnPipelineMetrics()
        metrics.start_http_server()
        
        logger.info("Metrics server started on port 8000")
        
        # Keep the server running by storing the process info
        context['task_instance'].xcom_push(key='metrics_server_started', value=True)
        
        return "Metrics server started successfully"
        
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        raise

def start_stream_processor(**context):
    """Start the real-time stream processor"""
    try:
        # Import and start the stream processor
        try:
            from streaming.consumers.real_time_processor import RealTimeStreamProcessor
        except ImportError:
            logging.warning("RealTimeStreamProcessor not available - streaming functionality limited")
            return "Stream processor not available - module not found"
        
        processor = RealTimeStreamProcessor()
        
        # Start processing in background thread
        import threading
        
        def run_processor():
            try:
                processor.run()
            except Exception as e:
                logger.error(f"Stream processor error: {e}")
        
        processor_thread = threading.Thread(target=run_processor, daemon=True)
        processor_thread.start()
        
        logger.info("Real-time stream processor started")
        
        # Store processor reference for monitoring
        context['task_instance'].xcom_push(key='stream_processor_started', value=True)
        
        return "Stream processor started successfully"
        
    except Exception as e:
        logger.error(f"Failed to start stream processor: {e}")
        raise

def generate_sample_events(**context):
    """Generate sample customer events for testing"""
    try:
        try:
            from streaming.producers.customer_event_producer import CustomerEventProducer
        except ImportError:
            logging.warning("CustomerEventProducer not available - event generation skipped")
            return "Event generation not available - module not found"
        
        producer = CustomerEventProducer()
        
        # Generate continuous stream of events
        logger.info("Starting sample event generation")
        producer.generate_and_send_events(num_events=100, delay_seconds=1.0)
        
        # Generate specific customer journeys
        for i in range(5):
            customer_id = f"CUST_{i+1:06d}"
            producer.simulate_customer_journey(customer_id, events_count=20)
        
        producer.close()
        
        logger.info("Sample event generation completed")
        return "Generated 100 sample events + 5 customer journeys"
        
    except Exception as e:
        logger.error(f"Failed to generate sample events: {e}")
        raise

def monitor_streaming_health(**context):
    """Monitor the health of streaming components"""
    try:
        import requests
        import json
        from kafka import KafkaConsumer
        
        # Try to import streaming module with fallback
        try:
            from streaming.schemas.customer_event import KAFKA_TOPICS, KAFKA_CONFIG
        except ImportError:
            logging.warning("Streaming module not found for health monitoring, using fallback")
            KAFKA_TOPICS = {
                'customer_events': 'customer-events',
                'churn_predictions': 'churn-predictions', 
                'feature_vectors': 'feature-vectors',
                'alerts': 'churn-alerts'
            }
            KAFKA_CONFIG = {
                'bootstrap_servers': ['localhost:9092'],
                'auto_offset_reset': 'latest'
            }
        
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'metrics_server': False,
            'kafka_connectivity': False,
            'topics_available': [],
            'consumer_lag': {},
            'message_throughput': {}
        }
        
        # Check metrics server
        try:
            response = requests.get('http://localhost:8000/metrics', timeout=5)
            if response.status_code == 200:
                health_report['metrics_server'] = True
                logger.info("Metrics server is healthy")
        except Exception as e:
            logger.warning(f"Metrics server check failed: {e}")
        
        # Check Kafka connectivity
        try:
            consumer = KafkaConsumer(
                bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
                auto_offset_reset='latest',
                consumer_timeout_ms=5000
            )
            
            # Get topic metadata
            metadata = consumer.list_consumer_group_offsets()
            available_topics = consumer.topics()
            
            health_report['kafka_connectivity'] = True
            health_report['topics_available'] = list(available_topics)
            
            logger.info(f"Kafka is healthy. Available topics: {available_topics}")
            
            consumer.close()
            
        except Exception as e:
            logger.warning(f"Kafka connectivity check failed: {e}")
        
        # Record health metrics
        metrics = get_metrics()
        metrics.record_task_run('monitor_streaming_health', 'success')
        
        # Store health report in XCom
        context['task_instance'].xcom_push(key='health_report', value=health_report)
        
        return f"Health check completed: {health_report}"
        
    except Exception as e:
        logger.error(f"Health monitoring failed: {e}")
        raise

def create_kafka_topics(**context):
    """Create required Kafka topics if they don't exist"""
    try:
        # First check if Kafka is available
        import socket
        import time
        
        def check_kafka_availability():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex(('localhost', 9092))
                sock.close()
                return result == 0
            except:
                return False
        
        if not check_kafka_availability():
            logging.warning("Kafka is not available at localhost:9092")
            return "Kafka topics creation skipped - Kafka not available"
        
        from kafka.admin import KafkaAdminClient, NewTopic
        from kafka.errors import TopicAlreadyExistsError
        
        # Try to import streaming module with fallback
        try:
            from streaming.schemas.customer_event import KAFKA_TOPICS, KAFKA_CONFIG
        except ImportError:
            # Fallback configuration if streaming module not found
            logging.warning("Streaming module not found, using fallback configuration")
            KAFKA_TOPICS = {
                'customer_events': 'customer-events',
                'churn_predictions': 'churn-predictions', 
                'feature_vectors': 'feature-vectors',
                'alerts': 'churn-alerts'
            }
            KAFKA_CONFIG = {
                'bootstrap_servers': ['localhost:9092'],
                'auto_offset_reset': 'latest',
                'enable_auto_commit': True,
                'group_id': 'churn-pipeline'
            }
        
        try:
            admin_client = KafkaAdminClient(
                bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
                client_id='airflow_topic_creator',
                request_timeout_ms=5000,
                connections_max_idle_ms=5000
            )
        except Exception as kafka_error:
            logging.warning(f"Kafka not available: {kafka_error}")
            return f"Kafka topics creation skipped - Kafka not available: {kafka_error}"
        
        # Define topics to create
        topics_to_create = [
            NewTopic(name=KAFKA_TOPICS['customer_events'], num_partitions=3, replication_factor=1),
            NewTopic(name=KAFKA_TOPICS['churn_predictions'], num_partitions=3, replication_factor=1),
            NewTopic(name=KAFKA_TOPICS['feature_vectors'], num_partitions=3, replication_factor=1),
            NewTopic(name=KAFKA_TOPICS['alerts'], num_partitions=1, replication_factor=1)
        ]
        
        created_topics = []
        
        for topic in topics_to_create:
            try:
                admin_client.create_topics(new_topics=[topic], validate_only=False)
                created_topics.append(topic.name)
                logger.info(f"Created Kafka topic: {topic.name}")
            except TopicAlreadyExistsError:
                logger.info(f"Topic {topic.name} already exists")
            except Exception as e:
                logger.warning(f"Failed to create topic {topic.name}: {e}")
        
        admin_client.close()
        
        return f"Topic creation completed. Created: {created_topics}"
        
    except Exception as e:
        logger.error(f"Failed to create Kafka topics: {e}")
        raise

def analyze_streaming_performance(**context):
    """Analyze streaming performance and generate reports"""
    try:
        import requests
        import re
        
        performance_report = {
            'timestamp': datetime.now().isoformat(),
            'message_throughput': 0,
            'processing_latency_p95': 0,
            'error_rate': 0,
            'active_consumers': 0,
            'recommendations': []
        }
        
        # Get metrics from Prometheus
        try:
            response = requests.get('http://localhost:8000/metrics', timeout=10)
            if response.status_code == 200:
                metrics_text = response.text
                
                # Parse key metrics
                throughput_match = re.search(r'churn_kafka_messages_consumed_total\s+(\d+(?:\.\d+)?)', metrics_text)
                if throughput_match:
                    performance_report['message_throughput'] = float(throughput_match.group(1))
                
                latency_match = re.search(r'churn_stream_processing_latency_seconds_bucket.*le="([0-9.]+)".*\s+(\d+)', metrics_text)
                if latency_match:
                    performance_report['processing_latency_p95'] = float(latency_match.group(1))
                
                # Generate recommendations based on metrics
                if performance_report['message_throughput'] > 1000:
                    performance_report['recommendations'].append("Consider scaling consumers")
                
                if performance_report['processing_latency_p95'] > 5:
                    performance_report['recommendations'].append("Optimize feature processing pipeline")
                
        except Exception as e:
            logger.warning(f"Failed to collect performance metrics: {e}")
        
        # Record performance metrics
        metrics = get_metrics()
        metrics.record_task_run('analyze_streaming_performance', 'success')
        
        logger.info(f"Performance analysis completed: {performance_report}")
        
        context['task_instance'].xcom_push(key='performance_report', value=performance_report)
        
        return f"Performance analysis: Throughput {performance_report['message_throughput']}, Latency P95: {performance_report['processing_latency_p95']}s"
        
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")
        raise

# Define tasks
create_topics_task = PythonOperator(
    task_id='create_kafka_topics',
    python_callable=create_kafka_topics,
    dag=dag
)

start_metrics_task = PythonOperator(
    task_id='start_metrics_server',
    python_callable=start_metrics_server,
    dag=dag
)

start_processor_task = PythonOperator(
    task_id='start_stream_processor',
    python_callable=start_stream_processor,
    dag=dag
)

generate_events_task = PythonOperator(
    task_id='generate_sample_events',
    python_callable=generate_sample_events,
    dag=dag
)

monitor_health_task = PythonOperator(
    task_id='monitor_streaming_health',
    python_callable=monitor_streaming_health,
    dag=dag
)

analyze_performance_task = PythonOperator(
    task_id='analyze_streaming_performance',
    python_callable=analyze_streaming_performance,
    dag=dag
)

# Set task dependencies
create_topics_task >> start_metrics_task >> start_processor_task >> generate_events_task >> monitor_health_task >> analyze_performance_task