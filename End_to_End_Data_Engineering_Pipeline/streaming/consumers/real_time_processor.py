"""
Real-time Stream Processor for Customer Events
Processes streaming customer events and generates real-time churn predictions
"""

import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading
from collections import defaultdict, deque

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import redis

from streaming.schemas.customer_event import (
    CustomerEvent, ChurnPrediction, FeatureVector, RiskLevel,
    KAFKA_TOPICS, KAFKA_CONFIG
)
from monitoring.metrics_collector import get_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeFeatureEngine:
    """Real-time feature engineering for streaming events"""
    
    def __init__(self, redis_client: redis.Redis = None):
        self.redis_client = redis_client
        self.feature_windows = {
            '1h': timedelta(hours=1),
            '24h': timedelta(hours=24),
            '7d': timedelta(days=7)
        }
        
        # In-memory fallback if Redis is unavailable
        self.memory_store = defaultdict(lambda: deque(maxlen=1000))
    
    def _store_event(self, customer_id: str, event: CustomerEvent):
        """Store event for feature calculation"""
        event_data = {
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type.value,
            'event_data': event.event_data
        }
        
        if self.redis_client:
            try:
                # Store in Redis with TTL
                key = f"customer_events:{customer_id}"
                self.redis_client.lpush(key, json.dumps(event_data))
                self.redis_client.expire(key, 604800)  # 7 days
            except Exception as e:
                logger.warning(f"Redis storage failed: {e}")
                self._store_in_memory(customer_id, event_data)
        else:
            self._store_in_memory(customer_id, event_data)
    
    def _store_in_memory(self, customer_id: str, event_data: Dict[str, Any]):
        """Fallback in-memory storage"""
        self.memory_store[customer_id].append(event_data)
    
    def _get_customer_events(self, customer_id: str, window: timedelta) -> List[Dict[str, Any]]:
        """Get customer events within time window"""
        cutoff_time = datetime.now() - window
        
        if self.redis_client:
            try:
                key = f"customer_events:{customer_id}"
                events_raw = self.redis_client.lrange(key, 0, -1)
                events = [json.loads(event) for event in events_raw]
            except Exception as e:
                logger.warning(f"Redis retrieval failed: {e}")
                events = list(self.memory_store[customer_id])
        else:
            events = list(self.memory_store[customer_id])
        
        # Filter by time window
        filtered_events = []
        for event in events:
            event_time = datetime.fromisoformat(event['timestamp'])
            if event_time >= cutoff_time:
                filtered_events.append(event)
        
        return filtered_events
    
    def extract_features(self, customer_id: str, current_event: CustomerEvent) -> Dict[str, float]:
        """Extract real-time features for a customer"""
        
        # Store current event
        self._store_event(customer_id, current_event)
        
        features = {}
        
        # Extract features for different time windows
        for window_name, window_duration in self.feature_windows.items():
            events = self._get_customer_events(customer_id, window_duration)
            
            if not events:
                # Set default values if no events
                features.update(self._get_default_features(window_name))
                continue
            
            # Count events by type
            event_type_counts = defaultdict(int)
            total_events = len(events)
            
            # Support call metrics
            support_calls = 0
            total_call_duration = 0
            billing_issues = 0
            complaints = 0
            
            # Usage metrics
            usage_spikes = 0
            usage_drops = 0
            
            # Financial metrics
            total_purchases = 0
            purchase_amount = 0.0
            payment_failures = 0
            
            for event in events:
                event_type = event['event_type']
                event_type_counts[event_type] += 1
                event_data = event.get('event_data', {})
                
                if event_type == 'support_call':
                    support_calls += 1
                    total_call_duration += event_data.get('call_duration_minutes', 0)
                elif event_type == 'billing_issue':
                    billing_issues += 1
                elif event_type == 'complaint':
                    complaints += 1
                elif event_type == 'usage_spike':
                    usage_spikes += 1
                elif event_type == 'usage_drop':
                    usage_drops += 1
                elif event_type == 'purchase':
                    total_purchases += 1
                    purchase_amount += event_data.get('purchase_amount', 0)
                elif event_type == 'payment_failed':
                    payment_failures += 1
            
            # Calculate features for this time window
            prefix = f"{window_name}_"
            
            # Event frequency features
            features[f"{prefix}total_events"] = total_events
            features[f"{prefix}support_calls"] = support_calls
            features[f"{prefix}billing_issues"] = billing_issues
            features[f"{prefix}complaints"] = complaints
            features[f"{prefix}payment_failures"] = payment_failures
            
            # Usage behavior features
            features[f"{prefix}usage_spikes"] = usage_spikes
            features[f"{prefix}usage_drops"] = usage_drops
            features[f"{prefix}usage_volatility"] = usage_spikes + usage_drops
            
            # Financial features
            features[f"{prefix}total_purchases"] = total_purchases
            features[f"{prefix}purchase_amount"] = purchase_amount
            features[f"{prefix}avg_purchase_amount"] = purchase_amount / max(total_purchases, 1)
            
            # Engagement features
            logins = event_type_counts.get('login', 0)
            logouts = event_type_counts.get('logout', 0)
            features[f"{prefix}login_frequency"] = logins
            features[f"{prefix}session_ratio"] = logins / max(logouts, 1)
            
            # Risk indicators
            negative_events = support_calls + billing_issues + complaints + payment_failures
            features[f"{prefix}negative_event_ratio"] = negative_events / max(total_events, 1)
            
            # Support metrics
            if support_calls > 0:
                features[f"{prefix}avg_call_duration"] = total_call_duration / support_calls
            else:
                features[f"{prefix}avg_call_duration"] = 0
        
        # Add current event context features
        features['current_event_type_support_call'] = 1 if current_event.event_type.value == 'support_call' else 0
        features['current_event_type_billing_issue'] = 1 if current_event.event_type.value == 'billing_issue' else 0
        features['current_event_type_complaint'] = 1 if current_event.event_type.value == 'complaint' else 0
        features['current_event_type_payment_failed'] = 1 if current_event.event_type.value == 'payment_failed' else 0
        
        # Add customer profile features from event data
        event_data = current_event.event_data
        features['tenure_months'] = event_data.get('tenure_months', 0)
        features['monthly_charges'] = event_data.get('monthly_charges', 0)
        
        # Contract type encoding (simplified)
        contract_type = event_data.get('contract_type', 'Month-to-month')
        features['contract_month_to_month'] = 1 if contract_type == 'Month-to-month' else 0
        features['contract_one_year'] = 1 if contract_type == 'One year' else 0
        features['contract_two_year'] = 1 if contract_type == 'Two year' else 0
        
        return features
    
    def _get_default_features(self, window_name: str) -> Dict[str, float]:
        """Get default feature values when no events exist"""
        prefix = f"{window_name}_"
        return {
            f"{prefix}total_events": 0,
            f"{prefix}support_calls": 0,
            f"{prefix}billing_issues": 0,
            f"{prefix}complaints": 0,
            f"{prefix}payment_failures": 0,
            f"{prefix}usage_spikes": 0,
            f"{prefix}usage_drops": 0,
            f"{prefix}usage_volatility": 0,
            f"{prefix}total_purchases": 0,
            f"{prefix}purchase_amount": 0,
            f"{prefix}avg_purchase_amount": 0,
            f"{prefix}login_frequency": 0,
            f"{prefix}session_ratio": 0,
            f"{prefix}negative_event_ratio": 0,
            f"{prefix}avg_call_duration": 0
        }

class RealTimeChurnPredictor:
    """Real-time churn prediction using streaming data"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or "/home/magicdash/astro-airflow/churn/models/churn_model.pkl"
        self.metadata_path = model_path or "/home/magicdash/astro-airflow/churn/models/model_metadata.pkl"
        
        self.model = None
        self.feature_columns = None
        self.scaler = None
        
        self._load_model()
        
        # Connect to Redis for caching
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            logger.info("Connected to Redis for feature caching")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        # Initialize feature engine
        self.feature_engine = RealTimeFeatureEngine(self.redis_client)
    
    def _load_model(self):
        """Load trained model and metadata"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.feature_columns = metadata.get('feature_columns', [])
            
            logger.info(f"Loaded model with {len(self.feature_columns)} features")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_churn(self, customer_event: CustomerEvent) -> ChurnPrediction:
        """Generate real-time churn prediction"""
        start_time = time.time()
        
        try:
            # Extract features from streaming data
            features = self.feature_engine.extract_features(
                customer_event.customer_id, 
                customer_event
            )
            
            # Prepare feature vector for model
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(features.get(col, 0.0))
            
            # Convert to numpy array and reshape for prediction
            X = np.array(feature_vector).reshape(1, -1)
            
            # Generate prediction
            churn_probability = self.model.predict_proba(X)[0][1]  # Probability of churn
            churn_prediction = self.model.predict(X)[0]
            
            # Determine risk level
            if churn_probability > 0.8:
                risk_level = RiskLevel.CRITICAL
                recommended_action = "IMMEDIATE_INTERVENTION"
            elif churn_probability > 0.6:
                risk_level = RiskLevel.HIGH
                recommended_action = "RETENTION_CAMPAIGN"
            elif churn_probability > 0.4:
                risk_level = RiskLevel.MEDIUM
                recommended_action = "PROACTIVE_ENGAGEMENT"
            else:
                risk_level = RiskLevel.LOW
                recommended_action = "CONTINUE_MONITORING"
            
            # Calculate confidence score
            confidence_score = max(churn_probability, 1 - churn_probability)
            
            # Create prediction result
            prediction_result = ChurnPrediction(
                customer_id=customer_event.customer_id,
                churn_probability=float(churn_probability),
                risk_level=risk_level,
                prediction_timestamp=datetime.now(),
                model_version="v1.0",
                features_used=features,
                confidence_score=float(confidence_score),
                next_action_recommended=recommended_action
            )
            
            # Record metrics
            processing_time = time.time() - start_time
            metrics = get_metrics()
            metrics.stream_processing_latency.labels(processor_type='churn_prediction').observe(processing_time)
            metrics.record_prediction('streaming', risk_level.value)
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Prediction failed for customer {customer_event.customer_id}: {e}")
            raise

class RealTimeStreamProcessor:
    """Main stream processor that orchestrates real-time churn prediction"""
    
    def __init__(self, consumer_config: Dict[str, Any] = None, producer_config: Dict[str, Any] = None):
        self.consumer_config = consumer_config or KAFKA_CONFIG.copy()
        self.producer_config = producer_config or KAFKA_CONFIG.copy()
        
        # Remove consumer-specific configs from producer
        producer_config_clean = {k: v for k, v in self.producer_config.items() 
                               if k not in ['group_id', 'auto_offset_reset', 'enable_auto_commit']}
        
        # Initialize Kafka consumer and producer
        self.consumer = KafkaConsumer(
            KAFKA_TOPICS['customer_events'],
            **self.consumer_config
        )
        
        self.producer = KafkaProducer(**producer_config_clean)
        
        # Initialize predictor
        self.predictor = RealTimeChurnPredictor()
        
        # Metrics
        self.metrics = get_metrics()
        
        # Processing statistics
        self.stats = {
            'messages_processed': 0,
            'predictions_generated': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
        # Control flags
        self.running = False
        
    def process_message(self, message) -> bool:
        """Process a single Kafka message"""
        try:
            # Parse customer event
            event_data = message.value
            customer_event = CustomerEvent.from_dict(event_data)
            
            # Record Kafka metrics
            self.metrics.record_kafka_message(
                topic=KAFKA_TOPICS['customer_events'],
                produced=False,
                consumer_group=self.consumer_config.get('group_id')
            )
            
            # Generate prediction
            prediction = self.predictor.predict_churn(customer_event)
            
            # Send prediction to output topic
            self.producer.send(
                KAFKA_TOPICS['churn_predictions'],
                value=prediction.to_dict(),
                key=customer_event.customer_id.encode('utf-8')
            )
            
            # Record metrics
            self.metrics.record_kafka_message(
                topic=KAFKA_TOPICS['churn_predictions'],
                produced=True
            )
            
            # Log high-risk customers
            if prediction.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                logger.warning(
                    f"HIGH RISK CUSTOMER: {customer_event.customer_id} "
                    f"({prediction.churn_probability:.3f} churn probability) "
                    f"- Action: {prediction.next_action_recommended}"
                )
                
                # Send alert
                alert_message = {
                    'customer_id': customer_event.customer_id,
                    'risk_level': prediction.risk_level.value,
                    'churn_probability': prediction.churn_probability,
                    'recommended_action': prediction.next_action_recommended,
                    'trigger_event': customer_event.event_type.value,
                    'alert_timestamp': datetime.now().isoformat()
                }
                
                self.producer.send(
                    KAFKA_TOPICS['alerts'],
                    value=alert_message,
                    key=customer_event.customer_id.encode('utf-8')
                )
            
            # Update statistics
            self.stats['messages_processed'] += 1
            self.stats['predictions_generated'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.stats['errors'] += 1
            return False
    
    def run(self):
        """Start the real-time stream processor"""
        self.running = True
        logger.info("Starting real-time stream processor...")
        
        # Start statistics logging thread
        stats_thread = threading.Thread(target=self._log_statistics, daemon=True)
        stats_thread.start()
        
        try:
            for message in self.consumer:
                if not self.running:
                    break
                
                self.process_message(message)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Stream processor error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the stream processor"""
        self.running = False
        
        if self.producer:
            self.producer.flush()
            self.producer.close()
        
        if self.consumer:
            self.consumer.close()
        
        logger.info("Stream processor stopped")
        self._log_final_statistics()
    
    def _log_statistics(self):
        """Log processing statistics periodically"""
        while self.running:
            time.sleep(60)  # Log every minute
            
            runtime = datetime.now() - self.stats['start_time']
            messages_per_minute = self.stats['messages_processed'] / max(runtime.total_seconds() / 60, 1)
            
            logger.info(
                f"Stream Processor Stats: "
                f"Messages: {self.stats['messages_processed']}, "
                f"Predictions: {self.stats['predictions_generated']}, "
                f"Errors: {self.stats['errors']}, "
                f"Rate: {messages_per_minute:.1f} msg/min, "
                f"Runtime: {runtime}"
            )
    
    def _log_final_statistics(self):
        """Log final statistics when stopping"""
        runtime = datetime.now() - self.stats['start_time']
        
        logger.info(
            f"Final Stream Processor Stats: "
            f"Total Messages: {self.stats['messages_processed']}, "
            f"Total Predictions: {self.stats['predictions_generated']}, "
            f"Total Errors: {self.stats['errors']}, "
            f"Success Rate: {(self.stats['messages_processed'] - self.stats['errors']) / max(self.stats['messages_processed'], 1):.2%}, "
            f"Total Runtime: {runtime}"
        )

def main():
    """Main function for standalone stream processor"""
    processor = RealTimeStreamProcessor()
    
    try:
        processor.run()
    except KeyboardInterrupt:
        logger.info("Stream processor interrupted by user")
    finally:
        processor.stop()

if __name__ == "__main__":
    main()