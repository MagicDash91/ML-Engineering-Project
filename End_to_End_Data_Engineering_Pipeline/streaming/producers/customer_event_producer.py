"""
Kafka Producer for Real-time Customer Events
Simulates real-time customer behavior for churn prediction
"""

import json
import time
import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import uuid

from kafka import KafkaProducer
from kafka.errors import KafkaError

from streaming.schemas.customer_event import (
    CustomerEvent, EventType, KAFKA_TOPICS, KAFKA_CONFIG
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerEventProducer:
    """Kafka producer for customer events"""
    
    def __init__(self, bootstrap_servers: List[str] = None):
        self.bootstrap_servers = bootstrap_servers or KAFKA_CONFIG['bootstrap_servers']
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=KAFKA_CONFIG['value_serializer'],
            retry_backoff_ms=100,
            retries=3
        )
        self.topic = KAFKA_TOPICS['customer_events']
        
        # Customer profiles for simulation
        self.customer_profiles = self._generate_customer_profiles()
        
    def _generate_customer_profiles(self) -> List[Dict[str, Any]]:
        """Generate realistic customer profiles for simulation"""
        profiles = []
        
        # Generate 1000 simulated customers
        for i in range(1000):
            customer_id = f"CUST_{i+1:06d}"
            
            # Customer characteristics that influence churn
            profile = {
                'customer_id': customer_id,
                'tenure_months': random.randint(1, 72),
                'monthly_charges': round(random.uniform(20, 120), 2),
                'contract_type': random.choice(['Month-to-month', 'One year', 'Two year']),
                'payment_method': random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']),
                'internet_service': random.choice(['DSL', 'Fiber optic', 'No']),
                'phone_service': random.choice(['Yes', 'No']),
                'device_type': random.choice(['Mobile', 'Desktop', 'Tablet']),
                'location': random.choice(['Urban', 'Suburban', 'Rural']),
                'churn_risk_score': random.uniform(0, 1),  # Pre-calculated risk for simulation
                'last_activity': datetime.now() - timedelta(days=random.randint(0, 30))
            }
            profiles.append(profile)
            
        return profiles
    
    def _get_customer_profile(self, customer_id: str = None) -> Dict[str, Any]:
        """Get or generate customer profile"""
        if customer_id:
            # Find specific customer
            for profile in self.customer_profiles:
                if profile['customer_id'] == customer_id:
                    return profile
        
        # Return random customer
        return random.choice(self.customer_profiles)
    
    def _generate_realistic_event(self, customer_profile: Dict[str, Any]) -> CustomerEvent:
        """Generate realistic customer event based on profile"""
        
        # Event probabilities based on customer risk
        risk_score = customer_profile['churn_risk_score']
        
        if risk_score > 0.7:
            # High-risk customer - more negative events
            event_weights = {
                EventType.SUPPORT_CALL: 0.3,
                EventType.BILLING_ISSUE: 0.2,
                EventType.COMPLAINT: 0.2,
                EventType.PAYMENT_FAILED: 0.15,
                EventType.USAGE_DROP: 0.1,
                EventType.LOGIN: 0.05
            }
        elif risk_score > 0.4:
            # Medium-risk customer - mixed events
            event_weights = {
                EventType.LOGIN: 0.3,
                EventType.SUPPORT_CALL: 0.2,
                EventType.PURCHASE: 0.15,
                EventType.USAGE_SPIKE: 0.1,
                EventType.BILLING_ISSUE: 0.1,
                EventType.LOGOUT: 0.15
            }
        else:
            # Low-risk customer - positive events
            event_weights = {
                EventType.LOGIN: 0.4,
                EventType.PURCHASE: 0.2,
                EventType.SERVICE_UPGRADE: 0.15,
                EventType.USAGE_SPIKE: 0.15,
                EventType.LOGOUT: 0.1
            }
        
        # Select event type based on weights
        events = list(event_weights.keys())
        weights = list(event_weights.values())
        event_type = random.choices(events, weights=weights)[0]
        
        # Generate event-specific data
        event_data = self._generate_event_data(event_type, customer_profile)
        
        return CustomerEvent(
            customer_id=customer_profile['customer_id'],
            event_type=event_type,
            timestamp=datetime.now(),
            event_data=event_data,
            session_id=str(uuid.uuid4()),
            device_type=customer_profile['device_type'],
            location=customer_profile['location']
        )
    
    def _generate_event_data(self, event_type: EventType, customer_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate event-specific data"""
        base_data = {
            'tenure_months': customer_profile['tenure_months'],
            'monthly_charges': customer_profile['monthly_charges'],
            'contract_type': customer_profile['contract_type']
        }
        
        if event_type == EventType.PURCHASE:
            base_data.update({
                'purchase_amount': round(random.uniform(10, 100), 2),
                'product_category': random.choice(['Add-on', 'Upgrade', 'Service'])
            })
        elif event_type == EventType.SUPPORT_CALL:
            base_data.update({
                'call_duration_minutes': random.randint(5, 45),
                'issue_type': random.choice(['Technical', 'Billing', 'General']),
                'resolution_status': random.choice(['Resolved', 'Escalated', 'Pending'])
            })
        elif event_type == EventType.BILLING_ISSUE:
            base_data.update({
                'issue_amount': round(random.uniform(10, 200), 2),
                'dispute_type': random.choice(['Overcharge', 'Unauthorized', 'Service Quality'])
            })
        elif event_type == EventType.USAGE_SPIKE:
            base_data.update({
                'usage_gb': round(random.uniform(50, 200), 2),
                'usage_type': random.choice(['Data', 'Voice', 'SMS'])
            })
        elif event_type == EventType.USAGE_DROP:
            base_data.update({
                'usage_gb': round(random.uniform(0, 20), 2),
                'drop_percentage': round(random.uniform(30, 80), 1)
            })
        
        return base_data
    
    def send_event(self, event: CustomerEvent) -> bool:
        """Send single event to Kafka"""
        try:
            future = self.producer.send(
                self.topic,
                value=event.to_dict(),
                key=event.customer_id.encode('utf-8')
            )
            
            # Wait for send to complete
            record_metadata = future.get(timeout=10)
            
            logger.info(f"Sent event {event.event_type.value} for customer {event.customer_id} "
                       f"to partition {record_metadata.partition} at offset {record_metadata.offset}")
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to send event: {e}")
            return False
    
    def generate_and_send_events(self, num_events: int = 100, delay_seconds: float = 1.0):
        """Generate and send multiple events with delay"""
        logger.info(f"Starting to generate {num_events} events with {delay_seconds}s delay")
        
        successful_sends = 0
        
        for i in range(num_events):
            try:
                # Select random customer profile
                customer_profile = self._get_customer_profile()
                
                # Generate realistic event
                event = self._generate_realistic_event(customer_profile)
                
                # Send event
                if self.send_event(event):
                    successful_sends += 1
                
                # Add delay between events
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{num_events} events")
                    
            except Exception as e:
                logger.error(f"Error generating event {i+1}: {e}")
        
        logger.info(f"Completed: {successful_sends}/{num_events} events sent successfully")
    
    def simulate_customer_journey(self, customer_id: str, events_count: int = 20):
        """Simulate a specific customer's journey"""
        customer_profile = self._get_customer_profile(customer_id)
        
        logger.info(f"Simulating journey for customer {customer_id} with {events_count} events")
        
        for i in range(events_count):
            event = self._generate_realistic_event(customer_profile)
            self.send_event(event)
            
            # Vary delay to simulate realistic timing
            delay = random.uniform(0.5, 3.0)
            time.sleep(delay)
    
    def close(self):
        """Close the producer"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Producer closed")


def main():
    """Main function for standalone testing"""
    producer = CustomerEventProducer()
    
    try:
        # Generate continuous stream of events
        producer.generate_and_send_events(num_events=50, delay_seconds=2.0)
        
        # Simulate specific customer journey
        producer.simulate_customer_journey("CUST_000001", events_count=10)
        
    finally:
        producer.close()


if __name__ == "__main__":
    main()