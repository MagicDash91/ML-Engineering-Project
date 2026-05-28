"""
Customer Event Schema Definitions for Real-time Streaming
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Optional, Any
import json
from enum import Enum

class EventType(Enum):
    """Customer event types for churn prediction"""
    LOGIN = "login"
    LOGOUT = "logout"
    PURCHASE = "purchase"
    SUPPORT_CALL = "support_call"
    BILLING_ISSUE = "billing_issue"
    SERVICE_DOWNGRADE = "service_downgrade"
    SERVICE_UPGRADE = "service_upgrade"
    PAYMENT_FAILED = "payment_failed"
    COMPLAINT = "complaint"
    USAGE_SPIKE = "usage_spike"
    USAGE_DROP = "usage_drop"

class RiskLevel(Enum):
    """Risk level for churn prediction"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class CustomerEvent:
    """Real-time customer event schema"""
    customer_id: str
    event_type: EventType
    timestamp: datetime
    event_data: Dict[str, Any]
    session_id: Optional[str] = None
    device_type: Optional[str] = None
    location: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CustomerEvent':
        """Create from dictionary"""
        data['event_type'] = EventType(data['event_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'CustomerEvent':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))

@dataclass
class ChurnPrediction:
    """Real-time churn prediction result"""
    customer_id: str
    churn_probability: float
    risk_level: RiskLevel
    prediction_timestamp: datetime
    model_version: str
    features_used: Dict[str, float]
    confidence_score: float
    next_action_recommended: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['risk_level'] = self.risk_level.value
        data['prediction_timestamp'] = self.prediction_timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())

@dataclass
class FeatureVector:
    """Feature vector for real-time scoring"""
    customer_id: str
    features: Dict[str, float]
    feature_timestamp: datetime
    data_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['feature_timestamp'] = self.feature_timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())

# Kafka Topic Configuration
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
    'group_id': 'churn-pipeline',
    'value_deserializer': lambda m: json.loads(m.decode('utf-8')),
    'value_serializer': lambda v: json.dumps(v).encode('utf-8')
}