#!/bin/bash

echo "🚀 Starting Infrastructure for End-to-End Data Engineering Pipeline"
echo "=================================================================="

# Navigate to the docker directory
cd /home/magicdash/astro-airflow/churn/docker

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found!"
    echo "Please make sure you're in the correct directory."
    exit 1
fi

echo "📋 Checking Docker containers status..."
docker-compose ps

echo ""
echo "🔧 Starting infrastructure services (Kafka, Redis, Prometheus, Grafana)..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 15

echo ""
echo "🔍 Checking service health..."

# Check Kafka
if curl -s localhost:8081 > /dev/null; then
    echo "✅ Kafka UI is running at http://localhost:8081"
else
    echo "⚠️  Kafka UI not responding"
fi

# Check Redis
if docker-compose exec -T redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
    echo "✅ Redis is running"
else
    echo "⚠️  Redis not responding"
fi

# Check Prometheus
if curl -s localhost:9090/-/healthy > /dev/null; then
    echo "✅ Prometheus is running at http://localhost:9090"
else
    echo "⚠️  Prometheus not responding"
fi

# Check Grafana
if curl -s localhost:3000/api/health > /dev/null; then
    echo "✅ Grafana is running at http://localhost:3000"
else
    echo "⚠️  Grafana not responding"
fi

echo ""
echo "🎯 Infrastructure Status Summary:"
echo "================================"
echo "Airflow:    http://astro-airflow.localhost:6563"
echo "Kafka UI:   http://localhost:8081" 
echo "Grafana:    http://localhost:3000 (admin/admin)"
echo "Prometheus: http://localhost:9090"
echo ""
echo "🔄 You can now trigger the DAGs in Airflow:"
echo "   - telco_churn_unified_pipeline (Main ML pipeline)"
echo "   - real_time_churn_streaming (Streaming infrastructure)"