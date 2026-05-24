#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/magicdash/airflow/churn')

import snowflake.connector
import pandas as pd
from config.snowflake_config import get_snowflake_connection_params, CHURN_TABLE

def test_snowflake_connection():
    """Test Snowflake connection and basic data retrieval."""
    try:
        print("🔍 Testing Snowflake connection...")
        
        # Get connection parameters
        conn_params = get_snowflake_connection_params()
        print(f"📡 Connecting to account: {conn_params['account']}")
        print(f"🗄️  Database: {conn_params['database']}")
        print(f"📋 Schema: {conn_params['schema']}")
        
        # Establish connection
        conn = snowflake.connector.connect(**conn_params)
        cursor = conn.cursor()
        
        # Test basic connectivity
        cursor.execute("SELECT CURRENT_VERSION()")
        version = cursor.fetchone()
        print(f"❄️  Snowflake version: {version[0]}")
        
        # Check if table exists
        cursor.execute(f"DESCRIBE TABLE {CHURN_TABLE}")
        columns = cursor.fetchall()
        print(f"\n📊 Table {CHURN_TABLE} schema:")
        for col in columns:
            print(f"  📝 {col[0]} ({col[1]})")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {CHURN_TABLE}")
        count = cursor.fetchone()
        print(f"\n📈 Total rows in {CHURN_TABLE}: {count[0]:,}")
        
        # Get sample data
        cursor.execute(f"SELECT * FROM {CHURN_TABLE} LIMIT 5")
        sample_data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        
        print("\n👀 Sample data:")
        df_sample = pd.DataFrame(sample_data, columns=column_names)
        print(df_sample)
        
        cursor.close()
        conn.close()
        
        print("\n✅ Snowflake connection test SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f"\n❌ Snowflake connection test FAILED: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_snowflake_connection()
    sys.exit(0 if success else 1)