import os
from typing import Dict

SNOWFLAKE_CONFIG = {
    'account': 'ORGKMBU-HU54176',
    'user': 'MAGICDASH',
    'password': os.getenv('SNOWFLAKE_PASSWORD', 'Permataputihg101'),
    'warehouse': 'COMPUTE_WH',
    'database': 'DATABASE',
    'schema': 'PUBLIC',
    'role': 'ACCOUNTADMIN',
    'account_url': 'ORGKMBU-HU54176.snowflakecomputing.com'
}

def get_snowflake_connection_params() -> Dict[str, str]:
    """Get Snowflake connection parameters."""
    return {
        'user': SNOWFLAKE_CONFIG['user'],
        'password': SNOWFLAKE_CONFIG['password'],
        'account': SNOWFLAKE_CONFIG['account'],
        'warehouse': SNOWFLAKE_CONFIG['warehouse'],
        'database': SNOWFLAKE_CONFIG['database'],
        'schema': SNOWFLAKE_CONFIG['schema'],
        'role': SNOWFLAKE_CONFIG['role']
    }

CHURN_TABLE = f"{SNOWFLAKE_CONFIG['database']}.{SNOWFLAKE_CONFIG['schema']}.CHURN"