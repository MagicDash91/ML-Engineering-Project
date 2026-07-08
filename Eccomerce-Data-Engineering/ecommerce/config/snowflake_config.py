"""
Snowflake configuration for the Olist E-Commerce data pipeline.

Credentials are read from environment variables. A `.env` file (gitignored) or a
secrets manager should supply SNOWFLAKE_PASSWORD in every environment. The
non-secret defaults below match the E_COMMERCE.PUBLIC project account and can be
overridden per-environment via env vars.
"""
import os
from typing import Dict

SNOWFLAKE_CONFIG = {
    'account': os.getenv('SNOWFLAKE_ACCOUNT', 'JNJUCIN-BE85650'),
    'user': os.getenv('SNOWFLAKE_USER', 'MAGICDASH'),
    # Never hardcode the real password. Supply via env / secrets manager.
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
    'database': os.getenv('SNOWFLAKE_DATABASE', 'E_COMMERCE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC'),
    # PRD recommends a dedicated least-privilege role (DBT_PIPELINE_ROLE).
    # ACCOUNTADMIN remains the fallback only until that role is provisioned.
    'role': os.getenv('SNOWFLAKE_ROLE', 'ACCOUNTADMIN'),
    'account_url': os.getenv(
        'SNOWFLAKE_ACCOUNT_URL', 'JNJUCIN-BE85650.snowflakecomputing.com'
    ),
}

# The nine raw Olist tables already loaded into E_COMMERCE.PUBLIC.
RAW_TABLES = [
    'CUSTOMERS',
    'GEOLOCATION',
    'ORDERS',
    'ORDER_ITEMS',
    'ORDER_PAYMENTS',
    'ORDER_REVIEWS',
    'PRODUCTS',
    'PRODUCT_CATEGORY',
    'SELLERS',
]


def get_snowflake_connection_params() -> Dict[str, str]:
    """Return kwargs for snowflake.connector.connect()."""
    params = {
        'user': SNOWFLAKE_CONFIG['user'],
        'password': SNOWFLAKE_CONFIG['password'],
        'account': SNOWFLAKE_CONFIG['account'],
        'warehouse': SNOWFLAKE_CONFIG['warehouse'],
        'database': SNOWFLAKE_CONFIG['database'],
        'schema': SNOWFLAKE_CONFIG['schema'],
        'role': SNOWFLAKE_CONFIG['role'],
    }
    if not params['password']:
        raise RuntimeError(
            'SNOWFLAKE_PASSWORD is not set. Export it or add it to a gitignored '
            '.env file before connecting to Snowflake.'
        )
    return params
