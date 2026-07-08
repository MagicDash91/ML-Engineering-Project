#!/usr/bin/env python3
"""Smoke-test the Snowflake connection and list the raw Olist tables.

Usage:
    export SNOWFLAKE_PASSWORD=...   # or set it in a gitignored .env
    python ecommerce/scripts/test_snowflake_connection.py
"""
import os
import sys

# Make the ecommerce package importable when run from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import snowflake.connector  # noqa: E402

from config.snowflake_config import (  # noqa: E402
    RAW_TABLES,
    get_snowflake_connection_params,
)


def main() -> int:
    print('🔍 Testing Snowflake connection to E_COMMERCE.PUBLIC ...')
    params = get_snowflake_connection_params()
    print(f'📡 account={params["account"]}  db={params["database"]}  role={params["role"]}')

    try:
        with snowflake.connector.connect(**params) as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT CURRENT_VERSION()')
                print(f'❄️  Snowflake version: {cur.fetchone()[0]}')

                print('\n📊 Raw table row counts:')
                grand_total = 0
                for table in RAW_TABLES:
                    cur.execute(f'SELECT COUNT(*) FROM E_COMMERCE.PUBLIC."{table}"')
                    count = cur.fetchone()[0]
                    grand_total += count
                    print(f'   {table:<18} {count:>12,}')
                print(f'   {"TOTAL":<18} {grand_total:>12,}')
        print('\n✅ Connection test SUCCESSFUL')
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f'\n❌ Connection test FAILED: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
