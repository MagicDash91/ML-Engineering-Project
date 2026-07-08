# dbt_ecommerce

dbt project transforming the raw Olist tables in `E_COMMERCE.PUBLIC` into a
star schema.

- **staging/** (`+materialized: view`, schema `PUBLIC_staging`) — one cleaning
  model per source: renames, casts, and the documented data-quality fixes.
- **marts/** (`+materialized: table`, schema `PUBLIC_marts`) — the star schema:
  `fct_order_items`, `fct_payments`, `fct_reviews` + `dim_customers`,
  `dim_products`, `dim_sellers`, `dim_geolocation`, `dim_date`.
- **tests/** — singular data tests (financial consistency, no future orders),
  in addition to the schema tests declared in `_staging.yml` / `_marts.yml`.

## Commands

```bash
export DBT_PROFILES_DIR="$PWD"     # profiles.yml lives in this dir
export SNOWFLAKE_PASSWORD=...
dbt deps && dbt build              # staging views + mart tables + all tests
```

Connection settings come from env vars (`SNOWFLAKE_*`) via `profiles.yml`; see
`../.env.example`.
