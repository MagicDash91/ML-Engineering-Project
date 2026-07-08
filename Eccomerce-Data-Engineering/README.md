# Olist E-Commerce Data Pipeline

End-to-end, reproducible data pipeline on the [Olist Brazilian e-commerce
dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce), built on the modern
data stack: **Terraform → Airflow → dbt → Snowflake**, with AWS S3 as the raw
landing zone.

Nine raw Olist CSVs are already loaded into `E_COMMERCE.PUBLIC`. This project
turns them into an analytics-ready **star schema**, orchestrates the build with
Airflow, and defines the supporting AWS infrastructure as disposable Terraform.

## Architecture

![Architecture](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Eccomerce-Data-Engineering/static/architecture.png)

```
                    ┌─────────────────────────────────────────────────────┐
                    │        Terraform  ·  Infrastructure as Code          │
                    │  VPC · Subnet · Internet Gateway · Security Group     │
                    │  IAM role · S3 landing bucket · EC2 host · Budgets    │
                    └───────────────────────────┬─────────────────────────┘
                                                │ provisions
                                                ▼
 ┌───────────────┐    ingest     ┌─────────────────────────────────────────┐
 │  Olist CSVs   │ ────────────► │   Snowflake  ·  E_COMMERCE.PUBLIC (raw)  │
 │ (9 raw files) │               │            9 raw source tables           │
 └───────────────┘               └───────────────────────┬─────────────────┘
                                                         │
 ┌───────────────────────────────────────────────────────┼───────────────────┐
 │  Apache Airflow  (Astronomer Runtime · Docker · EC2)   │   orchestration   │
 │                                                        ▼                   │
 │   validate_sources ─► land_raw_to_s3 ─► dbt_transform ─────► notify        │
 │                             │            (Cosmos: run+test)     │          │
 │                             ▼                                   ▼          │
 │                     ┌──────────────┐                    ┌──────────────┐   │
 │                     │  Amazon S3   │                    │   WhatsApp   │   │
 │                     │ raw landing  │                    │  Cloud API   │   │
 │                     └──────────────┘                    └──────────────┘   │
 └────────────────────────────────┬──────────────────────────────────────────┘
                                  │ dbt (via Cosmos)
                                  ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │                       dbt  ·  Snowflake                           │
     │   staging views (stg_*)  ──►  marts / star schema (fct_* dim_*)   │
     │                                                                   │
     │   Facts:  fct_order_items · fct_payments · fct_reviews            │
     │   Dims :  dim_customers · dim_products · dim_sellers ·            │
     │           dim_geolocation · dim_date                              │
     └──────────────────────────────────────────────────────────────────┘
```

## Screenshots

![Screenshot 1](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Eccomerce-Data-Engineering/static/a1.jpg)

![Screenshot 2](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Eccomerce-Data-Engineering/static/a2.jpg)

![Screenshot 3](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Eccomerce-Data-Engineering/static/a3.jpg)

![Screenshot 4](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Eccomerce-Data-Engineering/static/a4.jpg)

![Screenshot 5](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Eccomerce-Data-Engineering/static/a5.jpg)

![Screenshot 6](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Eccomerce-Data-Engineering/static/a6.jpg)

![Screenshot 6](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Eccomerce-Data-Engineering/static/a7.jpg)

## Tech stack

| Layer                            | Tools & technologies                                                                                                 |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Cloud data warehouse**         | Snowflake (`E_COMMERCE.PUBLIC`, `COMPUTE_WH`)                                                                        |
| **Transformation**               | dbt Core 1.11 · dbt-snowflake · dbt_utils                                                                            |
| **Orchestration**                | Apache Airflow (Astronomer Runtime 3.2) · Astronomer Cosmos (renders each dbt model + test as a native Airflow task) |
| **Containerization & local dev** | Docker · Astro CLI                                                                                                   |
| **Infrastructure as Code**       | Terraform (AWS provider ~> 5.0)                                                                                      |
| **Language & libraries**         | Python 3 · pandas · snowflake-connector-python · boto3 · requests                                                    |
| **Notifications**                | WhatsApp Cloud API (webhook; falls back to logging)                                                                  |
| **Source data**                  | Olist Brazilian E-Commerce dataset (9 raw CSVs)                                                                      |

### AWS services

| Service             | Role in the pipeline                                                                          |
| ------------------- | --------------------------------------------------------------------------------------------- |
| **Amazon S3**       | Raw landing bucket — versioned, SSE-encrypted, 30-day lifecycle expiry, public access blocked |
| **Amazon EC2**      | `t3.medium` self-hosted Airflow host (Docker + Astro CLI bootstrapped via user-data)          |
| **Amazon VPC**      | Dedicated VPC + public subnet + internet gateway + route table                                |
| **Security Groups** | SSH (22) and Airflow UI (8080) locked to a single allowed CIDR                                |
| **AWS IAM**         | Least-privilege role + instance profile for the Airflow host                                  |
| **AWS Budgets**     | Monthly cost alerts at $20 and $50                                                            |

## Repository layout

```
.
├── dags/
│   └── ecommerce_olist_pipeline.py     # Airflow DAG (Cosmos-rendered dbt tasks)
├── ecommerce/
│   ├── config/snowflake_config.py      # env-var-driven Snowflake config
│   ├── dbt_ecommerce/                   # dbt project
│   │   ├── models/staging/             # stg_* : rename, cast, DQ fixes
│   │   ├── models/marts/               # fct_* / dim_* star schema
│   │   ├── tests/                      # singular data tests
│   │   ├── dbt_project.yml
│   │   ├── packages.yml                # dbt_utils
│   │   └── profiles.yml
│   ├── terraform/                       # S3 + EC2 + IAM + Budgets (+ optional Snowflake)
│   ├── scripts/test_snowflake_connection.py
│   └── .env.example
├── include/dbt_profiles/profiles.yml    # profile used by the DAG in Docker
├── Dockerfile                           # ships ecommerce/ into the Astro image
└── requirements.txt
```

## Target data model (star schema)

| Model             | Grain                      | Notes                                                     |
| ----------------- | -------------------------- | --------------------------------------------------------- |
| `fct_order_items` | one row per order item     | central fact; joins order header for customer/status/date |
| `fct_payments`    | one row per payment record | FK to customer + date                                     |
| `fct_reviews`     | one row per review         | deduplicated by `review_id` (not unique in raw)           |
| `dim_customers`   | one row per `customer_id`  | enriched with geolocation lat/lng                         |
| `dim_products`    | one row per `product_id`   | joined to corrected PT→EN category mapping                |
| `dim_sellers`     | one row per `seller_id`    | enriched with geolocation lat/lng                         |
| `dim_geolocation` | one row per zip prefix     | deduplicated (avg lat/lng, modal city/state)              |
| `dim_date`        | one row per day            | 2016-01-01 → 2019-12-31, `date_key` = `YYYYMMDD`          |

### Data-quality fixes handled in staging

- **`PRODUCT_CATEGORY`** was loaded without headers (columns `C1`/`C2`, first data
  row is the literal header). `stg_product_category` drops that stray row and
  aliases `C1 → product_category_name`, `C2 → product_category_name_english`.
- **`PRODUCTS`** carries the source typo `PRODUCT_NAME_LENGHT` /
  `PRODUCT_DESCRIPTION_LENGHT`. Preserved in raw, corrected to `..._length` in staging.
- **`ORDER_REVIEWS.REVIEW_ID`** is not unique (99,224 rows / 98,410 ids).
  `stg_order_reviews` keeps the latest-answered row per `review_id`.
- **`GEOLOCATION`** has many rows per zip prefix; deduplicated to one representative
  point per prefix.

## Prerequisites

- Python 3.9+ with the project `venv` (or `pip install -r requirements.txt`)
- [Astro CLI](https://www.astronomer.io/docs/astro/cli/install-cli) + Docker (to run Airflow)
- Terraform ≥ 1.5 and AWS credentials (only for the infra layer)
- Snowflake access to `E_COMMERCE.PUBLIC`

## Setup

```bash
cp ecommerce/.env.example ecommerce/.env      # then fill in SNOWFLAKE_PASSWORD etc.
export SNOWFLAKE_PASSWORD=...                  # or `set -a; source ecommerce/.env; set +a`
```

Credentials are **never** hardcoded — everything is read from environment
variables, and `.env` is gitignored.

## Run dbt directly

```bash
cd ecommerce/dbt_ecommerce
export DBT_PROFILES_DIR="$PWD"
dbt deps           # install dbt_utils
dbt debug          # verify the Snowflake connection
dbt build          # run all models + tests (staging views + mart tables)
dbt docs generate && dbt docs serve   # browse the lineage/docs site
```

A clean `dbt build` produces **8 mart tables + 9 staging views** and runs
**88 tests** (PK `unique`/`not_null`, FK `relationships`, accepted values/ranges).
One intentional `warn` covers 13 products whose category is absent from the
mapping table.

## Run the pipeline in Airflow

```bash
astro dev start                 # builds the image (ships ecommerce/ into include/)
# open http://localhost:8080  →  enable & trigger `ecommerce_olist_pipeline`
```

DAG tasks: `validate_sources → land_raw_to_s3 → dbt_transform (Cosmos) → notify`.

- `validate_sources` — asserts all 9 raw tables exist and are non-empty.
- `land_raw_to_s3` — snapshots each raw table to CSV and uploads to the S3 landing
  bucket (skipped gracefully without AWS creds).
- `dbt_transform` — Cosmos renders every model + test as its own task (run→test).
- `notify` — logs a run summary; sends WhatsApp if `WHATSAPP_*` env vars are set.

## Provision / tear down AWS infra

```bash
cd ecommerce/terraform
terraform init
terraform apply      # S3 landing bucket, EC2 Airflow host, IAM role, Budgets
# ... run the pipeline / capture screenshots ...
terraform destroy    # buckets use force_destroy = true, so this is clean
```

Optional: rename `snowflake.tf.example → snowflake.tf` to provision the
least-privilege `DBT_PIPELINE_ROLE` and warehouse auto-suspend via
`terraform-provider-snowflake`.

## Cost & security controls

- **Budgets** alert at **$20** and **$50** of monthly AWS spend.
- **S3** landing objects expire after 30 days; buckets are private + encrypted.
- **EC2** should be **stopped between runs** (`aws ec2 stop-instances --instance-ids <id>`).
- **Snowflake** `COMPUTE_WH` should use auto-suspend (see optional Snowflake module).
- **No secrets in source**: all credentials come from env vars; `.env` is gitignored.
  Rotate any credential previously shared in plaintext (chat/screenshots) and move
  the pipeline to `DBT_PIPELINE_ROLE` instead of `ACCOUNTADMIN`.

## Teardown checklist (portfolio)

1. `dbt docs generate` and capture the lineage graph + a successful DAG run.
2. `astro dev stop`.
3. `terraform destroy` and confirm no orphaned S3 buckets / ENIs remain.
