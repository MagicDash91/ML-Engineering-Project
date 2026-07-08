FROM astrocrpublic.azurecr.io/runtime:3.2-4

# Ship the Olist e-commerce project (config + dbt project) into the image so the
# DAG can import config and run dbt against E_COMMERCE.PUBLIC.
COPY ecommerce/ /usr/local/airflow/include/ecommerce/
