-- Customer dimension. Grain: one row per customer_id (Olist's order-scoped id).
-- customer_unique_id is retained for cross-order customer analysis.
with customers as (

    select * from {{ ref('stg_customers') }}

),

geo as (

    select * from {{ ref('stg_geolocation') }}

)

select
    c.customer_id,
    c.customer_unique_id,
    c.customer_zip_code_prefix,
    c.customer_city,
    c.customer_state,
    g.latitude  as customer_latitude,
    g.longitude as customer_longitude

from customers c
left join geo g
    on c.customer_zip_code_prefix = g.zip_code_prefix
