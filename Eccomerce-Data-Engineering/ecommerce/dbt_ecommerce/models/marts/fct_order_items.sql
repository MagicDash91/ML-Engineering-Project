-- Central fact table. Grain: one row per order item.
-- Joins the order header onto each item to bring across order-level context
-- (customer, status, purchase date) and conform to the date dimension.
with order_items as (

    select * from {{ ref('stg_order_items') }}

),

orders as (

    select * from {{ ref('stg_orders') }}

)

select
    oi.order_item_sk,
    oi.order_id,
    oi.order_item_id,

    -- Foreign keys into the dimensions
    o.customer_id,
    oi.product_id,
    oi.seller_id,
    to_number(to_char(o.order_purchase_timestamp, 'YYYYMMDD')) as order_date_key,

    -- Order context
    o.order_status,
    o.order_purchase_timestamp,
    oi.shipping_limit_date,

    -- Measures
    oi.price,
    oi.freight_value,
    oi.total_item_value

from order_items oi
inner join orders o
    on oi.order_id = o.order_id
