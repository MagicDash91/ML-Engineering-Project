-- Payments fact. Grain: one row per payment record per order.
with payments as (

    select * from {{ ref('stg_order_payments') }}

),

orders as (

    select * from {{ ref('stg_orders') }}

)

select
    p.payment_sk,
    p.order_id,
    p.payment_sequential,

    -- Foreign keys
    o.customer_id,
    to_number(to_char(o.order_purchase_timestamp, 'YYYYMMDD')) as order_date_key,

    -- Payment attributes / measures
    p.payment_type,
    p.payment_installments,
    p.payment_value

from payments p
inner join orders o
    on p.order_id = o.order_id
