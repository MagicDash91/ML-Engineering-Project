with source as (

    select * from {{ source('raw', 'orders') }}

)

select
    order_id,
    customer_id,
    lower(trim(order_status))                        as order_status,
    order_purchase_timestamp,
    order_approved_at,
    order_delivered_carrier_date,
    order_delivered_customer_date,
    order_estimated_delivery_date,

    -- Derived delivery metrics
    datediff(
        'day', order_purchase_timestamp, order_delivered_customer_date
    )                                                as delivery_days,
    datediff(
        'day', order_delivered_customer_date, order_estimated_delivery_date
    )                                                as delivery_vs_estimate_days,
    case
        when order_delivered_customer_date is not null
             and order_delivered_customer_date > order_estimated_delivery_date
        then true
        when order_delivered_customer_date is not null
        then false
        else null
    end                                              as is_delivered_late

from source
