with source as (

    select * from {{ source('raw', 'order_items') }}

)

select
    -- Surrogate key for the (order_id, order_item_id) grain
    {{ dbt_utils.generate_surrogate_key(['order_id', 'order_item_id']) }} as order_item_sk,
    order_id,
    order_item_id,
    product_id,
    seller_id,
    shipping_limit_date,
    price,
    freight_value,
    price + freight_value as total_item_value

from source
