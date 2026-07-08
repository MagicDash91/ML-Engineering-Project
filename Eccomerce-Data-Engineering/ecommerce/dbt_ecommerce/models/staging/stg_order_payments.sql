with source as (

    select * from {{ source('raw', 'order_payments') }}

)

select
    -- Surrogate key for the (order_id, payment_sequential) grain
    {{ dbt_utils.generate_surrogate_key(['order_id', 'payment_sequential']) }} as payment_sk,
    order_id,
    payment_sequential,
    lower(trim(payment_type)) as payment_type,
    payment_installments,
    payment_value

from source
