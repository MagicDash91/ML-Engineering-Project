-- Reviews fact. Grain: one row per review (deduplicated by review_id upstream).
with reviews as (

    select * from {{ ref('stg_order_reviews') }}

),

orders as (

    select * from {{ ref('stg_orders') }}

)

select
    r.review_id,
    r.order_id,

    -- Foreign keys
    o.customer_id,
    to_number(to_char(r.review_creation_date, 'YYYYMMDD')) as review_date_key,

    -- Review attributes / measures
    r.review_score,
    r.has_comment,
    r.review_creation_date,
    r.review_answer_timestamp,
    r.response_time_hours

from reviews r
inner join orders o
    on r.order_id = o.order_id
