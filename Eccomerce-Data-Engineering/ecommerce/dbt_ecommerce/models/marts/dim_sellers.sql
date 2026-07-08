-- Seller dimension. Grain: one row per seller_id.
with sellers as (

    select * from {{ ref('stg_sellers') }}

),

geo as (

    select * from {{ ref('stg_geolocation') }}

)

select
    s.seller_id,
    s.seller_zip_code_prefix,
    s.seller_city,
    s.seller_state,
    g.latitude  as seller_latitude,
    g.longitude as seller_longitude

from sellers s
left join geo g
    on s.seller_zip_code_prefix = g.zip_code_prefix
