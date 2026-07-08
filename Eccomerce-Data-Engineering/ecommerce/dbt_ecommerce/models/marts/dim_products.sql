-- Product dimension joined to the corrected Portuguese -> English category
-- mapping. Grain: one row per product_id.
with products as (

    select * from {{ ref('stg_products') }}

),

category as (

    select * from {{ ref('stg_product_category') }}

)

select
    p.product_id,
    p.product_category_name,
    coalesce(cat.product_category_name_english, 'unknown') as product_category_name_english,
    p.product_name_length,
    p.product_description_length,
    p.product_photos_qty,
    p.product_weight_g,
    p.product_length_cm,
    p.product_height_cm,
    p.product_width_cm

from products p
left join category cat
    on p.product_category_name = cat.product_category_name
