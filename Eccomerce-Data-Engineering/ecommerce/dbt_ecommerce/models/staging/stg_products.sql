-- The raw PRODUCTS table carries a typo in the source column names
-- (PRODUCT_NAME_LENGHT / PRODUCT_DESCRIPTION_LENGHT). We preserve the raw
-- columns as-is upstream and fix the spelling here in staging.
with source as (

    select * from {{ source('raw', 'products') }}

)

select
    product_id,
    lower(trim(product_category_name)) as product_category_name,
    product_name_lenght                as product_name_length,
    product_description_lenght         as product_description_length,
    product_photos_qty,
    product_weight_g,
    product_length_cm,
    product_height_cm,
    product_width_cm

from source
