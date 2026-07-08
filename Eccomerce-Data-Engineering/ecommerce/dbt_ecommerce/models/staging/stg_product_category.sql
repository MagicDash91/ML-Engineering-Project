-- PRODUCT_CATEGORY was loaded without headers, so the raw columns are named
-- C1/C2 and the first data row is the literal header string. We drop that stray
-- header row and alias the columns to their intended names.
with source as (

    select * from {{ source('raw', 'product_category') }}

)

select
    lower(trim(c1)) as product_category_name,
    lower(trim(c2)) as product_category_name_english

from source
where lower(trim(c1)) != 'product_category_name'
