-- Geolocation dimension, one row per Brazilian zip code prefix.
select
    zip_code_prefix,
    latitude,
    longitude,
    city,
    state,
    point_count

from {{ ref('stg_geolocation') }}
