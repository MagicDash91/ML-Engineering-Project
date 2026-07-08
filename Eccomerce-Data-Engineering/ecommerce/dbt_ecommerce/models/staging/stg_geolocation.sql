-- Geolocation has many rows per zip code prefix. We deduplicate to one
-- representative point per prefix (average lat/lng, most common city/state)
-- so it can serve as a clean dimension keyed on the zip prefix.
with source as (

    select * from {{ source('raw', 'geolocation') }}

),

ranked as (

    select
        geolocation_zip_code_prefix                              as zip_code_prefix,
        avg(geolocation_lat)                                     as latitude,
        avg(geolocation_lng)                                     as longitude,
        mode(lower(trim(geolocation_city)))                     as city,
        mode(upper(trim(geolocation_state)))                    as state,
        count(*)                                                as point_count
    from source
    where geolocation_zip_code_prefix is not null
    group by geolocation_zip_code_prefix

)

select
    zip_code_prefix,
    latitude,
    longitude,
    city,
    state,
    point_count

from ranked
