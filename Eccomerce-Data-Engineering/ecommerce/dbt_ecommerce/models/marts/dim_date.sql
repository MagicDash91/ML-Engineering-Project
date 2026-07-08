-- Date dimension built from a Snowflake generator between the configured spine
-- bounds. Grain: one row per calendar day. date_key is an integer YYYYMMDD.
{% set start_date = var('date_spine_start') %}
{% set end_date = var('date_spine_end') %}

with spine as (

    select
        dateadd(
            day,
            seq4(),
            '{{ start_date }}'::date
        ) as date_day
    from table(generator(rowcount => 3653))  -- ~10 years of days; filtered below

)

select
    to_number(to_char(date_day, 'YYYYMMDD'))      as date_key,
    date_day,
    year(date_day)                                as year,
    quarter(date_day)                             as quarter,
    month(date_day)                               as month,
    monthname(date_day)                           as month_name,
    week(date_day)                                as week_of_year,
    day(date_day)                                 as day_of_month,
    dayofweek(date_day)                           as day_of_week,
    dayname(date_day)                             as day_name,
    case when dayofweek(date_day) in (0, 6) then true else false end as is_weekend

from spine
where date_day <= '{{ end_date }}'::date
