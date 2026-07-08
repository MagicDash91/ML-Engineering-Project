-- review_id is NOT unique in the raw ORDER_REVIEWS table (a handful of review
-- ids appear against more than one order). We keep the most recently answered
-- row per review_id so that review_id can serve as the fact primary key.
with source as (

    select * from {{ source('raw', 'order_reviews') }}

),

deduped as (

    select
        review_id,
        order_id,
        review_score,
        review_comment_title,
        review_comment_message,
        review_creation_date,
        review_answer_timestamp
    from source
    qualify row_number() over (
        partition by review_id
        order by review_answer_timestamp desc nulls last
    ) = 1

)

select
    review_id,
    order_id,
    review_score,
    review_comment_title,
    review_comment_message,
    -- Convenience flag: does the review carry any written feedback?
    case
        when nullif(trim(review_comment_title), '') is not null
          or nullif(trim(review_comment_message), '') is not null
        then true
        else false
    end                                             as has_comment,
    review_creation_date,
    review_answer_timestamp,
    datediff(
        'hour', review_creation_date, review_answer_timestamp
    )                                               as response_time_hours

from deduped
