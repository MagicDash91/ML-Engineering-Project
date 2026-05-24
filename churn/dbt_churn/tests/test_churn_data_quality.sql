-- Test: Ensure gender encoding is consistent
-- Both gender_male and gender_female should not be 1 for the same customer
select customer_id
from {{ ref('mart_churn_prediction') }}
where gender_male = 1 and gender_female = 1