-- Test: Monthly charges should be reasonable compared to total charges
-- For customers with tenure > 0, total_charges should generally be >= monthly_charges
select customer_id,
       monthly_charges,
       total_charges,
       tenure_months
from {{ ref('mart_churn_prediction') }}
where tenure_months > 0 
  and total_charges > 0
  and total_charges < monthly_charges * 0.5  -- Allow some flexibility for prorations