-- Financial consistency: every order item's total value must equal the sum of
-- its price and freight. Returns offending rows (test passes when empty).
select
    order_item_sk,
    price,
    freight_value,
    total_item_value
from {{ ref('fct_order_items') }}
where abs(total_item_value - (price + freight_value)) > 0.001
