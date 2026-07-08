-- Sanity check: no order should have been purchased in the future relative to
-- the latest purchase in the dataset's own history window.
select
    order_id,
    order_purchase_timestamp
from {{ ref('fct_order_items') }}
where order_purchase_timestamp > current_timestamp()
