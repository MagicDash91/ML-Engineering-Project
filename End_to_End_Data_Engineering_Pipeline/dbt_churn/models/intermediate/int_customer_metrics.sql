{{
  config(
    materialized='view'
  )
}}

with customer_features as (
    select * from {{ ref('int_customer_features') }}
),

-- Calculate aggregate metrics for context
aggregate_stats as (
    select
        avg(monthly_charges) as avg_monthly_charges_all,
        avg(total_charges) as avg_total_charges_all,
        avg(tenure_months) as avg_tenure_all,
        percentile_cont(0.5) within group (order by monthly_charges) as median_monthly_charges,
        percentile_cont(0.5) within group (order by total_charges) as median_total_charges,
        percentile_cont(0.5) within group (order by tenure_months) as median_tenure
    from customer_features
),

enhanced_metrics as (
    select
        cf.*,
        
        -- Relative metrics
        cf.monthly_charges / nullif(stats.avg_monthly_charges_all, 0) as monthly_charges_ratio_to_avg,
        cf.total_charges / nullif(stats.avg_total_charges_all, 0) as total_charges_ratio_to_avg,
        cf.tenure_months / nullif(stats.avg_tenure_all, 0) as tenure_ratio_to_avg,
        
        -- Percentile rankings
        case 
            when cf.monthly_charges >= stats.median_monthly_charges then 'above_median'
            else 'below_median'
        end as monthly_charges_percentile,
        
        case 
            when cf.total_charges >= stats.median_total_charges then 'above_median'
            else 'below_median'
        end as total_charges_percentile,
        
        case 
            when cf.tenure_months >= stats.median_tenure then 'above_median'
            else 'below_median'
        end as tenure_percentile,
        
        -- Risk indicators
        case 
            when cf.contract_term = 'monthly' and cf.tenure_months < 12 then 'high_risk'
            when cf.contract_term = 'monthly' and cf.payment_category = 'electronic' then 'medium_risk'
            else 'low_risk'
        end as churn_risk_indicator,
        
        -- Customer value score (simple scoring)
        (case when cf.tenure_months > 24 then 2 else 1 end +
         case when cf.monthly_charges > stats.avg_monthly_charges_all then 2 else 1 end +
         case when cf.internet_add_ons_count > 2 then 2 else 1 end +
         case when cf.contract_term != 'monthly' then 2 else 1 end) as customer_value_score
        
    from customer_features cf
    cross join aggregate_stats stats
)

select * from enhanced_metrics