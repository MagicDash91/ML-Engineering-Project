{{
  config(
    materialized='table'
  )
}}

with churn_data as (
    select * from {{ ref('mart_churn_prediction') }}
),

-- Customer segmentation for analytics
customer_segments as (
    select
        customer_id,
        
        -- Original features for analysis
        case when gender_male = 1 then 'Male' else 'Female' end as gender,
        case when is_senior_citizen = 1 then 'Senior' else 'Non-Senior' end as age_group,
        case when has_partner = 1 then 'Has Partner' else 'No Partner' end as partner_status,
        case when has_dependents = 1 then 'Has Dependents' else 'No Dependents' end as dependent_status,
        
        tenure_months,
        case 
            when is_new_customer = 1 then 'New (0-12 months)'
            when is_established_customer = 1 then 'Established (13-24 months)'
            when is_loyal_customer = 1 then 'Loyal (25-48 months)'
            else 'Long-term (48+ months)'
        end as tenure_category,
        
        case when has_internet_service = 1 then 'Has Internet' else 'No Internet' end as internet_status,
        case 
            when internet_fiber = 1 then 'Fiber Optic'
            when internet_dsl = 1 then 'DSL'
            else 'No Internet'
        end as internet_type,
        
        internet_add_ons_count,
        case 
            when internet_add_ons_count = 0 then 'No Add-ons'
            when internet_add_ons_count <= 2 then 'Few Add-ons'
            when internet_add_ons_count <= 4 then 'Many Add-ons'
            else 'All Add-ons'
        end as add_ons_category,
        
        case 
            when contract_monthly = 1 then 'Month-to-Month'
            when contract_annual = 1 then 'One Year'
            else 'Two Year'
        end as contract_type,
        
        case 
            when payment_electronic = 1 then 'Electronic Check'
            when payment_manual = 1 then 'Mailed Check'
            else 'Automatic Payment'
        end as payment_type,
        
        monthly_charges,
        total_charges,
        case 
            when is_low_spender = 1 then 'Low ($0-30)'
            when is_medium_spender = 1 then 'Medium ($30-65)'
            else 'High ($65+)'
        end as spending_tier,
        
        customer_value_score,
        case 
            when customer_value_score <= 4 then 'Low Value'
            when customer_value_score <= 6 then 'Medium Value'
            else 'High Value'
        end as value_segment,
        
        target_churned,
        case when target_churned = 1 then 'Churned' else 'Active' end as churn_status,
        
        case 
            when is_high_risk = 1 then 'High Risk'
            when is_medium_risk = 1 then 'Medium Risk'
            else 'Low Risk'
        end as risk_category,
        
        feature_created_at,
        mart_created_at
        
    from churn_data
),

-- Add summary statistics
summary_stats as (
    select
        *,
        
        -- Customer lifetime value estimate
        case 
            when target_churned = 0 then monthly_charges * 12 -- Assume 12 month future value for active customers
            else total_charges -- Historical value for churned customers
        end as estimated_clv,
        
        -- Churn risk score (0-100)
        case 
            when risk_category = 'High Risk' then 75 + (customer_value_score * 2)
            when risk_category = 'Medium Risk' then 50 + (customer_value_score * 3)
            else 25 + (customer_value_score * 2)
        end as churn_risk_score
        
    from customer_segments
)

select * from summary_stats