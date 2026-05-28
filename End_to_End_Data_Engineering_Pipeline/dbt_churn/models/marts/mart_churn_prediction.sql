{{
  config(
    materialized='table',
    indexes=[
        {'columns': ['customer_id'], 'unique': True},
        {'columns': ['has_churned']},
        {'columns': ['churn_risk_indicator']},
    ]
  )
}}

with customer_metrics as (
    select * from {{ ref('int_customer_metrics') }}
),

-- Create final ML-ready dataset
ml_features as (
    select
        customer_id,
        
        -- Encode categorical variables for ML
        case when gender = 'Male' then 1 else 0 end as gender_male,
        case when gender = 'Female' then 1 else 0 end as gender_female,
        
        case when is_senior_citizen then 1 else 0 end as is_senior_citizen,
        case when has_partner then 1 else 0 end as has_partner,
        case when has_dependents then 1 else 0 end as has_dependents,
        
        -- Family status encoding
        case when family_status = 'family_with_dependents' then 1 else 0 end as family_with_dependents,
        case when family_status = 'couple_no_dependents' then 1 else 0 end as couple_no_dependents,
        case when family_status = 'single_with_dependents' then 1 else 0 end as single_with_dependents,
        case when family_status = 'single_no_dependents' then 1 else 0 end as single_no_dependents,
        
        -- Tenure features
        tenure_months,
        case when tenure_segment = 'new_customer' then 1 else 0 end as is_new_customer,
        case when tenure_segment = 'established_customer' then 1 else 0 end as is_established_customer,
        case when tenure_segment = 'loyal_customer' then 1 else 0 end as is_loyal_customer,
        case when tenure_segment = 'long_term_customer' then 1 else 0 end as is_long_term_customer,
        
        -- Service features
        case when has_phone_service then 1 else 0 end as has_phone_service,
        case when has_multiple_lines then 1 else 0 end as has_multiple_lines,
        case when has_internet_service then 1 else 0 end as has_internet_service,
        
        -- Internet service type
        case when internet_service_type = 'DSL' then 1 else 0 end as internet_dsl,
        case when internet_service_type = 'Fiber optic' then 1 else 0 end as internet_fiber,
        case when internet_service_type = 'No' then 1 else 0 end as no_internet,
        
        -- Internet add-ons
        internet_add_ons_count,
        case when has_online_security then 1 else 0 end as has_online_security,
        case when has_online_backup then 1 else 0 end as has_online_backup,
        case when has_device_protection then 1 else 0 end as has_device_protection,
        case when has_tech_support then 1 else 0 end as has_tech_support,
        case when has_streaming_tv then 1 else 0 end as has_streaming_tv,
        case when has_streaming_movies then 1 else 0 end as has_streaming_movies,
        
        -- Streaming services
        case when streaming_services = 'both_streaming' then 1 else 0 end as has_both_streaming,
        case when streaming_services = 'tv_only' then 1 else 0 end as has_tv_only,
        case when streaming_services = 'movies_only' then 1 else 0 end as has_movies_only,
        case when streaming_services = 'no_streaming' then 1 else 0 end as no_streaming,
        
        -- Contract features
        case when contract_term = 'monthly' then 1 else 0 end as contract_monthly,
        case when contract_term = 'annual' then 1 else 0 end as contract_annual,
        case when contract_term = 'biennial' then 1 else 0 end as contract_biennial,
        
        case when has_paperless_billing then 1 else 0 end as has_paperless_billing,
        
        -- Payment method
        case when payment_category = 'electronic' then 1 else 0 end as payment_electronic,
        case when payment_category = 'manual' then 1 else 0 end as payment_manual,
        case when payment_category = 'automatic' then 1 else 0 end as payment_automatic,
        
        -- Financial features (normalized)
        monthly_charges,
        coalesce(total_charges, 0) as total_charges,
        coalesce(avg_monthly_charges, monthly_charges) as avg_monthly_charges,
        
        -- Spending segment
        case when spending_segment = 'low_spender' then 1 else 0 end as is_low_spender,
        case when spending_segment = 'medium_spender' then 1 else 0 end as is_medium_spender,
        case when spending_segment = 'high_spender' then 1 else 0 end as is_high_spender,
        
        -- Relative metrics
        coalesce(monthly_charges_ratio_to_avg, 1) as monthly_charges_ratio_to_avg,
        coalesce(total_charges_ratio_to_avg, 1) as total_charges_ratio_to_avg,
        coalesce(tenure_ratio_to_avg, 1) as tenure_ratio_to_avg,
        
        -- Risk indicators
        case when churn_risk_indicator = 'high_risk' then 1 else 0 end as is_high_risk,
        case when churn_risk_indicator = 'medium_risk' then 1 else 0 end as is_medium_risk,
        case when churn_risk_indicator = 'low_risk' then 1 else 0 end as is_low_risk,
        
        customer_value_score,
        coalesce(customer_lifetime_ratio, 1) as customer_lifetime_ratio,
        
        -- Target variable
        case when has_churned then 1 else 0 end as target_churned,
        
        -- Metadata
        dbt_created_at as feature_created_at,
        current_timestamp() as mart_created_at
        
    from customer_metrics
    where customer_id is not null
),

-- Add feature quality flags
final_features as (
    select
        *,
        
        -- Data quality flags
        case 
            when total_charges = 0 and tenure_months > 0 then 1 
            else 0 
        end as has_data_quality_issue,
        
        -- Feature completeness score
        case 
            when monthly_charges is null then 0
            when total_charges is null then 0.8
            else 1.0
        end as feature_completeness_score
        
    from ml_features
)

select * from final_features