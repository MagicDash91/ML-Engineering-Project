{{
  config(
    materialized='view'
  )
}}

with customer_base as (
    select * from {{ ref('stg_customer_churn') }}
),

feature_engineering as (
    select
        customer_id,
        
        -- Demographics
        gender,
        is_senior_citizen,
        has_partner,
        has_dependents,
        
        -- Create family status feature
        case 
            when has_partner and has_dependents then 'family_with_dependents'
            when has_partner and not has_dependents then 'couple_no_dependents'
            when not has_partner and has_dependents then 'single_with_dependents'
            else 'single_no_dependents'
        end as family_status,
        
        -- Service tenure
        tenure_months,
        case 
            when tenure_months <= 12 then 'new_customer'
            when tenure_months <= 24 then 'established_customer'
            when tenure_months <= 48 then 'loyal_customer'
            else 'long_term_customer'
        end as tenure_segment,
        
        -- Service features
        has_phone_service,
        has_multiple_lines,
        internet_service_type,
        case when internet_service_type = 'No' then false else true end as has_internet_service,
        
        -- Internet add-ons (count of additional services)
        (case when has_online_security then 1 else 0 end +
         case when has_online_backup then 1 else 0 end +
         case when has_device_protection then 1 else 0 end +
         case when has_tech_support then 1 else 0 end +
         case when has_streaming_tv then 1 else 0 end +
         case when has_streaming_movies then 1 else 0 end) as internet_add_ons_count,
        
        has_online_security,
        has_online_backup,
        has_device_protection,
        has_tech_support,
        has_streaming_tv,
        has_streaming_movies,
        
        -- Streaming services
        case 
            when has_streaming_tv and has_streaming_movies then 'both_streaming'
            when has_streaming_tv then 'tv_only'
            when has_streaming_movies then 'movies_only'
            else 'no_streaming'
        end as streaming_services,
        
        -- Contract and billing
        contract_type,
        case 
            when contract_type = 'Month-to-month' then 'monthly'
            when contract_type = 'One year' then 'annual'
            when contract_type = 'Two year' then 'biennial'
            else contract_type
        end as contract_term,
        has_paperless_billing,
        payment_method,
        
        -- Payment method categories
        case 
            when payment_method in ('Electronic check') then 'electronic'
            when payment_method in ('Mailed check') then 'manual'
            when payment_method in ('Bank transfer (automatic)', 'Credit card (automatic)') then 'automatic'
            else 'other'
        end as payment_category,
        
        -- Financial features
        monthly_charges,
        total_charges,
        case 
            when total_charges is null or tenure_months = 0 then monthly_charges
            else total_charges / nullif(tenure_months, 0)
        end as avg_monthly_charges,
        
        -- Price segments
        case 
            when monthly_charges < 30 then 'low_spender'
            when monthly_charges < 65 then 'medium_spender'
            else 'high_spender'
        end as spending_segment,
        
        -- Value metrics
        case 
            when total_charges is null then null
            when monthly_charges = 0 then null
            else total_charges / monthly_charges
        end as customer_lifetime_ratio,
        
        -- Target variable
        has_churned,
        
        dbt_created_at
        
    from customer_base
    where customer_id is not null
)

select * from feature_engineering