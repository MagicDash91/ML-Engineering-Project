{{
  config(
    materialized='view'
  )
}}

with source_data as (
    select * from {{ source('raw_data', 'churn') }}
),

cleaned_data as (
    select
        -- Customer identifiers
        customerid as customer_id,
        
        -- Demographics
        gender,
        case 
            when seniorcitizen = 1 then true
            when seniorcitizen = 0 then false
            else null
        end as is_senior_citizen,
        partner as has_partner,
        dependents as has_dependents,
        
        -- Service tenure
        tenure as tenure_months,
        
        -- Services
        phoneservice as has_phone_service,
        case 
            when multiplelines = 'No phone service' then false
            when multiplelines = 'No' then false
            when multiplelines = 'Yes' then true
            else null
        end as has_multiple_lines,
        
        -- Internet services
        internetservice as internet_service_type,
        case 
            when onlinesecurity = 'No internet service' then null
            when onlinesecurity = 'No' then false
            when onlinesecurity = 'Yes' then true
            else null
        end as has_online_security,
        case 
            when onlinebackup = 'No internet service' then null
            when onlinebackup = 'No' then false
            when onlinebackup = 'Yes' then true
            else null
        end as has_online_backup,
        case 
            when deviceprotection = 'No internet service' then null
            when deviceprotection = 'No' then false
            when deviceprotection = 'Yes' then true
            else null
        end as has_device_protection,
        case 
            when techsupport = 'No internet service' then null
            when techsupport = 'No' then false
            when techsupport = 'Yes' then true
            else null
        end as has_tech_support,
        case 
            when streamingtv = 'No internet service' then null
            when streamingtv = 'No' then false
            when streamingtv = 'Yes' then true
            else null
        end as has_streaming_tv,
        case 
            when streamingmovies = 'No internet service' then null
            when streamingmovies = 'No' then false
            when streamingmovies = 'Yes' then true
            else null
        end as has_streaming_movies,
        
        -- Contract and billing
        contract as contract_type,
        paperlessbilling as has_paperless_billing,
        paymentmethod as payment_method,
        
        -- Financial
        monthlycharges as monthly_charges,
        case 
            when totalcharges = ' ' then null
            when totalcharges = '' then null
            else try_cast(totalcharges as float)
        end as total_charges,
        
        -- Target variable
        churn as has_churned,
        
        -- Metadata
        current_timestamp() as dbt_created_at
        
    from source_data
)

select * from cleaned_data