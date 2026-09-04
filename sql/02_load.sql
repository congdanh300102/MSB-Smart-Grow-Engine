-- =====================================================================
-- MSB SMART GROWTH ENGINE - load synthetic CSVs into schema msb_sge (v2)
-- Run:  psql -h localhost -U msb -d msb_sge -v ON_ERROR_STOP=1 -f sql/02_load.sql
-- Expects CSVs under /data/csv inside the container (docker-compose mounts ./data -> /data).
-- Column lists are given explicitly (== CSV header order) because \copy maps
-- CSV columns positionally, not by name.
-- =====================================================================
SET search_path TO msb_sge;

TRUNCATE
    ai_model_metric, ai_model_coefficient, ai_model_registry, ml_propensity_training_set,
    ml_credit_card_training_set, rm_action_feedback, ai_recommendation,
    ai_score_reason, ai_customer_score, ai_feature_customer,
    customer_360_feature_mart, fact_campaign, fact_customer_service,
    fact_crm_interaction, fact_digital_activity, fact_insurance, fact_loan,
    fact_deposit, fact_card_monthly, agg_customer_transaction, fact_transaction,
    fact_casa_daily, dim_card, dim_customer, dim_campaign, dim_product, dim_rm
    RESTART IDENTITY CASCADE;

-- 1. dimensions --------------------------------------------------------
\copy dim_rm (rm_id,branch_id,rm_role,rm_segment,active_flag) FROM '/data/csv/dim_rm.csv' WITH (FORMAT csv, HEADER true, NULL '');

\copy dim_product (product_id,product_code,product_name,product_group,product_type,product_tier,active_flag) FROM '/data/csv/dim_product.csv' WITH (FORMAT csv, HEADER true, NULL '');

\copy dim_campaign (campaign_id,campaign_name,product_id,start_date,end_date,channel,campaign_type,target_segment,campaign_status) FROM '/data/csv/dim_campaign.csv' WITH (FORMAT csv, HEADER true, NULL '');

\copy dim_customer (customer_id,cif_id,customer_type,age_group,gender_code,occupation_group,industry_group,income_band,province_code,region_code,customer_segment,relationship_start_date,relationship_years,rm_id,branch_id,kyc_status,preferred_channel,customer_active_flag,marketing_consent_flag,consent_channels,do_not_contact_flag,record_effective_from,record_effective_to,current_flag) FROM '/data/csv/dim_customer.csv' WITH (FORMAT csv, HEADER true, NULL '');

\copy dim_card (card_id,customer_id,card_type,card_tier,card_status,issue_date,expiry_date,credit_limit) FROM '/data/csv/dim_card.csv' WITH (FORMAT csv, HEADER true, NULL '');

-- 2. raw / source facts -------------------------------------------------
\copy fact_casa_daily (snapshot_date,account_id,customer_id,account_type,currency,current_balance,available_balance,credit_amount_daily,debit_amount_daily,credit_txn_count,debit_txn_count,salary_credit_amount,salary_flag,account_status,avg_balance_30d,avg_balance_90d,avg_balance_180d,total_inflow_30d,total_outflow_30d,balance_growth_30d,balance_growth_90d,salary_stability_score) FROM '/data/csv/fact_casa_daily.csv' WITH (FORMAT csv, HEADER true, NULL '');

\copy fact_transaction (transaction_id,customer_id,account_id,transaction_timestamp,transaction_date,transaction_type,channel,debit_credit_flag,amount,currency,merchant_id,merchant_category_code,transaction_category,domestic_foreign_flag,counterparty_type,transaction_status) FROM '/data/csv/fact_transaction.csv' WITH (FORMAT csv, HEADER true, NULL '');

\copy fact_deposit (deposit_id,customer_id,open_date,maturity_date,deposit_type,term_months,currency,principal_amount,current_balance,interest_rate,auto_renew_flag,early_withdraw_flag,deposit_status,total_deposit_balance,deposit_count,maturity_30d_amount,maturity_60d_amount,days_to_nearest_maturity) FROM '/data/csv/fact_deposit.csv' WITH (FORMAT csv, HEADER true, NULL '');

\copy fact_loan (loan_id,customer_id,loan_type,start_date,maturity_date,original_amount,outstanding_balance,interest_rate,monthly_installment,secured_flag,payment_status,days_past_due,loan_status,has_active_loan,active_loan_count,total_loan_outstanding,monthly_debt_payment,loan_to_income_ratio) FROM '/data/csv/fact_loan.csv' WITH (FORMAT csv, HEADER true, NULL '');

\copy fact_insurance (policy_id,customer_id,insurance_type,provider_code,start_date,expiry_date,renewal_date,premium_amount,policy_value,payment_frequency,policy_status,insurance_product_count,active_insurance_flag,total_annual_premium,nearest_renewal_days) FROM '/data/csv/fact_insurance.csv' WITH (FORMAT csv, HEADER true, NULL '');

\copy fact_digital_activity (activity_date,customer_id,login_count,active_flag,session_count,transfer_count,qr_payment_count,bill_payment_count,digital_txn_count,feature_usage_count,push_received_count,push_open_count,last_login_timestamp,login_count_30d,active_days_30d,digital_txn_count_30d,qr_count_30d,digital_txn_ratio,push_open_rate,digital_engagement_score) FROM '/data/csv/fact_digital_activity.csv' WITH (FORMAT csv, HEADER true, NULL '');

\copy fact_crm_interaction (interaction_id,customer_id,rm_id,interaction_timestamp,interaction_type,channel,product_code,interaction_reason,lead_status,opportunity_stage,customer_response,next_followup_date,reason_lost,campaign_id,days_since_last_contact,contact_count_30d,contact_count_90d,last_customer_response,previous_product_interest,recent_rejection_flag) FROM '/data/csv/fact_crm_interaction.csv' WITH (FORMAT csv, HEADER true, NULL '');

\copy fact_customer_service (case_id,customer_id,created_timestamp,closed_timestamp,channel,case_type,issue_category,product_code,complaint_flag,severity,case_status,resolution_time_hours,csat_score,complaint_count_30d,complaint_count_90d,serious_complaint_7d,days_since_last_complaint,avg_csat,open_case_count) FROM '/data/csv/fact_customer_service.csv' WITH (FORMAT csv, HEADER true, NULL '');

\copy fact_campaign (campaign_customer_id,campaign_id,customer_id,product_code,campaign_date,channel,sent_flag,delivered_flag,opened_flag,clicked_flag,responded_flag,interested_flag,applied_flag,approved_flag,converted_flag,conversion_date) FROM '/data/csv/fact_campaign.csv' WITH (FORMAT csv, HEADER true, NULL '');

-- 3. aggregates ----------------------------------------------------------
\copy agg_customer_transaction (customer_id,snapshot_date,txn_count_30d,txn_count_90d,total_spend_30d,total_spend_90d,avg_txn_value_30d,online_spend_30d,travel_spend_90d,dining_spend_90d,international_spend_90d,qr_txn_count_30d,credit_inflow_30d,debit_outflow_30d,spending_growth_3m) FROM '/data/csv/agg_customer_transaction.csv' WITH (FORMAT csv, HEADER true, NULL '');

\copy fact_card_monthly (customer_id,card_id,month,card_spending,transaction_count,online_spending,international_spending,utilization_rate,payment_amount,late_payment_count) FROM '/data/csv/fact_card_monthly.csv' WITH (FORMAT csv, HEADER true, NULL '');

-- 4. feature / serving layer ---------------------------------------------
\copy customer_360_feature_mart (snapshot_date,customer_id,segment,age_group,income_band,relationship_years,region,rm_id,branch_id,avg_balance_30d,avg_balance_90d,balance_growth_3m,inflow_30d,outflow_30d,salary_flag,salary_amount_avg,txn_count_30d,txn_count_90d,spending_30d,spending_90d,spending_growth_3m,online_spending_90d,travel_spending_90d,international_spending_90d,product_count,has_credit_card,has_premium_card,has_active_loan,deposit_balance,insurance_active_flag,has_investment,income_monthly,net_cashflow_30d,feature_usage_30d,login_count_30d,active_days_30d,digital_txn_count_30d,digital_engagement_score,days_since_last_rm_contact,rm_contact_count_90d,last_customer_response,recent_rejection_flag,complaint_count_30d,serious_complaint_7d,avg_csat,campaign_count_6m,previous_campaign_response,previous_card_campaign_response,previous_conversion_flag,contact_count_7d,serious_complaint_15d,recent_rejection_30d_flag) FROM '/data/csv/customer_360_feature_mart.csv' WITH (FORMAT csv, HEADER true, NULL '');

\copy ai_feature_customer (snapshot_date,customer_id,product_propensity_feature_set,customer_value_score,intent_signal_score,engagement_score,timing_score,relationship_score,sales_suppression_flag) FROM '/data/csv/ai_feature_customer.csv' WITH (FORMAT csv, HEADER true, NULL '');

-- 5. model output ----------------------------------------------------------
\copy ai_customer_score (score_id,customer_id,snapshot_date,product_id,model_id,model_version,propensity_probability,product_propensity_score,customer_value_score,intent_signal_score,engagement_score,timing_score,relationship_score,smart_growth_score,priority_level,prediction_timestamp) FROM '/data/csv/ai_customer_score.csv' WITH (FORMAT csv, HEADER true, NULL '');

\copy ai_score_reason (reason_id,score_id,feature_name,feature_value,contribution_score,impact_direction,reason_rank) FROM '/data/csv/ai_score_reason.csv' WITH (FORMAT csv, HEADER true, NULL '');

-- 6. decision layer ---------------------------------------------------------
\copy ai_recommendation (recommendation_id,customer_id,score_id,recommended_product_id,recommended_action,recommended_channel,recommended_timing,message_angle,expected_conversion,priority,suppression_flag,recommendation_timestamp,status,priority_rank) FROM '/data/csv/ai_recommendation.csv' WITH (FORMAT csv, HEADER true, NULL '');

-- 7. feedback loop -----------------------------------------------------------
\copy rm_action_feedback (feedback_id,recommendation_id,customer_id,rm_id,action_timestamp,action_type,result_status,customer_response,appointment_flag,application_flag,converted_flag,reason_not_interested,rm_feedback,next_followup_date) FROM '/data/csv/rm_action_feedback.csv' WITH (FORMAT csv, HEADER true, NULL '');

-- 8. training data -------------------------------------------------------------
\copy ml_credit_card_training_set (customer_id,observation_date,age_group,segment,avg_balance_90d,income_band,spending_90d,txn_count_90d,salary_flag,digital_score,has_credit_card,campaign_response_history,days_since_last_rm_contact,complaint_count_30d,label_applied_90d,label_converted_90d) FROM '/data/csv/ml_credit_card_training_set.csv' WITH (FORMAT csv, HEADER true, NULL '');

\copy ml_propensity_training_set (customer_id,snapshot_date,product_id,product_group,x1_monthly_spending,x2_income,x3_digital_activity,x4_salary_account,x5_campaign_response,x6_product_gap,y_holds_product,y_adopt_next_90d) FROM '/data/csv/ml_propensity_training_set.csv' WITH (FORMAT csv, HEADER true, NULL '');

-- Model artefacts (ai_model_registry/coefficient/metric) are loaded separately
-- by sql/04_load_models.sql AFTER running:  py src/train_models.py --apply

ANALYZE;

SELECT 'dim_rm' t, count(*) FROM dim_rm
UNION ALL SELECT 'dim_product', count(*) FROM dim_product
UNION ALL SELECT 'dim_campaign', count(*) FROM dim_campaign
UNION ALL SELECT 'dim_customer', count(*) FROM dim_customer
UNION ALL SELECT 'dim_card', count(*) FROM dim_card
UNION ALL SELECT 'fact_casa_daily', count(*) FROM fact_casa_daily
UNION ALL SELECT 'fact_transaction', count(*) FROM fact_transaction
UNION ALL SELECT 'fact_deposit', count(*) FROM fact_deposit
UNION ALL SELECT 'fact_loan', count(*) FROM fact_loan
UNION ALL SELECT 'fact_insurance', count(*) FROM fact_insurance
UNION ALL SELECT 'fact_digital_activity', count(*) FROM fact_digital_activity
UNION ALL SELECT 'fact_crm_interaction', count(*) FROM fact_crm_interaction
UNION ALL SELECT 'fact_customer_service', count(*) FROM fact_customer_service
UNION ALL SELECT 'fact_campaign', count(*) FROM fact_campaign
UNION ALL SELECT 'agg_customer_transaction', count(*) FROM agg_customer_transaction
UNION ALL SELECT 'fact_card_monthly', count(*) FROM fact_card_monthly
UNION ALL SELECT 'customer_360_feature_mart', count(*) FROM customer_360_feature_mart
UNION ALL SELECT 'ai_feature_customer', count(*) FROM ai_feature_customer
UNION ALL SELECT 'ai_customer_score', count(*) FROM ai_customer_score
UNION ALL SELECT 'ai_score_reason', count(*) FROM ai_score_reason
UNION ALL SELECT 'ai_recommendation', count(*) FROM ai_recommendation
UNION ALL SELECT 'rm_action_feedback', count(*) FROM rm_action_feedback
UNION ALL SELECT 'ml_credit_card_training_set', count(*) FROM ml_credit_card_training_set
UNION ALL SELECT 'ml_propensity_training_set', count(*) FROM ml_propensity_training_set
ORDER BY 1;
