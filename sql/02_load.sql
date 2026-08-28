-- =====================================================================
-- MSB SMART GROWTH ENGINE - load synthetic CSVs into the schema
-- Run with psql:   psql -h localhost -U msb -d msb_sge -v ON_ERROR_STOP=1 -f sql/02_load.sql
-- Expects the CSV files under  /data/csv  inside the container
-- (docker-compose mounts ./data -> /data).
-- =====================================================================
SET search_path TO msb_sge;

TRUNCATE rm_action_feedback, ai_recommendation, ai_customer_score,
         customer_360_feature_mart, fact_campaign, fact_customer_service,
         fact_crm_interaction, fact_digital_activity, fact_insurance,
         fact_deposit, fact_loan, fact_card, fact_transaction,
         fact_casa_daily, dim_customer RESTART IDENTITY CASCADE;

\copy dim_customer               FROM '/data/csv/dim_customer.csv'               WITH (FORMAT csv, HEADER true, NULL '');
\copy fact_casa_daily            FROM '/data/csv/fact_casa_daily.csv'            WITH (FORMAT csv, HEADER true, NULL '');
\copy fact_transaction           FROM '/data/csv/fact_transaction.csv'           WITH (FORMAT csv, HEADER true, NULL '');
\copy fact_card                  FROM '/data/csv/fact_card.csv'                  WITH (FORMAT csv, HEADER true, NULL '');
\copy fact_loan                  FROM '/data/csv/fact_loan.csv'                  WITH (FORMAT csv, HEADER true, NULL '');
\copy fact_deposit               FROM '/data/csv/fact_deposit.csv'               WITH (FORMAT csv, HEADER true, NULL '');
\copy fact_insurance             FROM '/data/csv/fact_insurance.csv'             WITH (FORMAT csv, HEADER true, NULL '');
\copy fact_digital_activity      FROM '/data/csv/fact_digital_activity.csv'      WITH (FORMAT csv, HEADER true, NULL '');
\copy fact_crm_interaction       FROM '/data/csv/fact_crm_interaction.csv'       WITH (FORMAT csv, HEADER true, NULL '');
\copy fact_customer_service      FROM '/data/csv/fact_customer_service.csv'      WITH (FORMAT csv, HEADER true, NULL '');
\copy fact_campaign              FROM '/data/csv/fact_campaign.csv'              WITH (FORMAT csv, HEADER true, NULL '');
\copy customer_360_feature_mart  FROM '/data/csv/customer_360_feature_mart.csv'  WITH (FORMAT csv, HEADER true, NULL '');
\copy ai_customer_score          FROM '/data/csv/ai_customer_score.csv'          WITH (FORMAT csv, HEADER true, NULL '');
\copy ai_recommendation          FROM '/data/csv/ai_recommendation.csv'          WITH (FORMAT csv, HEADER true, NULL '');
\copy rm_action_feedback         FROM '/data/csv/rm_action_feedback.csv'         WITH (FORMAT csv, HEADER true, NULL '');

ANALYZE;

SELECT 'dim_customer' t, count(*) FROM dim_customer
UNION ALL SELECT 'fact_casa_daily', count(*) FROM fact_casa_daily
UNION ALL SELECT 'fact_transaction', count(*) FROM fact_transaction
UNION ALL SELECT 'fact_card', count(*) FROM fact_card
UNION ALL SELECT 'fact_loan', count(*) FROM fact_loan
UNION ALL SELECT 'fact_deposit', count(*) FROM fact_deposit
UNION ALL SELECT 'fact_insurance', count(*) FROM fact_insurance
UNION ALL SELECT 'fact_digital_activity', count(*) FROM fact_digital_activity
UNION ALL SELECT 'fact_crm_interaction', count(*) FROM fact_crm_interaction
UNION ALL SELECT 'fact_customer_service', count(*) FROM fact_customer_service
UNION ALL SELECT 'fact_campaign', count(*) FROM fact_campaign
UNION ALL SELECT 'customer_360_feature_mart', count(*) FROM customer_360_feature_mart
UNION ALL SELECT 'ai_customer_score', count(*) FROM ai_customer_score
UNION ALL SELECT 'ai_recommendation', count(*) FROM ai_recommendation
UNION ALL SELECT 'rm_action_feedback', count(*) FROM rm_action_feedback;
