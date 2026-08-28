-- =====================================================================
-- MSB SMART GROWTH ENGINE - Customer 360 Data Model (PostgreSQL)
-- AI-Powered Customer Intelligence & Sales Growth Platform
-- ---------------------------------------------------------------------
-- Layer 1  : DIM_CUSTOMER            - customer master
-- Layer 2  : FACT_*                  - raw fragmented source facts
-- Layer 3  : CUSTOMER_360_FEATURE_MART - engineered features (Customer 360)
-- Layer 4  : AI_CUSTOMER_SCORE       - model outputs (churn / propensity / risk / NBA)
-- Layer 5  : AI_RECOMMENDATION       - Next Best Product / Next Best Action
-- Layer 6  : RM_ACTION_FEEDBACK      - RM feedback loop (Interested / Converted)
-- =====================================================================

DROP SCHEMA IF EXISTS msb_sge CASCADE;
CREATE SCHEMA msb_sge;
SET search_path TO msb_sge;

-- ---------------------------------------------------------------------
-- DIM_CUSTOMER
-- ---------------------------------------------------------------------
CREATE TABLE dim_customer (
    customer_key     INTEGER      PRIMARY KEY,
    customer_id      VARCHAR(50)  NOT NULL UNIQUE,
    full_name        VARCHAR(150) NOT NULL,
    date_of_birth    DATE,
    gender           VARCHAR(20),
    segment_code     VARCHAR(30),
    region           VARCHAR(50),
    kyc_status       VARCHAR(30),
    onboarded_date   DATE,
    customer_status  VARCHAR(30)
);

-- ---------------------------------------------------------------------
-- FACT_CASA_DAILY  (daily current/savings account balance snapshots)
-- ---------------------------------------------------------------------
CREATE TABLE fact_casa_daily (
    fact_casa_daily_key  INTEGER       PRIMARY KEY,
    customer_key         INTEGER       NOT NULL REFERENCES dim_customer(customer_key),
    account_key          VARCHAR(50)   NOT NULL,
    balance_date         DATE          NOT NULL,
    daily_balance        DECIMAL(18,2),
    average_balance_30d  DECIMAL(18,2),
    min_balance_30d      DECIMAL(18,2),
    max_balance_30d      DECIMAL(18,2),
    overdraft_flag       BOOLEAN
);

-- ---------------------------------------------------------------------
-- FACT_TRANSACTION
-- ---------------------------------------------------------------------
CREATE TABLE fact_transaction (
    fact_transaction_key   INTEGER      PRIMARY KEY,
    customer_key           INTEGER      NOT NULL REFERENCES dim_customer(customer_key),
    transaction_id         VARCHAR(60)  NOT NULL,
    transaction_timestamp  TIMESTAMP,
    transaction_type       VARCHAR(40),
    amount                 DECIMAL(18,2),
    currency_code          VARCHAR(10),
    channel                VARCHAR(30),
    merchant_category      VARCHAR(80),
    is_fraud_suspected     BOOLEAN
);

-- ---------------------------------------------------------------------
-- FACT_CARD  (credit / debit cards)
-- ---------------------------------------------------------------------
CREATE TABLE fact_card (
    fact_card_key    INTEGER      PRIMARY KEY,
    customer_key     INTEGER      NOT NULL REFERENCES dim_customer(customer_key),
    card_id          VARCHAR(50)  NOT NULL,
    card_type        VARCHAR(30),
    card_status      VARCHAR(30),
    issue_date       DATE,
    expiry_date      DATE,
    credit_limit     DECIMAL(18,2),
    utilization_pct  DECIMAL(5,2),
    activation_flag  BOOLEAN
);

-- ---------------------------------------------------------------------
-- FACT_LOAN
-- ---------------------------------------------------------------------
CREATE TABLE fact_loan (
    fact_loan_key          INTEGER      PRIMARY KEY,
    customer_key           INTEGER      NOT NULL REFERENCES dim_customer(customer_key),
    loan_account_id        VARCHAR(50)  NOT NULL,
    loan_type              VARCHAR(40),
    disbursement_date      DATE,
    outstanding_principal  DECIMAL(18,2),
    interest_rate          DECIMAL(5,2),
    tenor_months           INTEGER,
    loan_status            VARCHAR(30),
    delinquency_days       INTEGER
);

-- ---------------------------------------------------------------------
-- FACT_DEPOSIT  (term deposits)
-- ---------------------------------------------------------------------
CREATE TABLE fact_deposit (
    fact_deposit_key    INTEGER      PRIMARY KEY,
    customer_key        INTEGER      NOT NULL REFERENCES dim_customer(customer_key),
    deposit_account_id  VARCHAR(50)  NOT NULL,
    deposit_date        DATE,
    deposit_amount      DECIMAL(18,2),
    term_months         INTEGER,
    interest_rate       DECIMAL(5,2),
    maturity_date       DATE,
    deposit_status      VARCHAR(30)
);

-- ---------------------------------------------------------------------
-- FACT_INSURANCE
-- ---------------------------------------------------------------------
CREATE TABLE fact_insurance (
    fact_insurance_key  INTEGER      PRIMARY KEY,
    customer_key        INTEGER      NOT NULL REFERENCES dim_customer(customer_key),
    policy_id           VARCHAR(50)  NOT NULL,
    policy_type         VARCHAR(40),
    premium_amount      DECIMAL(18,2),
    start_date          DATE,
    end_date            DATE,
    policy_status       VARCHAR(30),
    claim_count         INTEGER
);

-- ---------------------------------------------------------------------
-- FACT_DIGITAL_ACTIVITY  (monthly digital-banking engagement aggregates)
-- ---------------------------------------------------------------------
CREATE TABLE fact_digital_activity (
    fact_digital_activity_key  INTEGER      PRIMARY KEY,
    customer_key               INTEGER      NOT NULL REFERENCES dim_customer(customer_key),
    activity_id                VARCHAR(60)  NOT NULL,
    activity_date              TIMESTAMP,
    channel                    VARCHAR(30),
    session_count              INTEGER,
    login_count                INTEGER,
    app_open_count             INTEGER,
    feature_usage_count        INTEGER
);

-- ---------------------------------------------------------------------
-- FACT_CRM_INTERACTION
-- ---------------------------------------------------------------------
CREATE TABLE fact_crm_interaction (
    fact_crm_interaction_key  INTEGER      PRIMARY KEY,
    customer_key              INTEGER      NOT NULL REFERENCES dim_customer(customer_key),
    interaction_id            VARCHAR(60)  NOT NULL,
    interaction_date          TIMESTAMP,
    contact_channel           VARCHAR(30),
    case_type                 VARCHAR(50),
    case_status               VARCHAR(30),
    priority_level            VARCHAR(20),
    resolution_time_hours     DECIMAL(10,2)
);

-- ---------------------------------------------------------------------
-- FACT_CUSTOMER_SERVICE
-- ---------------------------------------------------------------------
CREATE TABLE fact_customer_service (
    fact_customer_service_key  INTEGER      PRIMARY KEY,
    customer_key               INTEGER      NOT NULL REFERENCES dim_customer(customer_key),
    service_request_id         VARCHAR(60)  NOT NULL,
    request_date               TIMESTAMP,
    service_type               VARCHAR(50),
    service_channel            VARCHAR(30),
    request_status             VARCHAR(30),
    sla_met_flag               BOOLEAN,
    handling_time_minutes      INTEGER
);

-- ---------------------------------------------------------------------
-- FACT_CAMPAIGN  (marketing campaign targeting + response)
-- ---------------------------------------------------------------------
CREATE TABLE fact_campaign (
    fact_campaign_key     INTEGER      PRIMARY KEY,
    customer_key          INTEGER      NOT NULL REFERENCES dim_customer(customer_key),
    campaign_id           VARCHAR(60)  NOT NULL,
    campaign_name         VARCHAR(120),
    campaign_start_date   DATE,
    campaign_end_date     DATE,
    campaign_channel      VARCHAR(30),
    response_flag         BOOLEAN,
    conversion_flag       BOOLEAN,
    response_date         DATE
);

-- ---------------------------------------------------------------------
-- CUSTOMER_360_FEATURE_MART  (engineered feature store / Customer 360)
--   campaign_key -> most-recent campaign targeting this customer (per ERD)
--   customer_key -> owning customer (added: the mart is 1 row / customer / snapshot)
-- ---------------------------------------------------------------------
CREATE TABLE customer_360_feature_mart (
    customer_360_feature_key   INTEGER       PRIMARY KEY,
    customer_key               INTEGER       NOT NULL REFERENCES dim_customer(customer_key),
    campaign_key               INTEGER       REFERENCES fact_campaign(fact_campaign_key),
    snapshot_date              DATE,
    total_balance              DECIMAL(18,2),
    total_transactions_30d     INTEGER,
    total_card_spend_30d       DECIMAL(18,2),
    total_deposit_value        DECIMAL(18,2),
    total_loan_outstanding     DECIMAL(18,2),
    digital_engagement_score   DECIMAL(10,2),
    service_contact_count_90d  INTEGER,
    campaign_response_rate     DECIMAL(5,2)
);

-- ---------------------------------------------------------------------
-- AI_CUSTOMER_SCORE  (model outputs, one row per feature-mart snapshot)
-- ---------------------------------------------------------------------
CREATE TABLE ai_customer_score (
    ai_customer_score_key     INTEGER       PRIMARY KEY,
    customer_360_feature_key  INTEGER       NOT NULL REFERENCES customer_360_feature_mart(customer_360_feature_key),
    score_date                DATE,
    churn_score               DECIMAL(10,4),
    propensity_to_buy_score   DECIMAL(10,4),
    credit_risk_score         DECIMAL(10,4),
    next_best_action_score    DECIMAL(10,4)
);

-- ---------------------------------------------------------------------
-- AI_RECOMMENDATION  (Next Best Product / ranked opportunities)
-- ---------------------------------------------------------------------
CREATE TABLE ai_recommendation (
    ai_recommendation_key   INTEGER       PRIMARY KEY,
    ai_customer_score_key   INTEGER       NOT NULL REFERENCES ai_customer_score(ai_customer_score_key),
    recommendation_date     DATE,
    recommendation_type     VARCHAR(80),
    recommendation_text     VARCHAR(500),
    confidence_score        DECIMAL(10,4),
    priority_rank           INTEGER
);

-- ---------------------------------------------------------------------
-- RM_ACTION_FEEDBACK  (Relationship Manager feedback loop)
-- ---------------------------------------------------------------------
CREATE TABLE rm_action_feedback (
    rm_action_feedback_key  INTEGER       PRIMARY KEY,
    ai_recommendation_key   INTEGER       NOT NULL REFERENCES ai_recommendation(ai_recommendation_key),
    feedback_date           DATE,
    action_taken            VARCHAR(120),
    action_outcome          VARCHAR(80),
    follow_up_required      BOOLEAN,
    rating                  INTEGER,
    notes                   VARCHAR(500)
);

-- ---------------------------------------------------------------------
-- Indexes on foreign keys / common filters
-- ---------------------------------------------------------------------
CREATE INDEX ix_casa_cust        ON fact_casa_daily(customer_key);
CREATE INDEX ix_casa_date        ON fact_casa_daily(balance_date);
CREATE INDEX ix_txn_cust         ON fact_transaction(customer_key);
CREATE INDEX ix_txn_ts           ON fact_transaction(transaction_timestamp);
CREATE INDEX ix_card_cust        ON fact_card(customer_key);
CREATE INDEX ix_loan_cust        ON fact_loan(customer_key);
CREATE INDEX ix_deposit_cust     ON fact_deposit(customer_key);
CREATE INDEX ix_insurance_cust   ON fact_insurance(customer_key);
CREATE INDEX ix_digital_cust     ON fact_digital_activity(customer_key);
CREATE INDEX ix_crm_cust         ON fact_crm_interaction(customer_key);
CREATE INDEX ix_service_cust     ON fact_customer_service(customer_key);
CREATE INDEX ix_campaign_cust    ON fact_campaign(customer_key);
CREATE INDEX ix_mart_cust        ON customer_360_feature_mart(customer_key);
CREATE INDEX ix_score_feat       ON ai_customer_score(customer_360_feature_key);
CREATE INDEX ix_reco_score       ON ai_recommendation(ai_customer_score_key);
CREATE INDEX ix_feedback_reco    ON rm_action_feedback(ai_recommendation_key);
