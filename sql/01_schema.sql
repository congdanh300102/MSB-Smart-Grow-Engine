-- =====================================================================
-- MSB SMART GROWTH ENGINE - Data Model v2  (PostgreSQL)
-- Theo tài liệu "MSB SMART GROWTH ENGINE V.01" - mục "I. THIẾT KẾ DATA TABLE"
-- và "2. Thiết kế hành trình AI" (12 Actions).
--
-- Luồng dữ liệu:
--   CUSTOMER DATA -> CUSTOMER 360 -> AI MODEL -> SCORE -> RECOMMENDATION
--                 -> RM ACTION -> ACTUAL RESULT -> TRAINING DATA -> MODEL
--
--   RAW / SOURCE        : dim_*, fact_* (grain nguồn)
--   AGGREGATE           : agg_customer_transaction, fact_card_monthly
--   FEATURE (serving)   : customer_360_feature_mart, ai_feature_customer
--   MODEL OUTPUT        : ai_customer_score, ai_score_reason
--   DECISION            : ai_recommendation  (Next Best Product/Action/Channel/Timing)
--   FEEDBACK LOOP       : rm_action_feedback
--   TRAINING            : ml_credit_card_training_set
--   JOURNEY GATES       : v_customer_product_eligibility, v_top_opportunities
-- =====================================================================

DROP SCHEMA IF EXISTS msb_sge CASCADE;
CREATE SCHEMA msb_sge;
SET search_path TO msb_sge;

-- =====================================================================
-- 1. DIMENSIONS
-- =====================================================================

-- DIM_RM ---------------------------------------------------------------
CREATE TABLE dim_rm (
    rm_id         VARCHAR(20) PRIMARY KEY,
    branch_id     VARCHAR(20) NOT NULL,
    rm_role       VARCHAR(30),
    rm_segment    VARCHAR(30),
    active_flag   BOOLEAN
);

-- DIM_PRODUCT ---------------------------------------------------------
CREATE TABLE dim_product (
    product_id     VARCHAR(20) PRIMARY KEY,
    product_code   VARCHAR(40) NOT NULL UNIQUE,
    product_name   VARCHAR(120),
    product_group  VARCHAR(40),   -- CARD / LOAN / DEPOSIT / INVESTMENT ...
    product_type   VARCHAR(40),   -- CREDIT_CARD / HOME_LOAN / TERM_DEPOSIT ...
    product_tier   VARCHAR(30),   -- PLATINUM / GOLD / STANDARD / NULL
    active_flag    BOOLEAN
);

-- DIM_CAMPAIGN ------------------------------------------------------
CREATE TABLE dim_campaign (
    campaign_id      VARCHAR(30) PRIMARY KEY,
    campaign_name    VARCHAR(150),
    product_id       VARCHAR(20) REFERENCES dim_product(product_id),
    start_date       DATE,
    end_date         DATE,
    channel          VARCHAR(30),
    campaign_type    VARCHAR(40),
    target_segment   VARCHAR(40),
    campaign_status  VARCHAR(30)
);

-- DIM_CUSTOMER  (SCD2 columns present; generator emits current rows) ----
CREATE TABLE dim_customer (
    customer_id             VARCHAR(50) PRIMARY KEY,
    cif_id                  VARCHAR(50) NOT NULL,
    customer_type           VARCHAR(20),          -- INDIVIDUAL / SME
    age_group               VARCHAR(20),
    gender_code             VARCHAR(10),
    occupation_group        VARCHAR(40),
    industry_group          VARCHAR(40),
    income_band             VARCHAR(20),          -- e.g. <10M, 10-20M, 20-40M, 40-80M, 80M+
    province_code           VARCHAR(10),
    region_code             VARCHAR(10),
    customer_segment        VARCHAR(30),          -- MASS / MASS_AFFLUENT / AFFLUENT / PRIVATE / SME
    relationship_start_date DATE,
    relationship_years      DECIMAL(5,2),
    rm_id                   VARCHAR(20) REFERENCES dim_rm(rm_id),
    branch_id               VARCHAR(20),
    kyc_status              VARCHAR(30),
    preferred_channel       VARCHAR(20),
    -- contactability / consent (phục vụ Decision Gate 1 & 2)
    customer_active_flag    BOOLEAN,
    marketing_consent_flag  BOOLEAN,
    consent_channels        VARCHAR(60),          -- csv: SMS,EMAIL,APP,ZALO
    do_not_contact_flag     BOOLEAN,
    -- SCD2
    record_effective_from   DATE,
    record_effective_to     DATE,
    current_flag            BOOLEAN
);

-- DIM_CARD  (không có trong doc v2 nhưng cần để biết holding thẻ / tier) --
CREATE TABLE dim_card (
    card_id        VARCHAR(40) PRIMARY KEY,
    customer_id    VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    card_type      VARCHAR(20),   -- CREDIT / DEBIT
    card_tier      VARCHAR(20),   -- PLATINUM / GOLD / STANDARD / NULL
    card_status    VARCHAR(20),
    issue_date     DATE,
    expiry_date    DATE,
    credit_limit   DECIMAL(18,2)
);

-- =====================================================================
-- 2. RAW / SOURCE FACTS
-- =====================================================================

-- FACT_CASA_DAILY ---------------------------------------------------
CREATE TABLE fact_casa_daily (
    snapshot_date          DATE        NOT NULL,
    account_id             VARCHAR(50) NOT NULL,
    customer_id            VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    account_type           VARCHAR(20),
    currency               VARCHAR(10),
    current_balance        DECIMAL(18,2),
    available_balance      DECIMAL(18,2),
    credit_amount_daily    DECIMAL(18,2),
    debit_amount_daily     DECIMAL(18,2),
    credit_txn_count       INTEGER,
    debit_txn_count        INTEGER,
    salary_credit_amount   DECIMAL(18,2),
    salary_flag            BOOLEAN,
    account_status         VARCHAR(20),
    avg_balance_30d        DECIMAL(18,2),
    avg_balance_90d        DECIMAL(18,2),
    avg_balance_180d       DECIMAL(18,2),
    total_inflow_30d       DECIMAL(18,2),
    total_outflow_30d      DECIMAL(18,2),
    balance_growth_30d     DECIMAL(9,4),
    balance_growth_90d     DECIMAL(9,4),
    salary_stability_score DECIMAL(6,2),
    PRIMARY KEY (snapshot_date, account_id)
);

-- FACT_DEPOSIT ----------------------------------------------------
CREATE TABLE fact_deposit (
    deposit_id               VARCHAR(50) PRIMARY KEY,
    customer_id              VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    open_date                DATE,
    maturity_date            DATE,
    deposit_type             VARCHAR(30),
    term_months              INTEGER,
    currency                 VARCHAR(10),
    principal_amount         DECIMAL(18,2),
    current_balance          DECIMAL(18,2),
    interest_rate            DECIMAL(6,3),
    auto_renew_flag          BOOLEAN,
    early_withdraw_flag      BOOLEAN,
    deposit_status           VARCHAR(20),
    -- customer-level aggregates (denormalised)
    total_deposit_balance    DECIMAL(18,2),
    deposit_count            INTEGER,
    maturity_30d_amount      DECIMAL(18,2),
    maturity_60d_amount      DECIMAL(18,2),
    days_to_nearest_maturity INTEGER
);

-- FACT_TRANSACTION  (raw - KHÔNG đưa trực tiếp vào ML) --------------
CREATE TABLE fact_transaction (
    transaction_id         VARCHAR(60) PRIMARY KEY,
    customer_id            VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    account_id             VARCHAR(50),
    transaction_timestamp  TIMESTAMP,
    transaction_date       DATE,
    transaction_type       VARCHAR(40),
    channel                VARCHAR(30),
    debit_credit_flag      VARCHAR(2),      -- D / C
    amount                 DECIMAL(18,2),
    currency               VARCHAR(10),
    merchant_id            VARCHAR(40),
    merchant_category_code VARCHAR(10),
    transaction_category   VARCHAR(40),
    domestic_foreign_flag  VARCHAR(3),      -- DOM / FOR
    counterparty_type      VARCHAR(30),
    transaction_status     VARCHAR(20)
);

-- FACT_LOAN -----------------------------------------------------
CREATE TABLE fact_loan (
    loan_id                VARCHAR(50) PRIMARY KEY,
    customer_id            VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    loan_type              VARCHAR(30),
    start_date             DATE,
    maturity_date          DATE,
    original_amount        DECIMAL(18,2),
    outstanding_balance    DECIMAL(18,2),
    interest_rate          DECIMAL(6,3),
    monthly_installment    DECIMAL(18,2),
    secured_flag           BOOLEAN,
    payment_status         VARCHAR(20),
    days_past_due          INTEGER,
    loan_status            VARCHAR(20),
    -- customer-level aggregates (denormalised)
    has_active_loan        BOOLEAN,
    active_loan_count      INTEGER,
    total_loan_outstanding DECIMAL(18,2),
    monthly_debt_payment   DECIMAL(18,2),
    loan_to_income_ratio   DECIMAL(9,4)
);

-- FACT_INSURANCE ----------------------------------------------
CREATE TABLE fact_insurance (
    policy_id               VARCHAR(50) PRIMARY KEY,
    customer_id             VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    insurance_type          VARCHAR(30),
    provider_code           VARCHAR(20),
    start_date              DATE,
    expiry_date             DATE,
    renewal_date            DATE,
    premium_amount          DECIMAL(18,2),
    policy_value            DECIMAL(18,2),
    payment_frequency       VARCHAR(20),
    policy_status           VARCHAR(20),
    insurance_product_count INTEGER,
    active_insurance_flag   BOOLEAN,
    total_annual_premium    DECIMAL(18,2),
    nearest_renewal_days    INTEGER
);

-- FACT_DIGITAL_ACTIVITY  (Digital Banking, grain = ngày x khách) ----
CREATE TABLE fact_digital_activity (
    activity_date           DATE        NOT NULL,
    customer_id             VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    login_count             INTEGER,
    active_flag             BOOLEAN,
    session_count           INTEGER,
    transfer_count          INTEGER,
    qr_payment_count        INTEGER,
    bill_payment_count      INTEGER,
    digital_txn_count       INTEGER,
    feature_usage_count     INTEGER,
    push_received_count     INTEGER,
    push_open_count         INTEGER,
    last_login_timestamp    TIMESTAMP,
    login_count_30d         INTEGER,
    active_days_30d         INTEGER,
    digital_txn_count_30d   INTEGER,
    qr_count_30d            INTEGER,
    digital_txn_ratio       DECIMAL(6,4),
    push_open_rate          DECIMAL(6,4),
    digital_engagement_score DECIMAL(6,2),
    PRIMARY KEY (activity_date, customer_id)
);

-- FACT_CRM_INTERACTION ---------------------------------------
CREATE TABLE fact_crm_interaction (
    interaction_id          VARCHAR(60) PRIMARY KEY,
    customer_id             VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    rm_id                   VARCHAR(20) REFERENCES dim_rm(rm_id),
    interaction_timestamp   TIMESTAMP,
    interaction_type        VARCHAR(30),
    channel                 VARCHAR(30),
    product_code            VARCHAR(40),
    interaction_reason      VARCHAR(60),
    lead_status             VARCHAR(30),
    opportunity_stage       VARCHAR(30),
    customer_response       VARCHAR(30),
    next_followup_date      DATE,
    reason_lost             VARCHAR(60),
    campaign_id             VARCHAR(30),
    days_since_last_contact INTEGER,
    contact_count_30d       INTEGER,
    contact_count_90d       INTEGER,
    last_customer_response  VARCHAR(30),
    previous_product_interest VARCHAR(40),
    recent_rejection_flag   BOOLEAN
);

-- FACT_CUSTOMER_SERVICE ------------------------------------
CREATE TABLE fact_customer_service (
    case_id                  VARCHAR(60) PRIMARY KEY,
    customer_id              VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    created_timestamp        TIMESTAMP,
    closed_timestamp         TIMESTAMP,
    channel                  VARCHAR(30),
    case_type                VARCHAR(40),
    issue_category           VARCHAR(40),
    product_code             VARCHAR(40),
    complaint_flag           BOOLEAN,
    severity                 VARCHAR(20),
    case_status              VARCHAR(20),
    resolution_time_hours    DECIMAL(10,2),
    csat_score               INTEGER,
    complaint_count_30d      INTEGER,
    complaint_count_90d      INTEGER,
    serious_complaint_7d     BOOLEAN,
    days_since_last_complaint INTEGER,
    avg_csat                 DECIMAL(4,2),
    open_case_count          INTEGER
);

-- FACT_CAMPAIGN  (campaign x customer, funnel flags) ------------
CREATE TABLE fact_campaign (
    campaign_customer_id  VARCHAR(70) PRIMARY KEY,
    campaign_id           VARCHAR(30) NOT NULL REFERENCES dim_campaign(campaign_id),
    customer_id           VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    product_code          VARCHAR(40),
    campaign_date         DATE,
    channel               VARCHAR(30),
    sent_flag             BOOLEAN,
    delivered_flag        BOOLEAN,
    opened_flag           BOOLEAN,
    clicked_flag          BOOLEAN,
    responded_flag        BOOLEAN,
    interested_flag       BOOLEAN,
    applied_flag          BOOLEAN,
    approved_flag         BOOLEAN,
    converted_flag        BOOLEAN,
    conversion_date       DATE
);

-- =====================================================================
-- 3. AGGREGATE TABLES
-- =====================================================================

-- AGG_CUSTOMER_TRANSACTION  (thay raw FACT_TRANSACTION cho ML) -------
CREATE TABLE agg_customer_transaction (
    customer_id             VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    snapshot_date           DATE        NOT NULL,
    txn_count_30d           INTEGER,
    txn_count_90d           INTEGER,
    total_spend_30d         DECIMAL(18,2),
    total_spend_90d         DECIMAL(18,2),
    avg_txn_value_30d       DECIMAL(18,2),
    online_spend_30d        DECIMAL(18,2),
    travel_spend_90d        DECIMAL(18,2),
    dining_spend_90d        DECIMAL(18,2),
    international_spend_90d  DECIMAL(18,2),
    qr_txn_count_30d        INTEGER,
    credit_inflow_30d       DECIMAL(18,2),
    debit_outflow_30d       DECIMAL(18,2),
    spending_growth_3m      DECIMAL(9,4),
    PRIMARY KEY (customer_id, snapshot_date)
);

-- FACT_CARD_MONTHLY -------------------------------------------
CREATE TABLE fact_card_monthly (
    customer_id            VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    card_id               VARCHAR(40) NOT NULL,
    month                 DATE        NOT NULL,   -- first day of month
    card_spending         DECIMAL(18,2),
    transaction_count     INTEGER,
    online_spending       DECIMAL(18,2),
    international_spending DECIMAL(18,2),
    utilization_rate      DECIMAL(6,4),
    payment_amount        DECIMAL(18,2),
    late_payment_count    INTEGER,
    PRIMARY KEY (customer_id, card_id, month)
);

-- =====================================================================
-- 4. FEATURE / SERVING LAYER
-- =====================================================================

-- CUSTOMER_360_FEATURE_MART  (bảng trực tiếp phục vụ AI) -----------
CREATE TABLE customer_360_feature_mart (
    snapshot_date              DATE        NOT NULL,
    customer_id                VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    -- PROFILE
    segment                    VARCHAR(30),
    age_group                  VARCHAR(20),
    income_band                VARCHAR(20),
    relationship_years         DECIMAL(5,2),
    region                     VARCHAR(10),
    rm_id                      VARCHAR(20),
    branch_id                  VARCHAR(20),
    -- CASA
    avg_balance_30d            DECIMAL(18,2),
    avg_balance_90d            DECIMAL(18,2),
    balance_growth_3m          DECIMAL(9,4),
    inflow_30d                 DECIMAL(18,2),
    outflow_30d                DECIMAL(18,2),
    salary_flag                BOOLEAN,
    salary_amount_avg          DECIMAL(18,2),
    -- TRANSACTION
    txn_count_30d              INTEGER,
    txn_count_90d              INTEGER,
    spending_30d               DECIMAL(18,2),
    spending_90d               DECIMAL(18,2),
    spending_growth_3m         DECIMAL(9,4),
    online_spending_90d        DECIMAL(18,2),
    travel_spending_90d        DECIMAL(18,2),
    international_spending_90d  DECIMAL(18,2),
    -- PRODUCT
    product_count              INTEGER,
    has_credit_card            BOOLEAN,
    has_premium_card           BOOLEAN,
    has_active_loan            BOOLEAN,
    deposit_balance            DECIMAL(18,2),
    insurance_active_flag      BOOLEAN,
    has_investment             BOOLEAN,
    income_monthly             DECIMAL(18,2),
    net_cashflow_30d           DECIMAL(18,2),
    feature_usage_30d          INTEGER,
    -- DIGITAL
    login_count_30d            INTEGER,
    active_days_30d            INTEGER,
    digital_txn_count_30d      INTEGER,
    digital_engagement_score   DECIMAL(6,2),
    -- CRM
    days_since_last_rm_contact INTEGER,
    rm_contact_count_90d       INTEGER,
    last_customer_response     VARCHAR(30),
    recent_rejection_flag      BOOLEAN,
    -- SERVICE
    complaint_count_30d        INTEGER,
    serious_complaint_7d       BOOLEAN,
    avg_csat                   DECIMAL(4,2),
    -- CAMPAIGN
    campaign_count_6m              INTEGER,
    previous_campaign_response    DECIMAL(6,4),
    previous_card_campaign_response DECIMAL(6,4),
    previous_conversion_flag      BOOLEAN,
    -- contactability helpers (Decision Gate 1)
    contact_count_7d              INTEGER,
    serious_complaint_15d         BOOLEAN,
    recent_rejection_30d_flag     BOOLEAN,
    -- tín hiệu nhu cầu cho danh mục sản phẩm mở rộng (src/products.py)
    customer_type                VARCHAR(12),
    occupation_group             VARCHAR(40),
    industry_group               VARCHAR(40),
    business_owner_flag          BOOLEAN,
    agri_flag                    BOOLEAN,
    family_flag                  BOOLEAN,
    auto_intent_flag             BOOLEAN,
    home_intent_flag             BOOLEAN,
    fx_active_flag               BOOLEAN,
    securities_value             DECIMAL(18,2),
    PRIMARY KEY (snapshot_date, customer_id)
);

-- AI_FEATURE_CUSTOMER  (feature set gọn cho model) ---------------
CREATE TABLE ai_feature_customer (
    snapshot_date                 DATE        NOT NULL,
    customer_id                   VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    product_propensity_feature_set JSONB,
    customer_value_score          DECIMAL(6,2),
    intent_signal_score           DECIMAL(6,2),
    engagement_score              DECIMAL(6,2),
    timing_score                  DECIMAL(6,2),
    relationship_score            DECIMAL(6,2),
    sales_suppression_flag        BOOLEAN,
    PRIMARY KEY (snapshot_date, customer_id)
);

-- =====================================================================
-- 5. MODEL OUTPUT
-- =====================================================================

-- AI_CUSTOMER_SCORE  (1 dòng / khách / sản phẩm / lần chạy model) ----
CREATE TABLE ai_customer_score (
    score_id                 VARCHAR(60) PRIMARY KEY,
    customer_id              VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    snapshot_date            DATE,
    product_id               VARCHAR(20) NOT NULL REFERENCES dim_product(product_id),
    model_id                 VARCHAR(30),
    model_version            VARCHAR(20),
    propensity_probability    DECIMAL(6,4),   -- 0..1
    product_propensity_score  DECIMAL(6,2),   -- 0..100
    customer_value_score      DECIMAL(6,2),
    intent_signal_score       DECIMAL(6,2),
    engagement_score          DECIMAL(6,2),
    timing_score              DECIMAL(6,2),
    relationship_score        DECIMAL(6,2),
    smart_growth_score        DECIMAL(6,2),   -- weighted 0..100
    priority_level            VARCHAR(20),    -- Very High / High / Medium / Low / Do not prioritize
    prediction_timestamp      TIMESTAMP
);

-- AI_SCORE_REASON  (explainability - "Why this customer") ----------
CREATE TABLE ai_score_reason (
    reason_id          VARCHAR(70) PRIMARY KEY,
    score_id           VARCHAR(60) NOT NULL REFERENCES ai_customer_score(score_id),
    feature_name       VARCHAR(60),
    feature_value      VARCHAR(60),
    contribution_score DECIMAL(9,4),
    impact_direction   VARCHAR(10),   -- POSITIVE / NEGATIVE
    reason_rank        INTEGER
);

-- =====================================================================
-- 6. DECISION LAYER
-- =====================================================================

-- AI_RECOMMENDATION  (Next Best Product / Action / Channel / Timing) --
CREATE TABLE ai_recommendation (
    recommendation_id        VARCHAR(60) PRIMARY KEY,
    customer_id              VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    score_id                 VARCHAR(60) NOT NULL REFERENCES ai_customer_score(score_id),
    recommended_product_id   VARCHAR(20) REFERENCES dim_product(product_id),
    recommended_action       VARCHAR(40),   -- RM_CALL / RM_ASSISTED_MESSAGE / IN_APP / PUSH / NURTURE / NO_CONTACT
    recommended_channel      VARCHAR(30),   -- RM_CALL / IN_APP / SMS / EMAIL / PUSH / ZALO
    recommended_timing       VARCHAR(40),   -- "Trong vòng 48 giờ" ...
    message_angle            VARCHAR(200),
    expected_conversion      DECIMAL(6,4),
    priority                 VARCHAR(20),
    suppression_flag         BOOLEAN,
    recommendation_timestamp TIMESTAMP,
    status                   VARCHAR(20),   -- NEW / SENT / RM_APPROVAL / BLOCKED / ACTIONED
    priority_rank            INTEGER        -- 1..3 trong Next Best Product
);

-- =====================================================================
-- 7. FEEDBACK LOOP
-- =====================================================================

-- RM_ACTION_FEEDBACK -------------------------------------------
CREATE TABLE rm_action_feedback (
    feedback_id            VARCHAR(60) PRIMARY KEY,
    recommendation_id      VARCHAR(60) NOT NULL REFERENCES ai_recommendation(recommendation_id),
    customer_id            VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    rm_id                  VARCHAR(20) REFERENCES dim_rm(rm_id),
    action_timestamp       TIMESTAMP,
    action_type            VARCHAR(40),   -- CALL / EMAIL / APPOINTMENT / SMS / NO_ACTION
    result_status          VARCHAR(30),   -- CONTACTED / NO_ANSWER / SCHEDULED / DONE
    customer_response      VARCHAR(30),   -- INTERESTED / NOT_INTERESTED / CALLBACK / APPLIED / CONVERTED / NO_RESPONSE
    appointment_flag       BOOLEAN,
    application_flag       BOOLEAN,
    converted_flag         BOOLEAN,
    reason_not_interested  VARCHAR(80),
    rm_feedback            VARCHAR(300),
    next_followup_date     DATE
);

-- =====================================================================
-- 8. TRAINING DATA
-- =====================================================================

-- ML_CREDIT_CARD_TRAINING_SET  (observation_date + forward label) ----
CREATE TABLE ml_credit_card_training_set (
    customer_id                VARCHAR(50) NOT NULL,
    observation_date           DATE        NOT NULL,
    age_group                  VARCHAR(20),
    segment                    VARCHAR(30),
    avg_balance_90d             DECIMAL(18,2),
    income_band                VARCHAR(20),
    spending_90d               DECIMAL(18,2),
    txn_count_90d              INTEGER,
    salary_flag                BOOLEAN,
    digital_score              DECIMAL(6,2),
    has_credit_card            BOOLEAN,
    campaign_response_history  DECIMAL(6,4),
    days_since_last_rm_contact INTEGER,
    complaint_count_30d        INTEGER,
    -- forward-looking labels (window = 90 days after observation_date)
    label_applied_90d          BOOLEAN,
    label_converted_90d        BOOLEAN,
    PRIMARY KEY (customer_id, observation_date)
);

-- ML_PROPENSITY_TRAINING_SET  (doc a. Product Propensity - logistic regression)
--   Z = b + w1*X1 + ... + w6*X6 ;  P = 1/(1+e^-Z) ;  score = P*100
--   y_holds_product = khách hiện đang sở hữu sản phẩm (nhãn train)
CREATE TABLE ml_propensity_training_set (
    customer_id           VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    snapshot_date         DATE        NOT NULL,
    product_id            VARCHAR(20) NOT NULL REFERENCES dim_product(product_id),
    product_group         VARCHAR(20),
    x1_monthly_spending   DECIMAL(9,6),   -- minmax(tổng chi tiêu/tháng)
    x2_income             DECIMAL(9,6),   -- minmax(thu nhập/tháng)
    x3_digital_activity   DECIMAL(9,6),   -- 40%ActiveDays + 35%DigitalTxn + 25%FeatureUsage
    x4_salary_account     DECIMAL(9,6),   -- nhận lương qua MSB (1/0)
    x5_campaign_response  DECIMAL(9,6),   -- 0 / .25 / .5 / .75 / .9 / 1 theo funnel
    x6_product_gap        DECIMAL(9,6),   -- 1 chưa có / 0.7 classic / 0 premium
    y_holds_product       INTEGER,        -- 1 nếu đang dùng sản phẩm (doc: y trạng thái đã dùng)
    y_adopt_next_90d      INTEGER,        -- nhãn train: 1 nếu mở sản phẩm trong 90 ngày tới (non-holder)
    PRIMARY KEY (customer_id, snapshot_date, product_id)
);

-- AI_MODEL_REGISTRY  (metadata mỗi model đã train)
CREATE TABLE ai_model_registry (
    model_id        VARCHAR(40) NOT NULL,
    model_version   VARCHAR(20) NOT NULL,
    product_id      VARCHAR(20) REFERENCES dim_product(product_id),
    algorithm       VARCHAR(40),
    trained_at      TIMESTAMP,
    n_train         INTEGER,
    n_test          INTEGER,
    positive_rate   DECIMAL(9,6),
    feature_list    VARCHAR(300),
    PRIMARY KEY (model_id, model_version, product_id)
);

-- AI_MODEL_COEFFICIENT  (b, w1..w6 model học được - explainability)
CREATE TABLE ai_model_coefficient (
    model_id        VARCHAR(40) NOT NULL,
    model_version   VARCHAR(20) NOT NULL,
    product_id      VARCHAR(20) NOT NULL,
    feature_name    VARCHAR(40) NOT NULL,   -- 'intercept' | 'x1_monthly_spending' | ...
    coefficient     DECIMAL(14,6),
    odds_ratio      DECIMAL(18,6),
    PRIMARY KEY (model_id, model_version, product_id, feature_name)
);

-- AI_MODEL_METRIC  (kết quả test model)
CREATE TABLE ai_model_metric (
    model_id        VARCHAR(40) NOT NULL,
    model_version   VARCHAR(20) NOT NULL,
    product_id      VARCHAR(20) NOT NULL,
    dataset_split   VARCHAR(20) NOT NULL,   -- 'train' | 'test' | 'cv'
    metric_name     VARCHAR(40) NOT NULL,   -- 'roc_auc' | 'pr_auc' | 'log_loss' | ...
    metric_value    DECIMAL(14,6),
    PRIMARY KEY (model_id, model_version, product_id, dataset_split, metric_name)
);

-- =====================================================================
-- 9. INDEXES
-- =====================================================================
CREATE INDEX ix_casa_cust      ON fact_casa_daily(customer_id);
CREATE INDEX ix_dep_cust       ON fact_deposit(customer_id);
CREATE INDEX ix_txn_cust       ON fact_transaction(customer_id);
CREATE INDEX ix_txn_date       ON fact_transaction(transaction_date);
CREATE INDEX ix_loan_cust      ON fact_loan(customer_id);
CREATE INDEX ix_ins_cust       ON fact_insurance(customer_id);
CREATE INDEX ix_dig_cust       ON fact_digital_activity(customer_id);
CREATE INDEX ix_crm_cust       ON fact_crm_interaction(customer_id);
CREATE INDEX ix_svc_cust       ON fact_customer_service(customer_id);
CREATE INDEX ix_camp_cust      ON fact_campaign(customer_id);
CREATE INDEX ix_camp_camp      ON fact_campaign(campaign_id);
CREATE INDEX ix_cardm_cust     ON fact_card_monthly(customer_id);
CREATE INDEX ix_score_cust     ON ai_customer_score(customer_id);
CREATE INDEX ix_score_prod     ON ai_customer_score(product_id);
CREATE INDEX ix_score_sgs      ON ai_customer_score(smart_growth_score);
CREATE INDEX ix_reason_score   ON ai_score_reason(score_id);
CREATE INDEX ix_reco_cust      ON ai_recommendation(customer_id);
CREATE INDEX ix_reco_score     ON ai_recommendation(score_id);
CREATE INDEX ix_fb_reco        ON rm_action_feedback(recommendation_id);
CREATE INDEX ix_dimcust_branch ON dim_customer(branch_id);
CREATE INDEX ix_proptrain_prod ON ml_propensity_training_set(product_id);
CREATE INDEX ix_modelcoef_prod ON ai_model_coefficient(product_id);
CREATE INDEX ix_modelmetric    ON ai_model_metric(product_id, dataset_split);

-- =====================================================================
-- 10. JOURNEY VIEWS  (Decision Gate 1 + Top-20)
-- =====================================================================

-- Eligibility per (customer, target product) - "Lọc tệp KH đủ điều kiện KD"
CREATE OR REPLACE VIEW v_customer_product_eligibility AS
WITH held AS (
    SELECT customer_id, 'CREDIT_CARD'::text grp FROM dim_card
      WHERE card_type = 'CREDIT' AND card_tier = 'PLATINUM' AND card_status = 'ACTIVE'
    UNION ALL
    SELECT customer_id, 'LOAN' FROM fact_loan WHERE loan_status = 'ACTIVE'
    UNION ALL
    SELECT customer_id, 'DEPOSIT' FROM fact_deposit WHERE deposit_status IN ('ACTIVE','MATURING_SOON')
    UNION ALL
    SELECT DISTINCT customer_id, 'INVESTMENT' FROM fact_campaign
      WHERE product_code LIKE 'INVEST%' AND converted_flag
)
SELECT
    m.snapshot_date,
    c.customer_id,
    c.branch_id,
    p.product_group                                   AS target_product_group,
    p.product_id                                      AS target_product_id,
    (c.customer_active_flag AND c.kyc_status = 'VERIFIED')                         AS rule_customer_active,
    (h.customer_id IS NULL)                                                        AS rule_target_product_not_held,
    COALESCE(c.marketing_consent_flag, false)                                      AS rule_marketing_consent,
    (NOT COALESCE(c.do_not_contact_flag, false))                                   AS rule_not_do_not_contact,
    (NOT COALESCE(m.serious_complaint_15d, false))                                 AS rule_no_serious_complaint_15d,
    (NOT COALESCE(m.recent_rejection_30d_flag, false))                             AS rule_no_recent_rejection_30d,
    (COALESCE(m.contact_count_7d, 0) < 2)                                          AS rule_contact_frequency_ok,
    (
        c.customer_active_flag AND c.kyc_status = 'VERIFIED'
        AND h.customer_id IS NULL
        AND COALESCE(c.marketing_consent_flag, false)
        AND NOT COALESCE(c.do_not_contact_flag, false)
        AND NOT COALESCE(m.serious_complaint_15d, false)
        AND NOT COALESCE(m.recent_rejection_30d_flag, false)
        AND COALESCE(m.contact_count_7d, 0) < 2
    )                                                                             AS is_eligible
FROM customer_360_feature_mart m
JOIN dim_customer c ON c.customer_id = m.customer_id
CROSS JOIN dim_product p
LEFT JOIN held h ON h.customer_id = c.customer_id AND h.grp = p.product_group
WHERE p.product_id IN ('P001','P004','P007','P009');   -- 4 sản phẩm giai đoạn đầu

-- Top opportunities = eligible khách + smart_growth_score, sort DESC
CREATE OR REPLACE VIEW v_top_opportunities AS
SELECT
    e.branch_id,
    dp.product_group,
    s.customer_id,
    dc.customer_segment,
    dc.age_group,
    dc.income_band,
    s.smart_growth_score,
    s.priority_level,
    s.propensity_probability,
    r.recommended_action,
    r.recommended_channel,
    r.recommended_timing,
    r.message_angle,
    row_number() OVER (
        PARTITION BY e.branch_id, dp.product_group
        ORDER BY s.smart_growth_score DESC, s.propensity_probability DESC
    ) AS branch_product_rank
FROM ai_customer_score s
JOIN dim_product dp ON dp.product_id = s.product_id
JOIN v_customer_product_eligibility e
      ON e.customer_id = s.customer_id AND e.target_product_id = s.product_id
JOIN dim_customer dc ON dc.customer_id = s.customer_id
LEFT JOIN ai_recommendation r ON r.score_id = s.score_id   -- reco for THIS product (if in top-3)
WHERE e.is_eligible
  AND s.smart_growth_score >= 60;
