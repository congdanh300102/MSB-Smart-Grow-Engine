-- =====================================================================
-- MSB SMART GROWTH ENGINE — AI Feedback Agent (Action 11–12)
-- RM phản hồi đề xuất AI  →  agent xem xét lại mô hình / giải thích / hiệu chỉnh
-- Sinh bởi: src/rm_feedback_agent.py [--apply]
-- Chạy sau 05_product_catalogue.sql:  psql ... -f sql/06_feedback_agent.sql
-- =====================================================================
SET search_path TO msb_sge;

DROP TABLE IF EXISTS ai_model_adjustment, ai_agent_review, rm_feedback_ai CASCADE;

CREATE TABLE rm_feedback_ai (
    feedback_id     VARCHAR(16) PRIMARY KEY,
    customer_id     VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    product_id      VARCHAR(12) REFERENCES dim_product_catalogue(product_id),
    product_code    VARCHAR(40),
    product_group   VARCHAR(20),
    segment         VARCHAR(30),
    priority_rank   INTEGER,
    priority_level  VARCHAR(20),
    propensity      DECIMAL(6,4),
    rm_id           VARCHAR(20),
    rm_verdict      VARCHAR(20),   -- AGREE / NOT_RELEVANT / CANT_AFFORD / ALREADY_HAS / WRONG_TIMING / NO_NEED_NOW
    agree_flag      BOOLEAN,
    rm_comment      VARCHAR(200),
    converted_flag  BOOLEAN,
    feedback_date   DATE
);

CREATE TABLE ai_agent_review (
    review_id             VARCHAR(12) PRIMARY KEY,
    run_date              DATE,
    scope                 VARCHAR(20),   -- product / product+segment
    product_code          VARCHAR(40),
    product_group         VARCHAR(20),
    segment               VARCHAR(30),
    n_feedback            INTEGER,
    agree_rate            DECIMAL(6,3),
    dominant_verdict      VARCHAR(20),
    dominant_share        DECIMAL(6,3),
    conversion_rate_acted DECIMAL(6,3),
    finding_type          VARCHAR(30),   -- RULE_TOO_LOOSE[_SEGMENT] / GATE_INCOME_TOO_LOOSE / HOLDINGS_GATE_MISSING / RM_SKEPTICISM / TIMING_RULE / DEMAND_SOFT / INSUFFICIENT_SIGNAL
    model_is_wrong        BOOLEAN,       -- agent kết luận "mô hình sai thật"
    explanation           TEXT,
    proposed_fix          TEXT,          -- JSON {kind, product_code, from, to, ...}
    impact_recos          INTEGER,
    status                VARCHAR(12)    -- PROPOSED / APPLIED / NO_ACTION
);

CREATE TABLE ai_model_adjustment (
    adjustment_id  VARCHAR(12) PRIMARY KEY,
    review_id      VARCHAR(12) REFERENCES ai_agent_review(review_id),
    run_date       DATE,
    product_code   VARCHAR(40),
    finding_type   VARCHAR(30),
    kind           VARCHAR(20),   -- min_fit / min_income / block_segments
    before         TEXT,
    after          TEXT,
    impact_recos   INTEGER
);

\copy rm_feedback_ai (feedback_id,customer_id,product_id,product_code,product_group,segment,priority_rank,priority_level,propensity,rm_id,rm_verdict,agree_flag,rm_comment,converted_flag,feedback_date) FROM '/data/csv/rm_feedback_ai.csv' WITH (FORMAT csv, HEADER true, NULL '');
\copy ai_agent_review (review_id,run_date,scope,product_code,product_group,segment,n_feedback,agree_rate,dominant_verdict,dominant_share,conversion_rate_acted,finding_type,model_is_wrong,explanation,proposed_fix,impact_recos,status) FROM '/data/csv/ai_agent_review.csv' WITH (FORMAT csv, HEADER true, NULL '');
\copy ai_model_adjustment (adjustment_id,review_id,run_date,product_code,finding_type,kind,before,after,impact_recos) FROM '/data/csv/ai_model_adjustment.csv' WITH (FORMAT csv, HEADER true, NULL '');

CREATE INDEX ix_rfb_cust ON rm_feedback_ai(customer_id);
CREATE INDEX ix_rfb_prod ON rm_feedback_ai(product_code);
ANALYZE;

-- ---------------------------------------------------------------------
-- Truy vấn mẫu
-- ---------------------------------------------------------------------
-- Q1. Đồng thuận RM ↔ AI theo nhóm sản phẩm
-- SELECT product_group, count(*) n, round(avg(agree_flag::int)*100,1) agree_pct,
--        round(avg(converted_flag::int)*100,1) conv_pct
-- FROM rm_feedback_ai GROUP BY 1 ORDER BY agree_pct;

-- Q2. Các vùng agent kết luận "mô hình sai thật" + fix
-- SELECT product_code, segment, finding_type, agree_rate, dominant_verdict,
--        impact_recos, proposed_fix
-- FROM ai_agent_review WHERE model_is_wrong ORDER BY impact_recos DESC;

-- Q3. Lịch sử hiệu chỉnh mô hình
-- SELECT * FROM ai_model_adjustment ORDER BY run_date, product_code;

-- Q4. Heatmap đồng thuận sản phẩm × phân khúc (đỏ = AI hay sai)
-- SELECT product_code, segment, count(*) n, round(avg(agree_flag::int),2) agree
-- FROM rm_feedback_ai GROUP BY 1,2 HAVING count(*) >= 25 ORDER BY agree;
