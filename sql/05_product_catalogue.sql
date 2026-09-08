-- =====================================================================
-- MSB SMART GROWTH ENGINE — danh mục sản phẩm mở rộng + phân tích phù hợp
-- Nguồn: MSB_products_description.xlsx (958 mã) -> 35 sản phẩm chuẩn hoá
-- Sinh bởi: src/products.py + src/product_analysis.py
-- Chạy sau 01_schema.sql / 02_load.sql:
--   psql ... -f sql/05_product_catalogue.sql
-- =====================================================================
SET search_path TO msb_sge;

DROP TABLE IF EXISTS ai_product_recommendation_v2, ai_product_fit,
    fact_customer_product_holding, agg_product_demand, dim_product_catalogue CASCADE;

CREATE TABLE dim_product_catalogue (
    product_id       VARCHAR(12) PRIMARY KEY,
    product_code     VARCHAR(40) NOT NULL UNIQUE,
    product_name     VARCHAR(120),
    product_group    VARCHAR(20),   -- CARD / CASA / FD / LENDING
    subgroup         VARCHAR(30),
    product_tier     VARCHAR(20),
    customer_type    VARCHAR(10),   -- IND / SME / BOTH
    target_need      VARCHAR(300),  -- "Khách hàng/Nhu cầu phù hợp"
    base_propensity  DECIMAL(6,4),
    active_flag      BOOLEAN
);

CREATE TABLE fact_customer_product_holding (
    customer_id  VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    product_id   VARCHAR(12) NOT NULL REFERENCES dim_product_catalogue(product_id),
    open_date    DATE,
    status       VARCHAR(20),
    balance      DECIMAL(18,2),
    PRIMARY KEY (customer_id, product_id)
);

CREATE TABLE ai_product_fit (
    customer_id        VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    product_id         VARCHAR(12) NOT NULL REFERENCES dim_product_catalogue(product_id),
    product_code       VARCHAR(40),
    product_group      VARCHAR(20),
    eligible           BOOLEAN,
    held               BOOLEAN,
    fit_score          DECIMAL(6,4),   -- 0..1 khớp rule "khách hàng phù hợp"
    propensity         DECIMAL(6,4),   -- hybrid LR + rule
    smart_growth_score DECIMAL(6,2),
    priority_level     VARCHAR(20),
    reason_1           VARCHAR(80),
    reason_2           VARCHAR(80),
    reason_3           VARCHAR(80),
    PRIMARY KEY (customer_id, product_id)
);

CREATE TABLE ai_product_recommendation_v2 (
    customer_id         VARCHAR(50) NOT NULL REFERENCES dim_customer(customer_id),
    priority_rank       INTEGER,
    product_id          VARCHAR(12) REFERENCES dim_product_catalogue(product_id),
    product_code        VARCHAR(40),
    product_name        VARCHAR(120),
    product_group       VARCHAR(20),
    propensity          DECIMAL(6,4),
    fit_score           DECIMAL(6,4),
    smart_growth_score  DECIMAL(6,2),
    priority_level      VARCHAR(20),
    reason_1            VARCHAR(80),
    reason_2            VARCHAR(80),
    reason_3            VARCHAR(80),
    expected_conversion DECIMAL(6,4),
    PRIMARY KEY (customer_id, priority_rank)
);

CREATE TABLE agg_product_demand (
    product_id            VARCHAR(12) REFERENCES dim_product_catalogue(product_id),
    product_code          VARCHAR(40),
    product_group         VARCHAR(20),
    segment               VARCHAR(30),
    eligible_customers    INTEGER,
    avg_propensity        DECIMAL(6,4),
    high_propensity       INTEGER,
    current_holders       INTEGER,
    expected_adopters_90d DECIMAL(12,1),
    expected_adopters_30d DECIMAL(12,1),
    expected_adopters_60d DECIMAL(12,1),
    PRIMARY KEY (product_id, segment)
);

\copy dim_product_catalogue (product_id,product_code,product_name,product_group,subgroup,product_tier,customer_type,target_need,base_propensity,active_flag) FROM '/data/csv/dim_product_catalogue.csv' WITH (FORMAT csv, HEADER true, NULL '');
\copy fact_customer_product_holding (customer_id,product_id,open_date,status,balance) FROM '/data/csv/fact_customer_product_holding.csv' WITH (FORMAT csv, HEADER true, NULL '');
\copy ai_product_fit (customer_id,product_id,product_code,product_group,eligible,held,fit_score,propensity,smart_growth_score,priority_level,reason_1,reason_2,reason_3) FROM '/data/csv/ai_product_fit.csv' WITH (FORMAT csv, HEADER true, NULL '');
\copy ai_product_recommendation_v2 (customer_id,priority_rank,product_id,product_code,product_name,product_group,propensity,fit_score,smart_growth_score,priority_level,reason_1,reason_2,reason_3,expected_conversion) FROM '/data/csv/ai_product_recommendation_v2.csv' WITH (FORMAT csv, HEADER true, NULL '');
\copy agg_product_demand (product_id,product_code,product_group,segment,eligible_customers,avg_propensity,high_propensity,current_holders,expected_adopters_90d,expected_adopters_30d,expected_adopters_60d) FROM '/data/csv/agg_product_demand.csv' WITH (FORMAT csv, HEADER true, NULL '');

CREATE INDEX ix_pfit_prod ON ai_product_fit(product_id);
CREATE INDEX ix_preco_cust ON ai_product_recommendation_v2(customer_id);
ANALYZE;

-- ---------------------------------------------------------------------
-- Truy vấn mẫu
-- ---------------------------------------------------------------------
-- Q1. Mỗi khách phù hợp nhất với sản phẩm nào (top-3, chưa sở hữu)
-- SELECT * FROM ai_product_recommendation_v2 WHERE priority_rank <= 3 ORDER BY customer_id, priority_rank;

-- Q2. Dự báo cầu 90 ngày theo sản phẩm
-- SELECT product_code, sum(eligible_customers) elig, sum(expected_adopters_90d) exp90
-- FROM agg_product_demand GROUP BY 1 ORDER BY exp90 DESC;

-- Q3. Sản phẩm #1 phổ biến nhất
-- SELECT product_code, count(*) n FROM ai_product_recommendation_v2
-- WHERE priority_rank = 1 GROUP BY 1 ORDER BY n DESC LIMIT 15;

-- Q4. Khoảng trống danh mục theo phân khúc
-- SELECT c.customer_segment, count(*) whitespace
-- FROM ai_product_fit f JOIN dim_customer c USING (customer_id)
-- WHERE f.eligible AND NOT f.held AND f.propensity >= 0.5
-- GROUP BY 1 ORDER BY 2 DESC;
