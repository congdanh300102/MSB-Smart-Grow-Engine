-- =====================================================================
-- MSB SMART GROWTH ENGINE - sample analytical queries
-- Each block answers one of the 5 core business questions from the brief.
-- =====================================================================
SET search_path TO msb_sge;

-- ---------------------------------------------------------------------
-- Q1. Khách hàng nào nên được ưu tiên tiếp cận?
--     -> Opportunity list: top khách theo next_best_action_score
-- ---------------------------------------------------------------------
SELECT c.customer_id, c.full_name, c.segment_code, c.region,
       m.total_balance, m.total_transactions_30d,
       s.next_best_action_score, s.propensity_to_buy_score, s.churn_score
FROM ai_customer_score s
JOIN customer_360_feature_mart m ON m.customer_360_feature_key = s.customer_360_feature_key
JOIN dim_customer c              ON c.customer_key = m.customer_key
ORDER BY s.next_best_action_score DESC
LIMIT 20;

-- ---------------------------------------------------------------------
-- Q2. Khách hàng đang quan tâm sản phẩm nào? (Next Best Product)
-- ---------------------------------------------------------------------
SELECT c.customer_id, c.full_name,
       r.priority_rank, r.confidence_score, r.recommendation_text
FROM ai_recommendation r
JOIN ai_customer_score s ON s.ai_customer_score_key = r.ai_customer_score_key
JOIN customer_360_feature_mart m ON m.customer_360_feature_key = s.customer_360_feature_key
JOIN dim_customer c ON c.customer_key = m.customer_key
WHERE c.customer_key = 1
ORDER BY r.priority_rank;

-- ---------------------------------------------------------------------
-- Q3. Xác suất phản hồi / chuyển đổi
--     -> propensity_to_buy_score + lịch sử phản hồi campaign thực tế
-- ---------------------------------------------------------------------
SELECT round(s.propensity_to_buy_score, 2) AS propensity_bucket,
       count(*)                              AS customers,
       round(avg(m.campaign_response_rate), 1) AS avg_hist_response_rate_pct
FROM ai_customer_score s
JOIN customer_360_feature_mart m ON m.customer_360_feature_key = s.customer_360_feature_key
GROUP BY 1 ORDER BY 1;

-- ---------------------------------------------------------------------
-- Q4. Thời điểm & kênh tiếp cận phù hợp
--     -> kênh campaign có tỉ lệ phản hồi cao nhất theo phân khúc
-- ---------------------------------------------------------------------
SELECT c.segment_code, fc.campaign_channel,
       count(*)                                        AS targeted,
       round(100.0 * avg(fc.response_flag::int), 1)    AS response_rate_pct,
       round(100.0 * avg(fc.conversion_flag::int), 1)  AS conversion_rate_pct
FROM fact_campaign fc
JOIN dim_customer c ON c.customer_key = fc.customer_key
GROUP BY 1, 2
HAVING count(*) > 30
ORDER BY 1, response_rate_pct DESC;

-- ---------------------------------------------------------------------
-- Q5. Sale nên làm gì tiếp theo? (Next Best Action + feedback loop)
-- ---------------------------------------------------------------------
SELECT f.action_taken, f.action_outcome,
       count(*)                        AS n,
       round(avg(f.rating), 2)         AS avg_rating,
       round(avg(r.confidence_score), 3) AS avg_ai_confidence
FROM rm_action_feedback f
JOIN ai_recommendation r ON r.ai_recommendation_key = f.ai_recommendation_key
GROUP BY 1, 2
ORDER BY n DESC;

-- ---------------------------------------------------------------------
-- Bonus. AI dynamic segmentation (phân khúc động)
--        Xếp khách theo tổ hợp score, không dùng segment tĩnh.
-- ---------------------------------------------------------------------
SELECT CASE
         WHEN s.churn_score > 0.6                                   THEN 'CHURN_RISK'
         WHEN s.credit_risk_score < 0.35 AND s.propensity_to_buy_score > 0.6
              AND m.total_loan_outstanding = 0                       THEN 'CREDIT_OPPORTUNITY'
         WHEN m.total_deposit_value > 0 AND s.propensity_to_buy_score > 0.6 THEN 'POTENTIAL_INVESTOR'
         WHEN m.digital_engagement_score > 40                        THEN 'DIGITAL_ACTIVE'
         WHEN m.total_balance > 300000000                            THEN 'HIGH_VALUE'
         ELSE 'STANDARD'
       END AS ai_segment,
       count(*) AS customers,
       round(avg(m.total_balance)) AS avg_balance,
       round(avg(s.next_best_action_score), 3) AS avg_nba
FROM ai_customer_score s
JOIN customer_360_feature_mart m ON m.customer_360_feature_key = s.customer_360_feature_key
GROUP BY 1 ORDER BY customers DESC;
