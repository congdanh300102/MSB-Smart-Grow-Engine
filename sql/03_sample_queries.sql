-- =====================================================================
-- MSB SMART GROWTH ENGINE - sample analytical queries (Data Model v2)
-- Phần A: 5 bài toán cốt lõi của đề bài
-- Phần B: 12 Actions của "Thiết kế hành trình AI"
-- Phần C: Xác định Top-20 khách hàng (2 tầng lọc) + Smart Growth Score
-- =====================================================================
SET search_path TO msb_sge;

-- =====================================================================
-- A. 5 BÀI TOÁN CỐT LÕI
-- =====================================================================

-- A1. Khách hàng nào nên được ưu tiên tiếp cận? (per RM, sort theo Smart Growth Score)
SELECT s.customer_id, c.customer_segment, c.branch_id, c.rm_id,
       dp.product_name, s.smart_growth_score, s.priority_level, s.propensity_probability
FROM ai_customer_score s
JOIN dim_customer c  ON c.customer_id = s.customer_id
JOIN dim_product dp  ON dp.product_id = s.product_id
WHERE s.smart_growth_score = (
      SELECT max(s2.smart_growth_score) FROM ai_customer_score s2 WHERE s2.customer_id = s.customer_id)
ORDER BY s.smart_growth_score DESC
LIMIT 20;

-- A2. Khách hàng đang quan tâm sản phẩm nào? (Next Best Product, top-3/khách)
SELECT r.customer_id, r.priority_rank, dp.product_name, r.expected_conversion, r.message_angle
FROM ai_recommendation r
JOIN dim_product dp ON dp.product_id = r.recommended_product_id
WHERE r.customer_id = (SELECT customer_id FROM dim_customer ORDER BY customer_id LIMIT 1)
ORDER BY r.priority_rank;

-- A3. Xác suất phản hồi / chuyển đổi theo dải propensity
SELECT width_bucket(s.propensity_probability, 0, 1, 10) / 10.0 AS propensity_bucket,
       count(*) customers,
       round(100.0 * avg(m.previous_campaign_response), 1) AS avg_hist_response_rate_pct
FROM ai_customer_score s
JOIN customer_360_feature_mart m ON m.customer_id = s.customer_id
GROUP BY 1 ORDER BY 1;

-- A4. Thời điểm & kênh tiếp cận phù hợp theo phân khúc
SELECT c.customer_segment, r.recommended_channel, r.recommended_timing,
       count(*) AS recommendations,
       round(100.0 * avg(r.expected_conversion), 1) AS avg_expected_conversion_pct
FROM ai_recommendation r
JOIN dim_customer c ON c.customer_id = r.customer_id
WHERE r.priority_rank = 1 AND NOT r.suppression_flag
GROUP BY 1, 2, 3
HAVING count(*) > 20
ORDER BY 1, avg_expected_conversion_pct DESC;

-- A5. Sale nên làm gì tiếp theo + hiệu quả feedback loop
SELECT f.action_type, f.customer_response, count(*) n,
       round(avg(r.expected_conversion), 3) avg_ai_expected_conversion,
       round(100.0 * avg(f.converted_flag::int), 1) actual_conversion_rate_pct
FROM rm_action_feedback f
JOIN ai_recommendation r ON r.recommendation_id = f.recommendation_id
GROUP BY 1, 2
ORDER BY n DESC;

-- =====================================================================
-- B. 12 ACTIONS - "Thiết kế hành trình AI"
-- =====================================================================

-- Action 1: Nhận tín hiệu khách hàng (Customer 360 snapshot mới nhất)
SELECT * FROM customer_360_feature_mart WHERE snapshot_date = (SELECT max(snapshot_date) FROM customer_360_feature_mart) LIMIT 5;

-- Action 2: AI phát hiện nhu cầu / opportunity (propensity + smart growth + priority)
SELECT s.customer_id, dp.product_group, s.product_propensity_score, s.smart_growth_score,
       CASE WHEN s.smart_growth_score >= 80 THEN 'High Opportunity' ELSE 'Standard' END AS opportunity_flag
FROM ai_customer_score s JOIN dim_product dp ON dp.product_id = s.product_id
ORDER BY s.smart_growth_score DESC LIMIT 10;

-- Action 3: Decision Gate 1 - Eligible / Suppress / Exclude  (xem view v_customer_product_eligibility)
SELECT is_eligible,
       CASE WHEN NOT rule_marketing_consent OR NOT rule_customer_active THEN 'EXCLUDE'
            WHEN NOT rule_not_do_not_contact OR NOT rule_no_serious_complaint_15d
                 OR NOT rule_no_recent_rejection_30d OR NOT rule_contact_frequency_ok THEN 'SUPPRESS'
            ELSE 'ELIGIBLE' END AS gate1_result,
       count(*) AS customers
FROM v_customer_product_eligibility
GROUP BY 1, 2 ORDER BY 3 DESC;

-- Action 4: Next Best Product - top 3 trong nhóm đủ điều kiện
SELECT e.customer_id, r.priority_rank, dp.product_name, r.expected_conversion
FROM v_customer_product_eligibility e
JOIN ai_customer_score s ON s.customer_id = e.customer_id AND s.product_id = e.target_product_id
JOIN ai_recommendation r ON r.score_id = s.score_id
JOIN dim_product dp ON dp.product_id = r.recommended_product_id
WHERE e.is_eligible
ORDER BY e.customer_id, r.priority_rank
LIMIT 30;

-- Action 5: Next Best Action theo nhóm khách (High Value / Digital Active / Low Priority)
SELECT c.customer_segment, r.recommended_action, count(*) AS n
FROM ai_recommendation r JOIN dim_customer c ON c.customer_id = r.customer_id
WHERE r.priority_rank = 1
GROUP BY 1, 2 ORDER BY 1, n DESC;

-- Action 6: Kênh & thời điểm tiếp cận
SELECT recommended_channel, recommended_timing, count(*) n
FROM ai_recommendation WHERE priority_rank = 1 AND NOT suppression_flag
GROUP BY 1, 2 ORDER BY n DESC;

-- Action 7: Nội dung cá nhân hoá (message_angle theo sản phẩm/khách)
SELECT r.customer_id, dp.product_name, r.message_angle
FROM ai_recommendation r JOIN dim_product dp ON dp.product_id = r.recommended_product_id
WHERE r.priority_rank = 1 LIMIT 10;

-- Action 8/9: Decision Gate 2 & 3 - trạng thái message trước khi gửi
SELECT status, suppression_flag, count(*) n
FROM ai_recommendation WHERE priority_rank = 1
GROUP BY 1, 2 ORDER BY n DESC;

-- Action 10/11: Message Orchestration & phản hồi khách hàng (funnel campaign thực tế)
SELECT channel,
       count(*) sent,
       round(100.0 * avg(delivered_flag::int), 1) delivered_pct,
       round(100.0 * avg(opened_flag::int), 1) opened_pct,
       round(100.0 * avg(clicked_flag::int), 1) clicked_pct,
       round(100.0 * avg(responded_flag::int), 1) responded_pct,
       round(100.0 * avg(applied_flag::int), 1) applied_pct,
       round(100.0 * avg(converted_flag::int), 1) converted_pct
FROM fact_campaign GROUP BY 1 ORDER BY sent DESC;

-- Action 12: Learning loop - feedback thực tế quay lại training set
SELECT label_applied_90d, label_converted_90d, count(*) n,
       round(avg(digital_score), 1) avg_digital_score,
       round(100.0 * avg(campaign_response_history), 1) avg_campaign_response_pct
FROM ml_credit_card_training_set GROUP BY 1, 2 ORDER BY n DESC;

-- =====================================================================
-- C. TOP-20 CƠ HỘI (2 tầng lọc + Smart Growth Score)
-- =====================================================================

-- C1. Top 20 toàn hàng cho 1 sản phẩm (vd Platinum Credit Card)
SELECT * FROM v_top_opportunities
WHERE product_group = 'CREDIT_CARD'
ORDER BY smart_growth_score DESC
LIMIT 20;

-- C2. Top 20 theo từng chi nhánh x sản phẩm (dùng cho RM Opportunity Desk)
SELECT * FROM v_top_opportunities
WHERE branch_product_rank <= 20
ORDER BY branch_id, product_group, branch_product_rank;

-- C3. Phân bố priority_level (đối chiếu bảng "Phân nhóm Score" trong tài liệu)
SELECT priority_level, count(*) customers,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
FROM (SELECT customer_id, priority_level,
             row_number() OVER (PARTITION BY customer_id ORDER BY smart_growth_score DESC) rn
      FROM ai_customer_score) x
WHERE rn = 1
GROUP BY 1
ORDER BY CASE priority_level WHEN 'Very High' THEN 1 WHEN 'High' THEN 2
              WHEN 'Medium' THEN 3 WHEN 'Low' THEN 4 ELSE 5 END;

-- =====================================================================
-- D. PRODUCT PROPENSITY - LOGISTIC REGRESSION  (doc "a. Product Propensity")
--    Cần chạy trước:  py src/train_models.py --apply  +  sql/04_load_models.sql
-- =====================================================================

-- D1. Hệ số b, w1..w6 model học được (Z = b + Σ wi*Xi)  + odds ratio
SELECT c.product_id, dp.product_name, c.feature_name, c.coefficient, c.odds_ratio
FROM ai_model_coefficient c JOIN dim_product dp ON dp.product_id = c.product_id
WHERE c.model_version = (SELECT max(model_version) FROM ai_model_coefficient)
ORDER BY c.product_id,
         CASE c.feature_name WHEN 'intercept' THEN 0 ELSE 1 END, c.feature_name;

-- D2. Kết quả test model (ROC-AUC / PR-AUC / LogLoss / Accuracy ... theo split)
SELECT m.product_id, dp.product_name, m.dataset_split, m.metric_name, m.metric_value
FROM ai_model_metric m JOIN dim_product dp ON dp.product_id = m.product_id
WHERE m.metric_name IN ('roc_auc','pr_auc','log_loss','brier','accuracy','f1','ks','positive_rate')
ORDER BY m.product_id, m.dataset_split, m.metric_name;

-- D3. Calibration: P dự đoán vs tỉ lệ adoption thực tế (nhãn model) theo decile,
--     chỉ trên tệp CHƯA sở hữu (đúng tệp train). P càng cao -> adoption thực càng cao.
SELECT dp.product_name,
       width_bucket(s.propensity_probability, 0, 1, 10) AS propensity_decile,
       count(*) AS customers,
       round(avg(s.propensity_probability), 3) AS avg_predicted_p,
       round(avg(t.y_adopt_next_90d::numeric), 3) AS actual_adopt_rate
FROM ai_customer_score s
JOIN dim_product dp ON dp.product_id = s.product_id
JOIN ml_propensity_training_set t
     ON t.customer_id = s.customer_id AND t.product_id = s.product_id
WHERE t.y_holds_product = 0
GROUP BY 1, 2 ORDER BY 1, 2;

-- D4. Model vs heuristic: Top-N cơ hội Credit Card có bị "đảo" nhiều không?
SELECT s.customer_id, s.propensity_probability, s.product_propensity_score,
       s.smart_growth_score, s.priority_level,
       t.x1_monthly_spending, t.x5_campaign_response, t.x6_product_gap
FROM ai_customer_score s
JOIN ml_propensity_training_set t
     ON t.customer_id = s.customer_id AND t.product_id = s.product_id
WHERE s.product_id = 'P001' AND t.y_holds_product = 0
ORDER BY s.propensity_probability DESC
LIMIT 20;

-- D5. Feature X1..X6: phân bố + tương quan với nhãn y_holds_product
SELECT product_group,
       round(avg(x1_monthly_spending), 3) x1, round(avg(x2_income), 3) x2,
       round(avg(x3_digital_activity), 3) x3, round(avg(x4_salary_account), 3) x4,
       round(avg(x5_campaign_response), 3) x5, round(avg(x6_product_gap), 3) x6,
       round(avg(y_holds_product::numeric), 3) hold_rate,
       round(avg(y_adopt_next_90d::numeric), 3) adopt_rate
FROM ml_propensity_training_set GROUP BY 1 ORDER BY 1;
