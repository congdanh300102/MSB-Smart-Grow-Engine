-- =====================================================================
-- MSB SMART GROWTH ENGINE - load logistic-regression model artefacts
-- Run AFTER:  py src/train_models.py --apply
--   psql ... -f sql/04_load_models.sql
-- Also reloads ai_customer_score / ai_recommendation which --apply rewrote
-- with model-based propensity.
-- =====================================================================
SET search_path TO msb_sge;

TRUNCATE ai_model_metric, ai_model_coefficient, ai_model_registry RESTART IDENTITY;

\copy ai_model_registry (model_id,model_version,product_id,algorithm,trained_at,n_train,n_test,positive_rate,feature_list) FROM '/data/csv/ai_model_registry.csv' WITH (FORMAT csv, HEADER true, NULL '');
\copy ai_model_coefficient (model_id,model_version,product_id,feature_name,coefficient,odds_ratio) FROM '/data/csv/ai_model_coefficient.csv' WITH (FORMAT csv, HEADER true, NULL '');
\copy ai_model_metric (model_id,model_version,product_id,dataset_split,metric_name,metric_value) FROM '/data/csv/ai_model_metric.csv' WITH (FORMAT csv, HEADER true, NULL '');

-- refresh scores/recommendations rewritten by --apply
TRUNCATE rm_action_feedback, ai_recommendation, ai_score_reason, ai_customer_score RESTART IDENTITY CASCADE;
\copy ai_customer_score (score_id,customer_id,snapshot_date,product_id,model_id,model_version,propensity_probability,product_propensity_score,customer_value_score,intent_signal_score,engagement_score,timing_score,relationship_score,smart_growth_score,priority_level,prediction_timestamp) FROM '/data/csv/ai_customer_score.csv' WITH (FORMAT csv, HEADER true, NULL '');
\copy ai_score_reason (reason_id,score_id,feature_name,feature_value,contribution_score,impact_direction,reason_rank) FROM '/data/csv/ai_score_reason.csv' WITH (FORMAT csv, HEADER true, NULL '');
\copy ai_recommendation (recommendation_id,customer_id,score_id,recommended_product_id,recommended_action,recommended_channel,recommended_timing,message_angle,expected_conversion,priority,suppression_flag,recommendation_timestamp,status,priority_rank) FROM '/data/csv/ai_recommendation.csv' WITH (FORMAT csv, HEADER true, NULL '');
\copy rm_action_feedback (feedback_id,recommendation_id,customer_id,rm_id,action_timestamp,action_type,result_status,customer_response,appointment_flag,application_flag,converted_flag,reason_not_interested,rm_feedback,next_followup_date) FROM '/data/csv/rm_action_feedback.csv' WITH (FORMAT csv, HEADER true, NULL '');

ANALYZE ai_model_registry, ai_model_coefficient, ai_model_metric, ai_customer_score, ai_recommendation;

SELECT product_id, dataset_split, metric_name, metric_value
FROM ai_model_metric WHERE metric_name IN ('roc_auc','pr_auc','log_loss','accuracy')
ORDER BY product_id, dataset_split, metric_name;
