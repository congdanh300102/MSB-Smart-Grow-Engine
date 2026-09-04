"""
Validate the v2 dataset (PK uniqueness, FK integrity, value domains) before
loading into PostgreSQL. Exits non-zero on any violation.

    py src/validate_data.py
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "data", "csv")

errors, warns = [], []


def load(name):
    return pd.read_csv(os.path.join(CSV, f"{name}.csv"))


TABLES = [
    "dim_rm", "dim_product", "dim_campaign", "dim_customer", "dim_card",
    "fact_casa_daily", "fact_transaction", "agg_customer_transaction", "fact_card_monthly",
    "fact_deposit", "fact_loan", "fact_insurance", "fact_digital_activity",
    "fact_crm_interaction", "fact_customer_service", "fact_campaign",
    "customer_360_feature_mart", "ai_feature_customer", "ai_customer_score",
    "ai_score_reason", "ai_recommendation", "rm_action_feedback",
    "ml_credit_card_training_set", "ml_propensity_training_set",
]
d = {t: load(t) for t in TABLES}

# ---- primary keys (single or composite) --------------------------------
PK = {
    "dim_rm": ["rm_id"], "dim_product": ["product_id"], "dim_campaign": ["campaign_id"],
    "dim_customer": ["customer_id"], "dim_card": ["card_id"],
    "fact_casa_daily": ["snapshot_date", "account_id"],
    "fact_transaction": ["transaction_id"],
    "agg_customer_transaction": ["customer_id", "snapshot_date"],
    "fact_card_monthly": ["customer_id", "card_id", "month"],
    "fact_deposit": ["deposit_id"], "fact_loan": ["loan_id"], "fact_insurance": ["policy_id"],
    "fact_digital_activity": ["activity_date", "customer_id"],
    "fact_crm_interaction": ["interaction_id"], "fact_customer_service": ["case_id"],
    "fact_campaign": ["campaign_customer_id"],
    "customer_360_feature_mart": ["snapshot_date", "customer_id"],
    "ai_feature_customer": ["snapshot_date", "customer_id"],
    "ai_customer_score": ["score_id"], "ai_score_reason": ["reason_id"],
    "ai_recommendation": ["recommendation_id"], "rm_action_feedback": ["feedback_id"],
    "ml_credit_card_training_set": ["customer_id", "observation_date"],
    "ml_propensity_training_set": ["customer_id", "snapshot_date", "product_id"],
}
for t, keys in PK.items():
    df = d[t]
    if df[keys].duplicated().any():
        errors.append(f"{t}: duplicate PK {keys}")
    if df[keys].isna().any().any():
        errors.append(f"{t}: null PK column in {keys}")

# ---- foreign keys -------------------------------------------------------
cust_ids = set(d["dim_customer"]["customer_id"])
rm_ids = set(d["dim_rm"]["rm_id"])
product_ids = set(d["dim_product"]["product_id"])
campaign_ids = set(d["dim_campaign"]["campaign_id"])

CUST_FK_TABLES = [
    "dim_card", "fact_casa_daily", "fact_transaction", "agg_customer_transaction",
    "fact_card_monthly", "fact_deposit", "fact_loan", "fact_insurance",
    "fact_digital_activity", "fact_crm_interaction", "fact_customer_service",
    "fact_campaign", "customer_360_feature_mart", "ai_feature_customer",
    "ai_customer_score", "ai_recommendation", "rm_action_feedback",
    "ml_propensity_training_set",
]
for t in CUST_FK_TABLES:
    bad = ~d[t]["customer_id"].isin(cust_ids)
    if bad.any():
        errors.append(f"{t}: {bad.sum()} rows with unknown customer_id")

for t in ["dim_customer"]:
    bad = ~d[t]["rm_id"].dropna().isin(rm_ids)
    if bad.any():
        errors.append(f"{t}: {bad.sum()} rows with unknown rm_id")
for t in ["fact_crm_interaction", "rm_action_feedback"]:
    bad = ~d[t]["rm_id"].dropna().isin(rm_ids)
    if bad.any():
        errors.append(f"{t}: {bad.sum()} rows with unknown rm_id")

bad = ~d["ai_customer_score"]["product_id"].isin(product_ids)
if bad.any():
    errors.append(f"ai_customer_score: {bad.sum()} unknown product_id")
bad = ~d["ai_recommendation"]["recommended_product_id"].dropna().isin(product_ids)
if bad.any():
    errors.append(f"ai_recommendation: {bad.sum()} unknown recommended_product_id")
bad = ~d["dim_campaign"]["product_id"].dropna().isin(product_ids)
if bad.any():
    errors.append(f"dim_campaign: {bad.sum()} unknown product_id")
bad = ~d["fact_campaign"]["campaign_id"].isin(campaign_ids)
if bad.any():
    errors.append(f"fact_campaign: {bad.sum()} unknown campaign_id")

score_ids = set(d["ai_customer_score"]["score_id"])
bad = ~d["ai_score_reason"]["score_id"].isin(score_ids)
if bad.any():
    errors.append(f"ai_score_reason: {bad.sum()} unknown score_id")
bad = ~d["ai_recommendation"]["score_id"].isin(score_ids)
if bad.any():
    errors.append(f"ai_recommendation: {bad.sum()} unknown score_id")

reco_ids = set(d["ai_recommendation"]["recommendation_id"])
bad = ~d["rm_action_feedback"]["recommendation_id"].isin(reco_ids)
if bad.any():
    errors.append(f"rm_action_feedback: {bad.sum()} unknown recommendation_id")

# ---- ml_propensity_training_set ------------------------------------
pt = d["ml_propensity_training_set"]
bad = ~pt["product_id"].isin(product_ids)
if bad.any():
    errors.append(f"ml_propensity_training_set: {bad.sum()} unknown product_id")
for c in ["x1_monthly_spending", "x2_income", "x3_digital_activity",
          "x4_salary_account", "x5_campaign_response", "x6_product_gap"]:
    if pt[c].min() < -1e-6 or pt[c].max() > 1 + 1e-6:
        errors.append(f"ml_propensity_training_set.{c} outside [0,1]: [{pt[c].min()},{pt[c].max()}]")
for c in ["y_holds_product", "y_adopt_next_90d"]:
    if not set(pt[c].dropna().unique()).issubset({0, 1}):
        errors.append(f"ml_propensity_training_set.{c} not binary")
if (pt.loc[pt["y_holds_product"] == 1, "y_adopt_next_90d"] == 1).any():
    warns.append("ml_propensity_training_set: some holders also flagged y_adopt_next_90d=1")
_pos = pt.groupby("product_group")["y_adopt_next_90d"].mean()
print("info: y_adopt_next_90d positive rate by product:",
      {k: round(v, 3) for k, v in _pos.items()})

# ---- value domains --------------------------------------------------
s = d["ai_customer_score"]
for col in ["product_propensity_score", "customer_value_score", "intent_signal_score",
            "engagement_score", "timing_score", "relationship_score", "smart_growth_score"]:
    if s[col].min() < 0 or s[col].max() > 100.0001:
        errors.append(f"ai_customer_score.{col} outside [0,100]: [{s[col].min()},{s[col].max()}]")
if s["propensity_probability"].min() < 0 or s["propensity_probability"].max() > 1:
    errors.append("ai_customer_score.propensity_probability outside [0,1]")

exp_prio = pd.cut(s["smart_growth_score"], [-1, 60, 70, 80, 90, 101], right=False,
                  labels=["Do not prioritize", "Low", "Medium", "High", "Very High"])
mismatch = (exp_prio.astype(str) != s["priority_level"].astype(str)).sum()
if mismatch:
    errors.append(f"ai_customer_score: {mismatch} rows with priority_level not matching score bucket")

r = d["ai_recommendation"]
bad_rank = r.groupby("customer_id")["priority_rank"].apply(
    lambda x: sorted(x) != list(range(1, len(x) + 1)))
if bad_rank.any():
    errors.append(f"ai_recommendation: {bad_rank.sum()} customers with non-1..k priority_rank")

card = d["dim_card"]
if (card["card_type"] == "CREDIT").sum() and card.loc[card.card_type == "CREDIT", "credit_limit"].min() < 0:
    errors.append("dim_card: negative credit_limit")

fb = d["rm_action_feedback"]
bad = fb["converted_flag"] & ~fb["application_flag"]
if bad.any():
    warns.append(f"rm_action_feedback: {bad.sum()} converted rows without application_flag")

# ---- coherence signals -----------------------------------------------
mart = d["customer_360_feature_mart"]
best_score = s.groupby("customer_id")["smart_growth_score"].max()
elig_like = mart.set_index("customer_id")
corr = elig_like["digital_engagement_score"].corr(best_score.reindex(elig_like.index))
print(f"info: corr(digital_engagement, best smart_growth_score) = {corr:+.3f}")
pct85 = (best_score >= 85).mean() * 100
print(f"info: customers with best smart_growth_score >= 85 : {pct85:.1f}%  "
      f"(n={ (best_score>=85).sum() })")
print(f"info: priority_level distribution (all score rows):")
print(s["priority_level"].value_counts().to_string())

print()
for t in TABLES:
    print(f"  {t:<30} {len(d[t]):>10,} rows")
print()
if warns:
    print("WARNINGS:")
    for w in warns:
        print("  !", w)
if errors:
    print("VALIDATION FAILED:")
    for e in errors:
        print("  x", e)
    sys.exit(1)
print("VALIDATION PASSED")
