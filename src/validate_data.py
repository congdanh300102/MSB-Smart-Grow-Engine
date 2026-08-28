"""
Validate the generated dataset against the schema constraints
(PK uniqueness, FK referential integrity, decimal precision, value domains)
BEFORE loading into PostgreSQL. Exits non-zero on any violation.

    py src/validate_data.py
"""
import os
import sys
import decimal

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "data", "csv")

errors = []


def load(name):
    return pd.read_csv(os.path.join(CSV, f"{name}.csv"))


TABLES = [
    "dim_customer", "fact_casa_daily", "fact_transaction", "fact_card", "fact_loan",
    "fact_deposit", "fact_insurance", "fact_digital_activity", "fact_crm_interaction",
    "fact_customer_service", "fact_campaign", "customer_360_feature_mart",
    "ai_customer_score", "ai_recommendation", "rm_action_feedback",
]
d = {t: load(t) for t in TABLES}

# ---- primary keys -------------------------------------------------------
PK = {
    "dim_customer": "customer_key",
    "fact_casa_daily": "fact_casa_daily_key",
    "fact_transaction": "fact_transaction_key",
    "fact_card": "fact_card_key",
    "fact_loan": "fact_loan_key",
    "fact_deposit": "fact_deposit_key",
    "fact_insurance": "fact_insurance_key",
    "fact_digital_activity": "fact_digital_activity_key",
    "fact_crm_interaction": "fact_crm_interaction_key",
    "fact_customer_service": "fact_customer_service_key",
    "fact_campaign": "fact_campaign_key",
    "customer_360_feature_mart": "customer_360_feature_key",
    "ai_customer_score": "ai_customer_score_key",
    "ai_recommendation": "ai_recommendation_key",
    "rm_action_feedback": "rm_action_feedback_key",
}
for t, k in PK.items():
    df = d[t]
    if df[k].duplicated().any():
        errors.append(f"{t}: duplicate PK {k}")
    if df[k].isna().any():
        errors.append(f"{t}: null PK {k}")

if d["dim_customer"]["customer_id"].duplicated().any():
    errors.append("dim_customer: customer_id not unique")

# ---- foreign keys -----------------------------------------------------
cust_keys = set(d["dim_customer"]["customer_key"])
for t in ["fact_casa_daily", "fact_transaction", "fact_card", "fact_loan", "fact_deposit",
          "fact_insurance", "fact_digital_activity", "fact_crm_interaction",
          "fact_customer_service", "fact_campaign", "customer_360_feature_mart"]:
    bad = ~d[t]["customer_key"].isin(cust_keys)
    if bad.any():
        errors.append(f"{t}: {bad.sum()} rows with unknown customer_key")

camp_keys = set(d["fact_campaign"]["fact_campaign_key"])
ck = d["customer_360_feature_mart"]["campaign_key"].dropna()
bad = ~ck.isin(camp_keys)
if bad.any():
    errors.append(f"customer_360_feature_mart: {bad.sum()} rows with unknown campaign_key")

feat_keys = set(d["customer_360_feature_mart"]["customer_360_feature_key"])
bad = ~d["ai_customer_score"]["customer_360_feature_key"].isin(feat_keys)
if bad.any():
    errors.append(f"ai_customer_score: {bad.sum()} rows with unknown customer_360_feature_key")

score_keys = set(d["ai_customer_score"]["ai_customer_score_key"])
bad = ~d["ai_recommendation"]["ai_customer_score_key"].isin(score_keys)
if bad.any():
    errors.append(f"ai_recommendation: {bad.sum()} rows with unknown ai_customer_score_key")

reco_keys = set(d["ai_recommendation"]["ai_recommendation_key"])
bad = ~d["rm_action_feedback"]["ai_recommendation_key"].isin(reco_keys)
if bad.any():
    errors.append(f"rm_action_feedback: {bad.sum()} rows with unknown ai_recommendation_key")

# ---- decimal precision (precision, scale) ----------------------------
DECIMALS = {
    ("fact_casa_daily", "daily_balance"): (18, 2),
    ("fact_transaction", "amount"): (18, 2),
    ("fact_card", "credit_limit"): (18, 2),
    ("fact_card", "utilization_pct"): (5, 2),
    ("fact_loan", "outstanding_principal"): (18, 2),
    ("fact_loan", "interest_rate"): (5, 2),
    ("fact_deposit", "deposit_amount"): (18, 2),
    ("customer_360_feature_mart", "campaign_response_rate"): (5, 2),
    ("customer_360_feature_mart", "digital_engagement_score"): (10, 2),
    ("ai_customer_score", "churn_score"): (10, 4),
    ("ai_customer_score", "propensity_to_buy_score"): (10, 4),
    ("ai_recommendation", "confidence_score"): (10, 4),
}
for (t, col), (prec, scale) in DECIMALS.items():
    s = d[t][col].dropna()
    max_int_digits = prec - scale
    over = s[s.abs() >= 10 ** max_int_digits]
    if len(over):
        errors.append(f"{t}.{col}: {len(over)} values exceed DECIMAL({prec},{scale}) "
                      f"(max abs {s.abs().max():,.2f})")

# ---- value domains --------------------------------------------------
for col in ["churn_score", "propensity_to_buy_score", "credit_risk_score", "next_best_action_score"]:
    s = d["ai_customer_score"][col]
    if s.min() < 0 or s.max() > 1:
        errors.append(f"ai_customer_score.{col} outside [0,1]: [{s.min()}, {s.max()}]")

r = d["ai_recommendation"]
if not r.groupby("ai_customer_score_key")["priority_rank"].apply(
        lambda x: sorted(x) == list(range(1, len(x) + 1))).all():
    errors.append("ai_recommendation: priority_rank not 1..k per customer")

if d["fact_card"]["utilization_pct"].max() > 100:
    errors.append("fact_card.utilization_pct > 100")

rating = d["rm_action_feedback"]["rating"]
if rating.min() < 1 or rating.max() > 5:
    errors.append(f"rm_action_feedback.rating outside 1..5: [{rating.min()},{rating.max()}]")

# ---- coherence signal: archetype-free sanity on scores --------------
mart = d["customer_360_feature_mart"].merge(
    d["ai_customer_score"], on="customer_360_feature_key")
corr = mart["digital_engagement_score"].corr(mart["propensity_to_buy_score"])
print(f"info: corr(digital_engagement, propensity_to_buy) = {corr:+.3f}")
corr2 = mart["total_loan_outstanding"].corr(mart["credit_risk_score"])
print(f"info: corr(loan_outstanding, credit_risk)         = {corr2:+.3f}")

# ---- report --------------------------------------------------------
print()
for t in TABLES:
    print(f"  {t:<28} {len(d[t]):>9,} rows")
print()
if errors:
    print("VALIDATION FAILED:")
    for e in errors:
        print("  x", e)
    sys.exit(1)
print("VALIDATION PASSED - all PK/FK/precision/domain checks OK")
