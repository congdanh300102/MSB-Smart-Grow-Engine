"""
MSB SMART GROWTH ENGINE - scoring rules (Data Model v2 / doc "V.01")
==================================================================

Deterministic implementation of the doc's formulas:

  Smart Growth Score = 25% Product Propensity + 20% Customer Value
                     + 20% Intent Signal      + 15% Engagement
                     + 10% Timing             + 10% Relationship         (0..100)

- Product Propensity is a **logistic regression** (see src/train_models.py):
      Z = b + w1*X1 + ... + w6*X6 ;  P = sigmoid(Z) ;  score = P * 100
  Features X1..X6 (normalised to 0..1) are built here by `build_propensity_dataset()`.
  Labels: `y_holds_product` (doc: đã dùng sản phẩm) and `y_adopt_next_90d` (forward
  label, model trains on this over non-holders to avoid X6 leaking the target).
- The other 5 blocks are deterministic weighted rules, built by `business_scores()`.
- `norm_score()` = PERCENTRANK.INC (doc mục a. cho phép thay cho min-max thô).

Everything is vectorised over the customer_360_feature_mart.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

AS_OF = date(2026, 8, 31)

# 4 target products of phase 1 (dim_product)
TARGET_PRODUCTS = {"CREDIT_CARD": "P001", "LOAN": "P004", "DEPOSIT": "P007", "INVESTMENT": "P009"}
GROUPS = ["CREDIT_CARD", "LOAN", "DEPOSIT", "INVESTMENT"]
PRODUCT_ID_TO_GROUP = {v: k for k, v in TARGET_PRODUCTS.items()}

SGS_WEIGHTS = {
    "product_propensity": 0.25, "customer_value": 0.20, "intent_signal": 0.20,
    "engagement": 0.15, "timing": 0.10, "relationship": 0.10,
}

PROPENSITY_FEATURES = ["x1_monthly_spending", "x2_income", "x3_digital_activity",
                       "x4_salary_account", "x5_campaign_response", "x6_product_gap"]

GROUP_OF_PCODE = {
    "CC_PLATINUM": "CREDIT_CARD", "CC_GOLD": "CREDIT_CARD", "CC_STANDARD": "CREDIT_CARD",
    "LOAN_HOME": "LOAN", "LOAN_AUTO": "LOAN", "LOAN_PERSONAL": "LOAN",
    "TD_STANDARD": "DEPOSIT", "TD_FLEXI": "DEPOSIT",
    "INVEST_FUND": "INVESTMENT", "INVEST_BOND": "INVESTMENT",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def minmax(s) -> pd.Series:
    """doc 'Công thức chung': Score = (X - Xmin) / (Xmax - Xmin)  -> 0..1 (raw min-max)."""
    s = pd.to_numeric(pd.Series(s).reset_index(drop=True), errors="coerce").astype(float)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(s)))
    return ((s - lo) / (hi - lo)).clip(0, 1)


def norm_score(s) -> pd.Series:
    """Chuẩn hoá 0..1 dùng PERCENTRANK.INC (doc a. cho phép: "PERCENTRANK.INC ...
    hoặc Công thức Score"). Percentile-rank trải đều 0..1 nên hợp với dữ liệu ngân
    hàng lệch phải (min-max thô dồn hầu hết KH về ~0). Dùng cho mọi Score 0..100
    và cho feature X1..X3 của model."""
    s = pd.to_numeric(pd.Series(s).reset_index(drop=True), errors="coerce").astype(float)
    if s.notna().sum() <= 1 or s.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)))
    return s.rank(method="average", pct=True).fillna(0.0).clip(0, 1)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# --------------------------------------------------------------------------
# campaign funnel signal per (customer, product group)
# --------------------------------------------------------------------------
def campaign_signals(fact_campaign: pd.DataFrame) -> pd.DataFrame:
    """Return per (customer_id, group): x5 funnel score (0..1) + days since last campaign."""
    fc = fact_campaign.copy()
    fc["grp"] = fc["product_code"].map(GROUP_OF_PCODE)
    fc = fc.dropna(subset=["grp"])
    for c in ["converted_flag", "applied_flag", "interested_flag", "clicked_flag",
              "opened_flag", "responded_flag"]:
        fc[c] = fc[c].astype("boolean").fillna(False).astype(bool)
    fc["fscore"] = np.select(
        [fc["converted_flag"], fc["applied_flag"], fc["interested_flag"],
         fc["clicked_flag"], fc["opened_flag"] | fc["responded_flag"]],
        [1.0, 0.9, 0.75, 0.5, 0.25], default=0.0)
    fc["cdate"] = pd.to_datetime(fc["campaign_date"], errors="coerce")
    g = fc.groupby(["customer_id", "grp"]).agg(
        x5_campaign_response=("fscore", "max"),
        last_campaign_date=("cdate", "max")).reset_index()
    g["days_since_last_campaign"] = (pd.Timestamp(AS_OF) - g["last_campaign_date"]).dt.days
    return g[["customer_id", "grp", "x5_campaign_response", "days_since_last_campaign"]]


# --------------------------------------------------------------------------
# X6 / product gap / holdings per group
# --------------------------------------------------------------------------
def _relationship_depth(mart: pd.DataFrame) -> np.ndarray:
    """0..1 - độ sâu quan hệ / mức độ sẵn sàng tiếp nhận sản phẩm mới."""
    return np.clip(
        0.30 * mart["salary_flag"].astype(bool).astype(float)
        + 0.25 * (mart["deposit_balance"].astype(float) > 0).astype(float)
        + 0.20 * (mart["product_count"].astype(int) >= 3).astype(float)
        + 0.15 * (mart["digital_engagement_score"].astype(float) > 25).astype(float)
        + 0.10 * mart["insurance_active_flag"].astype(bool).astype(float), 0, 1)


def _product_gap(mart: pd.DataFrame, grp: str) -> np.ndarray:
    """X6 - Product Gap / readiness in [0,1].

    Doc a.X6 dùng thang rời rạc 1 / 0.7 / 0 theo tình trạng sở hữu thẻ. Ở đây mở
    rộng thành thang liên tục (tình trạng sở hữu + độ sâu quan hệ) để mô hình học
    được trên tệp KHÁCH CHƯA SỞ HỮU (target_product = 0) mà không bị nhãn rò rỉ:
    cao = còn thiếu sản phẩm và có nền tảng quan hệ tốt để tiếp nhận.
    """
    depth = _relationship_depth(mart)
    if grp == "CREDIT_CARD":
        return np.clip(np.where(mart["has_premium_card"].astype(bool), 0.15,
                       np.where(mart["has_credit_card"].astype(bool), 0.70,   # có classic -> upgrade gap
                                0.45 + 0.55 * depth)), 0, 1)
    if grp == "LOAN":
        return np.clip(np.where(mart["has_active_loan"].astype(bool), 0.15, 0.40 + 0.60 * depth), 0, 1)
    if grp == "DEPOSIT":
        return np.clip(np.where(mart["deposit_balance"].astype(float) > 0, 0.20, 0.40 + 0.60 * depth), 0, 1)
    return np.clip(np.where(mart["has_investment"].astype(bool), 0.15, 0.35 + 0.65 * depth), 0, 1)


def _holds_product(mart: pd.DataFrame, grp: str) -> np.ndarray:
    if grp == "CREDIT_CARD":
        return mart["has_credit_card"].astype(bool).to_numpy()
    if grp == "LOAN":
        return mart["has_active_loan"].astype(bool).to_numpy()
    if grp == "DEPOSIT":
        return (mart["deposit_balance"].astype(float) > 0).to_numpy()
    return mart["has_investment"].astype(bool).to_numpy()


def _intent_product_gap(mart: pd.DataFrame, grp: str) -> np.ndarray:
    """doc c.ProductGap: premium 0 / credit-card-thuong 0.4 / chua co 1 (others 0/1)."""
    if grp == "CREDIT_CARD":
        return np.where(mart["has_premium_card"].astype(bool), 0.0,
                        np.where(mart["has_credit_card"].astype(bool), 0.4, 1.0))
    return _product_gap(mart, grp)   # 0 / 1


# --------------------------------------------------------------------------
# Product Propensity feature matrix  X1..X6  (+ label y)
# --------------------------------------------------------------------------
def build_propensity_dataset(mart: pd.DataFrame, fact_campaign: pd.DataFrame) -> pd.DataFrame:
    """One row per (customer, target product): X1..X6 in [0,1] and y_holds_product."""
    m = mart.reset_index(drop=True)
    x1 = norm_score(m["spending_30d"])
    x2 = norm_score(m["income_monthly"])
    x3 = (0.40 * norm_score(m["active_days_30d"])
          + 0.35 * norm_score(m["digital_txn_count_30d"])
          + 0.25 * norm_score(m["feature_usage_30d"])).clip(0, 1)
    x4 = m["salary_flag"].astype(bool).astype(float)

    camp = campaign_signals(fact_campaign)
    rows = []
    for grp in GROUPS:
        cg = camp[camp["grp"] == grp][["customer_id", "x5_campaign_response"]]
        x5 = (m[["customer_id"]].merge(cg, on="customer_id", how="left")["x5_campaign_response"]
              .fillna(0.0).to_numpy())
        x6 = _product_gap(m, grp)
        y = _holds_product(m, grp).astype(int)
        rows.append(pd.DataFrame({
            "customer_id": m["customer_id"].to_numpy(),
            "snapshot_date": AS_OF,
            "product_id": TARGET_PRODUCTS[grp],
            "product_group": grp,
            "x1_monthly_spending": np.round(x1.to_numpy(), 6),
            "x2_income": np.round(x2.to_numpy(), 6),
            "x3_digital_activity": np.round(x3.to_numpy(), 6),
            "x4_salary_account": np.round(x4.to_numpy(), 6),
            "x5_campaign_response": np.round(x5, 6),
            "x6_product_gap": np.round(x6, 6),
            "y_holds_product": y,
        }))
    return pd.concat(rows, ignore_index=True)


def build_score_frame(mart: pd.DataFrame, fact_campaign: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix + all 5 business scores, one row per (customer, target product).
    Used by train_models.py --apply to (re)build ai_customer_score."""
    feats = build_propensity_dataset(mart, fact_campaign)
    biz = business_scores(mart, fact_campaign)
    return feats.merge(biz, on=["customer_id", "product_id", "product_group"], how="left")


# --------------------------------------------------------------------------
# 5 deterministic business blocks -> per (customer, product group)
# --------------------------------------------------------------------------
def business_scores(mart: pd.DataFrame, fact_campaign: pd.DataFrame) -> pd.DataFrame:
    m = mart.reset_index(drop=True)
    n = len(m)

    # ---- shared (customer-level) sub-scores, 0..100 ----------------------
    income_s = norm_score(m["income_monthly"]) * 100
    balance_s = norm_score(m["avg_balance_90d"]) * 100
    cashflow_s = norm_score(m["net_cashflow_30d"]) * 100

    groups_owned = (
        1  # CASA
        + (m["deposit_balance"].astype(float) > 0).astype(int)          # Savings
        + m["has_credit_card"].astype(bool).astype(int)                 # Credit Card
        + (m["product_count"].astype(int) > 0).astype(int)              # Debit/other card proxy
        + m["has_active_loan"].astype(bool).astype(int)                 # Loan
        + m["insurance_active_flag"].astype(bool).astype(int)           # Insurance
        + m["has_investment"].astype(bool).astype(int)                  # Investment
        + m["salary_flag"].astype(bool).astype(int)                     # Payroll
    ).clip(0, 8)
    product_breadth = groups_owned / 8.0 * 100

    txn_usage = norm_score(m["txn_count_90d"]) * 100
    recency = np.where(m["txn_count_30d"].astype(int) > 0, 100.0,
                       np.where(m["txn_count_90d"].astype(int) > 0, 50.0, 20.0))
    active_ratio = np.clip(m["txn_count_30d"].to_numpy() / 20.0
                           + (m["digital_engagement_score"].to_numpy() > 20) * 0.3, 0, 1) * 100
    product_activity = 0.40 * active_ratio + 0.30 * txn_usage + 0.30 * recency
    product_value = norm_score(m["avg_balance_90d"].astype(float)
                           + m["deposit_balance"].astype(float)
                           + m["spending_90d"].astype(float) * 2.0) * 100
    product_relationship = 0.40 * product_breadth + 0.30 * product_activity + 0.30 * product_value

    customer_value = (0.30 * income_s + 0.30 * balance_s
                      + 0.20 * cashflow_s + 0.20 * product_relationship).clip(0, 100)

    # engagement (doc d)
    digital = (0.40 * norm_score(m["active_days_30d"]) + 0.25 * norm_score(m["login_count_30d"])
               + 0.25 * norm_score(m["digital_txn_count_30d"]) + 0.10 * norm_score(m["feature_usage_30d"])) * 100
    transaction_e = norm_score(m["txn_count_90d"]) * 100
    campaign_e = np.clip(m["previous_campaign_response"].to_numpy(), 0, 1) * 100
    resp_pos = m["last_customer_response"].isin(["INTERESTED", "CONVERTED", "CALLBACK"]).astype(float)
    rm_e = (0.5 * norm_score(m["rm_contact_count_90d"]) * 100 + 0.5 * resp_pos * 100)
    engagement = (0.40 * digital + 0.30 * transaction_e + 0.20 * campaign_e + 0.10 * rm_e).clip(0, 100)

    # timing (doc e) - shared part
    g_sp = m["spending_growth_3m"].to_numpy()
    g_bal = m["balance_growth_3m"].to_numpy()
    behavior_change = (((g_sp > 0.40).astype(float)
                        + (m["active_days_30d"].to_numpy() > np.median(m["active_days_30d"])).astype(float)
                        + (g_bal > 0.30).astype(float)) / 3.0) * 100
    financial_event = ((m["salary_flag"].astype(bool).astype(float).to_numpy()
                        + (g_bal > 0.20).astype(float)
                        + (m["net_cashflow_30d"].to_numpy() > 0).astype(float)) / 3.0) * 100
    last_contact = np.select(
        [m["days_since_last_rm_contact"] <= 3, m["days_since_last_rm_contact"] <= 7,
         m["days_since_last_rm_contact"] <= 14, m["days_since_last_rm_contact"] <= 30],
        [0.1, 0.3, 0.5, 0.75], default=0.9) * 100

    # relationship (doc g) - shared part
    ry = m["relationship_years"].to_numpy()
    tenure = np.select([ry < 1, ry < 2, ry < 4, ry < 7], [30, 50, 70, 85], default=100)
    salary_rel = m["salary_flag"].astype(bool).astype(float).to_numpy() * 100
    hist_response = np.clip(m["previous_campaign_response"].to_numpy(), 0, 1) * 100
    relationship = (0.25 * tenure + 0.25 * product_breadth.to_numpy()
                    + 0.25 * salary_rel + 0.25 * hist_response).clip(0, 100)

    # spending signal (intent, doc c) - shared
    spending_signal = np.select(
        [g_sp < 0.0, g_sp < 0.10, g_sp < 0.25, g_sp < 0.50],
        [20, 40, 60, 80], default=100).astype(float)
    financial_signal = ((0.30 * (g_bal > 0).astype(float)
                         + 0.30 * m["salary_flag"].astype(bool).astype(float).to_numpy()
                         + 0.40 * norm_score(m["net_cashflow_30d"]).to_numpy())) * 100

    camp = campaign_signals(fact_campaign)

    out = []
    for grp in GROUPS:
        cg = camp[camp["grp"] == grp][["customer_id", "x5_campaign_response", "days_since_last_campaign"]]
        j = m[["customer_id"]].merge(cg, on="customer_id", how="left")
        camp_sig = j["x5_campaign_response"].fillna(0.0).to_numpy() * 100
        dslc = j["days_since_last_campaign"].fillna(999).to_numpy()
        campaign_cooling = np.select([dslc <= 3, dslc <= 7, dslc <= 14, dslc <= 30],
                                     [0.1, 0.3, 0.6, 0.8], default=1.0) * 100
        pgap_intent = _intent_product_gap(m, grp) * 100

        intent = (0.30 * spending_signal + 0.25 * pgap_intent
                  + 0.25 * camp_sig + 0.20 * financial_signal).clip(0, 100)
        timing = (0.35 * behavior_change + 0.25 * financial_event
                  + 0.20 * last_contact + 0.20 * campaign_cooling).clip(0, 100)

        out.append(pd.DataFrame({
            "customer_id": m["customer_id"].to_numpy(),
            "product_id": TARGET_PRODUCTS[grp],
            "product_group": grp,
            "customer_value_score": np.round(customer_value, 2),
            "intent_signal_score": np.round(intent, 2),
            "engagement_score": np.round(engagement, 2),
            "timing_score": np.round(timing, 2),
            "relationship_score": np.round(relationship, 2),
        }))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# Smart Growth Score
# --------------------------------------------------------------------------
def smart_growth_score(product_propensity, customer_value, intent_signal,
                       engagement, timing, relationship):
    w = SGS_WEIGHTS
    return np.clip(
        w["product_propensity"] * np.asarray(product_propensity)
        + w["customer_value"] * np.asarray(customer_value)
        + w["intent_signal"] * np.asarray(intent_signal)
        + w["engagement"] * np.asarray(engagement)
        + w["timing"] * np.asarray(timing)
        + w["relationship"] * np.asarray(relationship), 0, 100)


def priority_level(sgs):
    sgs = np.round(np.asarray(sgs, dtype=float), 2)
    return np.select(
        [sgs >= 90, sgs >= 80, sgs >= 70, sgs >= 60],
        ["Very High", "High", "Medium", "Low"], default="Do not prioritize")
