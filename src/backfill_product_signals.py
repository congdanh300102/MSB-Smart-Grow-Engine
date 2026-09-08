"""
Bổ sung cột tín hiệu nhu cầu (customer_type, family_flag, auto_intent_flag,
home_intent_flag, agri_flag, fx_active_flag, securities_value, business_owner_flag,
occupation_group, industry_group) vào customer_360_feature_mart đã sinh trước đó —
để chạy src/product_analysis.py mà không phải regen toàn bộ.

Idempotent: nếu cột đã có thì bỏ qua.

    py src/backfill_product_signals.py
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PQ = os.path.join(ROOT, "data", "parquet")
CSV = os.path.join(ROOT, "data", "csv")
NEW = ["customer_type", "occupation_group", "industry_group", "business_owner_flag",
       "agri_flag", "family_flag", "auto_intent_flag", "home_intent_flag",
       "fx_active_flag", "securities_value"]


def main(seed=42):
    rng = np.random.default_rng(seed)
    mart = pd.read_parquet(os.path.join(PQ, "customer_360_feature_mart.parquet"))
    if all(c in mart.columns for c in NEW):
        print("Đã có đủ cột tín hiệu — bỏ qua.")
        return
    cust = pd.read_parquet(os.path.join(PQ, "dim_customer.parquet"))
    txn = pd.read_parquet(os.path.join(PQ, "fact_transaction.parquet"),
                          columns=["customer_id", "transaction_category"])
    dep = pd.read_parquet(os.path.join(PQ, "fact_deposit.parquet"),
                          columns=["customer_id", "principal_amount"])

    n = len(mart)
    idx = pd.Index(mart["customer_id"].to_numpy())
    c = cust.set_index("customer_id").reindex(idx)
    age = mart["age_group"].astype(str).to_numpy()

    mart["customer_type"] = c["customer_type"].values
    mart["occupation_group"] = c["occupation_group"].values
    mart["industry_group"] = c["industry_group"].values
    is_sme = (c["customer_type"].values == "SME") | (c["occupation_group"].values == "BUSINESS_OWNER")
    mart["business_owner_flag"] = is_sme
    mart["agri_flag"] = is_sme & (rng.random(n) < 0.18)
    mart["family_flag"] = rng.random(n) < np.where(
        np.isin(age, ["26-35", "36-45", "46-55"]), 0.55, 0.12)
    auto_cust = set(txn.loc[txn.transaction_category == "AUTOMOTIVE", "customer_id"])
    mart["auto_intent_flag"] = mart["customer_id"].isin(auto_cust).to_numpy()
    no_home_loan = ~mart["has_active_loan"].to_numpy()
    hb = np.where(np.isin(age, ["26-35", "36-45"]) & no_home_loan, 0.10, 0.03)
    hb = np.where((mart["balance_growth_3m"].to_numpy() > 0.15) & no_home_loan, hb + 0.06, hb)
    mart["home_intent_flag"] = rng.random(n) < hb
    mart["fx_active_flag"] = mart["international_spending_90d"].to_numpy() > 0
    dp = dep.groupby("customer_id")["principal_amount"].sum().reindex(idx).fillna(0.0)
    mart["securities_value"] = (dp.values * rng.uniform(0.8, 1.2, n)).round(2)

    mart.to_parquet(os.path.join(PQ, "customer_360_feature_mart.parquet"), index=False)
    mart.to_csv(os.path.join(CSV, "customer_360_feature_mart.csv"), index=False)
    print(f"OK — thêm {len(NEW)} cột vào customer_360_feature_mart ({n:,} dòng).")


if __name__ == "__main__":
    main()
