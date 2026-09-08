"""
MSB SMART GROWTH ENGINE — Phân tích / thống kê / dự báo mức độ phù hợp sản phẩm
============================================================================

Đọc dữ liệu đã sinh trong data/parquet/, dựng danh mục 35 sản phẩm chuẩn hoá
(src/products.py) và trả lời: **mỗi khách hàng phù hợp nhất với sản phẩm nào**.

Phương pháp (hybrid — theo lựa chọn "LR + rule"):
  propensity(khách, sp) =
      w_lr  · P_logistic(nhóm neo)          (thẻ tín dụng / vay / tiền gửi / đầu tư)
    + w_fit · fit_score(rule sản phẩm)      (mọi sản phẩm, kể cả sp ngách)
  Smart Growth Score sản phẩm = 25·propensity + 20·CustomerValue + 20·fit
                              + 15·Engagement + 10·fit + 10·Relationship   (0..100)

Outputs (data/parquet/ + data/csv/):
  dim_product_catalogue           35 sản phẩm + rule "khách hàng phù hợp"
  fact_customer_product_holding   sản phẩm khách đang sở hữu (mô phỏng)
  ai_product_fit                  khách × sản phẩm: fit, propensity, SGS, eligible, held
  ai_product_recommendation_v2    top-6 sản phẩm phù hợp nhất / khách (đã loại sp đã có)
  agg_product_demand              dự báo cầu theo sản phẩm × phân khúc
  agg_segment_product_affinity    ma trận phân khúc × sản phẩm
  models/product_analysis_report.md   báo cáo phân tích + dự báo

    py src/product_analysis.py [--seed 42]
"""
from __future__ import annotations

import argparse
import os
from datetime import date

import numpy as np
import pandas as pd

import products as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(ROOT, "data", "parquet")
CSV = os.path.join(ROOT, "data", "csv")
MODELS = os.path.join(ROOT, "models")
AS_OF = date(2026, 8, 31)

# neo LR: nhóm sản phẩm mới -> product_id cũ trong ai_customer_score
ANCHOR = {"CARD": "P001", "LENDING": "P004", "FD": "P007"}   # CASA: không có neo LR
W_LR = {"CARD": 0.45, "LENDING": 0.45, "FD": 0.45}
BASE_CONV_90D = 0.14           # tỉ lệ chuyển đổi nền của tệp đủ điều kiện được tiếp cận
CONTACT_CAPACITY = 0.55        # % tệp đủ điều kiện thực sự được RM/kênh tiếp cận trong quý


def _read(name):
    return pd.read_parquet(os.path.join(PARQUET, f"{name}.parquet"))


def build(seed=42):
    rng = np.random.default_rng(seed)
    mart = _read("customer_360_feature_mart")
    cust = _read("dim_customer")
    for c in ("customer_type", "occupation_group", "industry_group"):
        if c not in mart.columns:
            mart = mart.merge(cust[["customer_id", c]], on="customer_id", how="left")
    score = _read("ai_customer_score")

    dimp = P.dim_product_frame()
    dimp.to_parquet(os.path.join(PARQUET, "dim_product_catalogue.parquet"), index=False)
    dimp.to_csv(os.path.join(CSV, "dim_product_catalogue.csv"), index=False)

    # ---- fit + holdings ------------------------------------------------
    fit = P.fit_table(mart).merge(dimp[["product_id", "product_group", "product_code",
                                        "product_name", "customer_type"]], on="product_id")
    hold = P.build_holdings(mart, rng)
    hold.to_parquet(os.path.join(PARQUET, "fact_customer_product_holding.parquet"), index=False)
    hold.to_csv(os.path.join(CSV, "fact_customer_product_holding.csv"), index=False)
    held_pairs = set(zip(hold.customer_id, hold.product_id))
    fit["held"] = [(c, p) in held_pairs for c, p in zip(fit.customer_id, fit.product_id)]
    # "đã có sản phẩm cùng nhóm" — với thẻ tín dụng & vay thì không chào thêm biến thể
    hg = hold.merge(dimp[["product_id", "product_group"]], on="product_id")
    held_grp = set(zip(hg.customer_id, hg.product_group))
    SINGLE = {"LENDING"}  # nhóm chỉ nên sở hữu 1 (vay); CARD tín dụng xử lý riêng bên dưới
    is_credit_card = fit.product_id.map(lambda i: P.BY_ID[i]["subgroup"] == "CREDIT").fillna(False)
    held_cc = set(hg.loc[hg.product_id.map(lambda i: P.BY_ID.get(i, {}).get("subgroup")) == "CREDIT",
                         "customer_id"])
    fit["held_group"] = [
        (g in SINGLE and (c, g) in held_grp) or (bool(cc) and c in held_cc)
        for c, g, cc in zip(fit.customer_id, fit.product_group, is_credit_card)]

    # ---- anchor LR propensity + customer-level SGS parts -------------
    cust_parts = (score.groupby("customer_id")
                  .agg(customer_value_score=("customer_value_score", "max"),
                       engagement_score=("engagement_score", "max"),
                       relationship_score=("relationship_score", "max")).reset_index())
    anchor_p = {g: score.loc[score.product_id == pid, ["customer_id", "propensity_probability"]]
                .set_index("customer_id")["propensity_probability"]
                for g, pid in ANCHOR.items()}

    fit = fit.merge(cust_parts, on="customer_id", how="left")
    a = np.zeros(len(fit))
    for g, s in anchor_p.items():
        m = fit.product_group.values == g
        a[m] = fit.loc[m, "customer_id"].map(s).fillna(s.mean()).values
    w = fit.product_group.map(W_LR).fillna(0.0).values
    # thẻ ghi nợ / CASA / sp ngách: chỉ dùng rule; thẻ tín dụng & vay & tiền gửi: blend LR+rule
    is_debit_card = ((fit.product_group.values == "CARD")
                     & fit.product_id.map(lambda i: P.BY_ID[i]["subgroup"] == "DEBIT").values)
    is_lr = np.isin(fit.product_group.values, list(W_LR)) & ~is_debit_card
    w = np.where(is_lr, w, 0.0)
    prop = w * a + (1 - w) * fit.fit_score.values
    prop = np.clip(prop + rng.normal(0, 0.02, len(fit)), 0.01, 0.99)
    fit["propensity"] = np.round(prop, 4)

    cv = fit.customer_value_score.fillna(50).values
    eng = fit.engagement_score.fillna(50).values
    rel = fit.relationship_score.fillna(50).values
    f100 = fit.fit_score.values * 100
    fit["smart_growth_score"] = np.round(np.clip(
        0.25 * prop * 100 + 0.20 * cv + 0.20 * f100 + 0.15 * eng + 0.10 * f100 + 0.10 * rel,
        0, 100), 2)
    fit["priority_level"] = pd.cut(fit.smart_growth_score, [-1, 60, 70, 80, 90, 101], right=False,
                                   labels=["Do not prioritize", "Low", "Medium", "High", "Very High"])

    keep = ["customer_id", "product_id", "product_code", "product_group", "eligible", "held",
            "fit_score", "propensity", "smart_growth_score", "priority_level",
            "reason_1", "reason_2", "reason_3"]
    ai_fit = fit[keep].copy()
    ai_fit.to_parquet(os.path.join(PARQUET, "ai_product_fit.parquet"), index=False)
    ai_fit.to_csv(os.path.join(CSV, "ai_product_fit.csv"), index=False)

    # ---- recommendations: top-6 sp phù hợp nhất, loại sp đã có / cùng nhóm ----
    cust_i = mart.set_index("customer_id")
    cand = fit[fit.eligible & ~fit.held & ~fit.held_group].copy()
    cand = cand.sort_values(["customer_id", "propensity"], ascending=[True, False])
    cand["priority_rank"] = cand.groupby("customer_id").cumcount() + 1
    reco = cand[cand.priority_rank <= 6][[
        "customer_id", "priority_rank", "product_id", "product_code", "product_name",
        "product_group", "propensity", "fit_score", "smart_growth_score", "priority_level",
        "reason_1", "reason_2", "reason_3"]].copy()
    reco["expected_conversion"] = np.round(
        np.clip(reco.propensity * 0.75 * (1 + (reco.priority_rank == 1) * 0.15), 0.01, 0.95), 4)

    # Next Best Action / Channel / Timing / Message — theo Action 5,6,7 của hành trình AI
    seg = reco.customer_id.map(cust_i["segment"]).fillna("MASS")
    dig = reco.customer_id.map(cust_i["digital_engagement_score"]).fillna(0)
    supp = reco.customer_id.map(cust_i.get("recent_rejection_30d_flag", pd.Series(dtype=bool))).fillna(False) \
        | reco.customer_id.map(cust_i.get("serious_complaint_15d", pd.Series(dtype=bool))).fillna(False)
    hi = reco.priority_level.isin(["Very High", "High"])
    affluent = seg.isin(["AFFLUENT", "PRIVATE"])
    reco["recommended_action"] = np.select(
        [supp, affluent & (reco.priority_rank == 1), affluent,
         (dig >= 45) & hi, hi],
        ["NO_CONTACT", "RM_CALL", "RM_ASSISTED_MESSAGE", "IN_APP", "RM_ASSISTED_MESSAGE"],
        default="NURTURE")
    reco["recommended_channel"] = np.select(
        [reco.recommended_action.isin(["RM_CALL", "RM_ASSISTED_MESSAGE"]),
         reco.recommended_action.eq("IN_APP"), reco.recommended_action.eq("NURTURE")],
        ["RM_CALL", "IN_APP", "EMAIL"], default="SMS")
    reco["recommended_timing"] = np.select(
        [(reco.priority_rank == 1) & (reco.smart_growth_score >= 80),
         reco.priority_rank == 1, reco.priority_rank == 2],
        ["Trong vòng 48 giờ", "Trong tuần này", "Trong 2 tuần tới"], default="Nurture 30 ngày")
    ANGLE = {"CARD": "Hoàn tiền + ưu đãi chi tiêu, phù hợp mức chi tiêu và thu nhập.",
             "CASA": "Tài khoản/dịch vụ tối ưu dòng tiền và giao dịch hằng ngày.",
             "FD": "Cộng thêm lãi suất, tối ưu dòng tiền nhàn rỗi.",
             "LENDING": "Lãi suất ưu đãi, duyệt nhanh, phù hợp nhu cầu vốn."}
    reco["message_angle"] = reco.product_group.map(ANGLE)
    reco["status"] = np.where(supp, "BLOCKED",
                     np.where(affluent | (reco.recommended_action == "RM_CALL"), "RM_APPROVAL",
                     np.where(hi, "SENT", "NEW")))
    reco["branch_id"] = reco.customer_id.map(cust_i["branch_id"])
    reco["branch_product_rank"] = (reco.sort_values("smart_growth_score", ascending=False)
                                   .groupby(["branch_id", "product_id"]).cumcount() + 1)
    reco.to_parquet(os.path.join(PARQUET, "ai_product_recommendation_v2.parquet"), index=False)
    reco.drop(columns=[]).to_csv(os.path.join(CSV, "ai_product_recommendation_v2.csv"), index=False)

    # ---- demand forecast per product x segment ----------------------
    seg = mart.set_index("customer_id")["segment"]
    fit["segment"] = fit.customer_id.map(seg)
    whitespace = fit[fit.eligible & ~fit.held & ~fit.held_group]
    dem = (whitespace.groupby(["product_id", "product_code", "product_group", "segment"])
           .agg(eligible_customers=("customer_id", "count"),
                avg_propensity=("propensity", "mean"),
                high_propensity=("propensity", lambda s: int((s >= 0.6).sum())),
                sum_propensity=("propensity", "sum")).reset_index())
    holders = hold.merge(P.dim_product_frame()[["product_id"]], on="product_id")
    hcount = holders.groupby("product_id").size()
    dem["current_holders"] = dem.product_id.map(hcount).fillna(0).astype(int)
    dem["expected_adopters_90d"] = np.round(dem.sum_propensity * BASE_CONV_90D * CONTACT_CAPACITY, 1)
    dem["expected_adopters_30d"] = np.round(dem.expected_adopters_90d * 0.38, 1)
    dem["expected_adopters_60d"] = np.round(dem.expected_adopters_90d * 0.70, 1)
    dem["avg_propensity"] = dem.avg_propensity.round(4)
    dem = dem.drop(columns="sum_propensity").sort_values(
        ["product_group", "expected_adopters_90d"], ascending=[True, False])
    dem.to_parquet(os.path.join(PARQUET, "agg_product_demand.parquet"), index=False)
    dem.to_csv(os.path.join(CSV, "agg_product_demand.csv"), index=False)

    # ---- segment x product affinity matrix -------------------------
    aff = (fit.groupby(["segment", "product_code"]).propensity.mean().round(3)
           .reset_index().pivot(index="product_code", columns="segment", values="propensity"))
    aff.to_parquet(os.path.join(PARQUET, "agg_segment_product_affinity.parquet"))
    aff.to_csv(os.path.join(CSV, "agg_segment_product_affinity.csv"))

    _report(dimp, fit, reco, dem, hold, mart)
    print(f"OK  {len(P.CATALOGUE)} sản phẩm | {mart.customer_id.nunique():,} khách | "
          f"ai_product_fit {len(ai_fit):,} dòng | recommendation {len(reco):,} dòng")
    return ai_fit, reco, dem


def _report(dimp, fit, reco, dem, hold, mart):
    os.makedirs(MODELS, exist_ok=True)
    n = mart.customer_id.nunique()
    top_prod = (reco[reco.priority_rank == 1].groupby("product_code").size()
                .sort_values(ascending=False).head(12))
    demand = (dem.groupby(["product_code", "product_group"])
              .agg(eligible=("eligible_customers", "sum"),
                   exp90=("expected_adopters_90d", "sum"),
                   avg_p=("avg_propensity", "mean")).reset_index()
              .sort_values("exp90", ascending=False))
    hold_rate = hold.groupby("product_id").size().reindex(dimp.product_id).fillna(0) / n
    grp_hold = (hold.merge(dimp[["product_id", "product_group"]], on="product_id")
                .groupby("product_group").size())
    L = []
    L.append("# Phân tích mức độ phù hợp & dự báo cầu sản phẩm — MSB Smart Growth Engine\n")
    L.append(f"*Ngày phân tích: {AS_OF} · {n:,} khách hàng · {len(P.CATALOGUE)} sản phẩm chuẩn hoá "
             f"(từ {958} mã trong MSB_products_description.xlsx)*\n")
    L.append("## 1. Danh mục sản phẩm chuẩn hoá\n")
    L.append("| Nhóm | Số SP | KH đang sở hữu (mô phỏng) |")
    L.append("|---|--:|--:|")
    for g in P.GROUP_ORDER:
        L.append(f"| {g} | {(dimp.product_group == g).sum()} | {int(grp_hold.get(g, 0)):,} |")
    L.append("")
    L.append("## 2. Sản phẩm được đề xuất #1 nhiều nhất (Next Best Product)\n")
    L.append("| Sản phẩm | Số KH đứng #1 | % KH |")
    L.append("|---|--:|--:|")
    for code, cnt in top_prod.items():
        L.append(f"| {code} | {cnt:,} | {cnt / n * 100:.1f}% |")
    L.append("")
    L.append("## 3. Dự báo cầu 90 ngày (tệp đủ điều kiện, chưa sở hữu, có tiếp cận)\n")
    L.append(f"*Giả định: tỉ lệ chuyển đổi nền {BASE_CONV_90D:.0%} trên tệp được tiếp cận, "
             f"năng lực tiếp cận {CONTACT_CAPACITY:.0%} tệp đủ điều kiện/quý.*\n")
    L.append("| Sản phẩm | Nhóm | Tệp đủ điều kiện | Propensity TB | Dự báo mở mới 90N |")
    L.append("|---|---|--:|--:|--:|")
    for _, r in demand.head(20).iterrows():
        L.append(f"| {r.product_code} | {r.product_group} | {int(r.eligible):,} | "
                 f"{r.avg_p:.2f} | **{r.exp90:,.0f}** |")
    L.append(f"\n**Tổng dự báo mở mới 90 ngày: ~{demand.exp90.sum():,.0f} sản phẩm** "
             f"(30N: ~{dem.expected_adopters_30d.sum():,.0f}, 60N: ~{dem.expected_adopters_60d.sum():,.0f}).\n")
    L.append("## 4. Phân khúc phù hợp nhất theo nhóm sản phẩm\n")
    aff = fit.groupby(["segment", "product_group"]).propensity.mean().round(3).reset_index()
    piv = aff.pivot(index="segment", columns="product_group", values="propensity")
    L.append("| Phân khúc | " + " | ".join(piv.columns) + " |")
    L.append("|---|" + "|".join(["--:"] * len(piv.columns)) + "|")
    for s, row in piv.iterrows():
        L.append(f"| {s} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
    L.append("")
    L.append("## 5. Khoảng trống danh mục (portfolio gap)\n")
    gap = fit[fit.eligible & ~fit.held & ~fit.held_group & (fit.propensity >= 0.6)]
    # gap theo NHÓM: khách chưa có sp nào trong nhóm nhưng phù hợp >= 1 sp trong nhóm
    grp_gap = gap.groupby(["customer_id", "product_group"]).size().reset_index()
    per_cust = grp_gap.groupby("customer_id").size()
    L.append(f"- {per_cust.gt(0).sum():,} / {n:,} khách ({per_cust.gt(0).mean()*100:.0f}%) có "
             f"ít nhất 1 **nhóm sản phẩm** phù hợp (propensity ≥ 0.6) nhưng chưa sở hữu.")
    L.append(f"- Trung bình {per_cust.mean():.1f} nhóm whitespace / khách "
             f"(tối đa 4: CARD / CASA / FD / LENDING).")
    L.append("- Cơ hội cross-sell theo nhóm: " + ", ".join(
        f"{g} ({c:,} khách)" for g, c in
        grp_gap.groupby("product_group").customer_id.nunique().sort_values(ascending=False).items()) + ".")
    L.append(f"- Tổng cơ hội (khách × sản phẩm, propensity ≥ 0.6): {len(gap):,}.")
    L.append("")
    L.append("## 6. Cách dùng\n")
    L.append("- `ai_product_recommendation_v2` — top-6 sản phẩm phù hợp nhất mỗi khách, kèm lý do.")
    L.append("- `agg_product_demand` — dự báo cầu theo sản phẩm × phân khúc, đưa vào kế hoạch KPI/chiến dịch.")
    L.append("- `agg_segment_product_affinity` — ma trận phân khúc × sản phẩm cho định hướng danh mục.")
    L.append("- App Streamlit: trang **“Sản phẩm & Nhu cầu”**.")
    open(os.path.join(MODELS, "product_analysis_report.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    build(ap.parse_args().seed)
