"""
MSB SMART GROWTH ENGINE — AI Feedback Agent
==========================================

Vòng lặp Human-in-the-loop cho tài liệu Action 11–12 ("Khách hàng phản hồi" →
"Learning & Next Decision"): RM phản hồi với đề xuất của AI → agent **xem xét lại
mô hình**, **giải thích** vì sao AI đề xuất như vậy, phân biệt "AI sai thật" vs
"RM thận trọng", rồi **hiệu chỉnh** (rule override / thắt gate) khi mô hình sai thật.

Pipeline:
  build_feedback()  -> rm_feedback_ai            (RM verdict cho từng đề xuất)
  review()          -> ai_agent_review           (finding + giải thích + fix đề xuất + impact)
  apply()           -> models/rule_overrides.json + ai_model_adjustment
                       (products.py đọc file này ở lần chấm điểm sau)
  report()          -> models/ai_agent_report.md

    py src/rm_feedback_agent.py                 # build feedback + review (chưa sửa)
    py src/rm_feedback_agent.py --apply         # + áp hiệu chỉnh cho các finding "AI sai thật"
    py src/product_analysis.py                  # chấm lại với override -> RM Desk cập nhật
    py src/rm_feedback_agent.py                 # review lại -> agreement tăng
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

import products as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(ROOT, "data", "parquet")
CSV = os.path.join(ROOT, "data", "csv")
MODELS = os.path.join(ROOT, "models")
OVR_PATH = os.path.join(MODELS, "rule_overrides.json")
AS_OF = date(2026, 8, 31)

VERDICTS = ["AGREE", "NOT_RELEVANT", "ALREADY_HAS", "CANT_AFFORD", "WRONG_TIMING", "NO_NEED_NOW"]
MIN_N = 25                    # số phản hồi tối thiểu để agent kết luận
AGREE_FLOOR = 0.55            # dưới ngưỡng này mới nghi mô hình sai
DOMINANT_FLOOR = 0.40         # 1 lý do từ chối phải chiếm >= 40%
CONVERT_OK = 0.18             # timing-issue mà nhóm tiếp cận vẫn convert >= mức này -> RM thận trọng, KHÔNG sửa
TARGETING_VERDICTS = {"NOT_RELEVANT", "CANT_AFFORD", "ALREADY_HAS"}  # lý do = chọn sai KH


def _read(name):
    return pd.read_parquet(os.path.join(PARQUET, f"{name}.parquet"))


# ==========================================================================
# 1. RM feedback (giả lập, có CHÈN SẴN 3 lỗi mô hình để agent phát hiện)
# ==========================================================================
def build_feedback(seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 7)
    reco = _read("ai_product_recommendation_v2")
    mart = _read("customer_360_feature_mart")[
        ["customer_id", "segment", "income_monthly", "net_cashflow_30d",
         "has_active_loan", "travel_spending_90d", "family_flag"]]
    r = reco.merge(mart, on="customer_id", how="left")
    # RM chỉ review 1 phần danh mục (rank 1–3)
    r = r[r.priority_rank <= 3].sample(frac=0.35, random_state=seed).reset_index(drop=True)
    n = len(r)

    prop = r.propensity.to_numpy()
    p_agree = np.clip(0.42 + 0.9 * (prop - 0.5), 0.05, 0.9)

    seg = r.segment.fillna("MASS").to_numpy()
    code = r.product_code.to_numpy()
    inc = r.income_monthly.fillna(15e6).to_numpy()
    fam = r.family_flag.fillna(False).to_numpy()

    # ---- 3 LỖI MÔ HÌNH CHÈN SẴN để agent phát hiện (rule/gate quá lỏng) ----
    # LỖI 1: Tiết kiệm Măng non — gate = family_flag (cờ suy đoán theo tuổi, nhiễu)
    e1 = (code == "FD_SAV_KIDS") & fam & np.isin(seg, ["MASS", "MASS_AFFLUENT"])
    # LỖI 2: Vay mua ô tô — gate = auto_intent (bắt cả KH chỉ đổ xăng/bảo dưỡng xe)
    e2 = (code == "LEND_AUTO_NEW") & np.isin(seg, ["MASS", "MASS_AFFLUENT"])
    # LỖI 3: Vay tín chấp theo lương — gate không xét thu nhập thấp
    e3 = (code == "LEND_UNSECURED_PAYROLL") & (inc < 25e6)

    p_agree = np.where(e1, 0.22, p_agree)
    p_agree = np.where(e2, 0.26, p_agree)
    p_agree = np.where(e3, 0.20, p_agree)

    agree = rng.random(n) < p_agree
    dis = ~agree
    verdict = np.where(agree, "AGREE", "NO_NEED_NOW")
    # nhiễu chung: phần lớn lý do "mềm" (chưa quan tâm / sai thời điểm)
    verdict = np.where(dis, rng.choice(
        ["NO_NEED_NOW", "WRONG_TIMING", "ALREADY_HAS", "NOT_RELEVANT"],
        size=n, p=[0.5, 0.28, 0.14, 0.08]), verdict)
    # lý do từ chối theo pattern lỗi (đè lên nhiễu, ~85% số ca lỗi)
    hit = rng.random(n) < 0.9
    verdict = np.where(dis & e1 & hit, "NOT_RELEVANT", verdict)
    verdict = np.where(dis & e2 & hit, "NOT_RELEVANT", verdict)
    verdict = np.where(dis & e3 & hit, "CANT_AFFORD", verdict)

    # KH được RM theo đuổi (AGREE) -> converted theo propensity; nhóm CC_SIGNATURE
    # cố tình để RM hay nói WRONG_TIMING nhưng vẫn convert tốt (RM thận trọng, không phải lỗi)
    skeptic = (code == "CARD_CC_SIGNATURE")
    verdict = np.where(skeptic & dis & (rng.random(n) < 0.6), "WRONG_TIMING", verdict)
    p_conv = np.where(agree, np.clip(prop * 0.45, 0.02, 0.6), 0.0)
    p_conv = np.where(skeptic & agree, np.clip(prop * 0.6, 0.1, 0.7), p_conv)
    converted = rng.random(n) < p_conv

    comment_map = {
        "NOT_RELEVANT": "KH không có nhu cầu/đặc điểm phù hợp sản phẩm này.",
        "ALREADY_HAS": "KH đã có sản phẩm tương đương ở NH khác.",
        "CANT_AFFORD": "Dòng tiền/khả năng trả nợ chưa đủ điều kiện.",
        "WRONG_TIMING": "Sản phẩm ổn nhưng chưa đúng thời điểm.",
        "NO_NEED_NOW": "KH chưa quan tâm ở thời điểm hiện tại.",
        "AGREE": "Đề xuất hợp lý, đã tiếp cận KH.",
    }
    days = rng.integers(0, 45, n)
    return pd.DataFrame({
        "feedback_id": [f"RFB{i:09d}" for i in range(1, n + 1)],
        "customer_id": r.customer_id.to_numpy(),
        "product_id": r.product_id.to_numpy(),
        "product_code": code,
        "product_group": r.product_group.to_numpy(),
        "segment": seg,
        "priority_rank": r.priority_rank.to_numpy(),
        "priority_level": r.priority_level.to_numpy(),
        "propensity": prop.round(4),
        "rm_id": [f"RM{int(x):04d}" for x in rng.integers(1, 180, n)],
        "rm_verdict": verdict,
        "agree_flag": agree,
        "rm_comment": [comment_map[v] for v in verdict],
        "converted_flag": converted,
        "feedback_date": [AS_OF - timedelta(days=int(x)) for x in days],
    })


# ==========================================================================
# 2. Agent review — xem xét lại mô hình
# ==========================================================================
def _gate_text(code):
    g = {
        "FD_SAV_KIDS": "family_flag = 1 (cờ suy đoán theo độ tuổi, độ tin cậy thấp — không xác nhận KH có con)",
        "LEND_AUTO_NEW": "auto_intent = 1 (có giao dịch danh mục ô tô — bắt cả KH chỉ đổ xăng / bảo dưỡng)",
        "LEND_UNSECURED_PAYROLL": "nhận lương qua MSB / thu nhập > 15tr (không đặt sàn thu nhập, không xét khả năng trả nợ)",
    }
    return g.get(code, "rule fit tổng hợp theo cột 'Khách hàng/Nhu cầu phù hợp'")


def _diagnose(dom_verdict, dom_share, agree_rate, conv_rate, scoped_segment):
    # đồng thuận đủ cao -> mô hình ổn
    if agree_rate >= AGREE_FLOOR:
        return "OK", False
    # lý do "sai thời điểm" nhưng nhóm tiếp cận vẫn chuyển đổi tốt -> RM thận trọng, KHÔNG sửa
    if dom_verdict == "WRONG_TIMING":
        return ("RM_SKEPTICISM", False) if conv_rate >= CONVERT_OK else ("TIMING_RULE", False)
    if dom_share < DOMINANT_FLOOR:
        return "INSUFFICIENT_SIGNAL", False
    # lý do từ chối = "chọn SAI khách hàng" -> mô hình sai thật
    if dom_verdict == "CANT_AFFORD":
        return "GATE_INCOME_TOO_LOOSE", True
    if dom_verdict == "ALREADY_HAS":
        return "HOLDINGS_GATE_MISSING", True
    if dom_verdict == "NOT_RELEVANT":
        return ("RULE_TOO_LOOSE_SEGMENT" if scoped_segment else "RULE_TOO_LOOSE"), True
    # NO_NEED_NOW = nhu cầu mềm, không phải lỗi chọn KH
    return "DEMAND_SOFT", False


def review(feedback: pd.DataFrame) -> pd.DataFrame:
    fit = _read("ai_product_fit")[["customer_id", "product_id", "fit_score"]]
    reco = _read("ai_product_recommendation_v2")[["customer_id", "product_id", "priority_rank"]]
    cur_min_fit = P.load_overrides()
    rows = []
    rid = 1

    def _cell(sub, scope, code, group, seg):
        nonlocal rid
        if len(sub) < MIN_N:
            return
        vc = sub.rm_verdict.value_counts()
        agree_rate = float((sub.rm_verdict == "AGREE").mean())
        dom = vc.drop("AGREE", errors="ignore")
        dom_verdict = dom.index[0] if len(dom) else "—"
        dom_share = float(dom.iloc[0] / len(sub)) if len(dom) else 0.0
        acted = sub[sub.agree_flag]
        conv_rate = float(acted.converted_flag.mean()) if len(acted) else 0.0
        ftype, model_wrong = _diagnose(dom_verdict, dom_share, agree_rate, conv_rate, seg is not None)

        # impact: số đề xuất rank<=3 hiện tại sẽ bị loại nếu áp fix
        pool = reco[reco.product_id == code_to_id(code)]
        if seg is not None:
            segset = set(feedback.loc[feedback.segment == seg, "customer_id"])
            pool = pool[pool.customer_id.isin(segset)]
        impact_recos = int(len(pool))
        base_fit = cur_min_fit.get(code, {}).get("min_fit", 0.50)

        fix, expl = _fix_and_explain(ftype, code, group, seg, base_fit, agree_rate,
                                     dom_verdict, dom_share, conv_rate, len(sub))
        rows.append({
            "review_id": f"AGR{rid:05d}", "run_date": AS_OF, "scope": scope,
            "product_code": code, "product_group": group, "segment": seg or "(tất cả)",
            "n_feedback": len(sub), "agree_rate": round(agree_rate, 3),
            "dominant_verdict": dom_verdict, "dominant_share": round(dom_share, 3),
            "conversion_rate_acted": round(conv_rate, 3),
            "finding_type": ftype, "model_is_wrong": bool(model_wrong),
            "explanation": expl, "proposed_fix": json.dumps(fix, ensure_ascii=False),
            "impact_recos": impact_recos,
            "status": "PROPOSED" if model_wrong else "NO_ACTION",
        })
        rid += 1

    for code, sub in feedback.groupby("product_code"):
        grp = sub.product_group.iloc[0]
        _cell(sub, "product", code, grp, None)
        for seg, s2 in sub.groupby("segment"):
            _cell(s2, "product+segment", code, grp, seg)

    df = pd.DataFrame(rows)
    # giữ: mọi finding "sai thật" + finding có giải thích; bỏ nhiễu mẫu nhỏ
    keep = (df.model_is_wrong
            | df.finding_type.isin(["RM_SKEPTICISM", "TIMING_RULE", "DEMAND_SOFT"])
            | ((df.finding_type == "INSUFFICIENT_SIGNAL") & (df.n_feedback >= 80)))
    return (df[df.finding_type.ne("OK") & keep]
            .sort_values(["model_is_wrong", "n_feedback"], ascending=[False, False])
            .reset_index(drop=True))


def code_to_id(code):
    return P.BY_CODE[code]["product_id"]


def _fix_and_explain(ftype, code, group, seg, base_fit, agree_rate, dom, dom_share, conv, n):
    gate = _gate_text(code)
    head = (f"AI đề xuất **{P.BY_CODE[code]['product_name']}** "
            f"{'cho phân khúc ' + seg + ' ' if seg else ''}chủ yếu vì rule "
            f"\"{P.BY_CODE[code]['need']}\" và hard-gate: {gate}.")
    stat = (f"RM phản hồi: chỉ **{agree_rate:.0%} đồng ý** trên {n} lượt; "
            f"lý do từ chối chính là **{dom}** ({dom_share:.0%}); "
            f"tỉ lệ chuyển đổi của nhóm được RM theo đuổi = {conv:.0%}.")
    if ftype == "RM_SKEPTICISM":
        return ({"kind": "NONE"},
                f"{head} {stat} → Nhóm này **vẫn chuyển đổi tốt** khi RM tiếp cận "
                f"(> {CONVERT_OK:.0%}), nên đây là **RM thận trọng / lệch thời điểm**, "
                f"KHÔNG phải mô hình sai. Giữ nguyên, chỉ nhắc lại về timing.")
    if ftype == "GATE_INCOME_TOO_LOOSE":
        return ({"kind": "min_income", "product_code": code, "from": None, "to": 20_000_000,
                 "note": f"RM feedback {dom} — gate không kiểm tra dòng tiền ròng & dư nợ"},
                f"{head} {stat} → **Mô hình SAI THẬT**: gate không kiểm tra khả năng trả nợ "
                f"(dòng tiền ròng, dư nợ hiện có). Hiệu chỉnh: thêm điều kiện thu nhập ≥ 20tr "
                f"cho {code}.")
    if ftype == "HOLDINGS_GATE_MISSING":
        return ({"kind": "min_fit", "product_code": code, "from": base_fit, "to": round(base_fit + 0.08, 2),
                 "note": f"RM feedback {dom} — dữ liệu holdings có thể chưa đủ"},
                f"{head} {stat} → **Mô hình SAI THẬT**: KH đã sở hữu sản phẩm tương đương "
                f"mà hệ thống chưa ghi nhận. Hiệu chỉnh tạm: nâng ngưỡng fit ({base_fit:.2f} → "
                f"{base_fit + 0.08:.2f}); song song rà soát nguồn dữ liệu holdings.")
    if ftype == "RULE_TOO_LOOSE_SEGMENT":
        return ({"kind": "block_segments", "product_code": code, "add": [seg],
                 "note": f"RM feedback {dom} tập trung ở phân khúc {seg}"},
                f"{head} {stat} → **Mô hình SAI THẬT ở phân khúc {seg}**: gate quá lỏng, "
                f"bắt nhầm KH không thực sự phù hợp. Hiệu chỉnh: loại phân khúc {seg} khỏi "
                f"tệp đủ điều kiện của {code} (hoặc nâng ngưỡng gate theo tín hiệu thật).")
    if ftype == "RULE_TOO_LOOSE":
        return ({"kind": "min_fit", "product_code": code, "from": base_fit, "to": round(base_fit + 0.10, 2),
                 "note": f"RM feedback {dom} trải đều các phân khúc"},
                f"{head} {stat} → **Mô hình SAI THẬT**: rule fit quá lỏng. Hiệu chỉnh: "
                f"nâng ngưỡng fit_score {base_fit:.2f} → {base_fit + 0.10:.2f} cho {code}.")
    if ftype == "TIMING_RULE":
        return ({"kind": "NONE"},
                f"{head} {stat} → Sản phẩm phù hợp nhưng **sai thời điểm** — không phải lỗi "
                f"chọn sản phẩm. Đề xuất: chuyển sang nurture, đặt lại contact window.")
    if ftype == "DEMAND_SOFT":
        return ({"kind": "NONE"},
                f"{head} {stat} → Lý do từ chối chủ yếu là \"chưa quan tâm\" (nhu cầu mềm), "
                f"không phải chọn sai KH. Không sửa mô hình; tối ưu nội dung / ưu đãi / thời điểm.")
    return ({"kind": "NONE"},
            f"{head} {stat} → Chưa đủ tín hiệu nhất quán để kết luận. Tiếp tục thu thập feedback.")


# ==========================================================================
# 3. Apply — hiệu chỉnh mô hình khi "sai thật"
# ==========================================================================
def apply(reviews: pd.DataFrame):
    ovr = P.load_overrides()
    adj = []
    aid = 1
    for _, r in reviews[reviews.model_is_wrong & (reviews.status == "PROPOSED")].iterrows():
        fix = json.loads(r.proposed_fix)
        code = fix.get("product_code")
        if not code:
            continue
        cur = ovr.setdefault(code, {})
        before = json.dumps({k: cur.get(k) for k in ("min_fit", "block_segments", "min_income")},
                            ensure_ascii=False)
        if fix["kind"] == "min_fit":
            cur["min_fit"] = float(fix["to"])
        elif fix["kind"] == "min_income":
            cur["min_income"] = float(fix["to"])
        elif fix["kind"] == "block_segments":
            cur["block_segments"] = sorted(set(cur.get("block_segments", [])) | set(fix["add"]))
        cur["note"] = fix.get("note", "")
        cur["applied_by"] = "ai_feedback_agent"
        cur["applied_date"] = str(AS_OF)
        cur["review_id"] = r.review_id
        adj.append({
            "adjustment_id": f"ADJ{aid:05d}", "review_id": r.review_id, "run_date": AS_OF,
            "product_code": code, "finding_type": r.finding_type, "kind": fix["kind"],
            "before": before, "after": json.dumps(
                {k: cur.get(k) for k in ("min_fit", "block_segments", "min_income")},
                ensure_ascii=False),
            "impact_recos": int(r.impact_recos),
        })
        aid += 1
    os.makedirs(MODELS, exist_ok=True)
    with open(OVR_PATH, "w", encoding="utf-8") as fh:
        json.dump(ovr, fh, ensure_ascii=False, indent=2)
    adj_df = pd.DataFrame(adj)
    reviews = reviews.copy()
    reviews.loc[reviews.model_is_wrong & (reviews.status == "PROPOSED"), "status"] = "APPLIED"
    print(f"  Đã áp {len(adj)} hiệu chỉnh -> {os.path.relpath(OVR_PATH, ROOT)}")
    print("  Chạy tiếp:  py src/product_analysis.py   (chấm lại điểm với override)")
    return reviews, adj_df


# ==========================================================================
# 4. Report
# ==========================================================================
def report(feedback, reviews, adj_df):
    L = ["# AI Feedback Agent — Xem xét lại & hiệu chỉnh mô hình\n",
         f"*{AS_OF} · {len(feedback):,} lượt RM phản hồi · "
         f"{feedback.agree_flag.mean()*100:.0f}% đồng ý tổng thể*\n"]

    L.append("## 1. Mức độ đồng thuận RM ↔ AI theo nhóm sản phẩm\n")
    g = feedback.groupby("product_group").agg(
        n=("feedback_id", "count"), agree=("agree_flag", "mean"),
        conv=("converted_flag", "mean")).reset_index()
    L.append("| Nhóm | Lượt phản hồi | % đồng ý | % chuyển đổi (đã tiếp cận) |")
    L.append("|---|--:|--:|--:|")
    for _, r in g.iterrows():
        L.append(f"| {r.product_group} | {int(r.n):,} | {r.agree*100:.0f}% | {r.conv*100:.0f}% |")

    L.append("\n## 2. Phát hiện của agent\n")
    wrong = reviews[reviews.model_is_wrong]
    L.append(f"- Tổng {len(reviews)} vùng nghi vấn được rà soát; "
             f"**{len(wrong)} vùng kết luận MÔ HÌNH SAI THẬT** (đã đủ mẫu, lý do từ chối nhất quán, "
             f"và nhóm được tiếp cận vẫn không chuyển đổi).")
    L.append(f"- {len(reviews) - len(wrong)} vùng còn lại: RM thận trọng / sai thời điểm / "
             f"chưa đủ tín hiệu → **không sửa mô hình**.\n")
    for _, r in reviews.iterrows():
        flag = "🔴 SAI THẬT" if r.model_is_wrong else "⚪ không sửa"
        L.append(f"### {flag} — {r.product_code}"
                 + (f" · phân khúc {r.segment}" if r.segment != "(tất cả)" else ""))
        L.append(f"*{r.finding_type} · n={r.n_feedback} · đồng ý {r.agree_rate:.0%} · "
                 f"lý do chính {r.dominant_verdict} ({r.dominant_share:.0%}) · "
                 f"chuyển đổi nhóm tiếp cận {r.conversion_rate_acted:.0%}*\n")
        L.append(r.explanation + "\n")

    if adj_df is not None and len(adj_df):
        L.append("## 3. Hiệu chỉnh đã áp dụng (`models/rule_overrides.json`)\n")
        L.append("| Sản phẩm | Loại | Trước | Sau | Đề xuất bị ảnh hưởng |")
        L.append("|---|---|---|---|--:|")
        for _, a in adj_df.iterrows():
            L.append(f"| {a.product_code} | {a.kind} | `{a.before}` | `{a.after}` | {a.impact_recos:,} |")
        L.append("\n→ Chạy `py src/product_analysis.py` để chấm lại điểm; "
                 "`py src/rm_feedback_agent.py` để đo lại agreement (kỳ vọng tăng).")
    else:
        L.append("## 3. Hiệu chỉnh\n_Chạy với `--apply` để áp các fix cho vùng 'SAI THẬT'._")

    os.makedirs(MODELS, exist_ok=True)
    open(os.path.join(MODELS, "ai_agent_report.md"), "w", encoding="utf-8").write("\n".join(L))


def _dump(name, df):
    df.to_parquet(os.path.join(PARQUET, f"{name}.parquet"), index=False)
    df.to_csv(os.path.join(CSV, f"{name}.csv"), index=False)
    print(f"  {name:<26} {len(df):>7,} dòng")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply", action="store_true", help="áp hiệu chỉnh cho các finding 'SAI THẬT'")
    args = ap.parse_args()

    fb = build_feedback(args.seed)
    _dump("rm_feedback_ai", fb)
    rv = review(fb)
    adj_df = None
    if args.apply:
        rv, adj_df = apply(rv)
        _dump("ai_model_adjustment", adj_df)
    _dump("ai_agent_review", rv)
    report(fb, rv, adj_df)
    n_wrong = int(rv.model_is_wrong.sum())
    print(f"\nAgreement RM↔AI: {fb.agree_flag.mean()*100:.1f}%  |  "
          f"{len(rv)} vùng nghi vấn, {n_wrong} vùng 'mô hình sai thật'"
          + ("  → đã áp fix" if args.apply else "  (chạy --apply để sửa)"))
    print("Báo cáo: models/ai_agent_report.md")


if __name__ == "__main__":
    main()
