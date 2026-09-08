"""
MSB SMART GROWTH ENGINE — Danh mục sản phẩm chuẩn hoá + rule "khách hàng phù hợp"
==============================================================================

Chuẩn hoá từ `MSB_products_description.xlsx` (958 mã sản phẩm MSB, phần lớn là mã
legacy / nội bộ) → **35 sản phẩm bán được** cho khách hàng cá nhân (IND) và
hộ kinh doanh / DN nhỏ (SME), gom theo cột *"Nhóm sản phẩm chuẩn hoá"* và chỉ lấy
nhóm khuyến nghị *"Có thể tư vấn"* + *"Bán có điều kiện"*.

Mỗi sản phẩm có:
  - metadata: group (CARD/CASA/FD/LENDING), subgroup, tier, customer_type, need
  - `rules`: danh sách (điều_kiện, trọng_số, lý_do) — điều kiện là biểu thức trên
    customer_360_feature_mart (đã join dim_customer). Vectorised trên toàn bộ mart.
  - `base`: propensity nền (khi không có tín hiệu nào)

`fit_table(mart)` → DataFrame dài [customer_id, product_id, eligible, fit_score,
top_reasons] cho mọi (khách × sản phẩm).

`build_holdings(mart, rng)` → mô phỏng sản phẩm khách ĐANG sở hữu (phục vụ phân tích
gap danh mục + loại sản phẩm đã có khỏi đề xuất).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ helpers
INCOME_VND = {"<10M": 7e6, "10-20M": 15e6, "20-40M": 30e6, "40-80M": 60e6, "80M+": 120e6}
AFFLUENT_SEG = ("AFFLUENT", "PRIVATE")
YOUNG = ("18-25", "26-35")
FAMILY_AGE = ("26-35", "36-45", "46-55")


def _col(m, name, default=0):
    return m[name] if name in m.columns else pd.Series(default, index=m.index)


def _feat(m):
    """Các biến dẫn xuất dùng chung cho rule (Series cùng index với mart)."""
    seg = _col(m, "segment", "MASS").astype(str)
    ctype = _col(m, "customer_type", "INDIVIDUAL").astype(str)
    occ = _col(m, "occupation_group", "").astype(str)
    ind = _col(m, "industry_group", "").astype(str)
    inc = _col(m, "income_monthly", 0).astype(float)
    inc = inc.where(inc > 0, _col(m, "income_band", "10-20M").map(INCOME_VND).fillna(15e6))
    bal = _col(m, "avg_balance_90d", 0).astype(float)
    age = _col(m, "age_group", "").astype(str)
    return dict(
        seg=seg, ctype=ctype, occ=occ, ind=ind, income=inc, bal=bal, age=age,
        retired=age.isin(("56-65", "65+")) | occ.eq("RETIRED"),
        is_sme=(ctype.eq("SME") | occ.eq("BUSINESS_OWNER")),
        is_ind=ctype.ne("SME"),
        affluent=seg.isin(AFFLUENT_SEG),
        young=_col(m, "age_group", "").astype(str).isin(YOUNG),
        family=_col(m, "family_flag", False).astype(bool) | _col(m, "age_group", "").astype(str).isin(FAMILY_AGE) & (_col(m, "spending_90d", 0) > 20e6),
        salary=_col(m, "salary_flag", False).astype(bool),
        digital=_col(m, "digital_engagement_score", 0).astype(float),
        online_spend=_col(m, "online_spending_90d", 0).astype(float),
        travel_spend=_col(m, "travel_spending_90d", 0).astype(float),
        fx_spend=_col(m, "international_spending_90d", 0).astype(float),
        spend90=_col(m, "spending_90d", 0).astype(float),
        deposit_bal=_col(m, "deposit_balance", 0).astype(float),
        growth=_col(m, "balance_growth_3m", 0).astype(float),
        netcf=_col(m, "net_cashflow_30d", 0).astype(float),
        has_cc=_col(m, "has_credit_card", False).astype(bool),
        has_premium=_col(m, "has_premium_card", False).astype(bool),
        has_loan=_col(m, "has_active_loan", False).astype(bool),
        has_invest=_col(m, "has_investment", False).astype(bool),
        auto_intent=_col(m, "auto_intent_flag", False).astype(bool),
        home_intent=_col(m, "home_intent_flag", False).astype(bool),
        agri=_col(m, "agri_flag", False).astype(bool) | ind.str.contains("AGRI", case=False, na=False),
        securities=_col(m, "securities_value", 0).astype(float) + _col(m, "deposit_balance", 0).astype(float),
        rel_years=_col(m, "relationship_years", 0).astype(float),
        idle_cash=(_col(m, "avg_balance_90d", 0).astype(float) - _col(m, "spending_90d", 0).astype(float) / 3).clip(lower=0),
    )


# ------------------------------------------------------------------ catalogue
#  code, name_vi, group, subgroup, tier, customer_type, need, base_propensity, rules(f)->list[(cond, w, reason)]
def _catalogue():
    C = []

    def add(code, name, group, sub, tier, ctype, need, base, rules):
        C.append(dict(product_code=code, product_name=name, product_group=group,
                      subgroup=sub, product_tier=tier, customer_type=ctype,
                      need=need, base=base, rules=rules))

    # ---------------- CARD -------------------------------------------------
    add("CARD_DEBIT_DOMESTIC", "Thẻ ghi nợ nội địa Napas", "CARD", "DEBIT", None, "IND",
        "Cần thẻ để rút tiền, thanh toán và giao dịch hằng ngày", 0.35,
        lambda f: [(~f["has_cc"], .25, "chưa có thẻ nào"),
                   (f["salary"], .25, "có tài khoản nhận lương"),
                   (f["digital"] > 40, .2, "giao dịch số thường xuyên"),
                   (f["young"], .15, "khách hàng trẻ")])
    add("CARD_DEBIT_INTL_VISA", "Thẻ ghi nợ quốc tế Visa Classic", "CARD", "DEBIT", "CLASSIC", "IND",
        "Thanh toán quốc tế cơ bản, kiểm soát chi tiêu bằng số dư tài khoản", 0.22,
        lambda f: [(f["fx_spend"] > 0, .35, "có chi tiêu ngoại tệ"),
                   (f["travel_spend"] > 3e6, .25, "chi tiêu du lịch"),
                   (f["online_spend"] > 5e6, .2, "mua sắm online nhiều"),
                   (~f["has_premium"], .1, "chưa có thẻ quốc tế cao cấp")])
    add("CARD_DEBIT_PLATINUM_FCB", "Thẻ ghi nợ Mastercard Platinum FCB", "CARD", "DEBIT", "PLATINUM", "IND",
        "Khách ưu tiên cần thẻ ghi nợ quốc tế hạn mức / tiện ích cao", 0.15,
        lambda f: [(f["affluent"], .4, "phân khúc ưu tiên"),
                   (f["fx_spend"] + f["travel_spend"] > 10e6, .3, "chi tiêu quốc tế/du lịch cao"),
                   (f["bal"] > 200e6, .2, "số dư lớn")])
    add("CARD_CC_MDIGI", "Thẻ tín dụng Mastercard mDigi", "CARD", "CREDIT", "STANDARD", "IND",
        "Khách trẻ, chi tiêu ẩm thực, du lịch và dịch vụ số", 0.30,
        lambda f: [(f["young"], .3, "khách hàng trẻ"),
                   (f["online_spend"] > 4e6, .3, "chi tiêu số cao"),
                   (~f["has_cc"], .25, "chưa có thẻ tín dụng"),
                   (f["salary"], .15, "thu nhập ổn định")])
    add("CARD_CC_ONLINE", "Thẻ tín dụng Visa Online", "CARD", "CREDIT", "STANDARD", "IND",
        "Khách mua sắm / giải trí trực tuyến thường xuyên", 0.28,
        lambda f: [(f["online_spend"] > 6e6, .4, "mua sắm online thường xuyên"),
                   (~f["has_cc"], .3, "chưa có thẻ tín dụng"),
                   (f["digital"] > 50, .2, "tương tác số cao")])
    add("CARD_CC_FAMILY", "Thẻ tín dụng Mastercard Family", "CARD", "CREDIT", "GOLD", "IND",
        "Khách có gia đình, chi tiêu thường xuyên cho giáo dục / chăm sóc / mua sắm", 0.27,
        lambda f: [(f["family"], .4, "khách có gia đình"),
                   (f["spend90"] > 30e6, .3, "chi tiêu hộ gia đình lớn"),
                   (~f["has_premium"], .2, "chưa có thẻ tín dụng cao cấp")])
    add("CARD_CC_TRAVEL", "Thẻ tín dụng Visa Travel", "CARD", "CREDIT", "PLATINUM", "IND",
        "Khách thường xuyên du lịch, đặt vé / khách sạn và thanh toán quốc tế", 0.25,
        lambda f: [(f["travel_spend"] > 8e6, .45, "chi tiêu du lịch cao"),
                   (f["fx_spend"] > 5e6, .3, "thanh toán quốc tế"),
                   (f["income"] > 40e6, .15, "thu nhập khá/cao")])
    add("CARD_CC_SIGNATURE", "Thẻ tín dụng Visa Signature", "CARD", "CREDIT", "SIGNATURE", "IND",
        "Khách thu nhập khá / cao, ưu tiên hoàn tiền cho chi tiêu thiết yếu và phong cách sống", 0.30,
        lambda f: [(f["income"] > 40e6, .35, "thu nhập khá/cao"),
                   (f["spend90"] > 45e6, .3, "chi tiêu thẻ lớn"),
                   (~f["has_premium"], .2, "chưa có thẻ cao cấp"),
                   (f["affluent"], .15, "phân khúc ưu tiên")])
    add("CARD_CC_WORLD_ELITE", "Thẻ tín dụng Mastercard World Elite", "CARD", "CREDIT", "WORLD_ELITE", "IND",
        "Khách ưu tiên / cao cấp, thường xuyên du lịch và chi tiêu quốc tế", 0.18,
        lambda f: [(f["affluent"], .4, "phân khúc ưu tiên/cao cấp"),
                   (f["travel_spend"] + f["fx_spend"] > 20e6, .35, "du lịch & chi tiêu quốc tế lớn"),
                   (f["income"] > 80e6, .2, "thu nhập rất cao")])
    add("CARD_MERCHANT_POS", "Chấp nhận thanh toán Merchant / POS", "CARD", "ACQUIRING", None, "SME",
        "Hộ kinh doanh, doanh nghiệp có điểm bán hoặc thu tiền số", 0.30,
        lambda f: [(f["is_sme"], .5, "hộ/DN kinh doanh"),
                   (f["netcf"] > 0, .25, "dòng tiền vào đều"),
                   (f["digital"] > 40, .15, "sẵn sàng thu tiền số")])
    add("CARD_BUSINESS", "Thẻ doanh nghiệp", "CARD", "CREDIT", "BUSINESS", "SME",
        "Doanh nghiệp cần quản lý và kiểm soát chi tiêu công tác / thanh toán", 0.22,
        lambda f: [(f["is_sme"], .5, "khách hàng doanh nghiệp"),
                   (f["spend90"] > 50e6, .3, "chi phí vận hành lớn"),
                   (f["bal"] > 100e6, .15, "dòng tiền doanh nghiệp")])

    # ---------------- CASA -----------------------------------------------
    add("CASA_PERSONAL", "Tài khoản thanh toán cá nhân", "CASA", "PAYMENT", None, "IND",
        "Khách cá nhân có nhu cầu giao dịch hằng ngày", 0.30,
        lambda f: [(f["rel_years"] < 1, .4, "khách mới"),
                   (~f["salary"], .25, "chưa nhận lương qua MSB"),
                   (f["digital"] > 30, .2, "có giao dịch số")])
    add("CASA_PAYROLL", "M-Payroll / Tài khoản lương", "CASA", "PAYROLL", None, "IND",
        "Doanh nghiệp trả lương qua MSB và cán bộ nhân viên", 0.35,
        lambda f: [(~f["salary"], .45, "chưa nhận lương qua MSB"),
                   (f["income"] > 15e6, .3, "có thu nhập thường xuyên"),
                   (f["netcf"] > 0, .15, "dòng tiền dương")])
    add("CASA_M_FIRST", "Khách hàng ưu tiên M-First", "CASA", "PREMIUM", "M_FIRST", "IND",
        "Khách đủ tiêu chí phân hạng M-First (AUM / thu nhập)", 0.20,
        lambda f: [((f["bal"] > 300e6) | (f["deposit_bal"] > 500e6), .45, "AUM đạt ngưỡng ưu tiên"),
                   (f["income"] > 60e6, .3, "thu nhập cao"),
                   (~f["affluent"], .15, "chưa được phân hạng ưu tiên")])
    add("CASA_BIZ_MSMART", "Tài khoản doanh nghiệp M-Smart", "CASA", "BIZ_PAYMENT", None, "SME",
        "SME / doanh nghiệp cần giao dịch và quản lý dòng tiền số", 0.28,
        lambda f: [(f["is_sme"], .55, "khách hàng SME/DN"),
                   (f["digital"] > 35, .25, "quản lý dòng tiền số"),
                   (f["netcf"].abs() > 20e6, .15, "quy mô dòng tiền lớn")])
    add("CASA_ESCROW", "Tài khoản ký quỹ", "CASA", "ESCROW", None, "SME",
        "Khách có giao dịch cần ký quỹ / bảo đảm nghĩa vụ", 0.10,
        lambda f: [(f["is_sme"], .5, "khách hàng doanh nghiệp"),
                   (f["bal"] > 200e6, .3, "có nguồn để ký quỹ")])

    # ---------------- FD / tiết kiệm -------------------------------------
    add("FD_TD_STANDARD", "Tiền gửi có kỳ hạn cá nhân", "FD", "TERM_DEPOSIT", None, "IND",
        "Khách cá nhân có tiền nhàn rỗi", 0.35,
        lambda f: [(f["idle_cash"] > 50e6, .4, "tiền nhàn rỗi lớn"),
                   (f["growth"] > 0.1, .25, "số dư CASA tăng"),
                   (f["deposit_bal"] == 0, .2, "chưa có tiền gửi")])
    add("FD_TD_ONLINE", "Tiền gửi có kỳ hạn trực tuyến", "FD", "TERM_DEPOSIT", "ONLINE", "IND",
        "Khách nhàn rỗi + ưu tiên giao dịch số", 0.33,
        lambda f: [(f["idle_cash"] > 40e6, .35, "tiền nhàn rỗi"),
                   (f["digital"] > 55, .4, "ưu tiên kênh số"),
                   (f["deposit_bal"] == 0, .15, "chưa có tiền gửi")])
    add("FD_SAV_MAXRATE", "Tiết kiệm / Tiền gửi lãi suất cao nhất", "FD", "SAVINGS", "MAX_RATE", "IND",
        "Khách có tiền nhàn rỗi, ưu tiên tối ưu lãi suất", 0.30,
        lambda f: [(f["idle_cash"] > 100e6, .45, "tiền nhàn rỗi rất lớn"),
                   (f["deposit_bal"] > 0, .25, "đã quen sản phẩm tiền gửi"),
                   (f["affluent"], .15, "phân khúc ưu tiên")])
    add("FD_SAV_PARTIAL", "Tiết kiệm rút gốc từng phần", "FD", "SAVINGS", "PARTIAL_WD", "IND",
        "Khách cần linh hoạt sử dụng một phần tiền trước đáo hạn", 0.20,
        lambda f: [(f["idle_cash"] > 30e6, .3, "có khoản tích luỹ"),
                   (f["netcf"].abs() > 15e6, .3, "dòng tiền biến động, cần linh hoạt"),
                   (f["young"], .15, "nhu cầu linh hoạt")])
    add("FD_SAV_UPFRONT", "Tiết kiệm trả lãi ngay", "FD", "SAVINGS", "UPFRONT_INT", "IND",
        "Khách muốn nhận lãi sớm và giữ gốc đến đáo hạn", 0.18,
        lambda f: [(f["idle_cash"] > 50e6, .4, "có khoản gửi lớn"),
                   (f["retired"], .25, "cần dòng tiền lãi sớm")])
    add("FD_SAV_PERIODIC", "Tiền gửi trả lãi định kỳ", "FD", "SAVINGS", "PERIODIC_INT", "IND",
        "Khách cá nhân / tổ chức cần dòng tiền lãi định kỳ", 0.18,
        lambda f: [(f["idle_cash"] > 80e6, .4, "khoản gửi lớn"),
                   (f["seg"].isin(("PRIVATE", "AFFLUENT")), .3, "sống bằng thu nhập tài chính"),
                   (f["retired"], .2, "khách hưu trí")])
    add("FD_SAV_KIDS", "Tiết kiệm Măng non", "FD", "SAVINGS", "KIDS", "IND",
        "Cha mẹ / người giám hộ muốn tích luỹ dài hạn cho con", 0.15,
        lambda f: [(f["family"], .5, "khách có con nhỏ"),
                   (f["salary"], .25, "thu nhập ổn định để tích luỹ"),
                   (f["idle_cash"] > 20e6, .15, "có khả năng tích luỹ")])
    add("FD_CD", "Chứng chỉ tiền gửi", "FD", "CERTIFICATE", None, "IND",
        "Khách có tiền nhàn rỗi, ưu tiên đầu tư tiền gửi kỳ hạn dài", 0.16,
        lambda f: [(f["idle_cash"] > 150e6, .45, "nguồn nhàn rỗi lớn dài hạn"),
                   (f["affluent"], .3, "phân khúc đầu tư"),
                   (~f["has_invest"], .15, "chưa có sản phẩm đầu tư")])

    # ---------------- LENDING ------------------------------------------
    add("LEND_UNSECURED_PAYROLL", "Vay tín chấp theo lương / phân khúc", "LENDING", "UNSECURED", None, "IND",
        "Khách cá nhân có thu nhập ổn định / nhận lương hoặc thuộc phân khúc đủ điều kiện", 0.30,
        lambda f: [(f["salary"], .4, "nhận lương qua MSB"),
                   (f["income"] > 15e6, .25, "thu nhập đủ điều kiện"),
                   (~f["has_loan"], .2, "chưa có khoản vay"),
                   ((f["spend90"] > f["income"] * 2), .15, "chi tiêu vượt tích luỹ")])
    add("LEND_OVERDRAFT_PERSONAL", "Thấu chi cá nhân (M-Payroll / M-First)", "LENDING", "OVERDRAFT", None, "IND",
        "Khách cá nhân có thu nhập ổn định, cần nguồn tiền dự phòng ngắn hạn", 0.25,
        lambda f: [(f["salary"], .35, "có tài khoản lương"),
                   (f["netcf"] < 0, .3, "dòng tiền tháng âm, cần đệm"),
                   ((f["bal"] < f["income"]), .2, "số dư mỏng so với thu nhập"),
                   (~f["has_loan"], .1, "chưa có tín dụng")])
    add("LEND_CONSUMER", "Vay tiêu dùng", "LENDING", "CONSUMER", None, "IND",
        "Khách cá nhân có nhu cầu chi tiêu / mua sắm / kế hoạch cá nhân", 0.22,
        lambda f: [(f["spend90"] > 40e6, .3, "chi tiêu lớn"),
                   (f["salary"], .3, "thu nhập ổn định"),
                   (~f["has_loan"], .2, "chưa có khoản vay")])
    add("LEND_HOME_SECURED", "Vay mua bất động sản có sổ", "LENDING", "MORTGAGE", None, "IND",
        "Khách có nhu cầu mua nhà, đất hoặc bất động sản", 0.20,
        lambda f: [(f["home_intent"], .45, "có tín hiệu tìm mua bất động sản"),
                   (f["income"] > 30e6, .3, "thu nhập đủ trả góp"),
                   (f["affluent"], .15, "tích luỹ tốt"),
                   (~f["has_loan"], .1, "chưa có dư nợ")])
    add("LEND_HOME_PROJECT", "Vay mua nhà dự án", "LENDING", "MORTGAGE", "PROJECT", "IND",
        "Khách mua nhà / căn hộ tại dự án đủ điều kiện MSB", 0.16,
        lambda f: [(f["home_intent"], .4, "tín hiệu mua nhà"),
                   (f["young"] & (f["income"] > 30e6), .3, "khách trẻ mua căn hộ đầu tiên"),
                   (f["idle_cash"] > 100e6, .2, "có sẵn vốn đối ứng")])
    add("LEND_HOME_RENOVATE", "Vay xây sửa / hoàn thiện nhà", "LENDING", "MORTGAGE", "RENOVATE", "IND",
        "Khách có nhu cầu xây, sửa hoặc hoàn thiện nhà ở", 0.14,
        lambda f: [(f["home_intent"], .3, "tín hiệu liên quan nhà ở"),
                   ((f["rel_years"] > 3) & ~f["young"], .3, "đã ổn định, có nhà"),
                   (f["income"] > 25e6, .2, "thu nhập đủ trả góp")])
    add("LEND_AUTO_NEW", "Vay mua ô tô mới", "LENDING", "AUTO", "NEW", "IND",
        "Khách cá nhân có nhu cầu mua xe phục vụ đi lại / gia đình", 0.18,
        lambda f: [(f["auto_intent"], .5, "có giao dịch liên quan ô tô"),
                   (f["income"] > 30e6, .3, "thu nhập đủ trả góp xe"),
                   (f["family"], .15, "nhu cầu xe gia đình")])
    add("LEND_SECURITIES_PLEDGE", "Vay ứng vốn cầm cố giấy tờ có giá", "LENDING", "PLEDGE", None, "IND",
        "Khách có sổ tiết kiệm / GTCG cần thanh khoản ngắn hạn", 0.15,
        lambda f: [(f["securities"] > 100e6, .5, "có sổ TK / GTCG để cầm cố"),
                   (f["netcf"] < 0, .25, "cần thanh khoản ngắn hạn"),
                   (~f["has_loan"], .15, "chưa dùng tín dụng")])
    add("LEND_BIZ_WC", "Cho vay bổ sung vốn kinh doanh", "LENDING", "BIZ", "WORKING_CAPITAL", "SME",
        "Hộ kinh doanh / cá nhân kinh doanh cần vốn lưu động hoặc mở rộng hoạt động", 0.28,
        lambda f: [(f["is_sme"], .5, "hộ / cá nhân kinh doanh"),
                   (f["netcf"].abs() > 30e6, .25, "quy mô dòng tiền lớn"),
                   (~f["has_loan"], .15, "chưa có dư nợ vay vốn")])
    add("LEND_AGRI", "Vay sản xuất kinh doanh nông nghiệp", "LENDING", "BIZ", "AGRI", "SME",
        "Hộ kinh doanh / cá nhân sản xuất nông nghiệp cần vốn lưu động hoặc đầu tư", 0.20,
        lambda f: [(f["agri"], .55, "ngành nông nghiệp"),
                   (f["is_sme"], .25, "hộ sản xuất kinh doanh"),
                   (~f["has_loan"], .1, "chưa có dư nợ")])
    add("LEND_OVERDRAFT_BIZ", "Thấu chi kinh doanh / chuỗi cung ứng", "LENDING", "OVERDRAFT", "BIZ", "SME",
        "Doanh nghiệp, hộ kinh doanh hoặc nhà phân phối cần vốn lưu động ngắn hạn", 0.18,
        lambda f: [(f["is_sme"], .45, "khách hàng SME/DN"),
                   (f["netcf"] < 0, .3, "thiếu hụt vốn lưu động ngắn hạn"),
                   (f["digital"] > 35, .1, "quản lý dòng tiền số")])
    return C


# Hard gate: sản phẩm chỉ "đủ điều kiện" nếu có tín hiệu bắt buộc (ngoài customer_type).
GATES = {
    "CARD_DEBIT_PLATINUM_FCB": lambda f: f["affluent"] | (f["fx_spend"] + f["travel_spend"] > 8e6),
    "CARD_CC_TRAVEL":     lambda f: (f["travel_spend"] > 4e6) | (f["fx_spend"] > 3e6),
    "CARD_CC_WORLD_ELITE": lambda f: f["affluent"] & (f["travel_spend"] + f["fx_spend"] > 10e6),
    "CARD_CC_FAMILY":     lambda f: f["family"],
    "CARD_CC_SIGNATURE":  lambda f: (f["income"] > 40e6) | (f["spend90"] > 45e6),
    "CARD_CC_ONLINE":     lambda f: f["online_spend"] > 4e6,
    "CARD_CC_MDIGI":      lambda f: f["young"] | (f["online_spend"] > 3e6),
    "CARD_DEBIT_INTL_VISA": lambda f: (f["fx_spend"] > 0) | (f["travel_spend"] > 2e6) | (f["online_spend"] > 4e6),
    "CASA_M_FIRST":       lambda f: (f["bal"] > 250e6) | (f["deposit_bal"] > 400e6) | (f["income"] > 60e6),
    "FD_SAV_KIDS":        lambda f: f["family"],
    "FD_SAV_PERIODIC":    lambda f: f["retired"] | f["seg"].isin(("PRIVATE", "AFFLUENT")),
    "FD_CD":             lambda f: f["idle_cash"] > 120e6,
    "FD_SAV_MAXRATE":     lambda f: f["idle_cash"] > 80e6,
    "LEND_HOME_SECURED": lambda f: f["home_intent"] | (f["affluent"] & f["young"]),
    "LEND_HOME_PROJECT": lambda f: f["home_intent"] | (f["young"] & (f["income"] > 30e6)),
    "LEND_HOME_RENOVATE": lambda f: f["home_intent"] | ((f["rel_years"] > 3) & ~f["young"] & (f["income"] > 25e6)),
    "LEND_AUTO_NEW":      lambda f: f["auto_intent"],
    "LEND_SECURITIES_PLEDGE": lambda f: f["securities"] > 80e6,
    "LEND_CONSUMER":      lambda f: (f["spend90"] > 35e6) | f["salary"],
    "LEND_AGRI":          lambda f: f["agri"],
    "CARD_MERCHANT_POS":  lambda f: f["is_sme"],
    "CARD_BUSINESS":      lambda f: f["is_sme"],
    "CASA_BIZ_MSMART":    lambda f: f["is_sme"],
    "CASA_ESCROW":        lambda f: f["is_sme"] & (f["bal"] > 150e6),
    "LEND_BIZ_WC":        lambda f: f["is_sme"],
    "LEND_OVERDRAFT_BIZ": lambda f: f["is_sme"],
}

CATALOGUE = _catalogue()
PRODUCT_IDS = [f"PR{ i+1:03d}" for i in range(len(CATALOGUE))]
for _i, _p in enumerate(CATALOGUE):
    _p["product_id"] = PRODUCT_IDS[_i]

GROUP_ORDER = ["CARD", "CASA", "FD", "LENDING"]
BY_ID = {p["product_id"]: p for p in CATALOGUE}
BY_CODE = {p["product_code"]: p for p in CATALOGUE}


def _id(code):
    return BY_CODE[code]["product_id"]


# ------------------------------------------------------------------ dim_product
def dim_product_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "product_id": p["product_id"],
        "product_code": p["product_code"],
        "product_name": p["product_name"],
        "product_group": p["product_group"],
        "subgroup": p["subgroup"],
        "product_tier": p["product_tier"],
        "customer_type": p["customer_type"],
        "target_need": p["need"],
        "base_propensity": p["base"],
        "active_flag": True,
    } for p in CATALOGUE])


# ------------------------------------------------------------------ fit table
def fit_table(mart: pd.DataFrame) -> pd.DataFrame:
    """
    mart: customer_360_feature_mart đã join dim_customer (customer_type,
    occupation_group, industry_group) + các cờ tín hiệu (family_flag,
    auto_intent_flag, home_intent_flag, agri_flag, securities_value...).

    Trả DataFrame dài: customer_id, product_id, eligible, fit_score (0..1),
    reason_1..reason_3.
    """
    f = _feat(mart)
    cid = mart["customer_id"].to_numpy()
    n = len(mart)
    out = []
    for p in CATALOGUE:
        # customer_type gate
        if p["customer_type"] == "SME":
            ok_type = f["is_sme"]
        elif p["customer_type"] == "IND":
            ok_type = f["is_ind"] | f["is_sme"]  # SME chủ hộ vẫn có thể mua sp cá nhân
        else:
            ok_type = pd.Series(True, index=mart.index)

        score = np.full(n, p["base"], dtype=float)
        reasons = [[] for _ in range(n)]
        wsum = p["base"]
        try:
            rules = p["rules"](f)
        except Exception:
            rules = []
        for cond, w, reason in rules:
            cond = np.asarray(cond.fillna(False) if hasattr(cond, "fillna") else cond, dtype=bool)
            score = score + w * cond
            wsum += w
            for i in np.where(cond)[0]:
                reasons[i].append((w, reason))
        score = np.clip(score / max(wsum, 1e-9), 0, 1)
        # eligible = đúng nhóm khách + qua hard-gate (nếu có) + fit_score >= 0.5
        gate = GATES.get(p["product_code"])
        if gate is not None:
            try:
                g = np.asarray(gate(f).fillna(False), dtype=bool)
            except Exception:
                g = np.ones(n, dtype=bool)
        else:
            g = np.ones(n, dtype=bool)
        elig = ok_type.to_numpy() & g & (score >= 0.50)
        top = [sorted(r, reverse=True)[:3] for r in reasons]
        out.append(pd.DataFrame({
            "customer_id": cid,
            "product_id": p["product_id"],
            "eligible": elig,
            "fit_score": np.round(score, 4),
            "reason_1": [t[0][1] if len(t) > 0 else None for t in top],
            "reason_2": [t[1][1] if len(t) > 1 else None for t in top],
            "reason_3": [t[2][1] if len(t) > 2 else None for t in top],
        }))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ holdings
def build_holdings(mart: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Mô phỏng sản phẩm khách ĐANG sở hữu: P(hold) ~ fit_score đã suy giảm + vài luật
    cứng (has_credit_card => 1 thẻ tín dụng, has_active_loan => 1 khoản vay,
    deposit_balance>0 => 1 tiền gửi, salary_flag => tài khoản lương...).
    """
    ft = fit_table(mart).merge(
        dim_product_frame()[["product_id", "product_code", "product_group", "subgroup"]],
        on="product_id", how="left")
    m = mart.set_index("customer_id")
    ft = ft.join(m[["has_credit_card", "has_active_loan", "deposit_balance", "salary_flag",
                    "has_investment"]], on="customer_id")
    # danh mục thực tế thưa: ~2-3 sp/khách → chỉ giữ sp fit mạnh, xác suất thấp
    p_hold = (ft["fit_score"] ** 2 * 0.12).clip(0, 0.25)
    is_credit = (ft.product_group == "CARD") & (ft.subgroup == "CREDIT")
    p_hold = np.where(is_credit & ft.has_credit_card.fillna(False), 0.28, p_hold)
    p_hold = np.where(is_credit & ~ft.has_credit_card.fillna(False), 0.0, p_hold)
    p_hold = np.where((ft.product_group == "LENDING") & ft.has_active_loan.fillna(False), 0.16, p_hold)
    p_hold = np.where((ft.product_group == "LENDING") & ~ft.has_active_loan.fillna(False), 0.01, p_hold)
    p_hold = np.where((ft.product_group == "FD") & (ft.deposit_balance.fillna(0) > 0), 0.14, p_hold)
    p_hold = np.where((ft.product_group == "FD") & (ft.deposit_balance.fillna(0) == 0), 0.0, p_hold)
    p_hold = np.where(ft.product_id.eq(_id("CASA_PERSONAL")), 0.85, p_hold)
    p_hold = np.where(ft.product_id.eq(_id("CASA_PAYROLL")) & ft.salary_flag.fillna(False), 0.85, p_hold)
    hold = rng.random(len(ft)) < p_hold
    h = ft.loc[hold, ["customer_id", "product_id"]].copy()
    h["open_date"] = pd.Timestamp("2026-08-31") - pd.to_timedelta(
        rng.integers(30, 2000, len(h)), unit="D")
    h["status"] = "ACTIVE"
    h["balance"] = np.round(rng.gamma(2.0, 30e6, len(h)), 2)
    return h.reset_index(drop=True)
