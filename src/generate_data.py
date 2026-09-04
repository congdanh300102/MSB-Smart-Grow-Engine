"""
MSB SMART GROWTH ENGINE - Synthetic data generator (Data Model v2)
=================================================================

Sinh bộ dữ liệu Customer-360 theo tài liệu "MSB SMART GROWTH ENGINE V.01"
- mục "I. THIẾT KẾ DATA TABLE" và "2. Thiết kế hành trình AI" (12 Actions).

Mỗi khách hàng có một *archetype ẩn* chi phối toàn bộ dữ liệu bên dưới
(facts -> aggregates -> feature mart -> AI feature -> AI score -> recommendation
-> RM feedback -> training set), nên 5 bài toán của đề đều trả lời được.

Output:  data/csv/<table>.csv  và  data/parquet/<table>.parquet  (1 file / bảng)

    py src/generate_data.py --customers 5000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import scoring   # doc formulas (X1..X6 propensity features, business sub-scores)

# --------------------------------------------------------------------------
AS_OF = date(2026, 8, 31)                    # snapshot_date
AS_OF_TS = datetime(2026, 8, 31, 18, 0, 0)
CASA_DAYS = 90      # overridable via --casa-days (kept >= 95 internally for rolling windows)
DIGITAL_DAYS = 90   # overridable via --digital-days
CARD_MONTHS = 6
CAMPAIGN_MONTHS = 6
OBS_LAG_DAYS = 120                           # observation_date của training set
LABEL_WINDOW = 90

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(ROOT, "data", "csv")
PARQUET_DIR = os.path.join(ROOT, "data", "parquet")

N_BRANCHES = 20
RM_PER_BRANCH = 9

ARCHETYPES = ["HIGH_VALUE", "CREDIT_OPPORTUNITY", "POTENTIAL_INVESTOR",
              "DIGITAL_ACTIVE", "CHURN_RISK", "STANDARD"]
ARCH_P = [0.14, 0.17, 0.15, 0.18, 0.12, 0.24]

SEGMENTS = ["MASS", "MASS_AFFLUENT", "AFFLUENT", "PRIVATE", "SME"]
AGE_GROUPS = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
INCOME_BANDS = ["<10M", "10-20M", "20-40M", "40-80M", "80M+"]
OCCUPATION = ["OFFICE", "BUSINESS_OWNER", "FREELANCE", "PUBLIC_SECTOR",
              "MANAGER", "RETIRED", "OTHER"]
INDUSTRY = ["IT", "FINANCE", "MANUFACTURING", "RETAIL", "HEALTHCARE",
            "EDUCATION", "CONSTRUCTION", "LOGISTICS", "OTHER"]
PROVINCES = ["01", "79", "48", "31", "92", "74", "75", "56", "40", "22"]
REGION_OF = {"01": "NORTH", "31": "NORTH", "22": "NORTH", "40": "NORTH",
             "48": "CENTRAL", "56": "CENTRAL",
             "79": "SOUTH", "92": "SOUTH", "74": "SOUTH", "75": "SOUTH"}
CHANNELS_PREF = ["RM_CALL", "IN_APP", "SMS", "EMAIL", "ZALO"]

# 4 sản phẩm mục tiêu giai đoạn đầu + biến thể
PRODUCTS = [
    ("P001", "CC_PLATINUM",   "Platinum Credit Card",   "CREDIT_CARD", "CREDIT_CARD", "PLATINUM"),
    ("P002", "CC_GOLD",        "Gold Credit Card",       "CREDIT_CARD", "CREDIT_CARD", "GOLD"),
    ("P003", "CC_STANDARD",    "Standard Credit Card",   "CREDIT_CARD", "CREDIT_CARD", "STANDARD"),
    ("P004", "LOAN_HOME",      "Vay mua nhà",            "LOAN",        "HOME_LOAN",   None),
    ("P005", "LOAN_AUTO",      "Vay mua ô tô",           "LOAN",        "AUTO_LOAN",   None),
    ("P006", "LOAN_PERSONAL",  "Vay tiêu dùng",          "LOAN",        "PERSONAL_LOAN", None),
    ("P007", "TD_STANDARD",    "Tiền gửi có kỳ hạn",     "DEPOSIT",     "TERM_DEPOSIT", None),
    ("P008", "TD_FLEXI",       "Tiền gửi linh hoạt",     "DEPOSIT",     "FLEXI_DEPOSIT", None),
    ("P009", "INVEST_FUND",    "Quỹ mở",                 "INVESTMENT",  "MUTUAL_FUND", None),
    ("P010", "INVEST_BOND",    "Trái phiếu",             "INVESTMENT",  "BOND",        None),
]
TARGET_PRODUCTS = {"CREDIT_CARD": "P001", "LOAN": "P004", "DEPOSIT": "P007", "INVESTMENT": "P009"}
GROUPS = ["CREDIT_CARD", "LOAN", "DEPOSIT", "INVESTMENT"]

BASE_PROPENSITY = {
    "HIGH_VALUE":         {"CREDIT_CARD": 0.58, "LOAN": 0.40, "DEPOSIT": 0.55, "INVESTMENT": 0.74},
    "CREDIT_OPPORTUNITY": {"CREDIT_CARD": 0.90, "LOAN": 0.80, "DEPOSIT": 0.22, "INVESTMENT": 0.30},
    "POTENTIAL_INVESTOR": {"CREDIT_CARD": 0.33, "LOAN": 0.20, "DEPOSIT": 0.86, "INVESTMENT": 0.90},
    "DIGITAL_ACTIVE":     {"CREDIT_CARD": 0.66, "LOAN": 0.35, "DEPOSIT": 0.42, "INVESTMENT": 0.46},
    "CHURN_RISK":         {"CREDIT_CARD": 0.18, "LOAN": 0.14, "DEPOSIT": 0.24, "INVESTMENT": 0.20},
    "STANDARD":           {"CREDIT_CARD": 0.45, "LOAN": 0.34, "DEPOSIT": 0.40, "INVESTMENT": 0.34},
}

# Smart Growth Score weights (theo tài liệu)
SGS_W = {"pp": 0.25, "cv": 0.20, "intent": 0.20, "eng": 0.15, "timing": 0.10, "rel": 0.10}

CAMPAIGNS = [
    ("CMP01", "Ưu đãi mở thẻ Platinum hoàn tiền 6%",       "P001", "CC_PLATINUM",  "ACQUISITION"),
    ("CMP02", "Nâng hạng thẻ Gold lên Platinum",           "P001", "CC_PLATINUM",  "UPSELL"),
    ("CMP03", "Vay mua nhà lãi suất 6.5% cố định 24T",      "P004", "LOAN_HOME",    "ACQUISITION"),
    ("CMP04", "Vay mua ô tô duyệt nhanh 8 giờ",             "P005", "LOAN_AUTO",    "ACQUISITION"),
    ("CMP05", "Tiền gửi online +0.5% lãi suất",             "P007", "TD_STANDARD",  "CROSS_SELL"),
    ("CMP06", "Tái tục tiền gửi nhận quà",                  "P007", "TD_STANDARD",  "RETENTION"),
    ("CMP07", "Trải nghiệm quỹ mở miễn phí phí quản lý",    "P009", "INVEST_FUND",  "ACQUISITION"),
    ("CMP08", "Danh mục trái phiếu cho khách Affluent",     "P010", "INVEST_BOND",  "CROSS_SELL"),
    ("CMP09", "Kích hoạt Mobile Banking nhận 100k",         "P003", "CC_STANDARD",  "ENGAGEMENT"),
    ("CMP10", "Ưu đãi tài khoản nhận lương",                "P007", "TD_FLEXI",     "ENGAGEMENT"),
    ("CMP11", "Chương trình tri ân khách hàng thân thiết",  "P001", "CC_PLATINUM",  "RETENTION"),
    ("CMP12", "Vay tiêu dùng tín chấp lãi suất ưu đãi",     "P006", "LOAN_PERSONAL","ACQUISITION"),
]


def _clip01(x):
    return np.clip(x, 0.01, 0.99)


def _norm(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanpercentile(x, 2), np.nanpercentile(x, 98)
    return np.clip((x - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(x)


# ==========================================================================
# Dimensions
# ==========================================================================
def gen_dim_rm(rng):
    rows = []
    for b in range(1, N_BRANCHES + 1):
        for r in range(RM_PER_BRANCH):
            rid = f"RM{(b - 1) * RM_PER_BRANCH + r + 1:04d}"
            rows.append((rid, f"BR{b:03d}",
                         rng.choice(["RM", "SENIOR_RM", "TEAM_LEAD"], p=[0.7, 0.22, 0.08]),
                         rng.choice(["MASS", "AFFLUENT", "SME"], p=[0.5, 0.35, 0.15]),
                         True))
    return pd.DataFrame(rows, columns=["rm_id", "branch_id", "rm_role", "rm_segment", "active_flag"])


def gen_dim_product():
    return pd.DataFrame(PRODUCTS, columns=[
        "product_id", "product_code", "product_name", "product_group", "product_type", "product_tier"
    ]).assign(active_flag=True)


def gen_dim_campaign(rng):
    rows = []
    for cid, name, pid, pcode, ctype in CAMPAIGNS:
        start = AS_OF - timedelta(days=int(rng.integers(30, CAMPAIGN_MONTHS * 30)))
        rows.append((cid, name, pid, start, start + timedelta(days=int(rng.choice([30, 45, 60, 90]))),
                     rng.choice(["EMAIL", "SMS", "APP_PUSH", "RM_OUTBOUND", "MULTI"]),
                     ctype, rng.choice(SEGMENTS + ["ALL"]), "COMPLETED"))
    return pd.DataFrame(rows, columns=[
        "campaign_id", "campaign_name", "product_id", "start_date", "end_date",
        "channel", "campaign_type", "target_segment", "campaign_status"])


def gen_dim_customer(n, rng, rm_df):
    keys = np.arange(1, n + 1)
    archetype = rng.choice(ARCHETYPES, size=n, p=ARCH_P)

    seg = np.empty(n, dtype=object)
    for i, a in enumerate(archetype):
        if a == "HIGH_VALUE":
            seg[i] = rng.choice(["AFFLUENT", "PRIVATE", "MASS_AFFLUENT"], p=[0.5, 0.3, 0.2])
        elif a == "POTENTIAL_INVESTOR":
            seg[i] = rng.choice(["MASS_AFFLUENT", "AFFLUENT", "MASS"], p=[0.45, 0.35, 0.20])
        elif a == "CREDIT_OPPORTUNITY":
            seg[i] = rng.choice(["MASS", "MASS_AFFLUENT", "SME"], p=[0.5, 0.3, 0.2])
        elif a == "CHURN_RISK":
            seg[i] = rng.choice(["MASS", "MASS_AFFLUENT", "AFFLUENT"], p=[0.6, 0.25, 0.15])
        else:
            seg[i] = rng.choice(SEGMENTS, p=[0.45, 0.25, 0.13, 0.05, 0.12])

    income = np.select(
        [seg == "PRIVATE", seg == "AFFLUENT", seg == "MASS_AFFLUENT", seg == "SME"],
        [rng.choice(["80M+", "40-80M"], n, p=[0.7, 0.3]),
         rng.choice(["40-80M", "80M+", "20-40M"], n, p=[0.5, 0.3, 0.2]),
         rng.choice(["20-40M", "40-80M", "10-20M"], n, p=[0.5, 0.3, 0.2]),
         rng.choice(["20-40M", "40-80M", "80M+"], n, p=[0.4, 0.4, 0.2])],
        default=rng.choice(["<10M", "10-20M", "20-40M"], n, p=[0.35, 0.45, 0.20]))

    age_years = rng.integers(20, 70, size=n)
    age_grp = pd.cut(age_years, [17, 25, 35, 45, 55, 65, 200], labels=AGE_GROUPS).astype(str)

    tenure_days = rng.integers(120, 14 * 365, size=n)
    rel_start = [AS_OF - timedelta(days=int(t)) for t in tenure_days]
    rel_years = np.round(tenure_days / 365.25, 2)

    branch = rng.integers(1, N_BRANCHES + 1, size=n)
    branch_id = [f"BR{b:03d}" for b in branch]
    rm_id = [rm_df[rm_df.branch_id == b].rm_id.iloc[int(rng.integers(0, RM_PER_BRANCH))] for b in branch_id]

    prov = rng.choice(PROVINCES, size=n, p=[0.24, 0.30, 0.09, 0.08, 0.06, 0.07, 0.06, 0.04, 0.03, 0.03])
    region = [REGION_OF[p] for p in prov]

    active = np.where(archetype == "CHURN_RISK",
                      rng.random(n) < 0.82, rng.random(n) < 0.985)
    kyc = np.where(rng.random(n) < 0.92, "VERIFIED",
                   rng.choice(["PENDING", "EXPIRED"], n))
    consent = rng.random(n) < 0.78
    dnc = (~consent) & (rng.random(n) < 0.4) | (rng.random(n) < 0.05)

    def _cc(i):
        opts = ["SMS", "EMAIL", "APP", "ZALO"]
        k = rng.integers(1, 5)
        return ",".join(sorted(rng.choice(opts, size=k, replace=False))) if consent[i] else ""

    df = pd.DataFrame({
        "customer_id": [f"CUS{k:08d}" for k in keys],
        "cif_id": [f"CIF{k:09d}" for k in keys],
        "customer_type": np.where(seg == "SME", "SME", "INDIVIDUAL"),
        "age_group": age_grp,
        "gender_code": rng.choice(["M", "F"], size=n, p=[0.52, 0.48]),
        "occupation_group": rng.choice(OCCUPATION, size=n),
        "industry_group": rng.choice(INDUSTRY, size=n),
        "income_band": income,
        "province_code": prov,
        "region_code": region,
        "customer_segment": seg,
        "relationship_start_date": rel_start,
        "relationship_years": rel_years,
        "rm_id": rm_id,
        "branch_id": branch_id,
        "kyc_status": kyc,
        "preferred_channel": rng.choice(CHANNELS_PREF, size=n, p=[0.15, 0.35, 0.25, 0.15, 0.10]),
        "customer_active_flag": active,
        "marketing_consent_flag": consent,
        "consent_channels": [_cc(i) for i in range(n)],
        "do_not_contact_flag": dnc,
        "record_effective_from": rel_start,
        "record_effective_to": date(9999, 12, 31),
        "current_flag": True,
    })
    df.attrs["archetype"] = archetype
    df.attrs["age_years"] = age_years
    return df


# --------------------------------------------------------------------------
def wealth_base(archetype, segment, rng):
    seg_mult = pd.Series(segment).map({
        "MASS": 1.0, "MASS_AFFLUENT": 2.4, "AFFLUENT": 5.5, "PRIVATE": 16.0, "SME": 4.0}).to_numpy()
    arch_mult = pd.Series(archetype).map({
        "HIGH_VALUE": 3.0, "POTENTIAL_INVESTOR": 2.1, "CREDIT_OPPORTUNITY": 0.9,
        "DIGITAL_ACTIVE": 1.1, "CHURN_RISK": 0.8, "STANDARD": 1.0}).to_numpy()
    return 12_000_000 * seg_mult * arch_mult * rng.lognormal(0.0, 0.5, len(archetype))


def income_mid(band):
    return pd.Series(band).map({
        "<10M": 7_000_000, "10-20M": 15_000_000, "20-40M": 30_000_000,
        "40-80M": 60_000_000, "80M+": 120_000_000}).to_numpy()


# ==========================================================================
# FACT_CASA_DAILY  (+ salary series used later)
# ==========================================================================
def gen_casa(cust, rng):
    arch = cust.attrs["archetype"]
    n = len(cust)
    seg = cust["customer_segment"].to_numpy()
    base = wealth_base(arch, seg, rng)
    inc = income_mid(cust["income_band"].to_numpy())
    has_casa = rng.random(n) < 0.98
    warm = max(185, CASA_DAYS + 30)
    dates = [AS_OF - timedelta(days=d) for d in range(warm - 1, -1, -1)]
    D = warm
    drift_map = {"HIGH_VALUE": 0.0006, "POTENTIAL_INVESTOR": 0.0016, "CREDIT_OPPORTUNITY": 0.0002,
                 "DIGITAL_ACTIVE": 0.0004, "CHURN_RISK": -0.0022, "STANDARD": 0.0001}

    out_rows = []
    salary_avg = np.zeros(n)
    bal_now = np.zeros(n)
    growth_3m = np.zeros(n)
    inflow30 = np.zeros(n)
    outflow30 = np.zeros(n)
    salary_flag_arr = np.zeros(n, dtype=bool)

    emit_from = D - CASA_DAYS
    day_dom = np.array([d.day for d in dates])
    pay_mask = (day_dom == 25).astype(float)
    emit_dates = dates[emit_from:]
    E = len(emit_dates)
    idxE = np.arange(emit_from, D)

    def _rollmean(series, w):
        cs = np.concatenate([[0.0], np.cumsum(series)])
        lo = np.maximum(0, idxE - w + 1)
        return (cs[idxE + 1] - cs[lo]) / (idxE - lo + 1)

    for i in range(n):
        if not has_casa[i]:
            continue
        a = arch[i]
        vol = 0.03 if a == "CHURN_RISK" else 0.02
        steps = rng.normal(drift_map[a], vol, size=D)
        series = base[i] * np.cumprod(1 + steps)
        has_salary = rng.random() < (0.85 if a in ("CREDIT_OPPORTUNITY", "DIGITAL_ACTIVE", "HIGH_VALUE") else 0.55)
        sal_amt = inc[i] * rng.uniform(0.8, 1.1) if has_salary else 0.0
        credit_daily = np.abs(rng.normal(0, base[i] * 0.01, D))
        debit_daily = np.abs(rng.normal(0, base[i] * 0.012, D))
        if has_salary:
            series = series + sal_amt * np.cumsum(pay_mask)     # mỗi kỳ lương nâng mặt bằng số dư
            credit_daily = credit_daily + sal_amt * pay_mask
        if a == "POTENTIAL_INVESTOR" and rng.random() < 0.55:
            k = int(rng.integers(warm - 60, warm - 10))
            bump = base[i] * rng.uniform(0.6, 1.6)
            series[k:] += bump
            credit_daily[k] += bump
        series = np.round(np.maximum(series, -base[i] * 0.05), 2)

        avg30 = _rollmean(series, 30)
        avg90 = _rollmean(series, 90)
        avg180 = _rollmean(series, 180)
        prev30 = np.where(idxE >= 30, series[np.maximum(idxE - 30, 0)], 0.0)
        prev90 = np.where(idxE >= 90, series[np.maximum(idxE - 90, 0)], 0.0)
        g30 = np.where(prev30 != 0, series[idxE] / prev30 - 1, 0.0)
        g90 = np.where(prev90 != 0, series[idxE] / prev90 - 1, 0.0)
        ccum = np.concatenate([[0.0], np.cumsum(credit_daily)])
        dcum = np.concatenate([[0.0], np.cumsum(debit_daily)])
        lo30 = np.maximum(0, idxE - 29)
        in30 = ccum[idxE + 1] - ccum[lo30]
        out30 = dcum[idxE + 1] - dcum[lo30]

        acc = f"CASA{i + 1:08d}"
        cid = cust["customer_id"].iat[i]
        acct_type = "PAYROLL" if has_salary else "CURRENT"
        acct_status = "ACTIVE" if cust["customer_active_flag"].iat[i] else "DORMANT"
        avail = series[idxE] * rng.uniform(0.9, 1.0, E)
        ctc = rng.integers(0, 6, E)
        dtc = rng.integers(0, 9, E)
        sss = np.clip(60 + 40 * (1 if has_salary else -1) * rng.uniform(0.3, 1.0, E), 0, 100)
        for j in range(E):
            d = idxE[j]
            out_rows.append((
                emit_dates[j], acc, cid, acct_type, "VND",
                round(float(series[d]), 2), round(float(avail[j]), 2),
                round(float(credit_daily[d]), 2), round(float(debit_daily[d]), 2),
                int(ctc[j]), int(dtc[j]),
                round(float(sal_amt if (has_salary and day_dom[d] == 25) else 0.0), 2),
                bool(has_salary), acct_status,
                round(float(avg30[j]), 2), round(float(avg90[j]), 2), round(float(avg180[j]), 2),
                round(float(in30[j]), 2), round(float(out30[j]), 2),
                round(float(g30[j]), 4), round(float(g90[j]), 4),
                round(float(sss[j]), 2),
            ))
        salary_avg[i] = sal_amt
        salary_flag_arr[i] = has_salary
        bal_now[i] = series[-1]
        growth_3m[i] = (series[-1] / series[-91] - 1) if series[-91] else 0.0
        inflow30[i] = float(credit_daily[-30:].sum())
        outflow30[i] = float(debit_daily[-30:].sum())

    cols = ["snapshot_date", "account_id", "customer_id", "account_type", "currency",
            "current_balance", "available_balance", "credit_amount_daily", "debit_amount_daily",
            "credit_txn_count", "debit_txn_count", "salary_credit_amount", "salary_flag",
            "account_status", "avg_balance_30d", "avg_balance_90d", "avg_balance_180d",
            "total_inflow_30d", "total_outflow_30d", "balance_growth_30d", "balance_growth_90d",
            "salary_stability_score"]
    df = pd.DataFrame(out_rows, columns=cols)
    helper = pd.DataFrame({
        "customer_id": cust["customer_id"], "has_casa": has_casa,
        "salary_avg": salary_avg, "salary_flag": salary_flag_arr, "casa_balance": bal_now,
        "casa_growth_3m": growth_3m, "inflow_30d": inflow30, "outflow_30d": outflow30,
        "wealth_base": base, "income_mid": inc,
    })
    return df, helper


# ==========================================================================
# FACT_TRANSACTION  +  AGG_CUSTOMER_TRANSACTION
# ==========================================================================
TXN_TYPES = ["PURCHASE", "TRANSFER_OUT", "TRANSFER_IN", "ATM_WITHDRAWAL",
             "BILL_PAYMENT", "SALARY_CREDIT", "FEE", "REFUND"]
TXN_TYPE_P = [0.34, 0.20, 0.15, 0.10, 0.12, 0.04, 0.03, 0.02]
CHANNELS = ["MOBILE_APP", "INTERNET_BANKING", "ATM", "POS", "BRANCH", "QR"]
CAT_MCC = {
    "GROCERY": "5411", "DINING": "5812", "FUEL": "5541", "TRAVEL": "4722", "AUTOMOTIVE": "5511",
    "ELECTRONICS": "5732", "HEALTHCARE": "8011", "EDUCATION": "8220", "UTILITIES": "4900",
    "ENTERTAINMENT": "7832", "FASHION": "5651", "TRANSFER": "6012", "ATM": "6011",
    "INSURANCE": "6300", "INVESTMENT": "6211"}
CATS = list(CAT_MCC)


def gen_transactions(cust, casa_helper, rng):
    arch = cust.attrs["archetype"]
    n = len(cust)
    base = casa_helper["wealth_base"].to_numpy()
    lam = pd.Series(arch).map({
        "HIGH_VALUE": 55, "POTENTIAL_INVESTOR": 38, "CREDIT_OPPORTUNITY": 46,
        "DIGITAL_ACTIVE": 70, "CHURN_RISK": 12, "STANDARD": 34}).to_numpy().astype(float)
    counts = np.clip(rng.poisson(lam), 1, None)
    total = int(counts.sum())
    pos = np.repeat(np.arange(n), counts)
    cid = cust["customer_id"].to_numpy()[pos]

    day_off = rng.integers(0, 90, size=total)
    sec = rng.integers(0, 86400, size=total)
    ts = [AS_OF_TS - timedelta(days=int(d), seconds=int(s)) for d, s in zip(day_off, sec)]
    tdate = [t.date() for t in ts]

    ttype = rng.choice(TXN_TYPES, size=total, p=TXN_TYPE_P)
    dc = np.where(np.isin(ttype, ["TRANSFER_IN", "SALARY_CREDIT", "REFUND"]), "C", "D")
    amt = np.abs(rng.normal(0.03, 0.05, total)) * base[pos] + 50_000
    amt = np.where(ttype == "SALARY_CREDIT", base[pos] * rng.uniform(0.2, 0.5, total), amt)
    amt = np.where(ttype == "FEE", rng.uniform(5_000, 150_000, total), amt)
    amt = np.round(amt, 2)

    cat = rng.choice(CATS, size=total)
    auto_arch = np.isin(arch[pos], ["CREDIT_OPPORTUNITY", "HIGH_VALUE"])
    cat = np.where(auto_arch & (rng.random(total) < 0.06), "AUTOMOTIVE", cat)
    mcc = pd.Series(cat).map(CAT_MCC).to_numpy()

    chan = rng.choice(CHANNELS, size=total, p=[0.42, 0.20, 0.12, 0.18, 0.03, 0.05])
    dig = np.isin(arch[pos], ["DIGITAL_ACTIVE"])
    chan = np.where(dig & (rng.random(total) < 0.5), "MOBILE_APP", chan)
    dom_for = np.where(np.isin(cat, ["TRAVEL"]) & (rng.random(total) < 0.5), "FOR", "DOM")

    txn = pd.DataFrame({
        "transaction_id": [f"TXN{i:013d}" for i in range(1, total + 1)],
        "customer_id": cid,
        "account_id": ["CASA" + c[3:] for c in cid],
        "transaction_timestamp": ts,
        "transaction_date": tdate,
        "transaction_type": ttype,
        "channel": chan,
        "debit_credit_flag": dc,
        "amount": amt,
        "currency": np.where(dom_for == "FOR", "USD", "VND"),
        "merchant_id": np.where(np.isin(ttype, ["PURCHASE", "BILL_PAYMENT"]),
                                [f"M{x:07d}" for x in rng.integers(1, 400000, total)], None),
        "merchant_category_code": mcc,
        "transaction_category": cat,
        "domestic_foreign_flag": dom_for,
        "counterparty_type": rng.choice(["MERCHANT", "INDIVIDUAL", "BILLER", "BANK", "SELF"], size=total),
        "transaction_status": rng.choice(["SUCCESS", "SUCCESS", "SUCCESS", "FAILED", "REVERSED"], size=total),
    }).sort_values("transaction_timestamp").reset_index(drop=True)

    # --- AGG_CUSTOMER_TRANSACTION -----------------------------------
    ok = txn[txn.transaction_status == "SUCCESS"].copy()
    ok["d"] = (AS_OF - pd.to_datetime(ok["transaction_date"]).dt.date).apply(lambda x: x.days)
    spend = ok[ok.debit_credit_flag == "D"]
    g = pd.DataFrame({"customer_id": cust["customer_id"]}).set_index("customer_id")
    g["txn_count_30d"] = spend[spend.d < 30].groupby("customer_id").size()
    g["txn_count_90d"] = spend.groupby("customer_id").size()
    g["total_spend_30d"] = spend[spend.d < 30].groupby("customer_id").amount.sum()
    g["total_spend_90d"] = spend.groupby("customer_id").amount.sum()
    g["avg_txn_value_30d"] = g["total_spend_30d"] / g["txn_count_30d"].replace(0, np.nan)
    g["online_spend_30d"] = spend[(spend.d < 30) & (spend.channel.isin(["MOBILE_APP", "INTERNET_BANKING", "QR"]))].groupby("customer_id").amount.sum()
    g["travel_spend_90d"] = spend[spend.transaction_category == "TRAVEL"].groupby("customer_id").amount.sum()
    g["dining_spend_90d"] = spend[spend.transaction_category == "DINING"].groupby("customer_id").amount.sum()
    g["international_spend_90d"] = spend[spend.domestic_foreign_flag == "FOR"].groupby("customer_id").amount.sum()
    g["qr_txn_count_30d"] = spend[(spend.d < 30) & (spend.channel == "QR")].groupby("customer_id").size()
    g["credit_inflow_30d"] = ok[(ok.d < 30) & (ok.debit_credit_flag == "C")].groupby("customer_id").amount.sum()
    g["debit_outflow_30d"] = spend[spend.d < 30].groupby("customer_id").amount.sum()
    g = g.fillna(0.0)
    prev_30_60 = spend[(spend.d >= 30) & (spend.d < 60)].groupby("customer_id").amount.sum()
    g["spending_growth_3m"] = (g["total_spend_30d"] / prev_30_60.reindex(g.index).replace(0, np.nan) - 1).fillna(0.0)
    agg = g.reset_index()
    agg.insert(1, "snapshot_date", AS_OF)
    for c in ["txn_count_30d", "txn_count_90d", "qr_txn_count_30d"]:
        agg[c] = agg[c].astype(int)
    money = ["total_spend_30d", "total_spend_90d", "avg_txn_value_30d", "online_spend_30d",
             "travel_spend_90d", "dining_spend_90d", "international_spend_90d",
             "credit_inflow_30d", "debit_outflow_30d"]
    agg[money] = agg[money].round(2)
    agg["spending_growth_3m"] = agg["spending_growth_3m"].round(4)
    return txn, agg


# ==========================================================================
# DIM_CARD  +  FACT_CARD_MONTHLY
# ==========================================================================
def gen_cards(cust, casa_helper, rng):
    arch = cust.attrs["archetype"]
    n = len(cust)
    inc = casa_helper["income_mid"].to_numpy()
    p_has = pd.Series(arch).map({
        "HIGH_VALUE": 0.9, "POTENTIAL_INVESTOR": 0.7, "CREDIT_OPPORTUNITY": 0.45,
        "DIGITAL_ACTIVE": 0.8, "CHURN_RISK": 0.6, "STANDARD": 0.62}).to_numpy()
    dim_rows, mon_rows = [], []
    cnum = 1
    months = [ (AS_OF.replace(day=1) - timedelta(days=1)).replace(day=1) ]
    for _ in range(CARD_MONTHS - 1):
        months.append((months[-1] - timedelta(days=1)).replace(day=1))
    months = sorted(months)

    for i in range(n):
        if rng.random() > p_has[i]:
            continue
        a = arch[i]
        ncards = 1 + (rng.random() < 0.25)
        for _ in range(ncards):
            is_credit = rng.random() < (0.72 if a in ("HIGH_VALUE", "CREDIT_OPPORTUNITY") else 0.45)
            if is_credit:
                # credit-opportunity: hiếm khi đã có Platinum (đó là whitespace)
                tier = rng.choice(["PLATINUM", "GOLD", "STANDARD"],
                                  p=[0.08, 0.32, 0.60] if a == "CREDIT_OPPORTUNITY" else [0.34, 0.36, 0.30])
                ctype = "CREDIT"
            else:
                tier = None
                ctype = "DEBIT"
            issue = AS_OF - timedelta(days=int(rng.integers(60, 2200)))
            status = "ACTIVE" if rng.random() < 0.9 else rng.choice(["BLOCKED", "EXPIRED"])
            limit = 0.0
            if is_credit:
                limit = float(np.clip(inc[i] * rng.uniform(2, 6), 20e6, 500e6))
                limit = round(limit, -6)
            cardid = f"CARD{cnum:09d}"
            dim_rows.append((cardid, cust["customer_id"].iat[i], ctype, tier, status,
                             issue, issue + timedelta(days=5 * 365), round(limit, 2)))
            if is_credit and status == "ACTIVE":
                if a == "CREDIT_OPPORTUNITY":
                    util = rng.uniform(0.03, 0.30)
                elif a == "CHURN_RISK":
                    util = rng.uniform(0.0, 0.12)
                else:
                    util = rng.uniform(0.12, 0.85)
                for m in months:
                    spend = limit * util * rng.uniform(0.6, 1.2)
                    mon_rows.append((cust["customer_id"].iat[i], cardid, m,
                                     round(spend, 2), int(rng.integers(3, 40)),
                                     round(spend * rng.uniform(0.2, 0.7), 2),
                                     round(spend * rng.uniform(0.0, 0.3), 2),
                                     round(float(util * rng.uniform(0.8, 1.15)), 4),
                                     round(spend * rng.uniform(0.7, 1.05), 2),
                                     int(rng.random() < (0.18 if a == "CHURN_RISK" else 0.05))))
            cnum += 1

    dim_card = pd.DataFrame(dim_rows, columns=[
        "card_id", "customer_id", "card_type", "card_tier", "card_status",
        "issue_date", "expiry_date", "credit_limit"])
    fact_cm = pd.DataFrame(mon_rows, columns=[
        "customer_id", "card_id", "month", "card_spending", "transaction_count",
        "online_spending", "international_spending", "utilization_rate",
        "payment_amount", "late_payment_count"])
    return dim_card, fact_cm


# ==========================================================================
# FACT_DEPOSIT
# ==========================================================================
def gen_deposits(cust, casa_helper, rng):
    arch = cust.attrs["archetype"]
    n = len(cust)
    base = casa_helper["wealth_base"].to_numpy()
    p_has = pd.Series(arch).map({
        "HIGH_VALUE": 0.65, "POTENTIAL_INVESTOR": 0.88, "CREDIT_OPPORTUNITY": 0.15,
        "DIGITAL_ACTIVE": 0.30, "CHURN_RISK": 0.25, "STANDARD": 0.35}).to_numpy()
    rows = []
    did = 1
    per_cust = {}
    for i in range(n):
        if rng.random() > p_has[i]:
            continue
        a = arch[i]
        ndep = 1 + (rng.random() < (0.5 if a == "POTENTIAL_INVESTOR" else 0.15))
        cust_deps = []
        for _ in range(ndep):
            term = int(rng.choice([1, 3, 6, 9, 12, 18, 24]))
            open_d = AS_OF - timedelta(days=int(rng.integers(10, term * 30 + 40)))
            mat = open_d + timedelta(days=term * 30)
            principal = round(base[i] * rng.uniform(0.8, 5), 2)
            if a == "POTENTIAL_INVESTOR" and rng.random() < 0.5:
                mat = AS_OF + timedelta(days=int(rng.integers(3, 28)))
                open_d = mat - timedelta(days=term * 30)
            if mat < AS_OF:
                status = rng.choice(["MATURED", "ROLLED_OVER", "WITHDRAWN"], p=[0.25, 0.55, 0.20])
            elif (mat - AS_OF).days <= 30:
                status = "MATURING_SOON"
            else:
                status = "ACTIVE"
            bal = principal * (1 + rng.uniform(0, 0.05)) if status in ("ACTIVE", "MATURING_SOON", "ROLLED_OVER") else 0.0
            rows.append([f"DEP{did:09d}", cust["customer_id"].iat[i], open_d, mat,
                         rng.choice(["TERM", "FLEXI"], p=[0.8, 0.2]), term, "VND",
                         principal, round(bal, 2), round(float(rng.uniform(3.5, 6.2)), 3),
                         bool(rng.random() < 0.6), bool(rng.random() < 0.1), status])
            cust_deps.append((mat, principal, bal, status))
            did += 1
        per_cust[cust["customer_id"].iat[i]] = cust_deps

    cols = ["deposit_id", "customer_id", "open_date", "maturity_date", "deposit_type",
            "term_months", "currency", "principal_amount", "current_balance", "interest_rate",
            "auto_renew_flag", "early_withdraw_flag", "deposit_status"]
    df = pd.DataFrame(rows, columns=cols)

    # customer-level aggregates
    agg_map = {}
    for c, deps in per_cust.items():
        tot = sum(b for _, _, b, _ in deps)
        m30 = sum(p for m, p, _, s in deps if s in ("ACTIVE", "MATURING_SOON") and 0 <= (m - AS_OF).days <= 30)
        m60 = sum(p for m, p, _, s in deps if s in ("ACTIVE", "MATURING_SOON") and 0 <= (m - AS_OF).days <= 60)
        fut = [(m - AS_OF).days for m, _, _, s in deps if s in ("ACTIVE", "MATURING_SOON") and (m - AS_OF).days >= 0]
        agg_map[c] = (round(tot, 2), len(deps), round(m30, 2), round(m60, 2),
                      min(fut) if fut else None)
    df["total_deposit_balance"] = df["customer_id"].map(lambda c: agg_map[c][0])
    df["deposit_count"] = df["customer_id"].map(lambda c: agg_map[c][1])
    df["maturity_30d_amount"] = df["customer_id"].map(lambda c: agg_map[c][2])
    df["maturity_60d_amount"] = df["customer_id"].map(lambda c: agg_map[c][3])
    df["days_to_nearest_maturity"] = df["customer_id"].map(lambda c: agg_map[c][4]).astype("Int64")
    return df


# ==========================================================================
# FACT_LOAN
# ==========================================================================
def gen_loans(cust, casa_helper, rng):
    arch = cust.attrs["archetype"]
    n = len(cust)
    base = casa_helper["wealth_base"].to_numpy()
    inc = casa_helper["income_mid"].to_numpy()
    p_has = pd.Series(arch).map({
        "HIGH_VALUE": 0.35, "POTENTIAL_INVESTOR": 0.18, "CREDIT_OPPORTUNITY": 0.28,
        "DIGITAL_ACTIVE": 0.26, "CHURN_RISK": 0.40, "STANDARD": 0.30}).to_numpy()
    rows = []
    lid = 1
    per_cust = {}
    for i in range(n):
        if rng.random() > p_has[i]:
            continue
        a = arch[i]
        nloan = 1 + (rng.random() < 0.12)
        clist = []
        for _ in range(nloan):
            lt = rng.choice(["HOME", "AUTO", "PERSONAL", "BUSINESS"], p=[0.32, 0.2, 0.34, 0.14])
            start = AS_OF - timedelta(days=int(rng.integers(90, 2600)))
            tenor = int(rng.choice([12, 24, 36, 48, 60, 120, 180, 240]))
            mat = start + timedelta(days=tenor * 30)
            orig = round(base[i] * rng.uniform(2, 9), 2)
            paid = min(1.0, (AS_OF - start).days / (tenor * 30.0))
            outst = round(max(0.0, orig * (1 - paid * rng.uniform(0.7, 1.0))), 2)
            rate = round(float(rng.uniform(6.5, 14.5)), 3)
            inst = round(orig / tenor * rng.uniform(1.05, 1.25), 2)
            if a == "CHURN_RISK" and rng.random() < 0.4:
                dpd = int(rng.choice([15, 30, 60, 90], p=[0.4, 0.3, 0.2, 0.1]))
                pstat, lstat = "LATE", "DELINQUENT"
            else:
                dpd, pstat = 0, "CURRENT"
                lstat = "ACTIVE" if outst > 0 else "CLOSED"
            rows.append([f"LOAN{lid:09d}", cust["customer_id"].iat[i], lt, start, mat,
                         orig, outst, rate, inst, bool(lt in ("HOME", "AUTO")), pstat, dpd, lstat])
            clist.append((outst, inst, lstat))
            lid += 1
        per_cust[cust["customer_id"].iat[i]] = clist

    cols = ["loan_id", "customer_id", "loan_type", "start_date", "maturity_date",
            "original_amount", "outstanding_balance", "interest_rate", "monthly_installment",
            "secured_flag", "payment_status", "days_past_due", "loan_status"]
    df = pd.DataFrame(rows, columns=cols)
    inc_map = dict(zip(cust["customer_id"], inc))
    amap = {}
    for c, ls in per_cust.items():
        active = [x for x in ls if x[2] in ("ACTIVE", "DELINQUENT")]
        tot = sum(x[0] for x in active)
        mdp = sum(x[1] for x in active)
        amap[c] = (len(active) > 0, len(active), round(tot, 2), round(mdp, 2),
                   round(mdp / (inc_map[c] + 1), 4))
    df["has_active_loan"] = df["customer_id"].map(lambda c: amap[c][0])
    df["active_loan_count"] = df["customer_id"].map(lambda c: amap[c][1])
    df["total_loan_outstanding"] = df["customer_id"].map(lambda c: amap[c][2])
    df["monthly_debt_payment"] = df["customer_id"].map(lambda c: amap[c][3])
    df["loan_to_income_ratio"] = df["customer_id"].map(lambda c: amap[c][4])
    return df


# ==========================================================================
# FACT_INSURANCE
# ==========================================================================
def gen_insurance(cust, casa_helper, rng):
    arch = cust.attrs["archetype"]
    n = len(cust)
    base = casa_helper["wealth_base"].to_numpy()
    p_has = pd.Series(arch).map({
        "HIGH_VALUE": 0.42, "POTENTIAL_INVESTOR": 0.32, "CREDIT_OPPORTUNITY": 0.15,
        "DIGITAL_ACTIVE": 0.2, "CHURN_RISK": 0.12, "STANDARD": 0.18}).to_numpy()
    rows = []
    pid = 1
    per_cust = {}
    for i in range(n):
        if rng.random() > p_has[i]:
            continue
        npol = 1 + (rng.random() < 0.2)
        plist = []
        for _ in range(npol):
            itype = rng.choice(["LIFE", "HEALTH", "MOTOR", "HOME", "TRAVEL"], p=[0.35, 0.3, 0.15, 0.1, 0.1])
            start = AS_OF - timedelta(days=int(rng.integers(30, 1500)))
            exp = start + timedelta(days=365 * int(rng.choice([1, 5, 10, 20])))
            renew = start + timedelta(days=365 * (((AS_OF - start).days // 365) + 1))
            prem = round(base[i] * rng.uniform(0.01, 0.05), 2)
            freq = rng.choice(["MONTHLY", "QUARTERLY", "ANNUAL"], p=[0.3, 0.2, 0.5])
            status = "ACTIVE" if exp > AS_OF else "LAPSED"
            rows.append([f"POL{pid:09d}", cust["customer_id"].iat[i], itype, f"PRV{rng.integers(1,6)}",
                         start, exp, renew, prem, round(prem * rng.uniform(8, 25), 2), freq, status])
            plist.append((status, prem, (renew - AS_OF).days))
            pid += 1
        per_cust[cust["customer_id"].iat[i]] = plist
    cols = ["policy_id", "customer_id", "insurance_type", "provider_code", "start_date",
            "expiry_date", "renewal_date", "premium_amount", "policy_value",
            "payment_frequency", "policy_status"]
    df = pd.DataFrame(rows, columns=cols)
    amap = {}
    for c, ps in per_cust.items():
        act = [p for p in ps if p[0] == "ACTIVE"]
        prem_annual = sum(p[1] for p in act)
        nearest = min([p[2] for p in act], default=None)
        amap[c] = (len(ps), len(act) > 0, round(prem_annual, 2), nearest)
    df["insurance_product_count"] = df["customer_id"].map(lambda c: amap[c][0])
    df["active_insurance_flag"] = df["customer_id"].map(lambda c: amap[c][1])
    df["total_annual_premium"] = df["customer_id"].map(lambda c: amap[c][2])
    df["nearest_renewal_days"] = df["customer_id"].map(lambda c: amap[c][3]).astype("Int64")
    return df


# ==========================================================================
# FACT_DIGITAL_ACTIVITY  (daily)
# ==========================================================================
def gen_digital(cust, rng):
    arch = cust.attrs["archetype"]
    n = len(cust)
    intensity = pd.Series(arch).map({
        "HIGH_VALUE": 1.5, "POTENTIAL_INVESTOR": 1.3, "CREDIT_OPPORTUNITY": 1.2,
        "DIGITAL_ACTIVE": 2.3, "CHURN_RISK": 0.35, "STANDARD": 0.7}).to_numpy()
    dates = [AS_OF - timedelta(days=d) for d in range(DIGITAL_DAYS - 1, -1, -1)]
    rows = []
    eng_now = np.zeros(n)
    login30 = np.zeros(n)
    activedays30 = np.zeros(n)
    dtxn30 = np.zeros(n)
    featusage30 = np.zeros(n)
    for i in range(n):
        it = intensity[i]
        if rng.random() > min(0.99, 0.45 + it / 3):
            continue
        a = arch[i]
        daily_p = min(0.95, 0.25 * it)
        seq_login = np.zeros(DIGITAL_DAYS)
        seq_dtxn = np.zeros(DIGITAL_DAYS)
        seq_feat = np.zeros(DIGITAL_DAYS)
        seq_active = np.zeros(DIGITAL_DAYS, dtype=bool)
        last_login = None
        for d in range(DIGITAL_DAYS):
            decay = 1.0
            if a == "CHURN_RISK":
                decay = max(0.1, 1.0 - 0.010 * (DIGITAL_DAYS - 1 - d))
            if rng.random() < daily_p * decay:
                lg = 1 + rng.poisson(2.5 * it * decay)
                seq_login[d] = lg
                seq_active[d] = True
                seq_dtxn[d] = rng.poisson(1.6 * it)
                seq_feat[d] = rng.poisson(1.2 * it)
                last_login = datetime(dates[d].year, dates[d].month, dates[d].day,
                                      int(rng.integers(6, 23)), int(rng.integers(0, 60)))
        for d in range(DIGITAL_DAYS):
            if not seq_active[d] and rng.random() > 0.25:
                continue
            w0 = max(0, d - 29)
            l30 = int(seq_login[w0:d + 1].sum())
            ad30 = int(seq_active[w0:d + 1].sum())
            dt30 = int(seq_dtxn[w0:d + 1].sum())
            qr30 = int(dt30 * rng.uniform(0.2, 0.5))
            push_recv = int(rng.integers(0, 4))
            push_open = int(push_recv * rng.uniform(0, 0.8))
            sess = int(seq_login[d] + rng.poisson(1))
            transf = int(seq_dtxn[d] * rng.uniform(0.2, 0.6))
            qrp = int(seq_dtxn[d] * rng.uniform(0.1, 0.4))
            billp = int(seq_dtxn[d] * rng.uniform(0.1, 0.3))
            feat = int(seq_feat[d])
            ft30 = int(seq_feat[w0:d + 1].sum())
            eng = float(np.clip(l30 * 1.6 + dt30 * 1.2 + ad30 * 1.5, 0, 100))
            rows.append((
                dates[d], cust["customer_id"].iat[i], int(seq_login[d]), bool(seq_active[d]),
                sess, transf, qrp, billp, int(seq_dtxn[d]), feat, push_recv, push_open,
                last_login, l30, ad30, dt30, qr30,
                round(float(dt30 / (l30 + 1)), 4), round(float(push_open / (push_recv + 1)), 4),
                round(eng, 2)))
            if d == DIGITAL_DAYS - 1:
                eng_now[i] = eng
                login30[i] = l30
                activedays30[i] = ad30
                dtxn30[i] = dt30
                featusage30[i] = ft30
        if eng_now[i] == 0:  # ensure last-row features captured even if last day skipped
            w0 = max(0, DIGITAL_DAYS - 30)
            login30[i] = int(seq_login[w0:].sum())
            activedays30[i] = int(seq_active[w0:].sum())
            dtxn30[i] = int(seq_dtxn[w0:].sum())
            featusage30[i] = int(seq_feat[w0:].sum())
            eng_now[i] = float(np.clip(login30[i] * 1.6 + dtxn30[i] * 1.2 + activedays30[i] * 1.5, 0, 100))

    cols = ["activity_date", "customer_id", "login_count", "active_flag", "session_count",
            "transfer_count", "qr_payment_count", "bill_payment_count", "digital_txn_count",
            "feature_usage_count", "push_received_count", "push_open_count", "last_login_timestamp",
            "login_count_30d", "active_days_30d", "digital_txn_count_30d", "qr_count_30d",
            "digital_txn_ratio", "push_open_rate", "digital_engagement_score"]
    df = pd.DataFrame(rows, columns=cols)
    helper = pd.DataFrame({
        "customer_id": cust["customer_id"], "digital_engagement_score": eng_now,
        "login_count_30d": login30.astype(int), "active_days_30d": activedays30.astype(int),
        "digital_txn_count_30d": dtxn30.astype(int), "feature_usage_30d": featusage30.astype(int),
    })
    return df, helper


# ==========================================================================
# FACT_CRM_INTERACTION
# ==========================================================================
def gen_crm(cust, rng):
    arch = cust.attrs["archetype"]
    n = len(cust)
    rm_by_cust = dict(zip(cust["customer_id"], cust["rm_id"]))
    lam = pd.Series(arch).map({
        "HIGH_VALUE": 2.6, "POTENTIAL_INVESTOR": 1.7, "CREDIT_OPPORTUNITY": 1.5,
        "DIGITAL_ACTIVE": 1.2, "CHURN_RISK": 2.8, "STANDARD": 1.0}).to_numpy()
    counts = rng.poisson(lam)
    rows = []
    k = 1
    feat = {}
    for i in range(n):
        cid = cust["customer_id"].iat[i]
        a = arch[i]
        offs = sorted(rng.integers(0, 365, size=counts[i]).tolist())
        last_resp = None
        rej_flag = False
        c30 = c90 = 0
        for j, off in enumerate(offs):
            its = AS_OF_TS - timedelta(days=int(off), seconds=int(rng.integers(0, 86400)))
            itype = rng.choice(["OUTBOUND_CALL", "INBOUND_CALL", "EMAIL", "MEETING", "APP_MESSAGE"],
                               p=[0.35, 0.2, 0.2, 0.1, 0.15])
            reason = rng.choice(["PRODUCT_OFFER", "SERVICE", "RETENTION", "ADVISORY", "ONBOARDING"],
                                p=[0.4, 0.2, 0.15, 0.15, 0.1])
            if a == "CHURN_RISK" and rng.random() < 0.4:
                reason = "RETENTION"
            resp = rng.choice(["INTERESTED", "NOT_INTERESTED", "CALLBACK", "NO_RESPONSE", "CONVERTED"],
                              p=[0.22, 0.25, 0.15, 0.3, 0.08])
            stage = rng.choice(["NEW", "CONTACTED", "QUALIFIED", "PROPOSAL", "WON", "LOST"],
                               p=[0.2, 0.3, 0.2, 0.12, 0.08, 0.1])
            lead = rng.choice(["OPEN", "WORKING", "CLOSED"], p=[0.3, 0.3, 0.4])
            pcode = rng.choice([p[1] for p in PRODUCTS])
            reason_lost = rng.choice(["PRICE", "TIMING", "NO_NEED", "COMPETITOR", None], p=[0.2, 0.2, 0.2, 0.1, 0.3]) if stage == "LOST" else None
            if off <= 30:
                c30 += 1
            if off <= 90:
                c90 += 1
            recent_rej = (resp == "NOT_INTERESTED") and (off <= 30)
            if recent_rej:
                rej_flag = True
            rows.append((
                f"CRM{k:010d}", cid, rm_by_cust[cid], its, itype,
                rng.choice(["PHONE", "EMAIL", "BRANCH", "APP"]), pcode, reason, lead, stage, resp,
                (its + timedelta(days=int(rng.integers(3, 30)))).date() if resp == "CALLBACK" else None,
                reason_lost, rng.choice([c[0] for c in CAMPAIGNS] + [None]),
                int(off) if j == len(offs) - 1 else int(offs[-1] - off),
                0, 0, None, pcode if reason == "PRODUCT_OFFER" else None, recent_rej))
            k += 1
            if j == len(offs) - 1:
                last_resp = resp
        days_since = offs[0] if offs else None  # smallest offset = most recent
        feat[cid] = {
            "days_since_last_rm_contact": int(days_since) if days_since is not None else 999,
            "rm_contact_count_90d": c90,
            "last_customer_response": last_resp,
            "recent_rejection_flag": rej_flag,
            "contact_count_7d": sum(1 for o in offs if o <= 7),
        }
    df = pd.DataFrame(rows, columns=[
        "interaction_id", "customer_id", "rm_id", "interaction_timestamp", "interaction_type",
        "channel", "product_code", "interaction_reason", "lead_status", "opportunity_stage",
        "customer_response", "next_followup_date", "reason_lost", "campaign_id",
        "days_since_last_contact", "contact_count_30d", "contact_count_90d",
        "last_customer_response", "previous_product_interest", "recent_rejection_flag"])
    # fill rolling counts per customer
    df = df.sort_values(["customer_id", "interaction_timestamp"])
    df["contact_count_30d"] = df.groupby("customer_id").cumcount()  # placeholder, recompute below
    tmp = df.copy()
    tmp["off"] = (AS_OF_TS - tmp["interaction_timestamp"]).dt.days
    c30 = tmp[tmp.off <= 30].groupby("customer_id").size()
    c90 = tmp[tmp.off <= 90].groupby("customer_id").size()
    df["contact_count_30d"] = df["customer_id"].map(c30).fillna(0).astype(int)
    df["contact_count_90d"] = df["customer_id"].map(c90).fillna(0).astype(int)
    df["last_customer_response"] = df["customer_id"].map({c: feat[c]["last_customer_response"] for c in feat})
    return df, feat


# ==========================================================================
# FACT_CUSTOMER_SERVICE
# ==========================================================================
def gen_service(cust, rng):
    arch = cust.attrs["archetype"]
    n = len(cust)
    lam = pd.Series(arch).map({
        "HIGH_VALUE": 1.2, "POTENTIAL_INVESTOR": 0.9, "CREDIT_OPPORTUNITY": 1.0,
        "DIGITAL_ACTIVE": 1.4, "CHURN_RISK": 2.2, "STANDARD": 0.8}).to_numpy()
    counts = rng.poisson(lam)
    rows = []
    k = 1
    feat = {}
    for i in range(n):
        cid = cust["customer_id"].iat[i]
        a = arch[i]
        offs = sorted(rng.integers(0, 180, size=counts[i]).tolist())
        cc30 = cc90 = 0
        serious15 = serious7 = False
        csats = []
        opencnt = 0
        for j, off in enumerate(offs):
            created = AS_OF_TS - timedelta(days=int(off), seconds=int(rng.integers(0, 86400)))
            sev = rng.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"], p=[0.4, 0.38, 0.17, 0.05])
            is_complaint = rng.random() < (0.55 if a == "CHURN_RISK" else 0.35)
            status = rng.choice(["OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"], p=[0.08, 0.12, 0.5, 0.3])
            closed = None if status in ("OPEN", "IN_PROGRESS") else created + timedelta(hours=float(rng.gamma(2, 8)))
            rt = round((closed - created).total_seconds() / 3600, 2) if closed else None
            csat = int(rng.integers(1, 6)) if status in ("RESOLVED", "CLOSED") else None
            if csat:
                csats.append(csat)
            if status in ("OPEN", "IN_PROGRESS"):
                opencnt += 1
            if is_complaint and off <= 30:
                cc30 += 1
            if is_complaint and off <= 90:
                cc90 += 1
            if is_complaint and sev in ("HIGH", "CRITICAL"):
                if off <= 15:
                    serious15 = True
                if off <= 7:
                    serious7 = True
            rows.append((
                f"CASE{k:010d}", cid, created, closed,
                rng.choice(["HOTLINE", "APP", "BRANCH", "EMAIL", "CHATBOT"]),
                rng.choice(["INQUIRY", "COMPLAINT", "REQUEST", "DISPUTE"]),
                rng.choice(["CARD", "ACCOUNT", "LOAN", "APP", "FEE", "OTHER"]),
                rng.choice([p[1] for p in PRODUCTS] + [None]),
                is_complaint, sev, status, rt, csat, 0, 0, serious7, None, None, opencnt))
            k += 1
        days_since_complaint = next((o for o in offs), None)
        feat[cid] = {
            "complaint_count_30d": cc30,
            "serious_complaint_7d": serious7,
            "serious_complaint_15d": serious15,
            "avg_csat": round(float(np.mean(csats)), 2) if csats else None,
        }
    df = pd.DataFrame(rows, columns=[
        "case_id", "customer_id", "created_timestamp", "closed_timestamp", "channel",
        "case_type", "issue_category", "product_code", "complaint_flag", "severity",
        "case_status", "resolution_time_hours", "csat_score", "complaint_count_30d",
        "complaint_count_90d", "serious_complaint_7d", "days_since_last_complaint",
        "avg_csat", "open_case_count"])
    tmp = df.copy()
    tmp["off"] = (AS_OF_TS - tmp["created_timestamp"]).dt.days
    cc30 = tmp[(tmp.off <= 30) & tmp.complaint_flag].groupby("customer_id").size()
    cc90 = tmp[(tmp.off <= 90) & tmp.complaint_flag].groupby("customer_id").size()
    dslc = tmp[tmp.complaint_flag].groupby("customer_id")["off"].min()
    df["complaint_count_30d"] = df["customer_id"].map(cc30).fillna(0).astype(int)
    df["complaint_count_90d"] = df["customer_id"].map(cc90).fillna(0).astype(int)
    df["days_since_last_complaint"] = df["customer_id"].map(dslc).astype("Int64")
    df["avg_csat"] = df["customer_id"].map({c: feat[c]["avg_csat"] for c in feat})
    return df, feat


# ==========================================================================
# DIM_CAMPAIGN already; FACT_CAMPAIGN (campaign x customer)
# ==========================================================================
def gen_campaign(cust, dim_campaign, rng):
    arch = cust.attrs["archetype"]
    n = len(cust)
    pref = {
        "HIGH_VALUE": ["INVESTMENT", "CREDIT_CARD", "CREDIT_CARD"],
        "CREDIT_OPPORTUNITY": ["CREDIT_CARD", "LOAN", "LOAN"],
        "POTENTIAL_INVESTOR": ["DEPOSIT", "INVESTMENT", "INVESTMENT"],
        "DIGITAL_ACTIVE": ["CREDIT_CARD", "DEPOSIT", "CREDIT_CARD"],
        "CHURN_RISK": ["DEPOSIT", "CREDIT_CARD", "DEPOSIT"],
        "STANDARD": ["DEPOSIT", "CREDIT_CARD", "LOAN"],
    }
    camp_by_group = {}
    for cid, name, pid, pcode, ctype in CAMPAIGNS:
        grp = next(p[3] for p in PRODUCTS if p[0] == pid)
        camp_by_group.setdefault(grp, []).append((cid, pcode))
    rows = []
    feat = {}
    k = 1
    for i in range(n):
        cid = cust["customer_id"].iat[i]
        a = arch[i]
        ncamp = int(rng.integers(1, 6))
        sent = resp = conv = 0
        card_resp = card_sent = 0
        last_conv = False
        for _ in range(ncamp):
            grp = rng.choice(pref[a])
            cand = camp_by_group.get(grp, camp_by_group["DEPOSIT"])
            camp_id, pcode = cand[int(rng.integers(0, len(cand)))]
            cdate = AS_OF - timedelta(days=int(rng.integers(10, CAMPAIGN_MONTHS * 30)))
            chan = rng.choice(["EMAIL", "SMS", "APP_PUSH", "RM_OUTBOUND"], p=[0.3, 0.28, 0.30, 0.12])
            p_resp = 0.10 + (BASE_PROPENSITY[a].get(grp, 0.3) - 0.3) * 0.7
            p_resp = float(np.clip(p_resp + rng.normal(0, 0.05), 0.02, 0.85))
            sent_f = True
            deliv = rng.random() < 0.95
            opened = deliv and rng.random() < 0.55
            clicked = opened and rng.random() < 0.4
            responded = clicked and rng.random() < (p_resp / 0.3)
            interested = responded and rng.random() < 0.6
            applied = interested and rng.random() < 0.5
            approved = applied and rng.random() < 0.7
            converted = approved and rng.random() < 0.75
            conv_date = cdate + timedelta(days=int(rng.integers(2, 40))) if converted else None
            rows.append((
                f"CC{k:010d}", camp_id, cid, pcode, cdate, chan,
                sent_f, deliv, opened, clicked, responded, interested, applied, approved,
                converted, conv_date))
            sent += 1
            resp += int(responded)
            conv += int(converted)
            if grp == "CREDIT_CARD":
                card_sent += 1
                card_resp += int(responded)
            last_conv = last_conv or converted
            k += 1
        feat[cid] = {
            "campaign_count_6m": sent,
            "previous_campaign_response": round(resp / sent, 4) if sent else 0.0,
            "previous_card_campaign_response": round(card_resp / card_sent, 4) if card_sent else 0.0,
            "previous_conversion_flag": last_conv,
        }
    df = pd.DataFrame(rows, columns=[
        "campaign_customer_id", "campaign_id", "customer_id", "product_code", "campaign_date",
        "channel", "sent_flag", "delivered_flag", "opened_flag", "clicked_flag", "responded_flag",
        "interested_flag", "applied_flag", "approved_flag", "converted_flag", "conversion_date"])
    return df, feat


# ==========================================================================
# CUSTOMER_360_FEATURE_MART
# ==========================================================================
def gen_feature_mart(cust, casa_h, agg, dim_card, loan, deposit, insurance, dig_h,
                     crm_feat, svc_feat, camp_feat, rng):
    n = len(cust)
    cid = cust["customer_id"].to_numpy()
    idx = pd.Index(cid)

    loan_agg = loan.groupby("customer_id").agg(
        has_active_loan=("has_active_loan", "max")).reindex(idx)
    prod_count = pd.Series(0, index=idx)
    for src, col in [(dim_card, "customer_id"), (loan, "customer_id"),
                     (deposit, "customer_id"), (insurance, "customer_id")]:
        prod_count = prod_count.add(src.groupby(col).size().reindex(idx).fillna(0), fill_value=0)

    has_cc = dim_card[dim_card.card_type == "CREDIT"].groupby("customer_id").size().reindex(idx).fillna(0) > 0
    has_premium = dim_card[(dim_card.card_type == "CREDIT") & (dim_card.card_tier == "PLATINUM")
                           & (dim_card.card_status == "ACTIVE")].groupby("customer_id").size().reindex(idx).fillna(0) > 0
    dep_bal = deposit.groupby("customer_id")["current_balance"].sum().reindex(idx).fillna(0.0)
    ins_active = insurance[insurance.policy_status == "ACTIVE"].groupby("customer_id").size().reindex(idx).fillna(0) > 0

    A = agg.set_index("customer_id").reindex(idx)
    CH = casa_h.set_index("customer_id").reindex(idx)
    DH = dig_h.set_index("customer_id").reindex(idx)

    arch = np.asarray(cust.attrs["archetype"])
    p_invest = pd.Series(arch).map({
        "HIGH_VALUE": 0.42, "POTENTIAL_INVESTOR": 0.38, "CREDIT_OPPORTUNITY": 0.05,
        "DIGITAL_ACTIVE": 0.12, "CHURN_RISK": 0.04, "STANDARD": 0.06}).to_numpy()
    has_invest = rng.random(n) < p_invest

    def cf(d, key, default=0):
        return pd.Series({c: d.get(c, {}).get(key, default) for c in cid}, index=idx)

    df = pd.DataFrame({
        "snapshot_date": AS_OF,
        "customer_id": cid,
        "segment": cust["customer_segment"].to_numpy(),
        "age_group": cust["age_group"].to_numpy(),
        "income_band": cust["income_band"].to_numpy(),
        "relationship_years": cust["relationship_years"].to_numpy(),
        "region": cust["region_code"].to_numpy(),
        "rm_id": cust["rm_id"].to_numpy(),
        "branch_id": cust["branch_id"].to_numpy(),
        "avg_balance_30d": CH["casa_balance"].fillna(0).values,   # approx: last balance
        "avg_balance_90d": CH["casa_balance"].fillna(0).values,
        "balance_growth_3m": CH["casa_growth_3m"].fillna(0).round(4).values,
        "inflow_30d": CH["inflow_30d"].fillna(0).round(2).values,
        "outflow_30d": CH["outflow_30d"].fillna(0).round(2).values,
        "salary_flag": CH["salary_flag"].fillna(False).astype(bool).values,
        "salary_amount_avg": CH["salary_avg"].fillna(0).round(2).values,
        "txn_count_30d": A["txn_count_30d"].fillna(0).astype(int).values,
        "txn_count_90d": A["txn_count_90d"].fillna(0).astype(int).values,
        "spending_30d": A["total_spend_30d"].fillna(0).values,
        "spending_90d": A["total_spend_90d"].fillna(0).values,
        "spending_growth_3m": A["spending_growth_3m"].fillna(0).values,
        "online_spending_90d": A["online_spend_30d"].fillna(0).values,
        "travel_spending_90d": A["travel_spend_90d"].fillna(0).values,
        "international_spending_90d": A["international_spend_90d"].fillna(0).values,
        "product_count": prod_count.astype(int).values,
        "has_credit_card": has_cc.values,
        "has_premium_card": has_premium.values,
        "has_active_loan": loan_agg["has_active_loan"].fillna(False).astype(bool).values,
        "deposit_balance": dep_bal.round(2).values,
        "insurance_active_flag": ins_active.values,
        "has_investment": has_invest,
        "income_monthly": CH["income_mid"].fillna(0).round(2).values,
        "net_cashflow_30d": (CH["inflow_30d"].fillna(0) - CH["outflow_30d"].fillna(0)).round(2).values,
        "feature_usage_30d": DH["feature_usage_30d"].fillna(0).astype(int).values,
        "login_count_30d": DH["login_count_30d"].fillna(0).astype(int).values,
        "active_days_30d": DH["active_days_30d"].fillna(0).astype(int).values,
        "digital_txn_count_30d": DH["digital_txn_count_30d"].fillna(0).astype(int).values,
        "digital_engagement_score": DH["digital_engagement_score"].fillna(0).round(2).values,
        "days_since_last_rm_contact": cf(crm_feat, "days_since_last_rm_contact", 999).astype(int).values,
        "rm_contact_count_90d": cf(crm_feat, "rm_contact_count_90d", 0).astype(int).values,
        "last_customer_response": cf(crm_feat, "last_customer_response", None).values,
        "recent_rejection_flag": cf(crm_feat, "recent_rejection_flag", False).astype(bool).values,
        "complaint_count_30d": cf(svc_feat, "complaint_count_30d", 0).astype(int).values,
        "serious_complaint_7d": cf(svc_feat, "serious_complaint_7d", False).astype(bool).values,
        "avg_csat": cf(svc_feat, "avg_csat", None).values,
        "campaign_count_6m": cf(camp_feat, "campaign_count_6m", 0).astype(int).values,
        "previous_campaign_response": cf(camp_feat, "previous_campaign_response", 0.0).values,
        "previous_card_campaign_response": cf(camp_feat, "previous_card_campaign_response", 0.0).values,
        "previous_conversion_flag": cf(camp_feat, "previous_conversion_flag", False).astype(bool).values,
        "contact_count_7d": cf(crm_feat, "contact_count_7d", 0).astype(int).values,
        "serious_complaint_15d": cf(svc_feat, "serious_complaint_15d", False).astype(bool).values,
        "recent_rejection_30d_flag": cf(crm_feat, "recent_rejection_flag", False).astype(bool).values,
    })
    money = ["spending_30d", "spending_90d", "online_spending_90d", "travel_spending_90d",
             "international_spending_90d"]
    df[money] = df[money].round(2)
    df["spending_growth_3m"] = df["spending_growth_3m"].round(4)
    return df


# ==========================================================================
# Sub-scores, AI_FEATURE_CUSTOMER, AI_CUSTOMER_SCORE, AI_SCORE_REASON
# ==========================================================================
def gen_ai_layer(cust, mart, dim_card, deposit, txn, rng):
    arch = np.asarray(cust.attrs["archetype"])
    n = len(cust)
    m = mart.set_index("customer_id").loc[cust["customer_id"].to_numpy()]

    bal = m["avg_balance_90d"].to_numpy()
    dep = m["deposit_balance"].to_numpy()
    dig = m["digital_engagement_score"].to_numpy()
    camp_resp = m["previous_campaign_response"].to_numpy()
    rel_years = m["relationship_years"].to_numpy()
    prod_count = m["product_count"].to_numpy()
    complaints = m["complaint_count_30d"].to_numpy()
    growth = m["balance_growth_3m"].to_numpy()
    spend30 = m["spending_30d"].to_numpy()
    has_premium = m["has_premium_card"].to_numpy()

    dep_maturing = set(deposit[deposit.deposit_status == "MATURING_SOON"]["customer_id"])
    auto_txn = set(txn[txn.transaction_category == "AUTOMOTIVE"]["customer_id"])
    inc_band = pd.Series(m["income_band"].to_numpy()).map(
        {"<10M": 0.1, "10-20M": 0.3, "20-40M": 0.55, "40-80M": 0.8, "80M+": 1.0}).to_numpy()

    # customer-level scores (0..100).
    # Sub-scores are only weakly correlated when built purely from independent
    # features, so almost nobody clears the "Very High" band. Add an explicit
    # archetype "quality boost" (same latent signal that makes the archetype a
    # real opportunity in the first place) so the top-fit customers land
    # consistently high across dimensions instead of by joint chance.
    QUALITY = {
        "HIGH_VALUE":         {"cv": 45, "eng": 14, "rel": 24},
        "CREDIT_OPPORTUNITY": {"cv": 16, "eng": 16, "rel": 14},
        "POTENTIAL_INVESTOR": {"cv": 34, "eng": 14, "rel": 20},
        "DIGITAL_ACTIVE":     {"cv": 10, "eng": 45, "rel": 14},
        "CHURN_RISK":         {"cv": -10, "eng": -15, "rel": -15},
        "STANDARD":           {"cv": 0,  "eng": 0,  "rel": 0},
    }
    cv_boost = np.array([QUALITY[a]["cv"] for a in arch])
    eng_boost = np.array([QUALITY[a]["eng"] for a in arch])
    rel_boost = np.array([QUALITY[a]["rel"] for a in arch])

    cv = np.clip(100 * (0.45 * _norm(bal) + 0.20 * _norm(dep) + 0.20 * inc_band
                        + 0.15 * _norm(rel_years)) + cv_boost + rng.normal(0, 2, n), 0, 100)
    eng = np.clip(100 * (0.7 * _norm(dig) + 0.3 * _norm(m["login_count_30d"].to_numpy()))
                  + eng_boost + rng.normal(0, 2, n), 0, 100)
    rel = np.clip(100 * (0.4 * _norm(rel_years) + 0.3 * _norm(prod_count)
                         + 0.3 * (1 - _norm(complaints))) + rel_boost + rng.normal(0, 2, n), 0, 100)

    cid = cust["customer_id"].to_numpy()
    prod_props, rows_feat, rows_score, rows_reason = {}, [], [], []
    sc_k = 1
    rs_k = 1
    ts = AS_OF_TS

    for i in range(n):
        a = arch[i]
        c = cid[i]
        base = BASE_PROPENSITY[a]
        props = {}
        intent_by_grp = {}
        for grp in GROUPS:
            v = base[grp]
            it = 0.35
            if grp == "CREDIT_CARD":
                if not has_premium[i]:
                    v += 0.14
                if spend30[i] > 8_000_000:
                    v += 0.10
                    it += 0.25
                if m["previous_card_campaign_response"].iat[i] > 0.3:
                    it += 0.15
            if grp == "LOAN" and c in auto_txn:
                v += 0.15
                it += 0.35
            if grp == "DEPOSIT" and c in dep_maturing:
                v += 0.18
                it += 0.4
            if grp == "DEPOSIT" and growth[i] > 0.15:
                it += 0.2
            if grp == "INVESTMENT" and bal[i] > 300_000_000:
                v += 0.12
                it += 0.2
            if grp == "INVESTMENT" and c in dep_maturing:
                it += 0.25
            if camp_resp[i] > 0.3:
                it += 0.1
            props[grp] = float(_clip01(v + rng.normal(0, 0.05)))
            intent_by_grp[grp] = float(np.clip(it + rng.normal(0, 0.05), 0.02, 0.99))
        prod_props[c] = props

        suppression = bool(cust["do_not_contact_flag"].iat[i]
                           or mart["serious_complaint_15d"].iat[i]
                           or mart["recent_rejection_30d_flag"].iat[i]
                           or mart["contact_count_7d"].iat[i] >= 2
                           or not cust["customer_active_flag"].iat[i])
        intent_overall = 100 * max(intent_by_grp.values())
        timing_overall = float(np.clip(intent_overall * rng.uniform(0.8, 1.05), 0, 100))

        rows_feat.append((
            AS_OF, c, json.dumps({k: round(v, 3) for k, v in props.items()}),
            round(float(cv[i]), 2), round(float(intent_overall), 2), round(float(eng[i]), 2),
            round(timing_overall, 2), round(float(rel[i]), 2), suppression))

        for grp in GROUPS:
            pid = TARGET_PRODUCTS[grp]
            pp = props[grp] * 100
            intent = intent_by_grp[grp] * 100
            timing = float(np.clip(intent * rng.uniform(0.8, 1.05), 0, 100))
            sgs = (pp * SGS_W["pp"] + cv[i] * SGS_W["cv"] + intent * SGS_W["intent"]
                   + eng[i] * SGS_W["eng"] + timing * SGS_W["timing"] + rel[i] * SGS_W["rel"])
            sgs = round(float(np.clip(sgs, 0, 100)), 2)   # round BEFORE bucketing so priority_level
            pl = ("Very High" if sgs >= 90 else "High" if sgs >= 80 else "Medium" if sgs >= 70   # always matches the stored (rounded) score
                  else "Low" if sgs >= 60 else "Do not prioritize")
            sid = f"SC{sc_k:09d}"
            rows_score.append((
                sid, c, AS_OF, pid, "sge_smart_growth", "v2.0",
                round(props[grp], 4), round(pp, 2), round(float(cv[i]), 2),
                round(intent, 2), round(float(eng[i]), 2), round(timing, 2),
                round(float(rel[i]), 2), round(sgs, 2), pl, ts))
            sc_k += 1

            if grp == max(props, key=props.get):  # reasons for the top product only
                reasons = _build_reasons(a, m.iloc[i], c, has_premium[i], dep_maturing, auto_txn, growth[i])
                for rank, (fn, fv, contrib, direction) in enumerate(reasons, 1):
                    rows_reason.append((f"RS{rs_k:010d}", sid, fn, str(fv),
                                        round(contrib, 4), direction, rank))
                    rs_k += 1

    feat_cols = ["snapshot_date", "customer_id", "product_propensity_feature_set",
                 "customer_value_score", "intent_signal_score", "engagement_score",
                 "timing_score", "relationship_score", "sales_suppression_flag"]
    score_cols = ["score_id", "customer_id", "snapshot_date", "product_id", "model_id",
                  "model_version", "propensity_probability", "product_propensity_score",
                  "customer_value_score", "intent_signal_score", "engagement_score",
                  "timing_score", "relationship_score", "smart_growth_score", "priority_level",
                  "prediction_timestamp"]
    reason_cols = ["reason_id", "score_id", "feature_name", "feature_value",
                   "contribution_score", "impact_direction", "reason_rank"]
    return (pd.DataFrame(rows_feat, columns=feat_cols),
            pd.DataFrame(rows_score, columns=score_cols),
            pd.DataFrame(rows_reason, columns=reason_cols),
            prod_props)


def _build_reasons(arch, mrow, cid, has_premium, dep_maturing, auto_txn, growth):
    r = []
    if not has_premium:
        r.append(("has_premium_card", False, 0.22, "POSITIVE"))
    if mrow["spending_30d"] > 8_000_000:
        r.append(("spending_30d", int(mrow["spending_30d"]), 0.18, "POSITIVE"))
    if mrow["salary_flag"]:
        r.append(("salary_flag", True, 0.15, "POSITIVE"))
    if growth > 0.1:
        r.append(("balance_growth_3m", round(float(growth), 3), 0.12, "POSITIVE"))
    if cid in dep_maturing:
        r.append(("days_to_nearest_maturity", "<30", 0.16, "POSITIVE"))
    if cid in auto_txn:
        r.append(("automotive_txn_90d", True, 0.14, "POSITIVE"))
    if mrow["digital_engagement_score"] > 40:
        r.append(("digital_engagement_score", round(float(mrow["digital_engagement_score"]), 1), 0.10, "POSITIVE"))
    if mrow["recent_rejection_flag"]:
        r.append(("recent_rejection_flag", True, 0.20, "NEGATIVE"))
    if mrow["complaint_count_30d"] > 0:
        r.append(("complaint_count_30d", int(mrow["complaint_count_30d"]), 0.12, "NEGATIVE"))
    if not r:
        r.append(("relationship_years", round(float(mrow["relationship_years"]), 1), 0.08, "POSITIVE"))
    return r[:6]


# ==========================================================================
# AI_RECOMMENDATION (Next Best Product / Action / Channel / Timing)
# ==========================================================================
PRODUCT_LABEL = {"CREDIT_CARD": "Platinum Credit Card", "LOAN": "Vay mua nhà / ô tô",
                 "DEPOSIT": "Tiền gửi có kỳ hạn ưu đãi", "INVESTMENT": "Sản phẩm đầu tư / quỹ mở"}


def gen_recommendations(cust, score_df, feat_df, prod_props, rng):
    arch = dict(zip(cust["customer_id"], cust.attrs["archetype"]))
    pref_ch = dict(zip(cust["customer_id"], cust["preferred_channel"]))
    supp = dict(zip(feat_df["customer_id"], feat_df["sales_suppression_flag"]))
    seg = dict(zip(cust["customer_id"], cust["customer_segment"]))

    # best score row per (customer, group)
    grp_of_pid = {v: k for k, v in TARGET_PRODUCTS.items()}
    score_df = score_df.copy()
    score_df["grp"] = score_df["product_id"].map(grp_of_pid)
    rows = []
    k = 1
    ts = AS_OF_TS
    for c, sub in score_df.groupby("customer_id"):
        a = arch[c]
        sub = sub.sort_values("smart_growth_score", ascending=False).reset_index(drop=True)
        s_supp = bool(supp.get(c, False))
        for rank in range(min(3, len(sub))):
            row = sub.iloc[rank]
            grp = row["grp"]
            sgs = row["smart_growth_score"]
            prio = row["priority_level"]
            if s_supp:
                action, status = "NO_CONTACT", "BLOCKED"
            elif seg[c] in ("PRIVATE", "AFFLUENT") or a == "HIGH_VALUE":
                action = "RM_CALL" if rank == 0 else "RM_ASSISTED_MESSAGE"
                status = "RM_APPROVAL"
            elif a == "DIGITAL_ACTIVE":
                action = "IN_APP" if rank == 0 else "PUSH"
                status = "SENT" if prio in ("Very High", "High") else "NEW"
            elif prio in ("Very High", "High"):
                action, status = "RM_ASSISTED_MESSAGE", "SENT"
            else:
                action, status = "NURTURE", "NEW"

            if action in ("RM_CALL", "RM_ASSISTED_MESSAGE"):
                channel = "RM_CALL"
            elif action in ("IN_APP",):
                channel = "IN_APP"
            elif action == "PUSH":
                channel = "PUSH"
            elif action == "NURTURE":
                channel = "EMAIL"
            else:
                channel = pref_ch.get(c, "SMS")

            timing = ("Trong vòng 48 giờ" if rank == 0 and sgs >= 80 else
                      "Trong tuần này" if rank == 0 else
                      "Trong 2 tuần tới" if rank == 1 else "Nurture 30 ngày")
            angle = _message_angle(grp, a)
            exp_conv = float(np.clip(row["propensity_probability"] * rng.uniform(0.7, 0.95), 0.01, 0.95))
            rows.append((
                f"REC{k:09d}", c, row["score_id"], row["product_id"], action, channel, timing,
                angle, round(exp_conv, 4), prio, s_supp, ts, status, rank + 1))
            k += 1
    return pd.DataFrame(rows, columns=[
        "recommendation_id", "customer_id", "score_id", "recommended_product_id",
        "recommended_action", "recommended_channel", "recommended_timing", "message_angle",
        "expected_conversion", "priority", "suppression_flag", "recommendation_timestamp",
        "status", "priority_rank"])


def _message_angle(grp, arch):
    if grp == "CREDIT_CARD":
        return "Cashback + ưu đãi du lịch, phù hợp mức chi tiêu cao và thu nhập ổn định."
    if grp == "LOAN":
        return "Lãi suất cố định ưu đãi, duyệt nhanh, giải ngân trong ngày."
    if grp == "DEPOSIT":
        return "Cộng thêm lãi suất khi gửi online, tối ưu dòng tiền nhàn rỗi sắp đáo hạn."
    return "Danh mục đầu tư cân bằng, miễn phí phí quản lý 6 tháng đầu."


# ==========================================================================
# RM_ACTION_FEEDBACK
# ==========================================================================
def gen_feedback(reco, cust, rng):
    rm_by_cust = dict(zip(cust["customer_id"], cust["rm_id"]))
    cand = reco[(reco.priority_rank == 1) & (reco.status != "BLOCKED")].reset_index(drop=True)
    mask = rng.random(len(cand)) < np.where(cand["status"] == "RM_APPROVAL", 0.6, 0.4)
    sub = cand[mask].reset_index(drop=True)
    k = len(sub)
    conf = sub["expected_conversion"].to_numpy()
    p_conv = np.clip(conf * 0.55, 0.03, 0.55)
    r = rng.random(k)
    resp = np.where(r < p_conv, "CONVERTED",
           np.where(r < p_conv + 0.12, "APPLIED",
           np.where(r < p_conv + 0.34, "INTERESTED",
           np.where(r < p_conv + 0.52, "CALLBACK",
           np.where(r < p_conv + 0.78, "NO_RESPONSE", "NOT_INTERESTED")))))
    action_type = np.where(sub["recommended_channel"].to_numpy() == "RM_CALL", "CALL",
                  np.where(sub["recommended_channel"].to_numpy() == "EMAIL", "EMAIL", "SMS"))
    action_type = np.where(resp == "CALLBACK", "CALL", action_type)
    result = np.where(np.isin(resp, ["CONVERTED", "APPLIED"]), "DONE",
             np.where(resp == "CALLBACK", "SCHEDULED",
             np.where(resp == "NO_RESPONSE", "NO_ANSWER", "CONTACTED")))
    days = rng.integers(0, 40, size=k)
    reasons = {"NOT_INTERESTED": rng.choice(["Chưa có nhu cầu", "Đang dùng NH khác", "Phí/lãi suất chưa hấp dẫn"], k)}
    notes_map = {
        "CONVERTED": "KH đồng ý, đã hoàn tất hồ sơ mở sản phẩm.",
        "APPLIED": "KH đã nộp hồ sơ, chờ phê duyệt.",
        "INTERESTED": "KH quan tâm, cần gửi thêm bảng minh hoạ lãi suất/phí.",
        "CALLBACK": "KH bận, hẹn gọi lại cuối tuần.",
        "NO_RESPONSE": "Gọi 2 lần không liên lạc được, sẽ thử kênh khác.",
        "NOT_INTERESTED": "KH từ chối ở thời điểm hiện tại.",
    }
    out = pd.DataFrame({
        "feedback_id": [f"FB{i:09d}" for i in range(1, k + 1)],
        "recommendation_id": sub["recommendation_id"].to_numpy(),
        "customer_id": sub["customer_id"].to_numpy(),
        "rm_id": [rm_by_cust[c] for c in sub["customer_id"]],
        "action_timestamp": [AS_OF_TS - timedelta(days=int(d), seconds=int(rng.integers(0, 86400))) for d in days],
        "action_type": action_type,
        "result_status": result,
        "customer_response": resp,
        "appointment_flag": np.isin(resp, ["INTERESTED", "APPLIED", "CONVERTED"]) & (rng.random(k) < 0.5),
        "application_flag": np.isin(resp, ["APPLIED", "CONVERTED"]),
        "converted_flag": resp == "CONVERTED",
        "reason_not_interested": np.where(resp == "NOT_INTERESTED",
                                          rng.choice(["Chưa có nhu cầu", "Đang dùng NH khác",
                                                      "Phí/lãi suất chưa hấp dẫn"], k), None),
        "rm_feedback": [notes_map[x] for x in resp],
        "next_followup_date": [ (AS_OF + timedelta(days=int(rng.integers(7, 45)))) if x in ("INTERESTED", "CALLBACK") else None for x in resp ],
    })
    return out


# ==========================================================================
# ML_CREDIT_CARD_TRAINING_SET
# ==========================================================================
def gen_training_set(cust, mart, dim_card, agg, camp_feat_df, campaign, rng):
    arch = np.asarray(cust.attrs["archetype"])
    n = len(cust)
    obs = AS_OF - timedelta(days=OBS_LAG_DAYS)
    m = mart.set_index("customer_id").loc[cust["customer_id"].to_numpy()]

    had_cc_now = m["has_credit_card"].to_numpy()
    # approximate "had cc at obs": assume same minus a few recent issues
    issued_after_obs = set(dim_card[(dim_card.card_type == "CREDIT")
                                    & (pd.to_datetime(dim_card.issue_date).dt.date > obs)]["customer_id"])
    had_cc_obs = np.array([bool(had_cc_now[i]) and cust["customer_id"].iat[i] not in issued_after_obs
                           for i in range(n)])

    # label: applied / converted a credit-card campaign within [obs, obs+90]
    cc_camp = campaign[campaign.product_code.str.startswith("CC_")].copy()
    cc_camp["cd"] = pd.to_datetime(cc_camp["campaign_date"]).dt.date
    win = cc_camp[(cc_camp.cd >= obs) & (cc_camp.cd <= obs + timedelta(days=LABEL_WINDOW))]
    applied = set(win[win.applied_flag]["customer_id"])
    converted = set(win[win.converted_flag]["customer_id"])

    base_p = pd.Series(arch).map({k: BASE_PROPENSITY[k]["CREDIT_CARD"] for k in BASE_PROPENSITY}).to_numpy()
    synth_apply = rng.random(n) < np.clip(base_p * 0.35 * (~had_cc_obs), 0, 1)

    A = agg.set_index("customer_id").reindex(cust["customer_id"].to_numpy())
    df = pd.DataFrame({
        "customer_id": cust["customer_id"].to_numpy(),
        "observation_date": obs,
        "age_group": cust["age_group"].to_numpy(),
        "segment": cust["customer_segment"].to_numpy(),
        "avg_balance_90d": m["avg_balance_90d"].to_numpy().round(2),
        "income_band": cust["income_band"].to_numpy(),
        "spending_90d": m["spending_90d"].to_numpy().round(2),
        "txn_count_90d": m["txn_count_90d"].to_numpy().astype(int),
        "salary_flag": m["salary_flag"].to_numpy().astype(bool),
        "digital_score": m["digital_engagement_score"].to_numpy().round(2),
        "has_credit_card": had_cc_obs,
        "campaign_response_history": m["previous_campaign_response"].to_numpy(),
        "days_since_last_rm_contact": m["days_since_last_rm_contact"].to_numpy().astype(int),
        "complaint_count_30d": m["complaint_count_30d"].to_numpy().astype(int),
    })
    df["label_applied_90d"] = [ (c in applied) or (c in converted) or bool(sa)
                                for c, sa in zip(df["customer_id"], synth_apply) ]
    df["label_converted_90d"] = [ (c in converted) or (bool(sa) and rng.random() < 0.4)
                                  for c, sa in zip(df["customer_id"], synth_apply) ]
    return df


# ==========================================================================
# forward-looking adoption label for the propensity model
# ==========================================================================
def add_adoption_label(prop_train, cust, campaign, rng):
    """y_adopt_next_90d = 1 if customer takes the product in the next 90 days.

    Driven by the same latent archetype propensity that shapes every other table,
    modulated by the doc features (product gap X6, campaign response X5, salary X4,
    spend/income/digital). Non-holders are the training population; holders get 0.
    """
    arch = dict(zip(cust["customer_id"], cust.attrs["archetype"]))
    grp_of_pid = {v: k for k, v in TARGET_PRODUCTS.items()}
    d = prop_train.copy()
    a = d["customer_id"].map(arch).to_numpy()
    grp = d["product_id"].map(grp_of_pid).to_numpy()
    base = np.array([BASE_PROPENSITY[ai][gi] for ai, gi in zip(a, grp)])

    x1 = d["x1_monthly_spending"].to_numpy()
    x2 = d["x2_income"].to_numpy()
    x3 = d["x3_digital_activity"].to_numpy()
    x4 = d["x4_salary_account"].to_numpy()
    x5 = d["x5_campaign_response"].to_numpy()
    x6 = d["x6_product_gap"].to_numpy()

    # proper logistic data-generating process: X1..X6 are the true drivers the
    # model must recover; the hidden archetype only nudges the intercept so the
    # label stays coherent with the rest of the dataset.
    z = (-3.7
         + 1.5 * x6 + 1.7 * x5 + 0.9 * x1 + 0.8 * x2 + 0.5 * x3 + 0.4 * x4
         + 1.3 * (base - 0.45)
         + rng.normal(0, 0.45, len(d)))
    p_adopt = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(len(d)) < p_adopt).astype(int)
    y = np.where(d["y_holds_product"].to_numpy() == 1, 0, y)   # holders can't adopt again
    d["y_adopt_next_90d"] = y
    return d


# ==========================================================================
# writer
# ==========================================================================
def dump(name, df):
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(PARQUET_DIR, exist_ok=True)
    # nullable-int columns come out of pandas as float64 ("4.0") which breaks
    # COPY into an INTEGER column - coerce whole-number float columns to Int64.
    df = df.copy()
    for c in df.columns:
        s = df[c]
        if s.dtype == float and s.notna().any() and s.isna().any():
            nn = s.dropna()
            if (nn % 1 == 0).all():
                df[c] = s.astype("Int64")
    df.to_csv(os.path.join(CSV_DIR, f"{name}.csv"), index=False, encoding="utf-8")
    try:
        d = df
        if df.attrs:
            d = df.copy()
            d.attrs = {}
        d.to_parquet(os.path.join(PARQUET_DIR, f"{name}.parquet"), index=False)
    except Exception as e:
        print(f"  ! parquet skipped for {name}: {e}")
    print(f"  {name:<30} {len(df):>10,} rows")


def main():
    global CASA_DAYS, DIGITAL_DAYS
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--casa-days", type=int, default=CASA_DAYS,
                    help="days of daily CASA history to emit (lower = faster/smaller for big runs)")
    ap.add_argument("--digital-days", type=int, default=DIGITAL_DAYS,
                    help="days of daily digital-activity history to emit")
    args = ap.parse_args()
    CASA_DAYS = args.casa_days
    DIGITAL_DAYS = args.digital_days
    rng = np.random.default_rng(args.seed)
    print(f"MSB Smart Growth Engine v2 - {args.customers:,} customers (seed {args.seed}), snapshot {AS_OF}"
          f"  [casa {CASA_DAYS}d / digital {DIGITAL_DAYS}d]")

    rm = gen_dim_rm(rng); dump("dim_rm", rm)
    prod = gen_dim_product(); dump("dim_product", prod)
    dim_campaign = gen_dim_campaign(rng); dump("dim_campaign", dim_campaign)

    cust = gen_dim_customer(args.customers, rng, rm)

    casa, casa_h = gen_casa(cust, rng)
    txn, agg = gen_transactions(cust, casa_h, rng)
    dim_card, fact_cm = gen_cards(cust, casa_h, rng)
    deposit = gen_deposits(cust, casa_h, rng)
    loan = gen_loans(cust, casa_h, rng)
    insurance = gen_insurance(cust, casa_h, rng)
    digital, dig_h = gen_digital(cust, rng)
    crm, crm_feat = gen_crm(cust, rng)
    service, svc_feat = gen_service(cust, rng)
    campaign, camp_feat = gen_campaign(cust, dim_campaign, rng)

    dump("dim_customer", cust)
    dump("dim_card", dim_card)
    dump("fact_casa_daily", casa)
    dump("fact_transaction", txn)
    dump("agg_customer_transaction", agg)
    dump("fact_card_monthly", fact_cm)
    dump("fact_deposit", deposit)
    dump("fact_loan", loan)
    dump("fact_insurance", insurance)
    dump("fact_digital_activity", digital)
    dump("fact_crm_interaction", crm)
    dump("fact_customer_service", service)
    dump("fact_campaign", campaign)

    mart = gen_feature_mart(cust, casa_h, agg, dim_card, loan, deposit, insurance, dig_h,
                            crm_feat, svc_feat, camp_feat, rng)
    dump("customer_360_feature_mart", mart)

    ai_feat, ai_score, ai_reason, prod_props = gen_ai_layer(cust, mart, dim_card, deposit, txn, rng)
    dump("ai_feature_customer", ai_feat)
    dump("ai_customer_score", ai_score)
    dump("ai_score_reason", ai_reason)

    reco = gen_recommendations(cust, ai_score, ai_feat, prod_props, rng)
    dump("ai_recommendation", reco)

    feedback = gen_feedback(reco, cust, rng)
    dump("rm_action_feedback", feedback)

    training = gen_training_set(cust, mart, dim_card, agg, camp_feat, campaign, rng)
    dump("ml_credit_card_training_set", training)

    # per-(customer, product) logistic-regression training set:
    #   X1..X6 (doc a.) + y_holds_product + y_adopt_next_90d (forward label)
    prop_train = scoring.build_propensity_dataset(mart, campaign)
    prop_train = add_adoption_label(prop_train, cust, campaign, rng)
    dump("ml_propensity_training_set", prop_train)

    print("\nDone. CSV -> data/csv/   Parquet -> data/parquet/")
    print("Next: py src/train_models.py --apply   (fit logistic regression, write model scores)")


if __name__ == "__main__":
    main()
