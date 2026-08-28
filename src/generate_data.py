"""
MSB SMART GROWTH ENGINE - Synthetic data generator
==================================================

Generates a coherent Customer-360 dataset for the "AI-Powered Customer
Intelligence & Sales Growth Platform" hackathon project.

Every customer is assigned a latent *archetype* that drives ALL downstream
facts, features, AI scores and recommendations, so that the 5 core business
questions are answerable on the generated data:

  1. Which customers to prioritise?          -> next_best_action_score
  2. Which product is the customer keen on?   -> ai_recommendation (NEXT_BEST_PRODUCT)
  3. Response / conversion probability?       -> propensity_to_buy_score / confidence_score
  4. Right timing & channel?                  -> recommendation timing + campaign channel
  5. What should the RM say / do next?        -> recommendation_text + rm_action_feedback

Outputs one file per table into  data/csv/*.csv  and  data/parquet/*.parquet.

Usage:
    py src/generate_data.py --customers 5000 --seed 42
"""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
AS_OF_DATE = date(2026, 8, 27)                 # "today" for the snapshot
AS_OF_TS = datetime(2026, 8, 27, 9, 0, 0)
CASA_HISTORY_DAYS = 90
TXN_HISTORY_DAYS = 90
DIGITAL_HISTORY_MONTHS = 6

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(ROOT, "data", "csv")
PARQUET_DIR = os.path.join(ROOT, "data", "parquet")

ARCHETYPES = [
    "HIGH_VALUE",          # AUM cao, dòng tiền ổn định, lifetime value cao
    "CREDIT_OPPORTUNITY",  # thu nhập tốt, ít sản phẩm tín dụng
    "POTENTIAL_INVESTOR",  # tiền nhàn rỗi, tiền gửi đáo hạn, CASA tăng mạnh
    "DIGITAL_ACTIVE",      # dùng mobile banking nhiều, tương tác số cao
    "CHURN_RISK",          # giảm số dư, giảm giao dịch, không tương tác
    "STANDARD",            # phổ thông
]
ARCHETYPE_P = [0.14, 0.17, 0.15, 0.18, 0.12, 0.24]

SEGMENTS = ["MASS", "MASS_AFFLUENT", "AFFLUENT", "PRIVATE", "SME_OWNER"]
REGIONS = [
    "Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ",
    "Bình Dương", "Đồng Nai", "Khánh Hòa", "Nghệ An", "Quảng Ninh",
]

SURNAMES = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ",
            "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý"]
MIDDLE_M = ["Văn", "Hữu", "Đức", "Công", "Minh", "Quang", "Thành", "Bá", "Xuân"]
MIDDLE_F = ["Thị", "Thùy", "Thu", "Ngọc", "Kim", "Phương", "Hải", "Mai", "Diệu"]
GIVEN_M = ["An", "Bình", "Cường", "Dũng", "Hùng", "Khoa", "Long", "Nam", "Phúc",
           "Quân", "Sơn", "Tuấn", "Việt", "Vinh", "Hoàng", "Đạt", "Huy", "Trí"]
GIVEN_F = ["Anh", "Chi", "Dung", "Hà", "Hằng", "Hoa", "Lan", "Linh", "My", "Nga",
           "Ngân", "Nhung", "Oanh", "Trang", "Uyên", "Vân", "Yến", "Thảo"]

PRODUCTS = ["CREDIT_CARD", "LOAN", "DEPOSIT", "INVESTMENT"]

# base per-product propensity by archetype (before feature adjustment + noise)
BASE_PROPENSITY = {
    "HIGH_VALUE":         {"CREDIT_CARD": 0.58, "LOAN": 0.40, "DEPOSIT": 0.55, "INVESTMENT": 0.72},
    "CREDIT_OPPORTUNITY": {"CREDIT_CARD": 0.90, "LOAN": 0.80, "DEPOSIT": 0.22, "INVESTMENT": 0.30},
    "POTENTIAL_INVESTOR": {"CREDIT_CARD": 0.33, "LOAN": 0.20, "DEPOSIT": 0.85, "INVESTMENT": 0.90},
    "DIGITAL_ACTIVE":     {"CREDIT_CARD": 0.66, "LOAN": 0.35, "DEPOSIT": 0.42, "INVESTMENT": 0.46},
    "CHURN_RISK":         {"CREDIT_CARD": 0.18, "LOAN": 0.14, "DEPOSIT": 0.24, "INVESTMENT": 0.20},
    "STANDARD":           {"CREDIT_CARD": 0.45, "LOAN": 0.34, "DEPOSIT": 0.40, "INVESTMENT": 0.34},
}

PRODUCT_LABEL_VI = {
    "CREDIT_CARD": "Thẻ tín dụng Platinum",
    "LOAN":        "Vay mua nhà / mua ô tô",
    "DEPOSIT":     "Tiền gửi có kỳ hạn ưu đãi",
    "INVESTMENT":  "Sản phẩm đầu tư / quỹ mở",
}


def _to_ts(d: date) -> pd.Timestamp:
    return pd.Timestamp(d)


# --------------------------------------------------------------------------
# 1. DIM_CUSTOMER
# --------------------------------------------------------------------------
def gen_customers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    keys = np.arange(1, n + 1)
    archetype = rng.choice(ARCHETYPES, size=n, p=ARCHETYPE_P)

    # segment correlated with archetype
    seg = np.empty(n, dtype=object)
    for i, a in enumerate(archetype):
        if a == "HIGH_VALUE":
            seg[i] = rng.choice(["AFFLUENT", "PRIVATE", "MASS_AFFLUENT"], p=[0.5, 0.3, 0.2])
        elif a == "POTENTIAL_INVESTOR":
            seg[i] = rng.choice(["MASS_AFFLUENT", "AFFLUENT", "MASS"], p=[0.45, 0.35, 0.20])
        elif a == "CREDIT_OPPORTUNITY":
            seg[i] = rng.choice(["MASS", "MASS_AFFLUENT", "SME_OWNER"], p=[0.5, 0.3, 0.2])
        elif a == "CHURN_RISK":
            seg[i] = rng.choice(["MASS", "MASS_AFFLUENT", "AFFLUENT"], p=[0.6, 0.25, 0.15])
        else:
            seg[i] = rng.choice(SEGMENTS, p=[0.45, 0.25, 0.15, 0.05, 0.10])

    gender = rng.choice(["MALE", "FEMALE"], size=n, p=[0.52, 0.48])

    # names
    full_name = np.empty(n, dtype=object)
    for i in range(n):
        s = rng.choice(SURNAMES)
        if gender[i] == "MALE":
            m, g = rng.choice(MIDDLE_M), rng.choice(GIVEN_M)
        else:
            m, g = rng.choice(MIDDLE_F), rng.choice(GIVEN_F)
        full_name[i] = f"{s} {m} {g}"

    # age 20..70
    age_years = rng.integers(20, 70, size=n)
    dob = [AS_OF_DATE - timedelta(days=int(y * 365.25) + int(rng.integers(0, 365))) for y in age_years]

    # onboarded 1..12 years ago (churn risk skews older tenure)
    tenure_days = rng.integers(120, 12 * 365, size=n)
    onboarded = [AS_OF_DATE - timedelta(days=int(t)) for t in tenure_days]

    kyc = rng.choice(["VERIFIED", "PENDING", "EXPIRED"], size=n, p=[0.9, 0.07, 0.03])

    status = np.where(
        archetype == "CHURN_RISK",
        rng.choice(["ACTIVE", "DORMANT", "ACTIVE"], size=n),
        rng.choice(["ACTIVE", "ACTIVE", "ACTIVE", "DORMANT"], size=n),
    )

    df = pd.DataFrame({
        "customer_key": keys,
        "customer_id": [f"MSB{k:08d}" for k in keys],
        "full_name": full_name,
        "date_of_birth": dob,
        "gender": gender,
        "segment_code": seg,
        "region": rng.choice(REGIONS, size=n, p=[0.24, 0.30, 0.09, 0.08, 0.05, 0.08, 0.06, 0.04, 0.03, 0.03]),
        "kyc_status": kyc,
        "onboarded_date": onboarded,
        "customer_status": status,
    })
    df.attrs["archetype"] = archetype
    return df


# --------------------------------------------------------------------------
# helpers for wealth scale per customer
# --------------------------------------------------------------------------
def wealth_base(archetype: np.ndarray, segment: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Baseline CASA balance in VND."""
    seg_mult = pd.Series(segment).map({
        "MASS": 1.0, "MASS_AFFLUENT": 2.5, "AFFLUENT": 6.0, "PRIVATE": 20.0, "SME_OWNER": 4.0,
    }).to_numpy()
    arch_mult = pd.Series(archetype).map({
        "HIGH_VALUE": 3.0, "POTENTIAL_INVESTOR": 2.2, "CREDIT_OPPORTUNITY": 0.9,
        "DIGITAL_ACTIVE": 1.1, "CHURN_RISK": 0.8, "STANDARD": 1.0,
    }).to_numpy()
    noise = rng.lognormal(mean=0.0, sigma=0.5, size=len(archetype))
    return 12_000_000 * seg_mult * arch_mult * noise


# --------------------------------------------------------------------------
# 2. FACT_CASA_DAILY
# --------------------------------------------------------------------------
def gen_casa_daily(cust: pd.DataFrame, rng: np.random.Generator):
    archetype = cust.attrs["archetype"]
    n = len(cust)
    has_casa = rng.random(n) < 0.97
    idx = np.where(has_casa)[0]
    base = wealth_base(archetype, cust["segment_code"].to_numpy(), rng)

    dates = [AS_OF_DATE - timedelta(days=int(d)) for d in range(CASA_HISTORY_DAYS - 1, -1, -1)]
    D = len(dates)

    # drift by archetype (per-day multiplicative)
    drift_map = {
        "HIGH_VALUE": 0.0006, "POTENTIAL_INVESTOR": 0.0018, "CREDIT_OPPORTUNITY": 0.0002,
        "DIGITAL_ACTIVE": 0.0004, "CHURN_RISK": -0.0025, "STANDARD": 0.0001,
    }

    rows_cust, rows_date, rows_bal = [], [], []
    for i in idx:
        a = archetype[i]
        drift = drift_map[a]
        vol = 0.02 if a != "CHURN_RISK" else 0.03
        steps = rng.normal(drift, vol, size=D)
        series = base[i] * np.cumprod(1 + steps)
        # potential investor: a maturing deposit lands mid-window -> balance spike
        if a == "POTENTIAL_INVESTOR" and rng.random() < 0.6:
            k = rng.integers(20, D - 10)
            series[k:] += base[i] * rng.uniform(0.5, 1.5)
        # occasional salary credits
        for pay_day in range(D - 1, -1, -30):
            series[pay_day:] += base[i] * rng.uniform(0.15, 0.4) * (np.arange(D)[pay_day:] == pay_day)
        series = np.round(series, 2)
        rows_cust.append(np.full(D, cust["customer_key"].iloc[i]))
        rows_date.append(np.array(dates))
        rows_bal.append(series)

    df = pd.DataFrame({
        "customer_key": np.concatenate(rows_cust),
        "account_key": None,
        "balance_date": np.concatenate(rows_date),
        "daily_balance": np.concatenate(rows_bal),
    })
    df["account_key"] = "CASA" + df["customer_key"].astype(str).str.zfill(8)
    df = df.sort_values(["customer_key", "balance_date"]).reset_index(drop=True)

    roll = df.groupby("customer_key")["daily_balance"].rolling(30, min_periods=1)
    df["average_balance_30d"] = roll.mean().round(2).reset_index(level=0, drop=True)
    df["min_balance_30d"] = roll.min().reset_index(level=0, drop=True)
    df["max_balance_30d"] = roll.max().reset_index(level=0, drop=True)
    df["overdraft_flag"] = df["daily_balance"] < 0

    df.insert(0, "fact_casa_daily_key", np.arange(1, len(df) + 1))
    df = df[["fact_casa_daily_key", "customer_key", "account_key", "balance_date",
             "daily_balance", "average_balance_30d", "min_balance_30d",
             "max_balance_30d", "overdraft_flag"]]
    return df, has_casa


# --------------------------------------------------------------------------
# 3. FACT_TRANSACTION
# --------------------------------------------------------------------------
TXN_TYPES = ["PURCHASE", "TRANSFER_OUT", "TRANSFER_IN", "ATM_WITHDRAWAL",
             "BILL_PAYMENT", "SALARY_CREDIT", "FEE", "REFUND"]
TXN_TYPE_P = [0.34, 0.20, 0.15, 0.10, 0.12, 0.04, 0.03, 0.02]
CHANNELS = ["MOBILE_APP", "INTERNET_BANKING", "ATM", "POS", "BRANCH", "QR"]
MCC = ["Groceries", "Dining", "Fuel", "Travel", "Automotive", "Electronics",
       "Healthcare", "Education", "Utilities", "Entertainment", "Fashion",
       "Transfer", "ATM", "Insurance", "Investment"]


def gen_transactions(cust: pd.DataFrame, rng: np.random.Generator):
    archetype = cust.attrs["archetype"]
    n = len(cust)
    base = wealth_base(archetype, cust["segment_code"].to_numpy(), rng)

    lam = pd.Series(archetype).map({
        "HIGH_VALUE": 55, "POTENTIAL_INVESTOR": 38, "CREDIT_OPPORTUNITY": 46,
        "DIGITAL_ACTIVE": 70, "CHURN_RISK": 12, "STANDARD": 34,
    }).to_numpy().astype(float)
    counts = rng.poisson(lam * (TXN_HISTORY_DAYS / 90.0))
    counts = np.clip(counts, 1, None)
    total = int(counts.sum())

    cust_pos = np.repeat(np.arange(n), counts)
    ckey = cust["customer_key"].to_numpy()[cust_pos]

    day_off = rng.integers(0, TXN_HISTORY_DAYS, size=total)
    sec_off = rng.integers(0, 86400, size=total)
    ts = [AS_OF_TS - timedelta(days=int(d), seconds=int(s)) for d, s in zip(day_off, sec_off)]

    ttype = rng.choice(TXN_TYPES, size=total, p=TXN_TYPE_P)
    # amount scaled to customer wealth
    amt = np.abs(rng.normal(0.03, 0.05, size=total)) * base[cust_pos] + 50_000
    amt = np.where(ttype == "SALARY_CREDIT", base[cust_pos] * rng.uniform(0.2, 0.5, total), amt)
    amt = np.where(ttype == "FEE", rng.uniform(5_000, 120_000, total), amt)
    amt = np.round(amt, 2)

    mcc = rng.choice(MCC, size=total)
    # inject automotive signal for credit-opportunity + some high-value (car-loan trigger)
    is_auto_arch = np.isin(archetype[cust_pos], ["CREDIT_OPPORTUNITY", "HIGH_VALUE"])
    auto_hit = is_auto_arch & (rng.random(total) < 0.06)
    mcc = np.where(auto_hit, "Automotive", mcc)

    chan = rng.choice(CHANNELS, size=total, p=[0.42, 0.20, 0.12, 0.18, 0.03, 0.05])
    digital_mask = np.isin(archetype[cust_pos], ["DIGITAL_ACTIVE"])
    chan = np.where(digital_mask & (rng.random(total) < 0.5), "MOBILE_APP", chan)

    fraud = rng.random(total) < 0.004

    df = pd.DataFrame({
        "fact_transaction_key": np.arange(1, total + 1),
        "customer_key": ckey,
        "transaction_id": [f"TXN{i:012d}" for i in range(1, total + 1)],
        "transaction_timestamp": ts,
        "transaction_type": ttype,
        "amount": amt,
        "currency_code": np.where(rng.random(total) < 0.03, "USD", "VND"),
        "channel": chan,
        "merchant_category": mcc,
        "is_fraud_suspected": fraud,
    }).sort_values("transaction_timestamp").reset_index(drop=True)
    df["fact_transaction_key"] = np.arange(1, len(df) + 1)
    return df


# --------------------------------------------------------------------------
# 4. FACT_CARD
# --------------------------------------------------------------------------
def gen_cards(cust: pd.DataFrame, rng: np.random.Generator):
    archetype = cust.attrs["archetype"]
    n = len(cust)
    # credit-opportunity customers mostly DON'T have a premium credit card yet
    p_has_card = pd.Series(archetype).map({
        "HIGH_VALUE": 0.9, "POTENTIAL_INVESTOR": 0.75, "CREDIT_OPPORTUNITY": 0.45,
        "DIGITAL_ACTIVE": 0.8, "CHURN_RISK": 0.6, "STANDARD": 0.65,
    }).to_numpy()

    rows = []
    cid = 1
    for i in range(n):
        if rng.random() > p_has_card[i]:
            continue
        n_cards = 1 + (rng.random() < 0.3)
        for _ in range(n_cards):
            a = archetype[i]
            is_credit = rng.random() < (0.7 if a in ("HIGH_VALUE", "CREDIT_OPPORTUNITY") else 0.45)
            ctype = rng.choice(["CREDIT_PLATINUM", "CREDIT_GOLD", "CREDIT_STANDARD"],
                               p=[0.3, 0.35, 0.35]) if is_credit else "DEBIT"
            issue = AS_OF_DATE - timedelta(days=int(rng.integers(60, 2000)))
            expiry = issue + timedelta(days=365 * 5)
            status = "ACTIVE"
            if expiry < AS_OF_DATE:
                status = "EXPIRED"
            elif rng.random() < 0.06:
                status = "BLOCKED"
            limit = 0.0
            util = 0.0
            act = True
            if is_credit:
                limit = float(rng.choice([20, 30, 50, 80, 120, 200, 300])) * 1_000_000
                if a == "CREDIT_OPPORTUNITY":
                    util = round(float(rng.uniform(5, 35)), 2)     # low utilisation -> headroom
                elif a == "CHURN_RISK":
                    util = round(float(rng.uniform(0, 10)), 2)
                else:
                    util = round(float(rng.uniform(15, 85)), 2)
                act = rng.random() < 0.9
            rows.append((cid, cust["customer_key"].iloc[i],
                         f"CARD{cid:010d}", ctype, status, issue, expiry,
                         round(limit, 2), util, act))
            cid += 1

    df = pd.DataFrame(rows, columns=[
        "fact_card_key", "customer_key", "card_id", "card_type", "card_status",
        "issue_date", "expiry_date", "credit_limit", "utilization_pct", "activation_flag"])
    return df


# --------------------------------------------------------------------------
# 5. FACT_LOAN
# --------------------------------------------------------------------------
def gen_loans(cust: pd.DataFrame, rng: np.random.Generator):
    archetype = cust.attrs["archetype"]
    n = len(cust)
    base = wealth_base(archetype, cust["segment_code"].to_numpy(), rng)
    p_has_loan = pd.Series(archetype).map({
        "HIGH_VALUE": 0.35, "POTENTIAL_INVESTOR": 0.18, "CREDIT_OPPORTUNITY": 0.30,
        "DIGITAL_ACTIVE": 0.28, "CHURN_RISK": 0.40, "STANDARD": 0.30,
    }).to_numpy()

    rows = []
    lid = 1
    for i in range(n):
        if rng.random() > p_has_loan[i]:
            continue
        a = archetype[i]
        ltype = rng.choice(["HOME", "AUTO", "PERSONAL", "BUSINESS", "CREDIT_LINE"],
                           p=[0.3, 0.2, 0.3, 0.1, 0.1])
        disb = AS_OF_DATE - timedelta(days=int(rng.integers(90, 2500)))
        tenor = int(rng.choice([12, 24, 36, 48, 60, 120, 180, 240]))
        orig = base[i] * rng.uniform(2, 9)
        paid_ratio = min(1.0, (AS_OF_DATE - disb).days / (tenor * 30.0))
        outstanding = round(max(0.0, orig * (1 - paid_ratio * rng.uniform(0.7, 1.0))), 2)
        rate = round(float(rng.uniform(6.5, 14.5)), 2)
        if a == "CHURN_RISK" and rng.random() < 0.4:
            delinq = int(rng.choice([15, 30, 60, 90], p=[0.4, 0.3, 0.2, 0.1]))
            lstatus = "DELINQUENT"
        else:
            delinq = 0
            lstatus = "ACTIVE" if outstanding > 0 else "CLOSED"
        rows.append((lid, cust["customer_key"].iloc[i], f"LOAN{lid:010d}", ltype,
                     disb, outstanding, rate, tenor, lstatus, delinq))
        lid += 1

    df = pd.DataFrame(rows, columns=[
        "fact_loan_key", "customer_key", "loan_account_id", "loan_type",
        "disbursement_date", "outstanding_principal", "interest_rate",
        "tenor_months", "loan_status", "delinquency_days"])
    return df


# --------------------------------------------------------------------------
# 6. FACT_DEPOSIT
# --------------------------------------------------------------------------
def gen_deposits(cust: pd.DataFrame, rng: np.random.Generator):
    archetype = cust.attrs["archetype"]
    n = len(cust)
    base = wealth_base(archetype, cust["segment_code"].to_numpy(), rng)
    p_has_dep = pd.Series(archetype).map({
        "HIGH_VALUE": 0.65, "POTENTIAL_INVESTOR": 0.85, "CREDIT_OPPORTUNITY": 0.15,
        "DIGITAL_ACTIVE": 0.30, "CHURN_RISK": 0.25, "STANDARD": 0.35,
    }).to_numpy()

    rows = []
    did = 1
    for i in range(n):
        if rng.random() > p_has_dep[i]:
            continue
        a = archetype[i]
        n_dep = 1 + (rng.random() < (0.5 if a == "POTENTIAL_INVESTOR" else 0.15))
        for _ in range(n_dep):
            term = int(rng.choice([1, 3, 6, 9, 12, 18, 24]))
            dep_date = AS_OF_DATE - timedelta(days=int(rng.integers(10, term * 30 + 40)))
            maturity = dep_date + timedelta(days=term * 30)
            amount = round(base[i] * rng.uniform(0.8, 5), 2)
            if maturity < AS_OF_DATE:
                status = rng.choice(["MATURED", "ROLLED_OVER", "WITHDRAWN"], p=[0.3, 0.5, 0.2])
            elif (maturity - AS_OF_DATE).days <= 30:
                status = "MATURING_SOON"
            else:
                status = "ACTIVE"
            # potential investors: bias toward "maturing soon" (a demand signal)
            if a == "POTENTIAL_INVESTOR" and rng.random() < 0.5:
                maturity = AS_OF_DATE + timedelta(days=int(rng.integers(3, 28)))
                dep_date = maturity - timedelta(days=term * 30)
                status = "MATURING_SOON"
            rate = round(float(rng.uniform(3.5, 6.2)), 2)
            rows.append((did, cust["customer_key"].iloc[i], f"DEP{did:010d}",
                         dep_date, amount, term, rate, maturity, status))
            did += 1

    df = pd.DataFrame(rows, columns=[
        "fact_deposit_key", "customer_key", "deposit_account_id", "deposit_date",
        "deposit_amount", "term_months", "interest_rate", "maturity_date", "deposit_status"])
    return df


# --------------------------------------------------------------------------
# 7. FACT_INSURANCE
# --------------------------------------------------------------------------
def gen_insurance(cust: pd.DataFrame, rng: np.random.Generator):
    archetype = cust.attrs["archetype"]
    n = len(cust)
    base = wealth_base(archetype, cust["segment_code"].to_numpy(), rng)
    p_has = pd.Series(archetype).map({
        "HIGH_VALUE": 0.4, "POTENTIAL_INVESTOR": 0.3, "CREDIT_OPPORTUNITY": 0.15,
        "DIGITAL_ACTIVE": 0.2, "CHURN_RISK": 0.12, "STANDARD": 0.18,
    }).to_numpy()
    rows = []
    pid = 1
    for i in range(n):
        if rng.random() > p_has[i]:
            continue
        ptype = rng.choice(["LIFE", "HEALTH", "MOTOR", "HOME", "TRAVEL"],
                           p=[0.35, 0.3, 0.15, 0.1, 0.1])
        start = AS_OF_DATE - timedelta(days=int(rng.integers(30, 1500)))
        end = start + timedelta(days=365 * int(rng.choice([1, 5, 10, 20])))
        prem = round(base[i] * rng.uniform(0.01, 0.06), 2)
        status = "ACTIVE" if end > AS_OF_DATE else "LAPSED"
        rows.append((pid, cust["customer_key"].iloc[i], f"POL{pid:010d}", ptype,
                     prem, start, end, status, int(rng.poisson(0.3))))
        pid += 1
    return pd.DataFrame(rows, columns=[
        "fact_insurance_key", "customer_key", "policy_id", "policy_type",
        "premium_amount", "start_date", "end_date", "policy_status", "claim_count"])


# --------------------------------------------------------------------------
# 8. FACT_DIGITAL_ACTIVITY  (monthly aggregates)
# --------------------------------------------------------------------------
def gen_digital(cust: pd.DataFrame, rng: np.random.Generator):
    archetype = cust.attrs["archetype"]
    n = len(cust)
    intensity = pd.Series(archetype).map({
        "HIGH_VALUE": 1.0, "POTENTIAL_INVESTOR": 0.8, "CREDIT_OPPORTUNITY": 0.9,
        "DIGITAL_ACTIVE": 2.2, "CHURN_RISK": 0.35, "STANDARD": 0.7,
    }).to_numpy()

    months = []
    for m in range(DIGITAL_HISTORY_MONTHS - 1, -1, -1):
        first = (AS_OF_DATE.replace(day=1) - timedelta(days=1))
        for _ in range(m):
            first = (first.replace(day=1) - timedelta(days=1))
        months.append(first.replace(day=1))

    rows = []
    aid = 1
    for i in range(n):
        if rng.random() > min(0.98, 0.5 + intensity[i] / 3):
            continue
        base_login = 18 * intensity[i]
        for mi, mdate in enumerate(months):
            # churn risk: declining engagement over time
            decay = 1.0
            if archetype[i] == "CHURN_RISK":
                decay = max(0.15, 1.0 - 0.16 * (DIGITAL_HISTORY_MONTHS - 1 - mi))
            login = max(0, int(rng.normal(base_login * decay, 4)))
            if login == 0 and rng.random() < 0.6:
                continue
            sess = login + int(rng.normal(login * 0.4, 3))
            appo = int(login * rng.uniform(1.1, 1.8))
            feat = int(login * rng.uniform(0.3, 1.2))
            ts = datetime(mdate.year, mdate.month, min(28, int(rng.integers(1, 28))),
                          int(rng.integers(6, 23)), int(rng.integers(0, 60)))
            chan = rng.choice(["MOBILE_APP", "INTERNET_BANKING"], p=[0.75, 0.25])
            rows.append((aid, cust["customer_key"].iloc[i], f"DActivity{aid:011d}",
                         ts, chan, max(0, sess), login, max(0, appo), max(0, feat)))
            aid += 1

    return pd.DataFrame(rows, columns=[
        "fact_digital_activity_key", "customer_key", "activity_id", "activity_date",
        "channel", "session_count", "login_count", "app_open_count", "feature_usage_count"])


# --------------------------------------------------------------------------
# 9. FACT_CRM_INTERACTION
# --------------------------------------------------------------------------
def gen_crm(cust: pd.DataFrame, rng: np.random.Generator):
    archetype = cust.attrs["archetype"]
    n = len(cust)
    lam = pd.Series(archetype).map({
        "HIGH_VALUE": 2.5, "POTENTIAL_INVESTOR": 1.6, "CREDIT_OPPORTUNITY": 1.4,
        "DIGITAL_ACTIVE": 1.2, "CHURN_RISK": 2.8, "STANDARD": 1.0,
    }).to_numpy()
    counts = rng.poisson(lam)
    total = int(counts.sum())
    cust_pos = np.repeat(np.arange(n), counts)
    ckey = cust["customer_key"].to_numpy()[cust_pos]

    day_off = rng.integers(0, 365, size=total)
    idt = [AS_OF_TS - timedelta(days=int(d), seconds=int(rng.integers(0, 86400))) for d in day_off]
    case_type = rng.choice(
        ["COMPLAINT", "PRODUCT_INQUIRY", "RETENTION", "ADVISORY", "ONBOARDING", "DISPUTE"],
        size=total, p=[0.2, 0.28, 0.12, 0.2, 0.1, 0.1])
    case_type = np.where(
        (archetype[cust_pos] == "CHURN_RISK") & (rng.random(total) < 0.5), "RETENTION", case_type)
    status = rng.choice(["OPEN", "IN_PROGRESS", "RESOLVED", "ESCALATED"],
                        size=total, p=[0.1, 0.15, 0.68, 0.07])
    prio = rng.choice(["LOW", "MEDIUM", "HIGH", "URGENT"], size=total, p=[0.3, 0.42, 0.22, 0.06])
    res_h = np.where(status == "RESOLVED",
                     np.round(rng.gamma(2.0, 6.0, total), 2), np.nan)

    return pd.DataFrame({
        "fact_crm_interaction_key": np.arange(1, total + 1),
        "customer_key": ckey,
        "interaction_id": [f"CRM{i:011d}" for i in range(1, total + 1)],
        "interaction_date": idt,
        "contact_channel": rng.choice(["PHONE", "EMAIL", "BRANCH", "CHAT", "RM_VISIT"],
                                      size=total, p=[0.4, 0.25, 0.15, 0.15, 0.05]),
        "case_type": case_type,
        "case_status": status,
        "priority_level": prio,
        "resolution_time_hours": res_h,
    })


# --------------------------------------------------------------------------
# 10. FACT_CUSTOMER_SERVICE
# --------------------------------------------------------------------------
def gen_service(cust: pd.DataFrame, rng: np.random.Generator):
    archetype = cust.attrs["archetype"]
    n = len(cust)
    lam = pd.Series(archetype).map({
        "HIGH_VALUE": 1.2, "POTENTIAL_INVESTOR": 0.9, "CREDIT_OPPORTUNITY": 1.0,
        "DIGITAL_ACTIVE": 1.4, "CHURN_RISK": 2.0, "STANDARD": 0.8,
    }).to_numpy()
    counts = rng.poisson(lam)
    total = int(counts.sum())
    cust_pos = np.repeat(np.arange(n), counts)
    ckey = cust["customer_key"].to_numpy()[cust_pos]

    day_off = rng.integers(0, 180, size=total)
    rdt = [AS_OF_TS - timedelta(days=int(d), seconds=int(rng.integers(0, 86400))) for d in day_off]
    stype = rng.choice(
        ["CARD_REISSUE", "PIN_RESET", "STATEMENT_REQUEST", "LIMIT_CHANGE",
         "ADDRESS_UPDATE", "DISPUTE", "APP_SUPPORT", "ACCOUNT_CLOSURE"],
        size=total, p=[0.15, 0.15, 0.15, 0.12, 0.1, 0.12, 0.16, 0.05])
    stype = np.where((archetype[cust_pos] == "CHURN_RISK") & (rng.random(total) < 0.15),
                     "ACCOUNT_CLOSURE", stype)
    handling = rng.integers(3, 90, size=total)
    sla = handling <= 45

    return pd.DataFrame({
        "fact_customer_service_key": np.arange(1, total + 1),
        "customer_key": ckey,
        "service_request_id": [f"SR{i:011d}" for i in range(1, total + 1)],
        "request_date": rdt,
        "service_type": stype,
        "service_channel": rng.choice(["HOTLINE", "APP", "BRANCH", "EMAIL", "CHATBOT"],
                                      size=total, p=[0.35, 0.28, 0.15, 0.12, 0.1]),
        "request_status": rng.choice(["NEW", "PROCESSING", "COMPLETED", "REJECTED"],
                                     size=total, p=[0.06, 0.12, 0.78, 0.04]),
        "sla_met_flag": sla,
        "handling_time_minutes": handling.astype(int),
    })


# --------------------------------------------------------------------------
# 11. FACT_CAMPAIGN
# --------------------------------------------------------------------------
CAMPAIGNS = [
    ("CC_PLATINUM_Q2", "Ưu đãi mở thẻ Platinum hoàn tiền 6%", "CREDIT_CARD"),
    ("HOME_LOAN_SUMMER", "Vay mua nhà lãi suất 6.5% cố định 24 tháng", "LOAN"),
    ("AUTO_LOAN_2026", "Vay mua ô tô duyệt nhanh 8 giờ", "LOAN"),
    ("DEPOSIT_BOOST", "Tiền gửi online +0.5% lãi suất", "DEPOSIT"),
    ("WEALTH_INVEST", "Trải nghiệm quỹ mở miễn phí phí quản lý", "INVESTMENT"),
    ("DIGITAL_ONB", "Kích hoạt Mobile Banking nhận 100k", "DIGITAL"),
    ("PAYROLL_UP", "Ưu đãi tài khoản nhận lương", "CASA"),
    ("RETAIN_VIP", "Chương trình tri ân khách hàng thân thiết", "RETENTION"),
]


def gen_campaign(cust: pd.DataFrame, rng: np.random.Generator):
    archetype = cust.attrs["archetype"]
    n = len(cust)
    rows = []
    fk = 1
    prod_pref = {  # which campaign product each archetype tends to be targeted with
        "HIGH_VALUE": ["INVESTMENT", "CREDIT_CARD", "RETENTION"],
        "CREDIT_OPPORTUNITY": ["CREDIT_CARD", "LOAN", "LOAN"],
        "POTENTIAL_INVESTOR": ["DEPOSIT", "INVESTMENT", "INVESTMENT"],
        "DIGITAL_ACTIVE": ["DIGITAL", "CREDIT_CARD", "CASA"],
        "CHURN_RISK": ["RETENTION", "CASA", "DEPOSIT"],
        "STANDARD": ["CASA", "CREDIT_CARD", "DEPOSIT"],
    }
    for i in range(n):
        a = archetype[i]
        n_c = rng.integers(1, 5)
        for _ in range(n_c):
            want = rng.choice(prod_pref[a])
            cands = [c for c in CAMPAIGNS if c[2] == want] or CAMPAIGNS
            cid, cname, cprod = cands[int(rng.integers(0, len(cands)))]
            start = AS_OF_DATE - timedelta(days=int(rng.integers(15, 300)))
            end = start + timedelta(days=int(rng.choice([14, 30, 45, 60])))
            chan = rng.choice(["EMAIL", "SMS", "APP_PUSH", "RM_OUTBOUND", "CALL_CENTER"],
                              p=[0.3, 0.25, 0.25, 0.12, 0.08])
            # response probability tied to archetype/product match
            p_resp = 0.12 + (BASE_PROPENSITY[a].get(cprod, 0.3) - 0.3) * 0.6
            p_resp = float(np.clip(p_resp + rng.normal(0, 0.05), 0.03, 0.85))
            responded = rng.random() < p_resp
            converted = responded and (rng.random() < 0.35)
            rdate = None
            if responded:
                rdate = start + timedelta(days=int(rng.integers(1, max(2, (end - start).days))))
            rows.append((fk, cust["customer_key"].iloc[i], cid, cname, start, end,
                         chan, responded, converted, rdate))
            fk += 1
    return pd.DataFrame(rows, columns=[
        "fact_campaign_key", "customer_key", "campaign_id", "campaign_name",
        "campaign_start_date", "campaign_end_date", "campaign_channel",
        "response_flag", "conversion_flag", "response_date"])


# --------------------------------------------------------------------------
# 12. CUSTOMER_360_FEATURE_MART
# --------------------------------------------------------------------------
def gen_feature_mart(cust, casa, txn, card, loan, deposit, digital, crm, service, campaign, rng):
    n = len(cust)
    ck = cust["customer_key"].to_numpy()

    # latest CASA balance
    last_casa = (casa.sort_values("balance_date")
                 .groupby("customer_key")["daily_balance"].last())

    # transactions last 30d
    t30 = txn[txn["transaction_timestamp"] >= _to_ts(AS_OF_DATE - timedelta(days=30))]
    txn_cnt_30 = t30.groupby("customer_key").size()
    card_spend_30 = (t30[t30["transaction_type"] == "PURCHASE"]
                     .groupby("customer_key")["amount"].sum())

    dep_active = deposit[deposit["deposit_status"].isin(["ACTIVE", "MATURING_SOON", "ROLLED_OVER"])]
    dep_val = dep_active.groupby("customer_key")["deposit_amount"].sum()

    loan_active = loan[loan["loan_status"].isin(["ACTIVE", "DELINQUENT"])]
    loan_out = loan_active.groupby("customer_key")["outstanding_principal"].sum()

    # digital engagement: latest month logins normalised
    dig_last = (digital.sort_values("activity_date")
                .groupby("customer_key")
                .agg(login=("login_count", "last"), feat=("feature_usage_count", "last")))
    dig_score = (dig_last["login"] * 2 + dig_last["feat"]).clip(0, 100)

    contact_90 = pd.concat([
        crm[crm["interaction_date"] >= _to_ts(AS_OF_DATE - timedelta(days=90))][["customer_key"]],
        service[service["request_date"] >= _to_ts(AS_OF_DATE - timedelta(days=90))][["customer_key"]],
    ]).groupby("customer_key").size()

    camp_stats = campaign.groupby("customer_key").agg(
        sent=("fact_campaign_key", "count"),
        resp=("response_flag", "sum"))
    camp_rate = (camp_stats["resp"] / camp_stats["sent"] * 100).round(2)

    # most recent campaign per customer -> campaign_key FK
    last_campaign = (campaign.sort_values("campaign_start_date")
                     .groupby("customer_key")["fact_campaign_key"].last())

    df = pd.DataFrame({"customer_key": ck})
    df["customer_360_feature_key"] = np.arange(1, n + 1)
    df["campaign_key"] = df["customer_key"].map(last_campaign)
    df["snapshot_date"] = AS_OF_DATE
    df["total_balance"] = df["customer_key"].map(last_casa).fillna(0.0)
    df["total_deposit_value"] = df["customer_key"].map(dep_val).fillna(0.0)
    df["total_balance"] = (df["total_balance"] + df["total_deposit_value"]).round(2)
    df["total_transactions_30d"] = df["customer_key"].map(txn_cnt_30).fillna(0).astype(int)
    df["total_card_spend_30d"] = df["customer_key"].map(card_spend_30).fillna(0.0).round(2)
    df["total_loan_outstanding"] = df["customer_key"].map(loan_out).fillna(0.0).round(2)
    df["digital_engagement_score"] = df["customer_key"].map(dig_score).fillna(0.0).round(2)
    df["service_contact_count_90d"] = df["customer_key"].map(contact_90).fillna(0).astype(int)
    df["campaign_response_rate"] = df["customer_key"].map(camp_rate).fillna(0.0)

    df["campaign_key"] = df["campaign_key"].astype("Int64")
    df = df[["customer_360_feature_key", "customer_key", "campaign_key", "snapshot_date",
             "total_balance", "total_transactions_30d", "total_card_spend_30d",
             "total_deposit_value", "total_loan_outstanding", "digital_engagement_score",
             "service_contact_count_90d", "campaign_response_rate"]]
    return df


# --------------------------------------------------------------------------
# 13. AI_CUSTOMER_SCORE
# --------------------------------------------------------------------------
def _norm(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanpercentile(x, 2), np.nanpercentile(x, 98)
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def gen_scores(cust, mart, loan, card, rng):
    archetype = cust.attrs["archetype"]
    n = len(cust)
    m = mart.set_index("customer_key")

    delinq = loan.groupby("customer_key")["delinquency_days"].max()
    util = card[card["card_type"].str.startswith("CREDIT")].groupby("customer_key")["utilization_pct"].mean()

    ck = cust["customer_key"].to_numpy()
    bal = m.loc[ck, "total_balance"].to_numpy()
    txn30 = m.loc[ck, "total_transactions_30d"].to_numpy()
    dig = m.loc[ck, "digital_engagement_score"].to_numpy()
    contacts = m.loc[ck, "service_contact_count_90d"].to_numpy()
    camp_rate = m.loc[ck, "campaign_response_rate"].to_numpy()
    loan_out = m.loc[ck, "total_loan_outstanding"].to_numpy()
    d = pd.Series(ck).map(delinq).fillna(0).to_numpy()
    u = pd.Series(ck).map(util).fillna(0).to_numpy()

    arch = np.asarray(archetype)

    # churn: high when balance/txn/digital low, contacts high, or archetype churn
    churn = (0.45 * (arch == "CHURN_RISK")
             + 0.20 * (1 - _norm(txn30))
             + 0.15 * (1 - _norm(dig))
             + 0.12 * _norm(contacts)
             + 0.08 * (1 - _norm(bal)))
    churn = np.clip(churn + rng.normal(0, 0.05, n), 0.01, 0.99)

    # propensity to buy (any product): high for credit-opp / investor / digital
    prop = (0.30 * np.isin(arch, ["CREDIT_OPPORTUNITY", "POTENTIAL_INVESTOR"])
            + 0.15 * (arch == "DIGITAL_ACTIVE")
            + 0.20 * _norm(dig)
            + 0.15 * _norm(camp_rate)
            + 0.20 * _norm(bal))
    prop = np.clip(prop + rng.normal(0, 0.05, n), 0.01, 0.99)

    # credit risk: delinquency + utilisation + leverage vs balance
    lev = np.where(bal > 0, loan_out / (bal + 1), loan_out / 1e7)
    crisk = (0.40 * _norm(d)
             + 0.25 * _norm(u)
             + 0.20 * _norm(lev)
             + 0.15 * (arch == "CHURN_RISK"))
    crisk = np.clip(crisk + rng.normal(0, 0.04, n), 0.01, 0.99)

    # next best action score = prioritisation: worth an RM touch now
    nba = np.clip(0.55 * prop + 0.30 * (1 - crisk) + 0.25 * churn * (bal > np.median(bal))
                  + rng.normal(0, 0.04, n), 0.01, 0.99)

    df = pd.DataFrame({
        "ai_customer_score_key": np.arange(1, n + 1),
        "customer_360_feature_key": mart["customer_360_feature_key"].to_numpy(),
        "score_date": AS_OF_DATE,
        "churn_score": np.round(churn, 4),
        "propensity_to_buy_score": np.round(prop, 4),
        "credit_risk_score": np.round(crisk, 4),
        "next_best_action_score": np.round(nba, 4),
    })
    df.attrs["propensity_vec"] = prop
    df.attrs["credit_risk_vec"] = crisk
    return df


# --------------------------------------------------------------------------
# 14. AI_RECOMMENDATION  (Next Best Product, ranked top-3)
# --------------------------------------------------------------------------
def gen_recommendations(cust, mart, score, card, deposit, txn, rng):
    archetype = cust.attrs["archetype"]
    n = len(cust)
    ck = cust["customer_key"].to_numpy()
    m = mart.set_index("customer_key")

    has_credit_card = set(card[card["card_type"].str.startswith("CREDIT")]["customer_key"])
    dep_maturing = set(deposit[deposit["deposit_status"] == "MATURING_SOON"]["customer_key"])
    auto_txn = set(txn[txn["merchant_category"] == "Automotive"]["customer_key"])

    rows = []
    rid = 1
    for i in range(n):
        c = ck[i]
        a = archetype[i]
        base = BASE_PROPENSITY[a]
        feats = {
            "bal": m.at[c, "total_balance"] if c in m.index else 0,
            "spend30": m.at[c, "total_card_spend_30d"] if c in m.index else 0,
            "dig": m.at[c, "digital_engagement_score"] if c in m.index else 0,
        }
        prop = {}
        for p in PRODUCTS:
            v = base[p]
            if p == "CREDIT_CARD" and c not in has_credit_card:
                v += 0.15
            if p == "CREDIT_CARD" and feats["spend30"] > 8_000_000:
                v += 0.10
            if p == "DEPOSIT" and c in dep_maturing:
                v += 0.18
            if p == "INVESTMENT" and feats["bal"] > 300_000_000:
                v += 0.12
            if p == "LOAN" and c in auto_txn:
                v += 0.15
            v = float(np.clip(v + rng.normal(0, 0.06), 0.01, 0.99))
            prop[p] = v

        top = sorted(prop, key=prop.get, reverse=True)[:3]
        sdate = AS_OF_DATE
        for rank, p in enumerate(top, start=1):
            conf = round(prop[p], 4)
            timing = "Trong vòng 48 giờ" if rank == 1 and conf > 0.6 else \
                     "Trong tuần này" if rank == 1 else "Trong 2 tuần tới"
            text = _reco_text(p, a, feats, c, has_credit_card, dep_maturing, auto_txn, timing)
            rows.append((rid, score["ai_customer_score_key"].iloc[i], sdate,
                         "NEXT_BEST_PRODUCT", text, conf, rank))
            rid += 1

    return pd.DataFrame(rows, columns=[
        "ai_recommendation_key", "ai_customer_score_key", "recommendation_date",
        "recommendation_type", "recommendation_text", "confidence_score", "priority_rank"])


def _reco_text(product, arch, feats, ckey, has_cc, dep_mat, auto, timing):
    label = PRODUCT_LABEL_VI[product]
    reasons = []
    if product == "CREDIT_CARD":
        if ckey not in has_cc:
            reasons.append("chưa sở hữu thẻ tín dụng cao cấp")
        if feats["spend30"] > 5_000_000:
            reasons.append("chi tiêu hàng tháng cao và ổn định")
        reasons.append("thu nhập ổn định, khả năng trả nợ tốt")
        msg = "Cashback, ưu đãi du lịch và tích điểm chi tiêu."
    elif product == "LOAN":
        if ckey in auto:
            reasons.append("có giao dịch liên quan đến ô tô gần đây")
        reasons.append("dòng tiền vào đều, tỷ lệ đòn bẩy còn thấp")
        msg = "Lãi suất cố định ưu đãi, duyệt nhanh, giải ngân trong ngày."
    elif product == "DEPOSIT":
        if ckey in dep_mat:
            reasons.append("có khoản tiền gửi sắp đáo hạn")
        reasons.append("số dư CASA nhàn rỗi tăng")
        msg = "Cộng thêm lãi suất khi gửi online, kỳ hạn linh hoạt."
    else:  # INVESTMENT
        reasons.append("tiền nhàn rỗi lớn, ít sử dụng sản phẩm đầu tư")
        if feats["bal"] > 200_000_000:
            reasons.append("AUM cao phù hợp phân bổ danh mục")
        msg = "Miễn phí phí quản lý quỹ trong 6 tháng đầu."
    why = "; ".join(dict.fromkeys(reasons))
    return f"Đề xuất {label}. Lý do: {why}. Thông điệp: {msg} Thời điểm tiếp cận: {timing}."


# --------------------------------------------------------------------------
# 15. RM_ACTION_FEEDBACK
# --------------------------------------------------------------------------
def gen_feedback(reco, rng):
    # RM acts mostly on rank-1 (and some rank-2) recommendations
    mask = ((reco["priority_rank"] == 1) & (rng.random(len(reco)) < 0.42)) | \
           ((reco["priority_rank"] == 2) & (rng.random(len(reco)) < 0.12))
    sub = reco[mask].reset_index(drop=True)
    k = len(sub)

    actions = ["RM call", "Gửi email tư vấn", "Đặt lịch hẹn tại chi nhánh",
               "SMS follow-up", "Gọi lại theo yêu cầu KH", "Chưa liên hệ"]
    outcomes = ["Interested", "Converted", "Not interested", "No response", "Callback requested"]

    conf = sub["confidence_score"].to_numpy()
    p_convert = np.clip(conf * 0.5, 0.05, 0.5)
    r = rng.random(k)
    outcome = np.where(r < p_convert, "Converted",
              np.where(r < p_convert + 0.25, "Interested",
              np.where(r < p_convert + 0.45, "Callback requested",
              np.where(r < p_convert + 0.70, "No response", "Not interested"))))

    action = np.where(outcome == "Chưa liên hệ", "Chưa liên hệ",
                      rng.choice(actions[:-1], size=k))
    rating = np.clip(np.round(conf * 5 + rng.normal(0, 0.8, k)).astype(int), 1, 5)
    follow = np.isin(outcome, ["Interested", "Callback requested"]) | (rng.random(k) < 0.2)

    fb_day = rng.integers(0, 40, size=k)
    fdate = [AS_OF_DATE - timedelta(days=int(x)) for x in fb_day]

    note_map = {
        "Converted": "KH đồng ý mở sản phẩm, đã hoàn tất hồ sơ.",
        "Interested": "KH quan tâm, cần gửi thêm bảng minh hoạ lãi suất/phí.",
        "Callback requested": "KH bận, hẹn gọi lại vào cuối tuần.",
        "No response": "Gọi 2 lần không liên lạc được, sẽ thử kênh email.",
        "Not interested": "KH cho biết chưa có nhu cầu ở thời điểm hiện tại.",
    }
    notes = [note_map[o] for o in outcome]

    return pd.DataFrame({
        "rm_action_feedback_key": np.arange(1, k + 1),
        "ai_recommendation_key": sub["ai_recommendation_key"].to_numpy(),
        "feedback_date": fdate,
        "action_taken": action,
        "action_outcome": outcome,
        "follow_up_required": follow,
        "rating": rating,
        "notes": notes,
    })


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------
def dump(name: str, df: pd.DataFrame):
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(PARQUET_DIR, exist_ok=True)
    csv_path = os.path.join(CSV_DIR, f"{name}.csv")
    pq_path = os.path.join(PARQUET_DIR, f"{name}.parquet")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    try:
        df_pq = df
        if df.attrs:                       # attrs carry internal ndarrays, drop them
            df_pq = df.copy()
            df_pq.attrs = {}
        df_pq.to_parquet(pq_path, index=False)
    except Exception as e:  # pragma: no cover
        print(f"  ! parquet skipped for {name}: {e}")
    print(f"  {name:<28} {len(df):>9,} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"MSB Smart Growth Engine - generating {args.customers:,} customers (seed={args.seed})")

    cust = gen_customers(args.customers, rng)
    dump("dim_customer", cust)

    casa, _ = gen_casa_daily(cust, rng)
    dump("fact_casa_daily", casa)

    txn = gen_transactions(cust, rng)
    dump("fact_transaction", txn)

    card = gen_cards(cust, rng)
    dump("fact_card", card)

    loan = gen_loans(cust, rng)
    dump("fact_loan", loan)

    deposit = gen_deposits(cust, rng)
    dump("fact_deposit", deposit)

    insurance = gen_insurance(cust, rng)
    dump("fact_insurance", insurance)

    digital = gen_digital(cust, rng)
    dump("fact_digital_activity", digital)

    crm = gen_crm(cust, rng)
    dump("fact_crm_interaction", crm)

    service = gen_service(cust, rng)
    dump("fact_customer_service", service)

    campaign = gen_campaign(cust, rng)
    dump("fact_campaign", campaign)

    mart = gen_feature_mart(cust, casa, txn, card, loan, deposit, digital, crm, service, campaign, rng)
    dump("customer_360_feature_mart", mart)

    score = gen_scores(cust, mart, loan, card, rng)
    dump("ai_customer_score", score)

    reco = gen_recommendations(cust, mart, score, card, deposit, txn, rng)
    dump("ai_recommendation", reco)

    feedback = gen_feedback(reco, rng)
    dump("rm_action_feedback", feedback)

    print("\nDone. CSV -> data/csv/   Parquet -> data/parquet/")


if __name__ == "__main__":
    main()
