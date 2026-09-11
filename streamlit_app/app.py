"""
MSB SMART GROWTH ENGINE — Streamlit demo
=======================================
Trực quan hoá công cụ "AI-Powered Customer Intelligence & Sales Growth Platform"
theo đúng "2. Thiết kế hành trình AI" (12 Actions) trong tài liệu MSB V.01.

Chạy:  streamlit run streamlit_app/app.py
Dữ liệu đọc từ  data/parquet/  (sinh bằng src/generate_data.py + src/train_models.py --apply).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# stlite/Pyodide: pyarrow (streamlit dùng để serialize st.dataframe) không hỗ trợ tốt
# cột 'category' -> "'str' object cannot be interpreted as an integer". Trên WASM giữ
# cột chuỗi ở dạng object (RAM trình duyệt dư sức); chỉ nén category khi chạy native.
# sys.platform không đáng tin cậy trên mọi bản Pyodide/stlite -> kiểm thêm module 'pyodide'
# (luôn có sẵn trong runtime Pyodide, kể cả khi app không tự import).
IS_WASM = (sys.platform == "emscripten" or "pyodide" in sys.modules
           or os.environ.get("FORCE_WASM") == "1")

# Lớp bảo vệ độc lập với IS_WASM (phòng khi detect môi trường sai): chặn tận gốc ở
# st.dataframe — luôn ép cột 'category' về string trước khi streamlit serialize bằng
# Arrow, vì đây là nguồn lỗi thật sự ("'str' object cannot be interpreted as an integer").
_orig_st_dataframe = st.dataframe


def _safe_dataframe(data=None, *args, **kwargs):
    if isinstance(data, pd.DataFrame):
        cat_cols = data.select_dtypes(include="category").columns
        if len(cat_cols):
            data = data.copy()
            for c in cat_cols:
                data[c] = data[c].astype(str)
    return _orig_st_dataframe(data, *args, **kwargs)


st.dataframe = _safe_dataframe

# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data" / "parquet"
MODELS = ROOT / "models"

# Số khách hàng tối đa nạp vào app. Mặc định 25000 = full (hợp Hugging Face Spaces,
# local, VPS). Trên Streamlit Community Cloud (~1GB RAM) đặt env APP_MAX_CUST=10000.
APP_MAX_CUST = int(os.environ.get("APP_MAX_CUST", "25000"))

MSB_RED = "#E4002B"
MSB_INK = "#1d2733"
PALETTE = ["#E4002B", "#F4A300", "#2E8B8B", "#3E7CB1", "#8E44AD", "#5B6770"]

TARGET_PRODUCTS = {"CREDIT_CARD": "P001", "LOAN": "P004", "DEPOSIT": "P007", "INVESTMENT": "P009"}
GROUP_LABEL = {"CREDIT_CARD": "Thẻ tín dụng", "LOAN": "Cho vay",
               "DEPOSIT": "Tiền gửi", "INVESTMENT": "Đầu tư"}
PRIORITY_ORDER = ["Very High", "High", "Medium", "Low", "Do not prioritize"]
PRIORITY_COLOR = {"Very High": "#E4002B", "High": "#F4A300", "Medium": "#3E7CB1",
                  "Low": "#5B6770", "Do not prioritize": "#c9ced6"}

FEATURE_LABEL = {
    "x1_monthly_spending": "X1 · Chi tiêu/tháng",
    "x2_income": "X2 · Thu nhập",
    "x3_digital_activity": "X3 · Hoạt động Digital",
    "x4_salary_account": "X4 · Nhận lương qua MSB",
    "x5_campaign_response": "X5 · Phản hồi campaign",
    "x6_product_gap": "X6 · Product gap / sẵn sàng",
    "intercept": "b · Hệ số chặn",
}

ACTIONS = [
    ("1", "Nhận tín hiệu khách hàng",
     "Smart Growth Engine lấy snapshot Customer 360 mới nhất + hành vi gần đây (CASA, Transaction, Card, Digital, CRM, Campaign)."),
    ("2", "AI phát hiện nhu cầu / opportunity",
     "Model tính propensity, customer value, intent signal, engagement, timing → Smart Growth Score 0–100."),
    ("3", "Decision Gate 1 — Có nên tiếp cận không?",
     "Eligible / Suppress / Exclude: KYC + đang hoạt động, chưa sở hữu sản phẩm, có consent, không DNC, không complaint nghiêm trọng 15 ngày, không vừa từ chối 30 ngày, chưa vượt tần suất."),
    ("4", "Next Best Product",
     "AI chọn sản phẩm relevance cao nhất trong nhóm khách đủ điều kiện — xếp hạng top 3."),
    ("5", "Next Best Action",
     "High Value → RM Call; Digital Active → In-app / Push; Low priority → Nurture hoặc chưa tiếp cận."),
    ("6", "Chọn kênh và thời điểm",
     "Channel + contact window dựa trên preference, engagement, policy."),
    ("7", "GenAI tạo nội dung cá nhân hoá",
     "LLM nhận structured context + template đã kiểm soát → message angle / CTA phù hợp."),
    ("8", "Decision Gate 2 — Kiểm soát trước khi gửi",
     "Consent check, Do-not-contact, Frequency cap, Product eligibility, Compliance template, Masking."),
    ("9", "Decision Gate 3 — Auto-send hay RM approval?",
     "Auto-send (low-risk) / RM Approval (high-value, cần tư vấn) / Block (không qua business/compliance gate)."),
    ("10", "Message Orchestration",
     "Message chuyển tới channel gateway tương ứng + ghi log đầy đủ."),
    ("11", "Khách hàng phản hồi",
     "Ghi nhận Delivered → Opened → Clicked → Interested → Applied → Converted / Opt-out."),
    ("12", "Learning & Next Decision",
     "Outcome thực tế quay lại Customer 360 + model để cải thiện lần ra quyết định tiếp theo."),
]


# --------------------------------------------------------------------------
st.set_page_config(page_title="MSB Smart Growth Engine", page_icon="📈", layout="wide")

st.markdown(f"""
<style>
 .stApp header {{ background: transparent; }}
 h1, h2, h3 {{ color: {MSB_INK}; }}
 div[data-testid="stMetricValue"] {{ color: {MSB_INK}; }}
 .msb-badge {{ display:inline-block; padding:2px 10px; border-radius:12px;
   background:#fde8ec; color:{MSB_RED}; font-weight:600; font-size:0.8rem; }}
 .msb-step {{ border-left:3px solid {MSB_RED}; padding:2px 0 2px 14px; margin:10px 0; }}
 .msb-step b {{ color:{MSB_RED}; }}
</style>
""", unsafe_allow_html=True)


# ── Bộ nhớ Streamlit Cloud (~1GB) ────────────────────────────────────────
# Chỉ nạp sẵn các bảng dùng ở nhiều trang; các bảng lớn ít dùng -> nạp lười
# (get_df) đúng trang cần. Mọi frame được downcast float32/int32/category.
def _shrink(df: pd.DataFrame) -> pd.DataFrame:
    n = max(len(df), 1)
    for c in df.columns:
        s = df[c]
        dt = str(s.dtype)
        if dt == "float64":
            df[c] = s.astype("float32")
        elif dt == "int64" and s.abs().max() < 2_000_000_000:
            df[c] = s.astype("int32")
        elif dt in ("object", "str", "string"):
            if not IS_WASM and s.nunique(dropna=False) < n * 0.6:
                df[c] = s.astype("category")   # str-dtype mặc định của pandas 3 vẫn tốn RAM
            elif dt != "object":
                df[c] = s.astype(object)      # arrow của stlite ổn định nhất với object
    return df


# nạp sẵn (small / dùng nhiều); ai_product_recommendation_v2 bỏ cột text nặng
EAGER = {
    "customer_360_feature_mart": None, "ai_customer_score": None, "dim_customer": None,
    "dim_product": None, "rm_action_feedback": None,
    "ai_model_coefficient": None, "ai_model_metric": None, "ai_model_registry": None,
    "dim_product_catalogue": None, "agg_product_demand": None,
    "agg_segment_product_affinity": None, "fact_customer_product_holding": None,
    "agg_product_propensity": None, "ai_agent_review": None, "ai_model_adjustment": None,
    "ai_product_recommendation_v2": [
        "customer_id", "priority_rank", "product_id", "product_code", "product_name",
        "product_group", "propensity", "fit_score", "smart_growth_score", "priority_level",
        "reason_1", "reason_2", "reason_3", "expected_conversion", "recommended_action",
        "recommended_channel", "recommended_timing", "status", "branch_id"],
    "rm_feedback_ai": ["feedback_id", "customer_id", "product_code", "product_group",
                       "segment", "rm_verdict", "agree_flag", "converted_flag"],
}
LAZY = {  # name -> columns (None = tất cả)
    "ai_score_reason": None,
    "ai_recommendation": None,
    "fact_campaign": ["channel", "sent_flag", "delivered_flag", "opened_flag", "clicked_flag",
                      "responded_flag", "interested_flag", "applied_flag", "converted_flag"],
    "ml_propensity_training_set": ["customer_id", "product_id", "y_holds_product", "y_adopt_next_90d"],
}
ANGLE = {"CARD": "Hoàn tiền + ưu đãi chi tiêu, phù hợp mức chi tiêu và thu nhập.",
         "CASA": "Tài khoản/dịch vụ tối ưu dòng tiền và giao dịch hằng ngày.",
         "FD": "Cộng thêm lãi suất, tối ưu dòng tiền nhàn rỗi.",
         "LENDING": "Lãi suất ưu đãi, duyệt nhanh, phù hợp nhu cầu vốn."}


def _table_path(name):
    for ext in (".parquet", ".csv.gz", ".csv"):
        p = PARQUET / f"{name}{ext}"
        if p.exists():
            return p
    return None


def _read_table(name, columns=None):
    """Đọc 1 bảng — parquet (chạy native) hoặc csv.gz (bản stlite/WASM, không cần pyarrow)."""
    p = _table_path(name)
    if p is None:
        return None
    if p.name.endswith(".parquet"):
        return pd.read_parquet(p, columns=columns)
    df = pd.read_csv(p, compression="gzip" if p.name.endswith(".gz") else "infer")
    for c in df.columns:                       # khôi phục kiểu từ CSV
        s = df[c]
        if s.dtype == object:
            u = set(s.dropna().unique()[:4])
            if u and u <= {"True", "False"}:
                df[c] = s.map({"True": True, "False": False}).astype("boolean").fillna(False).astype(bool)
            elif c.endswith(("_date", "_timestamp", "_from", "_to")):
                df[c] = pd.to_datetime(s, errors="coerce")
    return df[columns] if columns else df


@st.cache_data
def keep_ids():
    """Tập customer_id được nạp vào app (lấy mẫu đều nếu vượt APP_MAX_CUST)."""
    d = _read_table("dim_customer", ["customer_id"])
    if d is None:
        return None
    ids = sorted(d["customer_id"].tolist())
    if len(ids) <= APP_MAX_CUST:
        return None
    step = max(1, len(ids) // APP_MAX_CUST)
    return frozenset(ids[::step][:APP_MAX_CUST])


def _filt(df, keep):
    if keep is not None and df is not None and "customer_id" in df.columns:
        return df[df["customer_id"].isin(keep)].reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Đang tải dữ liệu…")
def load(_spec_key):
    keep = keep_ids()
    d = {}
    for name, cols in EAGER.items():
        df = _read_table(name, cols)
        if df is not None:
            d[name] = _shrink(_filt(df, keep))
    if "ai_product_recommendation_v2" in d:
        r = d["ai_product_recommendation_v2"]
        ang = r["product_group"].astype(str).map(ANGLE)
        r["message_angle"] = ang if IS_WASM else ang.astype("category")
    d["__available__"] = sorted({p.name.split(".")[0]
                                 for p in PARQUET.glob("*") if p.suffix in (".parquet", ".gz", ".csv")})
    d["__n_cust__"] = 0 if keep is None else len(keep)
    return d


@st.cache_data(show_spinner=False)
def get_df(name):
    df = _read_table(name, LAZY.get(name))
    return None if df is None else _shrink(_filt(df, keep_ids()))


@st.cache_data
def load_model_report():
    p = MODELS / "model_report.md"
    return p.read_text(encoding="utf-8") if p.exists() else None


try:
    D = load(tuple(sorted(EAGER)))
except Exception as e:  # pragma: no cover
    st.error(f"Không đọc được data/parquet/. Chạy `py src/generate_data.py` + "
             f"`py src/train_models.py --apply` trước.\n\n{e}")
    st.stop()

_core = ("customer_360_feature_mart", "ai_customer_score", "dim_customer", "dim_product")
_missing = [t for t in _core if t not in D]
if _missing:
    st.error("Thiếu file dữ liệu lõi: " + ", ".join(f"`{m}.parquet`" for m in _missing)
             + ".\n\nCó trong repo: " + ", ".join(D.get("__available__", []) or ["(không có file nào)"])
             + ".\n\nTrên Streamlit Cloud: mở menu **⋮ → Reboot app** để pull commit mới nhất.")
    st.stop()

MART = D["customer_360_feature_mart"]
SCORE = D["ai_customer_score"]
CUST = D["dim_customer"]
PROD = D["dim_product"]
FB = D.get("rm_action_feedback")
COEF = D.get("ai_model_coefficient")
METRIC = D.get("ai_model_metric")
REGISTRY = D.get("ai_model_registry")
CAT = D.get("dim_product_catalogue")
PRECO = D.get("ai_product_recommendation_v2")
PDEM = D.get("agg_product_demand")
PAFF = D.get("agg_segment_product_affinity")
PHOLD = D.get("fact_customer_product_holding")
PCOMP = D.get("agg_product_propensity")
RFB = D.get("rm_feedback_ai")
AGR = D.get("ai_agent_review")
ADJ = D.get("ai_model_adjustment")

PID_NAME = dict(zip(PROD.product_id, PROD.product_name))
PID_GROUP = dict(zip(PROD.product_id, PROD.product_group))
N_CUST = MART.customer_id.nunique()


# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def eligibility_frame():
    """Tái hiện view v_customer_product_eligibility bằng pandas (nạp lười, cache)."""
    m = MART.merge(
        CUST[["customer_id", "customer_active_flag", "kyc_status",
              "marketing_consent_flag", "do_not_contact_flag"]], on="customer_id", how="left")
    held = {
        "CREDIT_CARD": m["has_premium_card"].astype(bool),
        "LOAN": m["has_active_loan"].astype(bool),
        "DEPOSIT": m["deposit_balance"].astype(float) > 0,
        "INVESTMENT": m["has_investment"].astype(bool),
    }
    out = []
    for grp, pid in TARGET_PRODUCTS.items():
        r_active = m["customer_active_flag"].astype(bool) & (m["kyc_status"] == "VERIFIED")
        r_gap = ~held[grp]
        r_consent = m["marketing_consent_flag"].astype(bool)
        r_dnc = ~m["do_not_contact_flag"].astype(bool)
        r_cmp = ~m["serious_complaint_15d"].astype(bool)
        r_rej = ~m["recent_rejection_30d_flag"].astype(bool)
        r_freq = m["contact_count_7d"].fillna(0).astype(int) < 2
        elig = r_active & r_gap & r_consent & r_dnc & r_cmp & r_rej & r_freq
        gate = np.where(~(r_consent & r_active), "EXCLUDE",
               np.where(~(r_dnc & r_cmp & r_rej & r_freq), "SUPPRESS", "ELIGIBLE"))
        out.append(pd.DataFrame({
            "customer_id": m["customer_id"], "product_id": pid, "product_group": grp,
            "branch_id": m["branch_id"], "is_eligible": elig, "gate1_result": gate,
            "rule_customer_active": r_active, "rule_product_gap": r_gap,
            "rule_marketing_consent": r_consent, "rule_not_dnc": r_dnc,
            "rule_no_serious_complaint_15d": r_cmp, "rule_no_recent_rejection_30d": r_rej,
            "rule_contact_frequency_ok": r_freq}))
    return pd.concat(out, ignore_index=True)


def fmt_vnd(x):
    try:
        x = float(x)
    except Exception:
        return "—"
    if abs(x) >= 1e9:
        return f"{x/1e9:.2f} tỷ"
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f} tr"
    return f"{x:,.0f}"


# ==========================================================================
# SIDEBAR
# ==========================================================================
st.sidebar.markdown(f"### <span style='color:{MSB_RED}'>◤ MSB</span> Smart Growth Engine",
                    unsafe_allow_html=True)
st.sidebar.caption("AI Evaluation Dashboard · Synthetic / Masked")
PAGE = st.sidebar.radio("Điều hướng", [
    "🏠 Giới thiệu & Mục đích",
    "🧭 Hành trình AI — 12 Actions",
    "🎯 RM Opportunity Desk",
    "👤 Customer 360",
    "🛍️ Sản phẩm & Nhu cầu",
    "🔁 AI Agent · Feedback & Hiệu chỉnh",
    "📊 Manager Intelligence",
    "🤖 Mô hình Propensity",
])
st.sidebar.divider()
st.sidebar.metric("Khách hàng trong app", f"{N_CUST:,}")
if D.get("__n_cust__"):
    st.sidebar.caption(f"(lấy mẫu đều từ 25.000 KH để vừa RAM Streamlit Cloud; "
                       f"dữ liệu đầy đủ trong `data/parquet/`)")
st.sidebar.metric("Cơ hội High (SGS ≥ 80)",
                  f"{(SCORE.groupby('customer_id').smart_growth_score.max() >= 80).sum():,}")
st.sidebar.caption(f"Model: `{REGISTRY.model_id.iloc[0]}` · `{REGISTRY.model_version.iloc[0]}`"
                   if REGISTRY is not None else "")


# ==========================================================================
# PAGE 1 — INTRO
# ==========================================================================
if PAGE.startswith("🏠"):
    st.markdown("""
<style>
 .block-container { max-width: 1200px; padding-top: 2rem; }
 .stMain h1, .stMain h2, .stMain h3, .stMain h4, .stMain h5,
 .stMain [data-testid="stMetricValue"] { font-family: inherit; }
 .stMain h1 { font-size: 1.8rem; }
 .stMain h2 { font-size: 1.3rem; }
 .stMain h3 { font-size: 1.15rem; }
 .stMain h4, .stMain h5 { font-size: 1rem; }
 .stMain [data-testid="stMetricValue"] { font-size: 1.5rem; }
 .msb-step { margin: 6px 0; }
</style>
""", unsafe_allow_html=True)
    st.title("MSB Smart Growth Engine")
    st.markdown("#### AI-Powered Customer Intelligence & Sales Growth Platform")
    st.markdown("<span class='msb-badge'>DEMO DATA · Synthetic / Masked</span>",
                unsafe_allow_html=True)

    st.markdown("""
Nền tảng AI **dự báo nhu cầu khách hàng** và **tối ưu hiệu quả bán hàng** — biến dữ liệu khách hàng
đang phân mảnh (Hồ sơ, CASA, Giao dịch, Thẻ, Vay, Bảo hiểm, Digital Banking, CRM, Chăm sóc KH,
Chiến dịch) thành **cơ hội bán có thể hành động ngay**.
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
##### <span style='color:{MSB_RED}'>AI FOR CUSTOMERS</span>
Giúp ngân hàng hiểu khách hàng sâu hơn, dự đoán nhu cầu để cung cấp đúng sản phẩm — từ bước
trải nghiệm đến gắn bó.
""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
##### <span style='color:{MSB_RED}'>AI FOR MY TEAM</span>
Hoạt động như một Sales + nhà quản lý: giúp đội kinh doanh chủ động tìm thông tin, giảm thao tác
thủ công, tập trung đúng khách hàng tiềm năng.
""", unsafe_allow_html=True)

    st.divider()
    st.subheader("5 bài toán công cụ giải quyết")
    q = ["**Khách hàng nào** nên được ưu tiên tiếp cận?",
         "Khách hàng đang quan tâm **sản phẩm nào**?",
         "**Xác suất** khách hàng phản hồi / chuyển đổi là bao nhiêu?",
         "Đâu là **thời điểm & kênh** tiếp cận phù hợp?",
         "Sale nên **nói gì & làm gì tiếp theo**?"]
    for i, x in enumerate(q, 1):
        st.markdown(f"<div class='msb-step'><b>{i}.</b> {x}</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Kiến trúc — Thiết kế hành trình AI")
    st.markdown("`Nhận tín hiệu → đánh giá khách hàng → quyết định có nên tiếp cận → "
                "chọn sản phẩm / kênh / thời điểm → tạo nội dung → kiểm tra rule & consent → "
                "gửi message → ghi nhận phản hồi → AI học lại`")
    st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent"; node [shape=box style="rounded,filled" fontname="Arial"
    fillcolor="#f5f6f8" color="#c9ced6" fontsize=11];
  data [label="CUSTOMER DATA" fillcolor="#eef1f4"];
  c360 [label="CUSTOMER 360"];
  model [label="AI MODEL\\n(logistic reg.)" fillcolor="#fde8ec" color="#E4002B"];
  score [label="SCORE\\nSmart Growth 0–100"];
  reco  [label="RECOMMENDATION\\nNBP / NBA / kênh / thời điểm"];
  rm    [label="RM ACTION"];
  result[label="ACTUAL RESULT"];
  train [label="TRAINING DATA"];
  data -> c360 -> model -> score -> reco -> rm -> result -> train;
  train -> model [style=dashed label="học lại" fontsize=10 color="#E4002B" fontcolor="#E4002B"];
}
""")

    st.divider()
    st.subheader("Sản phẩm giai đoạn đầu & mô hình dữ liệu")
    for grp, pid in TARGET_PRODUCTS.items():
        with st.expander(GROUP_LABEL[grp]):
            st.markdown(f"- {PID_NAME[pid]}")
    st.caption("Data model: 27 bảng PostgreSQL (dim / fact / aggregate / feature mart / "
               "AI feature / model output / recommendation / feedback / training) + 2 view "
               "(eligibility gate, top-20). Chi tiết: `sql/01_schema.sql`, `README.md`.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Khách hàng", f"{N_CUST:,}")
    m2.metric("Dòng feature mart", f"{len(MART):,}")
    m3.metric("Điểm AI (KH × sản phẩm)", f"{len(SCORE):,}")
    m4.metric("Khuyến nghị (35 SP)", f"{len(PRECO):,}" if PRECO is not None else "—")


# ==========================================================================
# PAGE 2 — 12 ACTIONS
# ==========================================================================
elif PAGE.startswith("🧭"):
    st.title("Hành trình AI — 12 Actions")
    st.caption("Mỗi bước gắn với dữ liệu / model thật trong công cụ.")

    best = SCORE.sort_values("smart_growth_score").groupby("customer_id").tail(1)

    step = st.select_slider("Chọn Action",
                            options=[a[0] for a in ACTIONS],
                            format_func=lambda x: f"Action {x}")
    idx = int(step) - 1
    _, title, desc = ACTIONS[idx]
    st.markdown(f"### Action {step} · {title}")
    st.info(desc)

    if step == "1":
        cid = st.selectbox("Khách hàng", MART.customer_id.head(500))
        row = MART[MART.customer_id == cid].iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Số dư CASA (avg 90d)", fmt_vnd(row.avg_balance_90d))
        b.metric("Chi tiêu 30d", fmt_vnd(row.spending_30d))
        c.metric("Digital engagement", f"{row.digital_engagement_score:.0f}")
        d.metric("Sản phẩm đang dùng", int(row.product_count))
        sig = []
        if row.balance_growth_3m > 0.15: sig.append("Số dư tăng đáng kể")
        if not row.has_premium_card: sig.append("Chưa sở hữu Credit Card Premium")
        if row.salary_flag: sig.append("Có nhận lương qua MSB")
        if row.recent_rejection_flag: sig.append("Vừa từ chối một offer gần đây")
        st.write("**Recent signals:** " + (" · ".join(sig) if sig else "—"))
        st.dataframe(pd.DataFrame({"feature": row.index.astype(str), "giá trị": row.astype(str).values}), width="stretch", height=460, hide_index=True)

    elif step == "2":
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.histogram(best, x="smart_growth_score", nbins=40,
                               color_discrete_sequence=[MSB_RED])
            fig.add_vline(x=80, line_dash="dash", annotation_text="High")
            fig.update_layout(title="Phân bố Smart Growth Score (điểm cao nhất / khách)",
                              bargap=0.02, height=380)
            st.plotly_chart(fig, width="stretch")
        with c2:
            comp = best[["product_propensity_score", "customer_value_score", "intent_signal_score",
                         "engagement_score", "timing_score", "relationship_score"]].mean().reset_index()
            comp.columns = ["thành phần", "điểm"]
            comp["thành phần"] = ["Propensity 25%", "Cust. Value 20%", "Intent 20%",
                                  "Engagement 15%", "Timing 10%", "Relationship 10%"]
            fig = px.bar(comp, x="điểm", y="thành phần", orientation="h",
                         color_discrete_sequence=[MSB_INK])
            fig.update_layout(title="Điểm bình quân từng thành phần", showlegend=False, height=380)
            st.plotly_chart(fig, width="stretch")
        st.caption("Smart Growth Score = 25%·Propensity + 20%·CustomerValue + 20%·Intent "
                   "+ 15%·Engagement + 10%·Timing + 10%·Relationship")

    elif step == "3":
        _elig = eligibility_frame()
        g = _elig.groupby("gate1_result").size().reindex(["ELIGIBLE", "SUPPRESS", "EXCLUDE"]).fillna(0)
        fig = go.Figure(go.Funnel(y=g.index, x=g.values,
                                  marker_color=["#2E8B8B", "#F4A300", "#5B6770"]))
        fig.update_layout(title="Decision Gate 1 — (khách × sản phẩm mục tiêu)", height=340)
        st.plotly_chart(fig, width="stretch")
        rules = [c for c in _elig.columns if c.startswith("rule_")]
        fail = ((~_elig[rules]).mean().sort_values(ascending=False) * 100).reset_index()
        fail.columns = ["điều kiện", "pct"]
        fail["điều kiện"] = fail["điều kiện"].str.replace("rule_", "").str.replace("_", " ")
        fig = px.bar(fail, x="pct", y="điều kiện", orientation="h",
                     color_discrete_sequence=[MSB_RED])
        fig.update_layout(title="% (khách × sản phẩm) rớt từng điều kiện", showlegend=False, height=320)
        st.plotly_chart(fig, width="stretch")

    elif step == "4":
        R = PRECO
        cid = st.selectbox("Khách hàng", R.customer_id.drop_duplicates().head(500))
        r = R[R.customer_id == cid].sort_values("priority_rank")
        for _, x in r.iterrows():
            name = x.get("product_name") or PID_NAME.get(x.get("recommended_product_id"), "")
            p = x.get("propensity", x.get("expected_conversion", 0))
            st.markdown(f"**#{int(x.priority_rank)} · {name}**  — Propensity **{p*100:.0f}%** · "
                        f"SGS {x.get('smart_growth_score', 0):.0f} · "
                        f"exp. conversion {x.expected_conversion*100:.0f}%")
            st.caption(x.get("message_angle", ""))
        if PRECO is not None:
            st.caption("Xếp hạng trên toàn bộ **35 sản phẩm MSB** (đã loại sản phẩm khách đã sở hữu).")

    elif step == "5":
        R = PRECO
        t = R[R.priority_rank == 1].merge(CUST[["customer_id", "customer_segment"]], on="customer_id")
        piv = t.groupby(["customer_segment", "recommended_action"]).size().reset_index(name="n")
        fig = px.bar(piv, x="customer_segment", y="n", color="recommended_action",
                     color_discrete_sequence=PALETTE)
        fig.update_layout(title="Next Best Action theo phân khúc", height=420)
        st.plotly_chart(fig, width="stretch")

    elif step == "6":
        R = PRECO
        t = R[R.priority_rank == 1]
        t = t[t.status != "BLOCKED"] if "status" in t.columns else t
        piv = t.groupby(["recommended_channel", "recommended_timing"]).agg(
            n=("customer_id", "count"), exp=("expected_conversion", "mean")).reset_index()
        fig = px.scatter(piv, x="recommended_timing", y="recommended_channel", size="n",
                         color="exp", color_continuous_scale="Reds", size_max=48)
        fig.update_layout(title="Kênh × thời điểm (size = số lượng, màu = expected conversion)",
                          height=420)
        st.plotly_chart(fig, width="stretch")

    elif step == "7":
        R = PRECO
        st.write("Message angle được GenAI sinh từ structured context + template kiểm soát:")
        if PRECO is not None:
            st.dataframe(R[["customer_id", "product_name", "recommended_channel", "message_angle"]]
                         .head(25), width="stretch", hide_index=True)
        else:
            st.dataframe(R[["customer_id", "recommended_product_id", "recommended_channel", "message_angle"]]
                         .head(25).assign(recommended_product_id=lambda d: d.recommended_product_id.map(PID_NAME)),
                         width="stretch", hide_index=True)

    elif step in ("8", "9"):
        R = PRECO
        t = R[R.priority_rank == 1].copy()
        t["blocked"] = t.status.eq("BLOCKED") if "status" in t.columns else t.get("suppression_flag", False)
        piv = t.groupby(["status", "blocked"]).size().reset_index(name="n")
        fig = px.bar(piv, x="status", y="n", color="blocked",
                     color_discrete_sequence=[MSB_INK, MSB_RED])
        fig.update_layout(title="Trạng thái message trước khi gửi (Gate 2 & 3)", height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption("BLOCKED = suppression (consent / DNC / complaint / frequency). "
                   "RM_APPROVAL = khách high-value cần tư vấn. SENT/NEW = auto-send / nurture.")

    elif step in ("10", "11"):
        _camp = get_df("fact_campaign")
        if _camp is not None:
            f = _camp[["sent_flag", "delivered_flag", "opened_flag", "clicked_flag",
                       "responded_flag", "interested_flag", "applied_flag", "converted_flag"]].mean() * 100
            f.index = ["Sent", "Delivered", "Opened", "Clicked", "Responded",
                       "Interested", "Applied", "Converted"]
            fig = go.Figure(go.Funnel(y=f.index, x=f.values, marker_color=MSB_RED))
            fig.update_layout(title="Funnel phản hồi khách hàng (fact_campaign, %)", height=420)
            st.plotly_chart(fig, width="stretch")

    elif step == "12":
        _rv1 = get_df("ai_recommendation")
        if FB is not None and _rv1 is not None:
            m = FB.merge(_rv1[["recommendation_id", "expected_conversion"]], on="recommendation_id")
            piv = m.groupby("customer_response").agg(
                n=("feedback_id", "count"),
                ai_expected=("expected_conversion", "mean")).reset_index().sort_values("ai_expected")
            fig = px.bar(piv, x="customer_response", y="ai_expected", text="n",
                         color="ai_expected", color_continuous_scale="Reds")
            fig.update_layout(title="Outcome thực tế vs AI expected conversion → quay lại train model",
                              height=400, yaxis_title="AI expected conversion (bình quân)")
            st.plotly_chart(fig, width="stretch")
            st.caption("KH `CONVERTED` có expected conversion cao rõ rệt so với `NOT_INTERESTED` "
                       "→ tín hiệu đúng, model học lại ở vòng sau.")

    st.divider()
    with st.expander("Xem toàn bộ 12 Actions"):
        for n, t, d in ACTIONS:
            st.markdown(f"<div class='msb-step'><b>Action {n} · {t}</b><br>{d}</div>",
                        unsafe_allow_html=True)


# ==========================================================================
# PAGE 3 — RM OPPORTUNITY DESK
# ==========================================================================
elif PAGE.startswith("🎯"):
    st.title("RM Opportunity Desk")
    st.caption("2 tầng lọc: (1) đủ điều kiện kinh doanh — Decision Gate 1  →  "
               "(2) Smart Growth Score DESC → Top N / chi nhánh / sản phẩm. "
               "Danh mục: 35 sản phẩm MSB chuẩn hoá.")

    if PRECO is None or CAT is None:
        st.warning("Chưa nạp được `ai_product_recommendation_v2` / `dim_product_catalogue`.\n\n"
                   "File có trong data/parquet/: " + ", ".join(D.get("__available__", [])) + "\n\n"
                   "→ Chạy `py src/product_analysis.py`, hoặc trên Streamlit Cloud: **⋮ → Reboot app**.")
        st.stop()

    prods = CAT.sort_values(["product_group", "product_name"])
    GLAB = {"CARD": "Thẻ", "CASA": "Tài khoản", "FD": "Tiền gửi", "LENDING": "Cho vay"}
    opt = list(prods.product_id)
    pname = dict(zip(prods.product_id, prods.product_name))
    pgrp = dict(zip(prods.product_id, prods.product_group))

    c0, c1, c2, c3 = st.columns([1, 2, 2, 1.4])
    gsel = c0.selectbox("Nhóm", ["(tất cả)"] + list(GLAB), format_func=lambda g: GLAB.get(g, g))
    opt2 = opt if gsel == "(tất cả)" else [p for p in opt if pgrp[p] == gsel]
    pid = c1.selectbox("Sản phẩm", opt2,
                       format_func=lambda p: f"{GLAB.get(pgrp[p], pgrp[p])} · {pname[p]}")
    branches = ["(Toàn hàng)"] + sorted(PRECO.branch_id.dropna().unique())
    branch = c2.selectbox("Chi nhánh", branches)
    topn = c3.slider("Top N", 5, 50, 20, 5)

    pool = PRECO[PRECO.product_id == pid].copy()
    pool = pool.merge(CUST[["customer_id", "customer_segment"]], on="customer_id", how="left")
    view = pool if branch == "(Toàn hàng)" else pool[pool.branch_id == branch]
    view = view.sort_values("smart_growth_score", ascending=False).head(topn).reset_index(drop=True)
    view["#"] = view.index + 1

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cơ hội (đủ ĐK, chưa sở hữu)", f"{len(pool):,}")
    k2.metric("Very High / High", f"{view.priority_level.isin(['Very High','High']).sum()}")
    k3.metric("SGS bình quân (Top N)", f"{view.smart_growth_score.mean():.0f}" if len(view) else "—")
    k4.metric("Propensity bình quân", f"{view.propensity.mean()*100:.0f}%" if len(view) else "—")

    disp = view.assign(
        Propensity=lambda d: (d.propensity * 100).round(0),
        SGS=lambda d: d.smart_growth_score.round(0),
        ExpConv=lambda d: (d.expected_conversion * 100).round(0),
        Reason=lambda d: d[["reason_1", "reason_2"]].apply(
            lambda r: " · ".join(x for x in r if isinstance(x, str)), axis=1),
    )[["#", "customer_id", "customer_segment", "SGS", "priority_level", "Propensity", "ExpConv",
       "recommended_action", "recommended_channel", "recommended_timing", "Reason", "status"]].rename(columns={
        "customer_segment": "Segment", "priority_level": "Priority",
        "recommended_action": "Next Best Action", "recommended_channel": "Kênh",
        "recommended_timing": "Thời điểm", "status": "Trạng thái"})
    st.dataframe(disp, width="stretch", hide_index=True, height=560)
    st.download_button("Tải danh sách (CSV)", disp.to_csv(index=False).encode("utf-8"),
                       f"top_{pid}_{branch}.csv", "text/csv")


# ==========================================================================
# PAGE 4 — CUSTOMER 360
# ==========================================================================
elif PAGE.startswith("👤"):
    st.title("Customer 360")
    _src = PRECO
    default_list = _src.customer_id.drop_duplicates().head(300).tolist()
    cid = st.selectbox("Chọn khách hàng", default_list
                       + [c for c in MART.customer_id.head(300) if c not in default_list])

    row = MART[MART.customer_id == cid].iloc[0]
    prof = CUST[CUST.customer_id == cid].iloc[0]
    sc = SCORE[SCORE.customer_id == cid].copy()
    sc["product"] = sc.product_id.map(PID_NAME)
    prc = (PRECO[PRECO.customer_id == cid].sort_values("priority_rank")
           if PRECO is not None else pd.DataFrame())
    _rv1 = get_df("ai_recommendation")
    rc = (_rv1[_rv1.customer_id == cid].sort_values("priority_rank")
          if _rv1 is not None else pd.DataFrame())

    a, b, c, d = st.columns(4)
    a.metric("Phân khúc", prof.customer_segment)
    b.metric("Quan hệ", f"{prof.relationship_years:.1f} năm")
    c.metric("Số dư CASA", fmt_vnd(row.avg_balance_90d))
    d.metric("Digital engagement", f"{row.digital_engagement_score:.0f}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Mức độ phù hợp theo sản phẩm (35 SP MSB)")
        if not prc.empty:
            b = prc.sort_values("smart_growth_score")
            fig = px.bar(b, x="smart_growth_score", y="product_name", orientation="h",
                         color="smart_growth_score", color_continuous_scale="Reds",
                         text="priority_level")
            fig.update_layout(height=320, showlegend=False, xaxis_title="Smart Growth Score",
                              yaxis_title="")
            st.plotly_chart(fig, width="stretch")
        else:
            fig = px.bar(sc.sort_values("smart_growth_score"), x="smart_growth_score", y="product",
                         orientation="h", color="smart_growth_score",
                         color_continuous_scale="Reds", text="priority_level")
            fig.update_layout(height=300, showlegend=False, xaxis_title="Smart Growth Score")
            st.plotly_chart(fig, width="stretch")

        best_pid = sc.sort_values("smart_growth_score").iloc[-1]
        radar = best_pid[["product_propensity_score", "customer_value_score", "intent_signal_score",
                          "engagement_score", "timing_score", "relationship_score"]].astype(float)
        radar.index = ["Propensity", "Cust.Value", "Intent", "Engagement", "Timing", "Relationship"]
        fig = go.Figure(go.Scatterpolar(r=radar.values, theta=radar.index, fill="toself",
                                        line_color=MSB_RED))
        fig.update_layout(title=f"Hồ sơ điểm — {best_pid['product']}",
                          polar=dict(radialaxis=dict(range=[0, 100])), height=340)
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Next Best Product / Action")
        _rows = prc if not prc.empty else rc.assign(
            product_name=rc.recommended_product_id.map(PID_NAME))
        for _, x in _rows.iterrows():
            tag = "🔴" if x.priority_rank == 1 else "▫️"
            reasons = " · ".join(str(x[c]) for c in ("reason_1", "reason_2", "reason_3")
                                 if c in x.index and isinstance(x[c], str))
            st.markdown(f"{tag} **#{int(x.priority_rank)} {x.get('product_name','')}** — "
                        f"{x.recommended_action} · {x.recommended_channel} · {x.recommended_timing}")
            st.caption(f"{x.get('message_angle','')}  \n"
                       f"*Propensity {x.get('propensity', x.get('expected_conversion',0))*100:.0f}% · "
                       f"Expected conversion {x.expected_conversion*100:.0f}% · status {x.get('status','')}*"
                       + (f"  \nLý do: {reasons}" if reasons else ""))

        st.subheader("Why this customer  (đóng góp logit theo feature)")
        best_sid = sc.sort_values("smart_growth_score").iloc[-1]["score_id"]
        _reason = get_df("ai_score_reason")
        rs = _reason[_reason.score_id == best_sid].sort_values("contribution_score").copy()
        rs["feat"] = rs.feature_name.map(FEATURE_LABEL).fillna(rs.feature_name)
        fig = px.bar(rs, x="contribution_score", y="feat", orientation="h",
                     color="impact_direction",
                     color_discrete_map={"POSITIVE": MSB_RED, "NEGATIVE": "#5B6770"})
        fig.update_layout(height=300, yaxis_title="", xaxis_title="đóng góp vào log-odds")
        st.plotly_chart(fig, width="stretch")

    with st.expander("Toàn bộ feature Customer 360"):
        st.dataframe(pd.DataFrame({"feature": row.index.astype(str), "giá trị": row.astype(str).values}), width="stretch", height=500, hide_index=True)


# ==========================================================================
# PAGE 5 — MANAGER INTELLIGENCE
# ==========================================================================
elif PAGE.startswith("📊"):
    st.title("Manager Intelligence")
    st.caption("Toàn bộ trên danh mục 35 sản phẩm MSB (`ai_product_recommendation_v2`).")

    best = (PRECO[PRECO.priority_rank == 1].copy() if PRECO is not None
            else SCORE.sort_values("smart_growth_score").groupby("customer_id").tail(1))
    best = best.merge(CUST[["customer_id", "customer_segment"]], on="customer_id", how="left")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Danh mục khách hàng", f"{N_CUST:,}")
    k2.metric("High opportunities (SGS ≥ 80)", f"{(best.smart_growth_score >= 80).sum():,}")
    k3.metric("Đủ điều kiện tiếp cận", f"{int(eligibility_frame().is_eligible.sum()):,}")
    if FB is not None:
        k4.metric("Tỉ lệ chuyển đổi (RM feedback)", f"{FB.converted_flag.mean()*100:.0f}%")

    c1, c2 = st.columns(2)
    with c1:
        pr = (best.priority_level.value_counts().reindex(PRIORITY_ORDER).fillna(0)
              .rename_axis("priority").reset_index(name="customers"))
        fig = px.bar(pr, x="priority", y="customers", color="priority",
                     color_discrete_map=PRIORITY_COLOR)
        fig.update_layout(title="Phân bố Priority (cơ hội #1 / khách)",
                          showlegend=False, height=360, xaxis_title="")
        st.plotly_chart(fig, width="stretch")
    with c2:
        piv = (best.groupby("customer_segment").smart_growth_score.mean()
               .sort_values().reset_index())
        fig = px.bar(piv, x="smart_growth_score", y="customer_segment", orientation="h",
                     color_discrete_sequence=[MSB_RED])
        fig.update_layout(title="Smart Growth Score bình quân theo phân khúc",
                          showlegend=False, height=360)
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        pcol = "product_name" if "product_name" in best.columns else "recommended_product_id"
        nbp = best[pcol].value_counts()
        top = nbp.head(9).rename_axis("product").reset_index(name="n")
        if len(nbp) > 9:
            top = pd.concat([top, pd.DataFrame([{"product": "Khác", "n": int(nbp.iloc[9:].sum())}])])
        fig = px.pie(top, values="n", names="product", color_discrete_sequence=PALETTE, hole=0.5)
        fig.update_layout(title="Next Best Product #1 — cơ cấu (35 SP)", height=380)
        st.plotly_chart(fig, width="stretch")
    with c4:
        if PDEM is not None:
            d = (PDEM.groupby("product_code").expected_adopters_90d.sum()
                 .sort_values(ascending=False).head(12).rename_axis("product").reset_index(name="exp90"))
            fig = px.bar(d, x="exp90", y="product", orientation="h",
                         color_discrete_sequence=[MSB_INK])
            fig.update_layout(title="Dự báo mở mới 90 ngày (top 12)", height=380,
                              yaxis=dict(autorange="reversed"), yaxis_title="")
            st.plotly_chart(fig, width="stretch")

    c5, c6 = st.columns(2)
    with c5:
        grp = best.get("product_group")
        if grp is not None:
            g = grp.value_counts().rename_axis("nhóm").reset_index(name="n")
            fig = px.bar(g, x="nhóm", y="n", color="nhóm", color_discrete_sequence=PALETTE)
            fig.update_layout(title="Cơ hội #1 theo nhóm sản phẩm", showlegend=False, height=340)
            st.plotly_chart(fig, width="stretch")
    with c6:
        if FB is not None:
            oc = FB.customer_response.value_counts().rename_axis("response").reset_index(name="n")
            fig = px.bar(oc, x="response", y="n", color_discrete_sequence=[MSB_INK])
            fig.update_layout(title="Kết quả hành động RM (feedback loop)",
                              showlegend=False, height=340, xaxis_title="")
            st.plotly_chart(fig, width="stretch")

    st.subheader("Xếp hạng chi nhánh theo số cơ hội High")
    src = PRECO
    hi = src[src.priority_level.isin(["Very High", "High"])]
    if "branch_id" not in hi.columns:
        hi = hi.merge(CUST[["customer_id", "branch_id"]], on="customer_id", how="left")
    br = (hi.groupby("branch_id").size().sort_values(ascending=False).head(20)
          .rename_axis("branch_id").reset_index(name="n"))
    fig = px.bar(br, x="branch_id", y="n", color_discrete_sequence=[MSB_RED])
    fig.update_layout(showlegend=False, height=340, xaxis_title="", yaxis_title="cơ hội High")
    st.plotly_chart(fig, width="stretch")


# ==========================================================================
# PAGE 6 — MODEL
# ==========================================================================
elif PAGE.startswith("🤖"):
    st.title("Mô hình Propensity — Hybrid (Logistic Regression + Rule fit)")
    st.markdown(
        r"$P_{\text{sp}} = w\cdot P_{\text{LR}}(\text{nhóm neo}) + (1-w)\cdot \text{fit\_score}(\text{rule})$"
        "  \n"
        r"$P_{\text{LR}} = \sigma(b + \sum_i w_i X_i)$  — 6 mô hình neo: Thẻ tín dụng · Vay · "
        "Tiền gửi · Đầu tư ($w=0.45$).  31 sản phẩm còn lại: $w=0$ → chỉ rule fit theo "
        "\"Khách hàng/Nhu cầu phù hợp\".")

    if PCOMP is not None:
        st.subheader("Cấu thành propensity — 35 sản phẩm MSB")
        cc = PCOMP.copy()
        cc["kiểu"] = np.where(cc.lr_blended, "LR + rule (w=0.45)", "chỉ rule (w=0)")
        m = cc.melt(id_vars=["product_code", "product_group", "kiểu"],
                    value_vars=["avg_fit_score", "avg_propensity"],
                    var_name="chỉ số", value_name="giá trị")
        m["chỉ số"] = m["chỉ số"].map({"avg_fit_score": "fit_score (rule)",
                                       "avg_propensity": "propensity (hybrid)"})
        fig = px.bar(m.sort_values("giá trị"), x="giá trị", y="product_code", color="chỉ số",
                     barmode="group", color_discrete_sequence=[MSB_INK, MSB_RED], height=760)
        fig.update_layout(title="fit_score (rule) vs propensity (hybrid) theo sản phẩm",
                          yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("Chênh lệch propensity − fit_score = đóng góp của mô hình LR nhóm neo "
                   "(âm khi P_LR < fit, dương khi P_LR > fit).")

    if METRIC is None or COEF is None:
        st.info("Chưa có artefact LR (`py src/train_models.py --apply`) — chỉ hiển thị phần hybrid ở trên.")
        st.stop()
    st.divider()
    st.subheader("4 mô hình neo — Logistic Regression (doc mục a., feature X1..X6)")

    mt = METRIC.copy()
    mt["product"] = mt.product_id.map(PID_NAME)
    key = mt[mt.metric_name.isin(["roc_auc", "pr_auc", "log_loss", "accuracy", "f1", "ks"])]
    piv = key.pivot_table(index=["product", "dataset_split"], columns="metric_name",
                          values="metric_value").reset_index()
    st.subheader("Kết quả test")
    st.dataframe(piv.round(3), width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        auc = mt[(mt.metric_name == "roc_auc")]
        fig = px.bar(auc, x="product", y="metric_value", color="dataset_split", barmode="group",
                     color_discrete_sequence=PALETTE)
        fig.update_layout(title="ROC-AUC theo split (train ≈ test ≈ CV = không overfit)",
                          height=380, yaxis_range=[0.5, 0.85])
        st.plotly_chart(fig, width="stretch")
    with c2:
        prod_sel = st.selectbox("Sản phẩm", COEF.product_id.unique(),
                                format_func=lambda p: PID_NAME.get(p, p))
        cf = COEF[COEF.product_id == prod_sel].copy()
        cf["feat"] = cf.feature_name.map(FEATURE_LABEL).fillna(cf.feature_name)
        cf = cf[cf.feature_name != "intercept"].sort_values("coefficient")
        fig = px.bar(cf, x="coefficient", y="feat", orientation="h",
                     color="coefficient", color_continuous_scale="RdBu_r")
        fig.update_layout(title=f"Hệ số wᵢ — {PID_NAME.get(prod_sel, prod_sel)}",
                          height=380, yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("Hệ số quy đổi về thang gốc X∈[0,1]; odds ratio = eʷ.")

    _pt = get_df("ml_propensity_training_set")
    if _pt is not None:
        st.subheader("Calibration — P dự đoán vs adoption thực tế (tệp chưa sở hữu)")
        s = SCORE.merge(_pt[["customer_id", "product_id", "y_holds_product",
                             "y_adopt_next_90d"]], on=["customer_id", "product_id"])
        s = s[s.y_holds_product == 0].copy()
        s["decile"] = (s.propensity_probability * 10).clip(0, 9.999).astype(int) + 1
        cal = s.groupby(["product_id", "decile"]).agg(
            pred=("propensity_probability", "mean"),
            actual=("y_adopt_next_90d", "mean")).reset_index()
        cal["product"] = cal.product_id.map(PID_NAME)
        fig = px.line(cal, x="pred", y="actual", color="product", markers=True,
                      color_discrete_sequence=PALETTE)
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
        fig.update_layout(height=420, xaxis_title="P dự đoán (decile)",
                          yaxis_title="tỉ lệ adoption thực tế")
        st.plotly_chart(fig, width="stretch")

    rep = load_model_report()
    if rep:
        with st.expander("models/model_report.md"):
            st.markdown(rep)


# ==========================================================================
# PAGE 5b — SẢN PHẨM & NHU CẦU
# ==========================================================================
elif PAGE.startswith("🛍️"):
    st.title("Sản phẩm & Nhu cầu — Khách hàng phù hợp với sản phẩm nào?")
    st.caption("Danh mục 35 sản phẩm chuẩn hoá từ MSB_products_description.xlsx (958 mã). "
               "Propensity = hybrid Logistic Regression (nhóm neo) + rule fit theo "
               "'Khách hàng/Nhu cầu phù hợp'.")

    if PRECO is None or CAT is None or PDEM is None:
        st.warning("Chưa nạp được dữ liệu phân tích sản phẩm.\n\n"
                   "File có trong data/parquet/: " + ", ".join(D.get("__available__", [])) + "\n\n"
                   "→ Chạy `py src/product_analysis.py`, hoặc trên Streamlit Cloud: **⋮ → Reboot app**.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Danh mục", "📈 Dự báo cầu", "🔥 Phân khúc × Sản phẩm", "👤 Gợi ý theo khách hàng"])

    # ---- Danh mục -----------------------------------------------------
    with tab1:
        c = CAT.copy()
        gsel = st.multiselect("Nhóm", P_GROUPS := list(c.product_group.unique()),
                              default=P_GROUPS)
        st.dataframe(
            c[c.product_group.isin(gsel)][["product_code", "product_name", "product_group",
                                           "subgroup", "product_tier", "customer_type",
                                           "target_need", "base_propensity"]],
            width="stretch", hide_index=True, height=520)
        k1, k2, k3 = st.columns(3)
        k1.metric("Sản phẩm chuẩn hoá", len(c))
        if PHOLD is not None:
            k2.metric("SP đang sở hữu / khách (TB)", f"{len(PHOLD)/MART.customer_id.nunique():.1f}")
        k3.metric("Đề xuất/khách (top-6)", f"{len(PRECO)/PRECO.customer_id.nunique():.1f}")

    # ---- Dự báo cầu -------------------------------------------------
    with tab2:
        d = (PDEM.groupby(["product_code", "product_group"])
             .agg(eligible=("eligible_customers", "sum"),
                  holders=("current_holders", "max"),
                  exp30=("expected_adopters_30d", "sum"),
                  exp60=("expected_adopters_60d", "sum"),
                  exp90=("expected_adopters_90d", "sum"),
                  avg_p=("avg_propensity", "mean")).reset_index()
             .sort_values("exp90", ascending=False))
        c1, c2, c3 = st.columns(3)
        c1.metric("Dự báo mở mới 30 ngày", f"{d.exp30.sum():,.0f}")
        c2.metric("60 ngày", f"{d.exp60.sum():,.0f}")
        c3.metric("90 ngày", f"{d.exp90.sum():,.0f}")
        fig = px.bar(d.head(20), x="exp90", y="product_code", orientation="h", color="product_group",
                     color_discrete_sequence=PALETTE, text="exp90")
        fig.update_layout(title="Dự báo mở mới 90 ngày theo sản phẩm (top 20)", height=560,
                          yaxis_title="", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        seg_sel = st.selectbox("Xem chi tiết theo phân khúc", ["(tất cả)"] + sorted(PDEM.segment.dropna().unique()))
        dd = PDEM if seg_sel == "(tất cả)" else PDEM[PDEM.segment == seg_sel]
        st.dataframe(dd.sort_values("expected_adopters_90d", ascending=False)[
            ["product_code", "product_group", "segment", "eligible_customers", "current_holders",
             "avg_propensity", "high_propensity", "expected_adopters_90d"]],
            width="stretch", hide_index=True, height=360)

    # ---- Phân khúc × Sản phẩm --------------------------------------
    with tab3:
        aff = PAFF.set_index(PAFF.columns[0]) if PAFF.columns[0] != "product_code" else PAFF.set_index("product_code")
        aff = aff.select_dtypes("number")
        fig = px.imshow(aff, aspect="auto",
                        color_continuous_scale=["#eff6ff", "#bfdbfe", "#60a5fa", "#2563eb", "#1e3a8a"],
                        labels=dict(color="propensity TB"))
        fig.update_layout(title="Ma trận phân khúc × sản phẩm (propensity trung bình)",
                          height=620,
                          margin=dict(l=60, r=30, t=70, b=60),
                          coloraxis_colorbar=dict(thickness=14, len=0.8, tickformat=".0%"))
        st.plotly_chart(fig, width="stretch")

    # ---- Gợi ý theo khách hàng ------------------------------------
    with tab4:
        cid = st.selectbox("Khách hàng", PRECO.customer_id.drop_duplicates().head(800))
        r = PRECO[PRECO.customer_id == cid].sort_values("priority_rank")
        m = MART[MART.customer_id == cid]
        if not m.empty:
            mm = m.iloc[0]
            cc = st.columns(4)
            cc[0].metric("Phân khúc", str(mm.get("segment", "")))
            cc[1].metric("Số dư ~", f"{mm.get('avg_balance_90d', 0)/1e6:,.0f}tr")
            cc[2].metric("SP đang sở hữu",
                         int((PHOLD.customer_id == cid).sum()) if PHOLD is not None else "—")
            cc[3].metric("Digital score", f"{mm.get('digital_engagement_score', 0):.0f}")
        for _, row in r.iterrows():
            reasons = " · ".join([x for x in (row.reason_1, row.reason_2, row.reason_3) if isinstance(x, str)])
            st.markdown(
                f"**#{int(row.priority_rank)} — {row['product_name']}**  "
                f"<span class='msb-badge'>{row.priority_level}</span>  \n"
                f"Propensity **{row.propensity:.0%}** · fit {row.fit_score:.0%} · "
                f"SGS {row.smart_growth_score:.0f} · kỳ vọng chuyển đổi {row.expected_conversion:.0%}  \n"
                f"<span style='color:#5B6770'>Lý do: {reasons}</span>",
                unsafe_allow_html=True)
        st.divider()
        st.dataframe(r[["priority_rank", "product_code", "product_group", "propensity",
                        "fit_score", "smart_growth_score", "expected_conversion"]],
                     width="stretch", hide_index=True)


# ==========================================================================
# PAGE 5c — AI AGENT · FEEDBACK & HIỆU CHỈNH  (Action 11–12)
# ==========================================================================
elif PAGE.startswith("🔁"):
    st.title("AI Agent — RM Feedback & Hiệu chỉnh mô hình")
    st.caption("Action 11–12 của hành trình AI: RM phản hồi với đề xuất → agent **xem xét lại "
               "mô hình**, **giải thích**, phân biệt \"AI sai thật\" vs \"RM thận trọng\", rồi "
               "**hiệu chỉnh rule/gate** khi sai thật → chấm lại điểm.")

    if RFB is None or AGR is None:
        st.warning("Chưa có dữ liệu agent. Chạy `py src/rm_feedback_agent.py --apply` "
                   "rồi `py src/product_analysis.py`.\n\n"
                   "Có trong data/parquet/: " + ", ".join(D.get("__available__", [])))
        st.stop()

    wrong = AGR[AGR.model_is_wrong]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Lượt RM phản hồi", f"{len(RFB):,}")
    k2.metric("Đồng thuận RM ↔ AI", f"{RFB.agree_flag.mean()*100:.0f}%")
    k3.metric("Vùng nghi vấn", f"{len(AGR)}")
    k4.metric("Kết luận MÔ HÌNH SAI", f"{len(wrong)}")
    k5.metric("Đề xuất bị loại sau fix",
              f"{int(ADJ.impact_recos.sum()):,}" if ADJ is not None and len(ADJ) else "0")

    t1, t2, t3, t4 = st.tabs(["📊 Đồng thuận", "🔎 Phát hiện & giải thích",
                              "🔧 Hiệu chỉnh đã áp", "✍️ RM gửi feedback"])

    with t1:
        g = (RFB.groupby("product_group").agg(n=("feedback_id", "count"),
             agree=("agree_flag", "mean"), conv=("converted_flag", "mean")).reset_index())
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(g, x="product_group", y="agree", color="product_group",
                         color_discrete_sequence=PALETTE, text=g.agree.map(lambda v: f"{v:.0%}"))
            fig.update_layout(title="% RM đồng ý với đề xuất AI, theo nhóm", showlegend=False,
                              height=360, yaxis_tickformat=".0%", xaxis_title="")
            st.plotly_chart(fig, width="stretch")
        with c2:
            v = RFB.rm_verdict.value_counts().rename_axis("verdict").reset_index(name="n")
            fig = px.bar(v, x="n", y="verdict", orientation="h", color_discrete_sequence=[MSB_INK])
            fig.update_layout(title="Phân bố phản hồi RM", height=360, yaxis_title="")
            st.plotly_chart(fig, width="stretch")
        heat = (RFB.groupby(["product_code", "segment"]).agree_flag.mean().reset_index()
                .pivot(index="product_code", columns="segment", values="agree_flag"))
        fig = px.imshow(heat, color_continuous_scale="RdYlGn", zmin=0.2, zmax=0.9, aspect="auto",
                        labels=dict(color="% đồng ý"))
        fig.update_layout(title="Heatmap đồng thuận: sản phẩm × phân khúc (đỏ = AI sai nhiều)",
                          height=760)
        st.plotly_chart(fig, width="stretch")

    with t2:
        st.markdown("Agent chỉ kết luận **“mô hình sai thật”** khi: đủ mẫu (≥25 phản hồi) · "
                    "một lý do từ chối *chọn sai khách* (NOT_RELEVANT / CANT_AFFORD / ALREADY_HAS) "
                    "chiếm ≥40% · và nhóm được RM tiếp cận **vẫn không chuyển đổi**. "
                    "Ngược lại → RM thận trọng / sai thời điểm / thiếu tín hiệu → **không sửa**.")
        for _, r in AGR.iterrows():
            badge = "🔴 MÔ HÌNH SAI THẬT" if r.model_is_wrong else "⚪ Không sửa mô hình"
            scope = f" · phân khúc {r.segment}" if r.segment != "(tất cả)" else ""
            with st.expander(f"{badge} — {r.product_code}{scope}  ·  {r.finding_type}"):
                m = st.columns(4)
                m[0].metric("Phản hồi", int(r.n_feedback))
                m[1].metric("% đồng ý", f"{r.agree_rate:.0%}")
                m[2].metric(f"Lý do chính · {r.dominant_verdict}", f"{r.dominant_share:.0%}")
                m[3].metric("Chuyển đổi (đã tiếp cận)", f"{r.conversion_rate_acted:.0%}")
                st.markdown(r.explanation)
                st.code(r.proposed_fix, language="json")

    with t3:
        if ADJ is None or not len(ADJ):
            st.info("Chưa áp hiệu chỉnh nào. Chạy `py src/rm_feedback_agent.py --apply`.")
        else:
            st.dataframe(ADJ[["product_code", "finding_type", "kind", "before", "after",
                              "impact_recos"]], width="stretch", hide_index=True)
            st.caption("Ghi vào `models/rule_overrides.json` — `src/products.py` đọc file này ở "
                       "lần chấm điểm kế tiếp (thắt ngưỡng fit / chặn phân khúc / đặt sàn thu nhập).")
            ovr = MODELS / "rule_overrides.json"
            if ovr.exists():
                with st.expander("models/rule_overrides.json"):
                    st.code(ovr.read_text(encoding="utf-8"), language="json")
        rep = MODELS / "ai_agent_report.md"
        if rep.exists():
            with st.expander("models/ai_agent_report.md"):
                st.markdown(rep.read_text(encoding="utf-8"))

    with t4:
        st.caption("Demo: chọn 1 đề xuất #1 của khách và gửi phản hồi. Lưu tạm trong phiên "
                   "(không ghi vào file) — pipeline thật: append vào `rm_feedback_ai`.")
        if "rm_fb_demo" not in st.session_state:
            st.session_state.rm_fb_demo = []
        cid = st.selectbox("Khách hàng", PRECO.customer_id.drop_duplicates().head(400))
        top = PRECO[(PRECO.customer_id == cid) & (PRECO.priority_rank == 1)]
        if not top.empty:
            x = top.iloc[0]
            st.markdown(f"**AI đề xuất #1:** {x['product_name']} · propensity {x.propensity:.0%} · "
                        f"SGS {x.smart_growth_score:.0f}  \n*{x.message_angle}*")
            with st.form("rm_fb"):
                verdict = st.radio("Phản hồi của RM", [
                    "AGREE — hợp lý, sẽ tiếp cận",
                    "NOT_RELEVANT — KH không phù hợp sản phẩm này",
                    "CANT_AFFORD — chưa đủ khả năng tài chính",
                    "ALREADY_HAS — KH đã có sản phẩm tương đương",
                    "WRONG_TIMING — đúng SP, sai thời điểm",
                    "NO_NEED_NOW — KH chưa quan tâm"], horizontal=False)
                note = st.text_input("Ghi chú")
                if st.form_submit_button("Gửi phản hồi"):
                    st.session_state.rm_fb_demo.insert(0, {
                        "customer_id": cid, "product": x["product_name"],
                        "verdict": verdict.split(" — ")[0], "note": note})
        if st.session_state.rm_fb_demo:
            st.dataframe(pd.DataFrame(st.session_state.rm_fb_demo), width="stretch", hide_index=True)
