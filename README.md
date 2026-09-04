# MSB SMART GROWTH ENGINE — Data Model, Synthetic Data & Propensity Models

*AI-Powered Customer Intelligence & Sales Growth Platform — MSB Hackathon*

Triển khai theo tài liệu **"MSB SMART GROWTH ENGINE V.01"**:
- **I. Thiết kế Data Table** → schema PostgreSQL (`sql/01_schema.sql`, 27 bảng + 2 view)
- **2. Thiết kế hành trình AI** (12 Actions) → query mẫu (`sql/03_sample_queries.sql`, phần B)
- **a. Product Propensity** → **hồi quy logistic** 1 model/sản phẩm (`src/train_models.py`)
- **b–g. Customer Value / Intent / Engagement / Timing / Relationship** → công thức trọng số (`src/scoring.py`)
- **Smart Growth Score 0–100 → RANKING → TOP 20 → Next Best Action**

## Luồng dữ liệu

```
CUSTOMER DATA → CUSTOMER 360 → AI MODEL (logistic reg.) → SCORE → RECOMMENDATION → RM ACTION
                                                                                     │
                                                          ACTUAL RESULT → TRAINING DATA → MODEL
```

| Lớp | Bảng |
|---|---|
| Dimension | `dim_customer`, `dim_card`, `dim_rm`, `dim_product`, `dim_campaign` |
| Raw fact | `fact_casa_daily`, `fact_transaction`, `fact_deposit`, `fact_loan`, `fact_insurance`, `fact_digital_activity`, `fact_crm_interaction`, `fact_customer_service`, `fact_campaign` |
| Aggregate | `agg_customer_transaction`, `fact_card_monthly` |
| Feature | `customer_360_feature_mart` (48 cột), `ai_feature_customer` |
| Training | `ml_propensity_training_set` (X1..X6 + nhãn), `ml_credit_card_training_set` |
| Model output | `ai_customer_score`, `ai_score_reason`, `ai_model_registry`, `ai_model_coefficient`, `ai_model_metric` |
| Decision | `ai_recommendation` (Next Best Product/Action/Channel/Timing) |
| Feedback | `rm_action_feedback` |
| Journey views | `v_customer_product_eligibility` (Decision Gate 1), `v_top_opportunities` (Top-20) |

## Product Propensity — hồi quy logistic (doc "a.")

```
Z = b + w1·X1 + w2·X2 + w3·X3 + w4·X4 + w5·X5 + w6·X6
P = 1 / (1 + e^−Z)                       (sigmoid)
Loss = −[ y·log(p) + (1−y)·log(1−p) ]    (binary cross-entropy)
```

| Feature | Ý nghĩa (doc) | Cách tính trong `src/scoring.py` |
|---|---|---|
| X1 | Monthly Spending Score | `norm_score(spending_30d)` (PERCENTRANK.INC) |
| X2 | Income Score | `norm_score(income_monthly)` |
| X3 | Digital Activity Score | 40%·activeDays30 + 35%·digitalTxn30 + 25%·featureUsage30 (đều norm) |
| X4 | Salary Account | `salary_flag` (1/0) |
| X5 | Campaign Response | funnel cao nhất từng đạt: 0 / .25 / .5 / .75 / .9 / 1 |
| X6 | Product Gap | mức "còn thiếu & sẵn sàng": tình trạng sở hữu + độ sâu quan hệ (0..1) |

- **1 model / sản phẩm mục tiêu**: `P001` Platinum Credit Card, `P004` Vay mua nhà, `P007` Tiền gửi, `P009` Quỹ mở.
- Nhãn train mặc định = `y_adopt_next_90d` (mở sản phẩm trong 90 ngày tới), train trên **tệp chưa sở hữu**
  (điều kiện "target_product = 0" của doc). `y_holds_product` cũng có sẵn để phân tích / đổi nhãn.
- `sklearn` `LogisticRegression` + `StandardScaler`, `class_weight="balanced"`, split 75/25 stratified, 5-fold CV.
- Hệ số được **quy đổi về thang gốc X∈[0,1]** để khớp trực tiếp công thức `Z = b + Σ wᵢ·Xᵢ` và lưu vào
  `ai_model_coefficient` (kèm odds ratio). Metric test lưu vào `ai_model_metric`.
- `P × 100` = `product_propensity_score`, đưa vào Smart Growth Score.

> **X6 — khác với doc:** doc quy định thang rời rạc 1 / 0.7 / 0 theo "đã có thẻ hay chưa".
> Nếu dùng đúng vậy và train nhãn "đang sở hữu" thì X6 ≡ nhãn ⇒ AUC = 1.0 (rò rỉ, model vô nghĩa).
> Ở đây X6 = thang liên tục "còn thiếu sản phẩm + độ sâu quan hệ", nhãn = adoption 90 ngày ⇒ model học thật.

## Smart Growth Score (0–100)

```
SGS = 25%·ProductPropensity + 20%·CustomerValue + 20%·IntentSignal
     + 15%·Engagement       + 10%·Timing        + 10%·Relationship
```
Bucket: ≥90 Very High · 80–89 High · 70–79 Medium · 60–69 Low · <60 Do not prioritize.
5 khối business là **công thức trọng số tất định** theo doc mục b–g (`src/scoring.py`), chuẩn hoá bằng
PERCENTRANK.INC (doc mục a. cho phép, hợp dữ liệu ngân hàng lệch phải).

## Top-20 khách hàng — 2 tầng lọc

1. **Decision Gate 1** (`v_customer_product_eligibility`): `customer_active`, chưa sở hữu sản phẩm mục tiêu,
   `marketing_consent`, `do_not_contact=0`, `serious_complaint_15d=0`, `recent_rejection_30d=0`,
   `contact_frequency` chưa vượt (`contact_count_7d < 2`).
2. **Sort Smart Growth Score DESC → Top 20** trong tệp đã lọc, theo từng chi nhánh × sản phẩm (`v_top_opportunities`).

## Cấu trúc thư mục

```
sql/
  01_schema.sql          DDL (schema msb_sge): dim / fact / aggregate / feature / training /
                         model-output / decision / feedback + view eligibility & top-20
  02_load.sql            \copy CSV -> bảng (cột liệt kê tường minh theo thứ tự CSV)
  03_sample_queries.sql  A. 5 bài toán  B. 12 Actions  C. Top-20  D. Logistic-regression model
  04_load_models.sql     nạp ai_model_* + ai_customer_score/recommendation sau train_models --apply
src/
  generate_data.py       Sinh dữ liệu giả lập theo archetype (numpy/pandas, offline)
  scoring.py             Công thức doc: X1..X6 + 5 business sub-scores + Smart Growth Score
  train_models.py        Hồi quy logistic 1 model/sản phẩm + test + artefacts (+ --apply)
  validate_data.py       PK/FK/domain/priority-bucket + kiểm tra feature X∈[0,1], nhãn nhị phân
streamlit_app/app.py     App trực quan 6 trang theo 12 Actions (đọc data/parquet/, không cần DB)
models/                  propensity_<sp>.joblib, metrics.json, model_report.md, roc/calibration *.png
data/csv/*.csv  data/parquet/*.parquet     1 file / bảng
docker-compose.yml       PostgreSQL 16
```

## App trực quan (Streamlit)

```bash
pip install -r requirements.txt        # + streamlit, plotly
streamlit run streamlit_app/app.py     # http://localhost:8501
```

6 trang: **Giới thiệu & Mục đích** · **Hành trình AI — 12 Actions** (slider từng bước, chart dữ liệu thật) ·
**RM Opportunity Desk** (Top-N/chi nhánh/sản phẩm) · **Customer 360** (điểm + "Why this customer") ·
**Manager Intelligence** (KPI danh mục, feedback loop) · **Mô hình Propensity** (hệ số, metric, calibration).
Chi tiết: [streamlit_app/README.md](streamlit_app/README.md).

## Chạy

```bash
pip install -r requirements.txt

# 1. sinh dữ liệu (mặc định 5.000 KH; --customers 20000 cho bộ lớn)
py src/generate_data.py --customers 20000 --seed 42 --casa-days 45 --digital-days 45
py src/validate_data.py

# 2. train + test 4 model hồi quy logistic, ghi score vào CSV
py src/train_models.py --apply
#   -> models/model_report.md , models/metrics.json , data/csv/ai_model_*.csv

# 3. nạp PostgreSQL
docker compose up -d
docker exec -e PGPASSWORD=msb msb_sge_db psql -U msb -d msb_sge -f /sql/01_schema.sql
docker exec -e PGPASSWORD=msb msb_sge_db psql -U msb -d msb_sge -f /sql/02_load.sql
docker exec -e PGPASSWORD=msb msb_sge_db psql -U msb -d msb_sge -f /sql/04_load_models.sql
docker exec -e PGPASSWORD=msb msb_sge_db psql -U msb -d msb_sge -f /sql/03_sample_queries.sql
```

Kết nối: `postgresql://msb:msb@localhost:5433/msb_sge` (schema `msb_sge`).
Docker Desktop đang pause/tắt → `docker desktop restart` rồi `docker compose up -d`.

## Khối lượng dữ liệu (seed 42, 25.000 KH, casa/digital 45 ngày)

| Bảng | Số dòng | | Bảng | Số dòng |
|---|---:|---|---|---:|
| dim_customer | 25.000 | | fact_campaign | 75.062 |
| dim_card | 21.072 | | customer_360_feature_mart | 25.000 |
| fact_casa_daily | 1.102.725 | | ai_feature_customer | 25.000 |
| fact_transaction | 1.085.231 | | ai_customer_score | 100.000 (25k × 4 sp) |
| agg_customer_transaction | 25.000 | | ai_score_reason | 115.753 |
| fact_card_monthly | 59.862 | | ai_recommendation | 75.000 (top-3/KH) |
| fact_deposit / loan / insurance | 12.880 / 8.139 / 6.746 | | rm_action_feedback | 9.001 |
| fact_digital_activity | 459.632 | | ml_propensity_training_set | 100.000 |
| fact_crm_interaction / customer_service | 41.809 / 29.420 | | ml_credit_card_training_set | 25.000 |

Sinh mất ~8 phút; nạp Postgres qua bind-mount `\copy` ~8–10 phút (chủ yếu 3 bảng theo ngày).
Bộ nhẹ để demo nhanh: `--customers 5000` (~40s sinh).

## Tính nhất quán

Mỗi KH có **archetype ẩn** (`HIGH_VALUE`, `CREDIT_OPPORTUNITY`, `POTENTIAL_INVESTOR`, `DIGITAL_ACTIVE`,
`CHURN_RISK`, `STANDARD`) chi phối toàn bộ facts → feature mart → nhãn adoption → model → recommendation →
feedback. Nhờ đó:
- Nhãn `y_adopt_next_90d` sinh từ một **DGP logistic thật** trên X1..X6 ⇒ model hồi quy học lại được:

  | Sản phẩm | test ROC-AUC | CV-AUC | LogLoss | KS |
  |---|--:|--:|--:|--:|
  | Platinum Credit Card | 0.708 | 0.725 | 0.619 | 0.31 |
  | Vay mua nhà | 0.687 | 0.693 | 0.639 | 0.27 |
  | Tiền gửi kỳ hạn | 0.705 | 0.698 | 0.624 | 0.31 |
  | Quỹ mở | 0.759 | 0.760 | 0.582 | 0.39 |

  train ≈ test ≈ CV (không overfit). Calibration đơn điệu: decile P dự đoán 0.16→0.92 khớp
  adoption thực 0.08→0.88 (xem `models/model_report.md`, `models/calibration_*.png`, hoặc query D3).
- `CREDIT_OPPORTUNITY`: thu nhập ổn định, chưa có thẻ Platinum, có giao dịch `AUTOMOTIVE` → NBP Credit Card / Loan.
- `POTENTIAL_INVESTOR`: CASA tăng, tiền gửi `MATURING_SOON` → NBP Deposit / Investment.
- `CHURN_RISK`: số dư & digital giảm dần, nhiều complaint, `sales_suppression_flag` cao → loại ở Gate 1.
- Smart Growth Score (công thức doc, không boost): mean ~50, ~0.4% KH ≥85, ~1.4% ≥80 —
  phân bố "phễu cơ hội" hẹp. Top-20/chi nhánh vẫn chạy tốt vì `v_top_opportunities` xếp hạng
  *trong tệp đủ điều kiện*, không phụ thuộc ngưỡng tuyệt đối.

## Lưu ý

- Dữ liệu 100% giả lập bằng `numpy` (seed cố định), không dùng dữ liệu thật.
- `dim_customer` **không lưu tên/SĐT/email** — chỉ thuộc tính phân khúc/nhân khẩu học đã ẩn danh.
- `\copy` liệt kê cột tường minh vì COPY ánh xạ theo **vị trí cột**, không theo tên.
- `ai_model_*` chỉ có sau khi chạy `train_models.py --apply`; `02_load.sql` không phụ thuộc chúng.
- **`data/csv/` được `.gitignore`** (raw CSV ~500MB, `fact_casa_daily.csv` >100MB vượt giới hạn GitHub).
  Repo giữ bản **Parquet**. Sinh lại CSV để chạy `02_load.sql`: `py src/generate_data.py --customers 25000 --seed 42 --casa-days 45 --digital-days 45` (tái lập chính xác nhờ seed cố định).
