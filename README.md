---
title: MSB Smart Growth Engine
emoji: 📈
colorFrom: red
colorTo: gray
sdk: static
pinned: false
license: mit
short_description: AI Customer Intelligence & Sales Growth (stlite demo)
---

# MSB SMART GROWTH ENGINE

*AI-Powered Customer Intelligence & Sales Growth Platform — MSB Hackathon*

Công cụ giúp RM và cấp quản lý của MSB trả lời: **khách hàng nào, nên chào sản phẩm gì,
qua kênh nào, vào lúc nào** — và học lại từ phản hồi thực tế của RM.

- **Customer 360** từ dữ liệu CASA, giao dịch, thẻ, tiền gửi, vay, digital, CRM, campaign.
- **Propensity hybrid** (hồi quy logistic + rule "khách hàng phù hợp") trên **35 sản phẩm MSB**
  chuẩn hoá từ 958 mã trong `MSB_products_description.xlsx`.
- **Smart Growth Score 0–100** → xếp hạng → **Next Best Product / Action / Channel / Timing**.
- **AI Feedback Agent**: RM phản hồi → agent giải thích, phân biệt "AI sai thật" với "RM thận trọng",
  và hiệu chỉnh rule khi cần.
- **Dashboard Streamlit 8 trang**. Bản demo chạy hoàn toàn trong trình duyệt (stlite / WebAssembly)
  trên Hugging Face Static Space, dùng đủ 25.000 khách hàng.

> Dữ liệu **100% giả lập** (numpy, seed cố định). Không có tên, SĐT hay email khách hàng.

Triển khai theo tài liệu **"MSB SMART GROWTH ENGINE V.01"** (`MSB SMART GROWTH ENGINE V.01.docx`).

---

## Mục lục

1. [Kiến trúc & luồng dữ liệu](#1-kiến-trúc--luồng-dữ-liệu)
2. [Dashboard Streamlit](#2-dashboard-streamlit)
3. [Mô hình](#3-mô-hình)
4. [Cài đặt & chạy pipeline](#4-cài-đặt--chạy-pipeline)
5. [Deploy](#5-deploy)
6. [Cấu trúc thư mục](#6-cấu-trúc-thư-mục)
7. [Dữ liệu & tính nhất quán](#7-dữ-liệu--tính-nhất-quán)
8. [Lưu ý](#8-lưu-ý)

---

## 1. Kiến trúc & luồng dữ liệu

```
CUSTOMER DATA → CUSTOMER 360 → AI MODEL (LR + rule) → SCORE → RECOMMENDATION → RM ACTION
                                                                                  │
                                    MODEL ← TRAINING DATA ← RM FEEDBACK / ACTUAL RESULT
```

Hành trình AI gồm **12 Actions**: nhận tín hiệu → phát hiện nhu cầu → *Decision Gate 1*
(có nên tiếp cận?) → Next Best Product → Next Best Action → kênh & thời điểm → GenAI tạo nội dung
→ *Gate 2* (kiểm soát trước khi gửi) → *Gate 3* (auto-send hay RM duyệt) → orchestration
→ khách phản hồi → learning loop.

| Lớp | Bảng |
|---|---|
| Dimension | `dim_customer`, `dim_card`, `dim_rm`, `dim_product`, `dim_campaign`, `dim_product_catalogue` (35 SP) |
| Raw fact | `fact_casa_daily`, `fact_transaction`, `fact_deposit`, `fact_loan`, `fact_insurance`, `fact_digital_activity`, `fact_crm_interaction`, `fact_customer_service`, `fact_campaign`, `fact_customer_product_holding` |
| Aggregate | `agg_customer_transaction`, `fact_card_monthly`, `agg_product_demand`, `agg_product_propensity`, `agg_segment_product_affinity` |
| Feature | `customer_360_feature_mart`, `ai_feature_customer` |
| Training | `ml_propensity_training_set` (X1..X6 + nhãn), `ml_credit_card_training_set` |
| Model output | `ai_customer_score`, `ai_score_reason`, `ai_model_registry`, `ai_model_coefficient`, `ai_model_metric`, `ai_product_fit` |
| Decision | `ai_recommendation`, `ai_product_recommendation_v2` (top-6 SP / khách) |
| Feedback & agent | `rm_action_feedback`, `rm_feedback_ai`, `ai_agent_review`, `ai_model_adjustment` |
| Views | `v_customer_product_eligibility` (Gate 1), `v_top_opportunities`, `v_top_opportunities_v2` |

Schema PostgreSQL: `sql/01_schema.sql`, `sql/05_product_catalogue.sql`, `sql/06_feedback_agent.sql`.

---

## 2. Dashboard Streamlit

App chỉ đọc `data/parquet/`, không cần PostgreSQL.

| Trang | Nội dung |
|---|---|
| 🏠 **Giới thiệu & Mục đích** | Bài toán, 5 vấn đề công cụ giải quyết, kiến trúc hành trình AI |
| 🧭 **Hành trình AI — 12 Actions** | Đi qua từng Action, mỗi bước gắn dữ liệu và biểu đồ thật |
| 🎯 **RM Opportunity Desk** | Top-N cơ hội theo chi nhánh × nhóm × sản phẩm (qua Gate 1 + SGS), tải CSV |
| 👤 **Customer 360** | Hồ sơ khách, mức phù hợp 35 SP, NBP/NBA, "Why this customer" (đóng góp logit) |
| 🛍️ **Sản phẩm & Nhu cầu** | Danh mục 35 SP, dự báo mở mới 30/60/90 ngày, heatmap phân khúc × SP, gợi ý theo khách |
| 🔁 **AI Agent · Feedback & Hiệu chỉnh** | Mức đồng thuận RM, kết luận "mô hình sai thật", các hiệu chỉnh đã áp, form RM gửi feedback |
| 📊 **Manager Intelligence** | KPI danh mục, priority, SGS theo phân khúc, cơ cấu NBP, dự báo, feedback RM, xếp hạng chi nhánh |
| 🤖 **Mô hình Propensity** | Cấu thành propensity (Top-N), **kế hoạch phát triển theo kỳ**, giải thích tham số, metric, hệ số, calibration của LR |

### Tính năng phân tích

- **🔎 Bộ lọc riêng cho từng biểu đồ** (Manager Intelligence, Sản phẩm & Nhu cầu, AI Agent):
  lọc nhiều giá trị (phân khúc, nhóm SP, priority, vùng…), tìm kiếm **không dấu**, Top N,
  nút xoá bộ lọc. Bộ lọc của biểu đồ nào chỉ áp cho biểu đồ đó.
- **Cấu thành propensity**: xem Top N sản phẩm (mặc định 10), xếp theo propensity, fit_score
  hoặc đóng góp của LR, lọc theo nhóm.
- **Hiệu chỉnh tham số kỳ vọng — kế hoạch theo kỳ**: chọn kỳ (tháng/quý/nửa năm) và sản phẩm
  trọng tâm, chỉnh trọng số LR `w`, tỉ lệ chuyển đổi, độ phủ tiếp cận; nhập **uplift chiến dịch**
  và **chỉ tiêu** cho từng kỳ × sản phẩm. App trả về dự báo, % đạt (heatmap), lũy kế so với chỉ tiêu,
  đánh giá *Đạt / Sát chỉ tiêu / Rủi ro* và **uplift cần có để đạt chỉ tiêu**.
- **"Vì sao chọn các tham số này?"**: giải thích từng tham số, kèm số liệu đối chiếu lấy từ dữ liệu.

### Chạy cục bộ

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py        # http://localhost:8501
```

Biến môi trường `APP_MAX_CUST` (mặc định `25000`) giới hạn số khách được nạp. Trên máy ít RAM,
ví dụ Streamlit Community Cloud khoảng 1 GB, đặt `APP_MAX_CUST=10000`.

---

## 3. Mô hình

### 3.1 Product Propensity — hồi quy logistic (doc mục a.)

```
Z = b + w1·X1 + w2·X2 + w3·X3 + w4·X4 + w5·X5 + w6·X6
P = 1 / (1 + e^−Z)
Loss = −[ y·log(p) + (1−y)·log(1−p) ]
```

| Feature | Ý nghĩa | Cách tính (`src/scoring.py`) |
|---|---|---|
| X1 | Monthly Spending | `norm_score(spending_30d)` (PERCENTRANK.INC) |
| X2 | Income | `norm_score(income_monthly)` |
| X3 | Digital Activity | 40%·activeDays30 + 35%·digitalTxn30 + 25%·featureUsage30 |
| X4 | Salary Account | `salary_flag` (1/0) |
| X5 | Campaign Response | mức cao nhất từng đạt trong funnel: 0 / .25 / .5 / .75 / .9 / 1 |
| X6 | Product Gap | mức "còn thiếu & sẵn sàng": tình trạng sở hữu + độ sâu quan hệ (0..1) |

- **1 mô hình cho mỗi sản phẩm mục tiêu**: `P001` thẻ tín dụng, `P004` vay, `P007` tiền gửi, `P009` quỹ mở.
- Nhãn `y_adopt_next_90d` (mở sản phẩm trong 90 ngày tới), train trên **tệp chưa sở hữu**.
- `LogisticRegression` + `StandardScaler`, `class_weight="balanced"`, split 75/25 stratified, CV 5-fold.
- Hệ số quy đổi về thang gốc X ∈ [0,1], kèm odds ratio (`ai_model_coefficient`).

| Sản phẩm | test ROC-AUC | CV-AUC | LogLoss | KS |
|---|--:|--:|--:|--:|
| Thẻ tín dụng (P001) | 0.708 | 0.725 | 0.619 | 0.31 |
| Vay (P004) | 0.687 | 0.693 | 0.639 | 0.27 |
| Tiền gửi (P007) | 0.705 | 0.698 | 0.624 | 0.31 |
| Quỹ mở (P009) | 0.759 | 0.760 | 0.582 | 0.39 |

Train ≈ test ≈ CV (không overfit), calibration đơn điệu (`models/model_report.md`).

> **X6 khác với doc:** doc dùng thang rời rạc 1 / 0.7 / 0 theo "đã có thẻ hay chưa". Nếu làm đúng
> như vậy thì X6 trùng với nhãn "đang sở hữu" ⇒ AUC = 1.0 (rò rỉ nhãn). Ở đây X6 là thang liên tục
> và nhãn là adoption 90 ngày, nên mô hình học được thật.

### 3.2 Propensity hybrid trên 35 sản phẩm

```
P_sp = w · P_LR(nhóm neo) + (1 − w) · fit_score(rule)
```

- **Nhóm neo** (`w = 0.45`): thẻ tín dụng ← P001, tiền gửi ← P007, vay ← P004. Tổng cộng **27 SP**.
- **8 SP còn lại** (CASA, thẻ ghi nợ): `w = 0`, chỉ dùng rule fit.
- `fit_score` là rule "Khách hàng/Nhu cầu phù hợp" kèm hard-gate, map từ danh mục MSB (`src/products.py`).
- **Vì sao hybrid:** LR chỉ có nhãn ở cấp *nhóm*, nên mọi SP cùng nhóm nhận cùng một P_LR.
  Rule phân biệt từng SP (ví dụ Thẻ Travel với Thẻ Family). `w < 0.5` để rule quyết định
  *SP nào hợp với khách*, còn LR điều chỉnh *khách nào dễ chuyển đổi hơn*.

Tham số dự báo cầu (`src/product_analysis.py`): tỉ lệ chuyển đổi nền 90 ngày **14%**, độ phủ tiếp cận
**55%**/quý, phân bổ 30/60/90 ngày là **38% / 70% / 100%**. Đây là **giả định khởi tạo** cho dữ liệu
giả lập; hiệu chỉnh lại theo kết quả chiến dịch thực tế.

### 3.3 Smart Growth Score (0–100)

```
SGS = 25%·ProductPropensity + 20%·CustomerValue + 20%·IntentSignal
    + 15%·Engagement        + 10%·Timing        + 10%·Relationship
```

Phân bậc: ≥90 Very High · 80–89 High · 70–79 Medium · 60–69 Low · <60 Do not prioritize.

**Top-N cơ hội — 2 tầng lọc:**
1. **Decision Gate 1**: khách đang hoạt động, chưa sở hữu SP, có consent, không DNC, không có
   complaint nghiêm trọng trong 15 ngày, không vừa từ chối trong 30 ngày, chưa vượt tần suất liên hệ.
2. Sắp xếp SGS giảm dần trong tệp đã lọc, theo chi nhánh × sản phẩm.

### 3.4 AI Feedback Agent (Action 11–12)

`src/rm_feedback_agent.py`:
- RM chấm từng đề xuất: `AGREE / NOT_RELEVANT / CANT_AFFORD / ALREADY_HAS / WRONG_TIMING / NO_NEED_NOW`.
- Agent chỉ kết luận **`model_is_wrong`** khi đủ 3 điều kiện: đủ mẫu (≥25), lý do "chọn sai khách"
  chiếm ≥40%, và nhóm được RM tiếp cận **vẫn không chuyển đổi**. Các trường hợp khác (RM thận trọng,
  sai thời điểm, nhu cầu mềm) thì **không sửa** mô hình.
- `--apply` ghi `models/rule_overrides.json` (siết `min_fit`, chặn phân khúc, đặt sàn thu nhập).
  Lần chấm điểm sau sẽ đọc file này.

---

## 4. Cài đặt & chạy pipeline

```bash
pip install -r requirements-dev.txt          # pipeline đầy đủ (requirements.txt chỉ dành cho app)

# 1. Sinh dữ liệu + kiểm tra
py src/generate_data.py --customers 25000 --seed 42 --casa-days 45 --digital-days 45
py src/validate_data.py

# 2. Train 4 mô hình LR, ghi score
py src/train_models.py --apply

# 3. Danh mục 35 SP, propensity hybrid, đề xuất, dự báo
py src/product_analysis.py

# 4. Feedback agent → áp hiệu chỉnh → chấm lại
py src/rm_feedback_agent.py --apply
py src/product_analysis.py

# 5. (tuỳ chọn) nạp PostgreSQL 16
docker compose up -d
docker exec -e PGPASSWORD=msb msb_sge_db psql -U msb -d msb_sge -f /sql/01_schema.sql
docker exec -e PGPASSWORD=msb msb_sge_db psql -U msb -d msb_sge -f /sql/02_load.sql
docker exec -e PGPASSWORD=msb msb_sge_db psql -U msb -d msb_sge -f /sql/04_load_models.sql
docker exec -e PGPASSWORD=msb msb_sge_db psql -U msb -d msb_sge -f /sql/03_sample_queries.sql
```

Kết nối DB: `postgresql://msb:msb@localhost:5433/msb_sge` (schema `msb_sge`).
Bộ nhỏ để demo nhanh: `--customers 5000` (khoảng 40 giây). Bộ 25.000 khách mất khoảng 8 phút.
Nếu dữ liệu đã được sinh từ phiên bản cũ, chạy `py src/backfill_product_signals.py` trước bước 3.

---

## 5. Deploy

### Hugging Face Static Space (bản đang dùng)

Hugging Face không còn Streamlit Space miễn phí, nên app được đóng gói bằng
[stlite](https://github.com/whitphx/stlite): Streamlit chạy bằng WebAssembly **trong trình duyệt**
và không cần server. Lần đầu tải khoảng 55 MB (Pyodide + thư viện + dữ liệu), các lần sau dùng cache.

```powershell
py -m pip install -U huggingface_hub
hf auth login                                   # token quyền Write

# build stlite/site/ (index.html + app + data .csv.gz) rồi upload lên Space
py stlite/build.py --upload danh30012002/msb-smart-growth-engine -m "update"
```

Demo: https://huggingface.co/spaces/danh30012002/msb-smart-growth-engine

- **Mỗi lần sửa `streamlit_app/app.py` hoặc dữ liệu** đều phải build lại và upload lại.
  `git push` lên GitHub **không** cập nhật Space.
- Nếu Space vẫn hiện bản cũ, nhấn Ctrl+Shift+R.
- Chi tiết và cách xử lý lỗi: [HUGGINGFACE.md](HUGGINGFACE.md).

### Streamlit Community Cloud

Trỏ tới `streamlit_app/app.py` và dùng `streamlit_app/requirements.txt`. Đặt `APP_MAX_CUST=10000`
trong Secrets để tránh lỗi hết RAM.

---

## 6. Cấu trúc thư mục

```
sql/
  01_schema.sql              DDL schema msb_sge + view eligibility / top-N
  02_load.sql                \copy CSV -> bảng
  03_sample_queries.sql      5 bài toán · 12 Actions · Top-N · mô hình LR
  04_load_models.sql         nạp ai_model_* + score/recommendation
  05_product_catalogue.sql   danh mục 35 SP + bảng phân tích sản phẩm
  06_feedback_agent.sql      bảng RM feedback / agent review / adjustment
src/
  generate_data.py           sinh dữ liệu giả lập theo archetype
  scoring.py                 công thức X1..X6, các điểm thành phần, Smart Growth Score
  train_models.py            hồi quy logistic cho từng SP + metric + artefact
  validate_data.py           kiểm tra PK/FK/domain/feature/nhãn
  products.py                chuẩn hoá 958 mã → 35 SP + rule "khách hàng phù hợp"
  product_analysis.py        propensity hybrid, NBP/NBA, dự báo cầu, cấu thành propensity
  rm_feedback_agent.py       AI Feedback Agent (review + hiệu chỉnh)
  backfill_product_signals.py  bổ sung cờ tín hiệu cho feature mart cũ
streamlit_app/app.py         dashboard 8 trang
stlite/build.py              đóng gói site tĩnh cho Hugging Face
models/                      model_report.md, metrics.json, ROC/calibration *.png,
                             product_analysis_report.md, ai_agent_report.md, rule_overrides.json
data/csv/  data/parquet/     1 file / bảng
stlite/site/                 site tĩnh do build.py sinh ra (không commit, upload lên Space)
docker-compose.yml           PostgreSQL 16
```

---

## 7. Dữ liệu & tính nhất quán

Mỗi khách có một **archetype ẩn** (`HIGH_VALUE`, `CREDIT_OPPORTUNITY`, `POTENTIAL_INVESTOR`,
`DIGITAL_ACTIVE`, `CHURN_RISK`, `STANDARD`). Archetype này chi phối toàn bộ chuỗi: fact → feature mart
→ nhãn → mô hình → đề xuất → feedback. Nhờ vậy mô hình học lại được tín hiệu thật:

- `CREDIT_OPPORTUNITY`: thu nhập ổn định, chưa có thẻ cao cấp, có giao dịch ô tô → NBP là thẻ hoặc vay.
- `POTENTIAL_INVESTOR`: CASA tăng, có tiền gửi sắp đáo hạn → NBP là tiền gửi hoặc đầu tư.
- `CHURN_RISK`: số dư và hoạt động digital giảm, nhiều complaint → bị loại ở Gate 1.

| Bảng | Số dòng | | Bảng | Số dòng |
|---|---:|---|---|---:|
| dim_customer | 25.000 | | fact_campaign | 75.062 |
| fact_casa_daily | 1.102.725 | | customer_360_feature_mart | 25.000 |
| fact_transaction | 1.085.231 | | ai_customer_score | 100.000 |
| fact_digital_activity | 459.632 | | ai_product_recommendation_v2 | ~143.000 |
| fact_deposit / loan / insurance | 12.880 / 8.139 / 6.746 | | rm_action_feedback | 9.001 |
| fact_crm_interaction / customer_service | 41.809 / 29.420 | | ml_propensity_training_set | 100.000 |

---

## 8. Lưu ý

- Dữ liệu giả lập hoàn toàn, tái lập được nhờ seed cố định (`--seed 42`).
- `dim_customer` không lưu thông tin định danh, chỉ có thuộc tính phân khúc đã ẩn danh.
- Các file lớn (`fact_casa_daily`, `fact_transaction`, `fact_digital_activity`) dùng **Git LFS**
  với `fetchexclude = *`, nên clone về sẽ không tự tải. Khi cần, chạy `git lfs pull`. App không cần các file này.
- `ai_model_*` chỉ có sau khi chạy `train_models.py --apply`.
- Tham số kinh doanh (tỉ lệ chuyển đổi, độ phủ, ngưỡng) là giá trị khởi tạo cho demo và cần hiệu chỉnh
  theo dữ liệu thật trước khi dùng cho kế hoạch.
