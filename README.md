# MSB SMART GROWTH ENGINE — Data Model & Synthetic Data

*AI-Powered Customer Intelligence & Sales Growth Platform — MSB Hackathon*

Nền tảng biến dữ liệu khách hàng phân mảnh thành **cơ hội bán hàng có thể hành động ngay**.
Repo này chứa **mô hình dữ liệu Customer‑360** (PostgreSQL) và **bộ dữ liệu giả lập** 5.000 khách hàng,
được thiết kế để trả lời trực tiếp 5 bài toán trong đề bài:

| # | Bài toán | Trả lời bằng |
|---|----------|--------------|
| 1 | Khách hàng nào nên ưu tiên tiếp cận? | `ai_customer_score.next_best_action_score` |
| 2 | Khách quan tâm sản phẩm nào? | `ai_recommendation` (type `NEXT_BEST_PRODUCT`, top‑3) |
| 3 | Xác suất phản hồi / chuyển đổi? | `propensity_to_buy_score`, `ai_recommendation.confidence_score` |
| 4 | Thời điểm & kênh phù hợp? | `recommendation_text` (timing) + `fact_campaign.campaign_channel` |
| 5 | Sale nên nói / làm gì tiếp theo? | `recommendation_text` + `rm_action_feedback` |

## Kiến trúc mô hình (6 lớp)

```
DIM_CUSTOMER                         ── customer master
   │  1:N
   ├─ FACT_CASA_DAILY                 ┐
   ├─ FACT_TRANSACTION                │
   ├─ FACT_CARD                       │
   ├─ FACT_LOAN                       │  Lớp nguồn thô (fragmented sources)
   ├─ FACT_DEPOSIT                    │  Hồ sơ / CASA / Giao dịch / Thẻ / Vay /
   ├─ FACT_INSURANCE                  │  Tiền gửi / Bảo hiểm / Digital / CRM /
   ├─ FACT_DIGITAL_ACTIVITY           │  Customer Service / Campaign
   ├─ FACT_CRM_INTERACTION            │
   ├─ FACT_CUSTOMER_SERVICE           │
   └─ FACT_CAMPAIGN ──────────────────┘
          │ 1:N (campaign gần nhất)
          ▼
   CUSTOMER_360_FEATURE_MART          ── feature store (Customer 360, 1 dòng / KH / snapshot)
          │ 1:N
          ▼
   AI_CUSTOMER_SCORE                  ── churn / propensity_to_buy / credit_risk / next_best_action
          │ 1:N
          ▼
   AI_RECOMMENDATION                  ── Next Best Product xếp hạng + kịch bản tư vấn
          │ 1:N
          ▼
   RM_ACTION_FEEDBACK                 ── vòng phản hồi của RM (Interested / Converted ...)
```

> Ghi chú so với ERD gốc: `CUSTOMER_360_FEATURE_MART` được bổ sung `customer_key`
> (khoá KH sở hữu) bên cạnh `campaign_key` như trong ERD, vì feature mart là **1 dòng / khách hàng / snapshot**.

## Cấu trúc thư mục

```
sql/
  01_schema.sql          DDL PostgreSQL (schema msb_sge, 15 bảng, PK/FK, index)
  02_load.sql            \copy CSV -> bảng (theo đúng thứ tự FK)
  03_sample_queries.sql  6 truy vấn mẫu cho 5 bài toán + phân khúc động
src/
  generate_data.py       Sinh dữ liệu giả lập (numpy/pandas, không cần internet)
  validate_data.py       Kiểm tra PK/FK/precision/domain trước khi nạp DB
data/
  csv/*.csv              15 file, 1 file / bảng  (UTF-8, có header)
  parquet/*.parquet      Bản Parquet tương ứng (cho pipeline ML/Spark)
docker-compose.yml       PostgreSQL 16 để nạp & truy vấn nhanh
```

## Chạy nhanh

### 1. Sinh lại dữ liệu (tuỳ chọn — dữ liệu đã có sẵn trong `data/`)

```bash
pip install -r requirements.txt
py src/generate_data.py --customers 5000 --seed 42
py src/validate_data.py
```

`--customers` tối đa khuyến nghị 10000. Seed cố định ⇒ dữ liệu tái lập được.

### 2. Nạp vào PostgreSQL bằng Docker

```bash
docker compose up -d
docker exec -e PGPASSWORD=msb msb_sge_db psql -U msb -d msb_sge -f /sql/01_schema.sql
docker exec -e PGPASSWORD=msb msb_sge_db psql -U msb -d msb_sge -f /sql/02_load.sql
docker exec -e PGPASSWORD=msb msb_sge_db psql -U msb -d msb_sge -f /sql/03_sample_queries.sql
```

Kết nối từ ngoài: `postgresql://msb:msb@localhost:5433/msb_sge` (schema `msb_sge`).

### 3. Dùng trực tiếp bằng Python / pandas (không cần DB)

```python
import pandas as pd
cust = pd.read_parquet("data/parquet/dim_customer.parquet")
score = pd.read_parquet("data/parquet/ai_customer_score.parquet")
```

## Khối lượng dữ liệu (seed 42, 5.000 KH)

| Bảng | Số dòng |
|------|--------:|
| dim_customer | 5.000 |
| fact_casa_daily | ~436.000 (90 ngày/KH) |
| fact_transaction | ~218.000 (~90 ngày) |
| fact_card | ~4.500 |
| fact_loan | ~1.500 |
| fact_deposit | ~2.600 |
| fact_insurance | ~1.100 |
| fact_digital_activity | ~23.000 (6 tháng, tổng hợp theo tháng) |
| fact_crm_interaction | ~8.100 |
| fact_customer_service | ~5.800 |
| fact_campaign | ~12.400 |
| customer_360_feature_mart | 5.000 |
| ai_customer_score | 5.000 |
| ai_recommendation | 15.000 (3 / KH) |
| rm_action_feedback | ~2.700 |

## Tính nhất quán của dữ liệu giả lập

Mỗi khách hàng được gán một **archetype ẩn** (`HIGH_VALUE`, `CREDIT_OPPORTUNITY`,
`POTENTIAL_INVESTOR`, `DIGITAL_ACTIVE`, `CHURN_RISK`, `STANDARD`) chi phối **toàn bộ** dữ liệu
xuống dưới, nên các mô hình học được tín hiệu thật:

- `CHURN_RISK` → số dư CASA giảm dần, giao dịch ít, tương tác digital suy giảm theo tháng,
  nhiều case `RETENTION`, `churn_score` cao.
- `CREDIT_OPPORTUNITY` → thu nhập/chi tiêu ổn định, **chưa có thẻ tín dụng cao cấp**,
  utilization thấp, có giao dịch `Automotive` → `AI_RECOMMENDATION` ưu tiên Thẻ / Vay.
- `POTENTIAL_INVESTOR` → CASA tăng, có **tiền gửi sắp đáo hạn** (`MATURING_SOON`),
  ít sản phẩm đầu tư → đề xuất Tiền gửi / Đầu tư.
- `DIGITAL_ACTIVE` → login/app_open cao → phù hợp digital cross‑sell.

Kiểm chứng nhanh (từ `validate_data.py`):
`corr(digital_engagement, propensity_to_buy) ≈ +0.44`,
`Converted` gắn với `confidence_score ≈ 0.80` còn `Not interested ≈ 0.37`.

## Lưu ý

Toàn bộ dữ liệu là **giả lập** bằng `numpy` (seed cố định), **không dùng dữ liệu thật của ngân hàng**.
Tên khách hàng, ID, số dư… đều được sinh ngẫu nhiên.
