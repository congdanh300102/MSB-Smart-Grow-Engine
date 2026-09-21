# Phân tích mức độ phù hợp & dự báo cầu sản phẩm — MSB Smart Growth Engine

*Ngày phân tích: 2026-08-31 · 25,000 khách hàng · 35 sản phẩm chuẩn hoá (từ 958 mã trong MSB_products_description.xlsx)*

## 1. Danh mục sản phẩm chuẩn hoá

| Nhóm | Số SP | KH đang sở hữu (mô phỏng) |
|---|--:|--:|
| CARD | 11 | 24,011 |
| CASA | 5 | 38,885 |
| FD | 8 | 10,819 |
| LENDING | 11 | 14,630 |

## 2. Sản phẩm được đề xuất #1 nhiều nhất (Next Best Product)

| Sản phẩm | Số KH đứng #1 | % KH |
|---|--:|--:|
| CARD_DEBIT_DOMESTIC | 8,594 | 34.4% |
| CARD_DEBIT_INTL_VISA | 4,950 | 19.8% |
| CASA_PAYROLL | 3,991 | 16.0% |
| CASA_BIZ_MSMART | 1,340 | 5.4% |
| CARD_DEBIT_PLATINUM_FCB | 996 | 4.0% |
| CASA_ESCROW | 679 | 2.7% |
| CARD_CC_ONLINE | 636 | 2.5% |
| FD_TD_STANDARD | 576 | 2.3% |
| CARD_CC_FAMILY | 500 | 2.0% |
| CASA_M_FIRST | 419 | 1.7% |
| LEND_CONSUMER | 356 | 1.4% |
| LEND_UNSECURED_PAYROLL | 320 | 1.3% |

## 3. Dự báo cầu 90 ngày (tệp đủ điều kiện, chưa sở hữu, có tiếp cận)

*Giả định: tỉ lệ chuyển đổi nền 14% trên tệp được tiếp cận, năng lực tiếp cận 55% tệp đủ điều kiện/quý.*

| Sản phẩm | Nhóm | Tệp đủ điều kiện | Propensity TB | Dự báo mở mới 90N |
|---|---|--:|--:|--:|
| CARD_DEBIT_DOMESTIC | CARD | 21,109 | 0.81 | **1,304** |
| FD_TD_STANDARD | FD | 19,543 | 0.67 | **984** |
| FD_TD_ONLINE | FD | 20,273 | 0.64 | **974** |
| FD_SAV_UPFRONT | FD | 19,552 | 0.62 | **896** |
| FD_SAV_PARTIAL | FD | 19,114 | 0.62 | **876** |
| CARD_DEBIT_INTL_VISA | CARD | 12,679 | 0.85 | **804** |
| LEND_CONSUMER | LENDING | 13,581 | 0.70 | **703** |
| CARD_CC_MDIGI | CARD | 11,822 | 0.75 | **661** |
| LEND_OVERDRAFT_PERSONAL | LENDING | 15,458 | 0.56 | **641** |
| FD_SAV_MAXRATE | FD | 13,329 | 0.62 | **636** |
| LEND_UNSECURED_PAYROLL | LENDING | 10,830 | 0.69 | **573** |
| FD_CD | FD | 11,645 | 0.63 | **570** |
| CARD_CC_ONLINE | CARD | 8,470 | 0.84 | **542** |
| CASA_PAYROLL | CASA | 8,858 | 0.79 | **531** |
| CARD_CC_FAMILY | CARD | 8,567 | 0.82 | **512** |
| FD_SAV_PERIODIC | FD | 9,503 | 0.62 | **463** |
| CARD_DEBIT_PLATINUM_FCB | CARD | 6,837 | 0.76 | **432** |
| CARD_CC_SIGNATURE | CARD | 7,172 | 0.76 | **422** |
| CASA_M_FIRST | CASA | 7,528 | 0.74 | **418** |
| CASA_BIZ_MSMART | CASA | 5,030 | 0.88 | **336** |

**Tổng dự báo mở mới 90 ngày: ~16,178 sản phẩm** (30N: ~6,147, 60N: ~11,324).

## 4. Phân khúc phù hợp nhất theo nhóm sản phẩm

| Phân khúc | CARD | CASA | FD | LENDING |
|---|--:|--:|--:|--:|
| AFFLUENT | 0.72 | 0.55 | 0.64 | 0.55 |
| MASS | 0.45 | 0.39 | 0.42 | 0.38 |
| MASS_AFFLUENT | 0.55 | 0.48 | 0.53 | 0.47 |
| PRIVATE | 0.81 | 0.59 | 0.69 | 0.60 |
| SME | 0.65 | 0.70 | 0.58 | 0.58 |

## 5. Khoảng trống danh mục (portfolio gap)

- 24,771 / 25,000 khách (100%) có ít nhất 1 **nhóm sản phẩm** phù hợp (propensity ≥ 0.6) nhưng chưa sở hữu.
- Trung bình 2.7 nhóm whitespace / khách (tối đa 4: CARD / CASA / FD / LENDING).
- Cơ hội cross-sell theo nhóm: CARD (22,438 khách), FD (17,302 khách), CASA (15,432 khách), LENDING (12,100 khách).
- Tổng cơ hội (khách × sản phẩm, propensity ≥ 0.6): 211,942.

## 6. Hệ số mùa vụ & ưu tiên kinh doanh đang áp dụng

*Quý áp dụng: **Q3** (theo AS_OF 2026-08-31) · nguồn: `models/priority_config.json` (chưa hiệu chỉnh — dùng hệ số minh hoạ mặc định)*

| Nhóm sản phẩm | Hệ số mùa vụ (Q3) |
|---|--:|
| CARD | 1.10× |
| CASA | 1.00× |
| FD | 0.95× |
| LENDING | 1.00× |

_Chưa có sản phẩm nào được đặt hệ số ưu tiên khác 1.0._

Cả 2 hệ số chỉ nhân vào **propensity** (ảnh hưởng thứ hạng đề xuất & dự báo cầu), không đổi `fit_score`/điều kiện đủ điều kiện sản phẩm. Hiệu chỉnh qua trang Streamlit *AI Agent · Feedback & Tuning → Ưu tiên kinh doanh & Mùa vụ*, rồi chạy lại `py src/product_analysis.py` để áp dụng.

## 7. Cách dùng

- `ai_product_recommendation_v2` — top-6 sản phẩm phù hợp nhất mỗi khách, kèm lý do.
- `agg_product_demand` — dự báo cầu theo sản phẩm × phân khúc, đưa vào kế hoạch KPI/chiến dịch.
- `agg_segment_product_affinity` — ma trận phân khúc × sản phẩm cho định hướng danh mục.
- App Streamlit: trang **“Sản phẩm & Nhu cầu”**.