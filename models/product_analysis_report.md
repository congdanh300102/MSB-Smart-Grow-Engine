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
| CARD_DEBIT_DOMESTIC | 6,203 | 24.8% |
| CASA_PAYROLL | 4,656 | 18.6% |
| CARD_DEBIT_INTL_VISA | 3,777 | 15.1% |
| CASA_BIZ_MSMART | 1,937 | 7.7% |
| CARD_DEBIT_PLATINUM_FCB | 1,321 | 5.3% |
| FD_TD_STANDARD | 1,138 | 4.6% |
| CASA_ESCROW | 864 | 3.5% |
| FD_SAV_KIDS | 740 | 3.0% |
| FD_SAV_UPFRONT | 572 | 2.3% |
| FD_TD_ONLINE | 537 | 2.1% |
| CASA_M_FIRST | 515 | 2.1% |
| LEND_CONSUMER | 391 | 1.6% |

## 3. Dự báo cầu 90 ngày (tệp đủ điều kiện, chưa sở hữu, có tiếp cận)

*Giả định: tỉ lệ chuyển đổi nền 14% trên tệp được tiếp cận, năng lực tiếp cận 55% tệp đủ điều kiện/quý.*

| Sản phẩm | Nhóm | Tệp đủ điều kiện | Propensity TB | Dự báo mở mới 90N |
|---|---|--:|--:|--:|
| CARD_DEBIT_DOMESTIC | CARD | 21,109 | 0.74 | **1,199** |
| FD_TD_STANDARD | FD | 19,543 | 0.71 | **1,036** |
| FD_TD_ONLINE | FD | 20,273 | 0.68 | **1,025** |
| FD_SAV_UPFRONT | FD | 19,552 | 0.65 | **943** |
| FD_SAV_PARTIAL | FD | 19,114 | 0.66 | **922** |
| LEND_UNSECURED_PAYROLL | LENDING | 15,274 | 0.68 | **767** |
| LEND_AUTO_NEW | LENDING | 15,438 | 0.69 | **761** |
| CARD_DEBIT_INTL_VISA | CARD | 12,679 | 0.80 | **756** |
| LEND_CONSUMER | LENDING | 13,581 | 0.70 | **703** |
| FD_SAV_KIDS | FD | 12,697 | 0.72 | **683** |
| FD_SAV_MAXRATE | FD | 13,329 | 0.66 | **670** |
| LEND_OVERDRAFT_PERSONAL | LENDING | 15,458 | 0.56 | **641** |
| CARD_CC_MDIGI | CARD | 11,822 | 0.68 | **602** |
| FD_CD | FD | 11,645 | 0.67 | **600** |
| CASA_PAYROLL | CASA | 8,858 | 0.79 | **531** |
| CARD_CC_ONLINE | CARD | 8,470 | 0.76 | **495** |
| FD_SAV_PERIODIC | FD | 9,503 | 0.65 | **488** |
| CARD_CC_FAMILY | CARD | 8,567 | 0.75 | **467** |
| CASA_M_FIRST | CASA | 7,528 | 0.74 | **418** |
| CARD_DEBIT_PLATINUM_FCB | CARD | 6,837 | 0.70 | **409** |

**Tổng dự báo mở mới 90 ngày: ~17,077 sản phẩm** (30N: ~6,490, 60N: ~11,953).

## 4. Phân khúc phù hợp nhất theo nhóm sản phẩm

| Phân khúc | CARD | CASA | FD | LENDING |
|---|--:|--:|--:|--:|
| AFFLUENT | 0.66 | 0.55 | 0.67 | 0.55 |
| MASS | 0.41 | 0.39 | 0.45 | 0.38 |
| MASS_AFFLUENT | 0.50 | 0.48 | 0.55 | 0.47 |
| PRIVATE | 0.74 | 0.59 | 0.72 | 0.60 |
| SME | 0.60 | 0.70 | 0.61 | 0.58 |

## 5. Khoảng trống danh mục (portfolio gap)

- 24,772 / 25,000 khách (100%) có ít nhất 1 **nhóm sản phẩm** phù hợp (propensity ≥ 0.6) nhưng chưa sở hữu.
- Trung bình 2.8 nhóm whitespace / khách (tối đa 4: CARD / CASA / FD / LENDING).
- Cơ hội cross-sell theo nhóm: CARD (21,497 khách), FD (19,058 khách), CASA (15,432 khách), LENDING (12,702 khách).
- Tổng cơ hội (khách × sản phẩm, propensity ≥ 0.6): 225,150.

## 6. Cách dùng

- `ai_product_recommendation_v2` — top-6 sản phẩm phù hợp nhất mỗi khách, kèm lý do.
- `agg_product_demand` — dự báo cầu theo sản phẩm × phân khúc, đưa vào kế hoạch KPI/chiến dịch.
- `agg_segment_product_affinity` — ma trận phân khúc × sản phẩm cho định hướng danh mục.
- App Streamlit: trang **“Sản phẩm & Nhu cầu”**.