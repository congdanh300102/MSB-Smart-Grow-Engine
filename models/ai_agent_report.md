# AI Feedback Agent — Xem xét lại & hiệu chỉnh mô hình

*2026-08-31 · 26,135 lượt RM phản hồi · 67% đồng ý tổng thể*

## 1. Mức độ đồng thuận RM ↔ AI theo nhóm sản phẩm

| Nhóm | Lượt phản hồi | % đồng ý | % chuyển đổi (đã tiếp cận) |
|---|--:|--:|--:|
| CARD | 10,511 | 71% | 27% |
| CASA | 5,217 | 75% | 29% |
| FD | 7,140 | 59% | 20% |
| LENDING | 3,267 | 56% | 19% |

## 2. Phát hiện của agent

- Tổng 10 vùng nghi vấn được rà soát; **7 vùng kết luận MÔ HÌNH SAI THẬT** (đã đủ mẫu, lý do từ chối nhất quán, và nhóm được tiếp cận vẫn không chuyển đổi).
- 3 vùng còn lại: RM thận trọng / sai thời điểm / chưa đủ tín hiệu → **không sửa mô hình**.

### 🔴 SAI THẬT — FD_SAV_KIDS
*RULE_TOO_LOOSE · n=1335 · đồng ý 39% · lý do chính NOT_RELEVANT (47%) · chuyển đổi nhóm tiếp cận 36%*

AI đề xuất **Tiết kiệm Măng non** chủ yếu vì rule "Cha mẹ / người giám hộ muốn tích luỹ dài hạn cho con" và hard-gate: family_flag = 1 (cờ suy đoán theo độ tuổi, độ tin cậy thấp — không xác nhận KH có con). RM phản hồi: chỉ **39% đồng ý** trên 1335 lượt; lý do từ chối chính là **NOT_RELEVANT** (47%); tỉ lệ chuyển đổi của nhóm được RM theo đuổi = 36%. → **Mô hình SAI THẬT**: rule fit quá lỏng. Hiệu chỉnh: nâng ngưỡng fit_score 0.50 → 0.60 cho FD_SAV_KIDS.

### 🔴 SAI THẬT — LEND_AUTO_NEW
*RULE_TOO_LOOSE · n=760 · đồng ý 41% · lý do chính NOT_RELEVANT (43%) · chuyển đổi nhóm tiếp cận 32%*

AI đề xuất **Vay mua ô tô mới** chủ yếu vì rule "Khách cá nhân có nhu cầu mua xe phục vụ đi lại / gia đình" và hard-gate: auto_intent = 1 (có giao dịch danh mục ô tô — bắt cả KH chỉ đổ xăng / bảo dưỡng). RM phản hồi: chỉ **41% đồng ý** trên 760 lượt; lý do từ chối chính là **NOT_RELEVANT** (43%); tỉ lệ chuyển đổi của nhóm được RM theo đuổi = 32%. → **Mô hình SAI THẬT**: rule fit quá lỏng. Hiệu chỉnh: nâng ngưỡng fit_score 0.50 → 0.60 cho LEND_AUTO_NEW.

### 🔴 SAI THẬT — FD_SAV_KIDS · phân khúc MASS
*RULE_TOO_LOOSE_SEGMENT · n=680 · đồng ý 29% · lý do chính NOT_RELEVANT (61%) · chuyển đổi nhóm tiếp cận 31%*

AI đề xuất **Tiết kiệm Măng non** cho phân khúc MASS chủ yếu vì rule "Cha mẹ / người giám hộ muốn tích luỹ dài hạn cho con" và hard-gate: family_flag = 1 (cờ suy đoán theo độ tuổi, độ tin cậy thấp — không xác nhận KH có con). RM phản hồi: chỉ **29% đồng ý** trên 680 lượt; lý do từ chối chính là **NOT_RELEVANT** (61%); tỉ lệ chuyển đổi của nhóm được RM theo đuổi = 31%. → **Mô hình SAI THẬT ở phân khúc MASS**: gate quá lỏng, bắt nhầm KH không thực sự phù hợp. Hiệu chỉnh: loại phân khúc MASS khỏi tệp đủ điều kiện của FD_SAV_KIDS (hoặc nâng ngưỡng gate theo tín hiệu thật).

### 🔴 SAI THẬT — FD_SAV_KIDS · phân khúc MASS_AFFLUENT
*RULE_TOO_LOOSE_SEGMENT · n=407 · đồng ý 38% · lý do chính NOT_RELEVANT (50%) · chuyển đổi nhóm tiếp cận 37%*

AI đề xuất **Tiết kiệm Măng non** cho phân khúc MASS_AFFLUENT chủ yếu vì rule "Cha mẹ / người giám hộ muốn tích luỹ dài hạn cho con" và hard-gate: family_flag = 1 (cờ suy đoán theo độ tuổi, độ tin cậy thấp — không xác nhận KH có con). RM phản hồi: chỉ **38% đồng ý** trên 407 lượt; lý do từ chối chính là **NOT_RELEVANT** (50%); tỉ lệ chuyển đổi của nhóm được RM theo đuổi = 37%. → **Mô hình SAI THẬT ở phân khúc MASS_AFFLUENT**: gate quá lỏng, bắt nhầm KH không thực sự phù hợp. Hiệu chỉnh: loại phân khúc MASS_AFFLUENT khỏi tệp đủ điều kiện của FD_SAV_KIDS (hoặc nâng ngưỡng gate theo tín hiệu thật).

### 🔴 SAI THẬT — LEND_UNSECURED_PAYROLL · phân khúc MASS
*GATE_INCOME_TOO_LOOSE · n=290 · đồng ý 35% · lý do chính CANT_AFFORD (44%) · chuyển đổi nhóm tiếp cận 29%*

AI đề xuất **Vay tín chấp theo lương / phân khúc** cho phân khúc MASS chủ yếu vì rule "Khách cá nhân có thu nhập ổn định / nhận lương hoặc thuộc phân khúc đủ điều kiện" và hard-gate: nhận lương qua MSB / thu nhập > 15tr (không đặt sàn thu nhập, không xét khả năng trả nợ). RM phản hồi: chỉ **35% đồng ý** trên 290 lượt; lý do từ chối chính là **CANT_AFFORD** (44%); tỉ lệ chuyển đổi của nhóm được RM theo đuổi = 29%. → **Mô hình SAI THẬT**: gate không kiểm tra khả năng trả nợ (dòng tiền ròng, dư nợ hiện có). Hiệu chỉnh: thêm điều kiện thu nhập ≥ 20tr cho LEND_UNSECURED_PAYROLL.

### 🔴 SAI THẬT — LEND_AUTO_NEW · phân khúc MASS_AFFLUENT
*RULE_TOO_LOOSE_SEGMENT · n=269 · đồng ý 25% · lý do chính NOT_RELEVANT (68%) · chuyển đổi nhóm tiếp cận 39%*

AI đề xuất **Vay mua ô tô mới** cho phân khúc MASS_AFFLUENT chủ yếu vì rule "Khách cá nhân có nhu cầu mua xe phục vụ đi lại / gia đình" và hard-gate: auto_intent = 1 (có giao dịch danh mục ô tô — bắt cả KH chỉ đổ xăng / bảo dưỡng). RM phản hồi: chỉ **25% đồng ý** trên 269 lượt; lý do từ chối chính là **NOT_RELEVANT** (68%); tỉ lệ chuyển đổi của nhóm được RM theo đuổi = 39%. → **Mô hình SAI THẬT ở phân khúc MASS_AFFLUENT**: gate quá lỏng, bắt nhầm KH không thực sự phù hợp. Hiệu chỉnh: loại phân khúc MASS_AFFLUENT khỏi tệp đủ điều kiện của LEND_AUTO_NEW (hoặc nâng ngưỡng gate theo tín hiệu thật).

### 🔴 SAI THẬT — LEND_AUTO_NEW · phân khúc MASS
*RULE_TOO_LOOSE_SEGMENT · n=215 · đồng ý 26% · lý do chính NOT_RELEVANT (66%) · chuyển đổi nhóm tiếp cận 16%*

AI đề xuất **Vay mua ô tô mới** cho phân khúc MASS chủ yếu vì rule "Khách cá nhân có nhu cầu mua xe phục vụ đi lại / gia đình" và hard-gate: auto_intent = 1 (có giao dịch danh mục ô tô — bắt cả KH chỉ đổ xăng / bảo dưỡng). RM phản hồi: chỉ **27% đồng ý** trên 215 lượt; lý do từ chối chính là **NOT_RELEVANT** (66%); tỉ lệ chuyển đổi của nhóm được RM theo đuổi = 16%. → **Mô hình SAI THẬT ở phân khúc MASS**: gate quá lỏng, bắt nhầm KH không thực sự phù hợp. Hiệu chỉnh: loại phân khúc MASS khỏi tệp đủ điều kiện của LEND_AUTO_NEW (hoặc nâng ngưỡng gate theo tín hiệu thật).

### ⚪ không sửa — LEND_OVERDRAFT_PERSONAL
*INSUFFICIENT_SIGNAL · n=199 · đồng ý 42% · lý do chính NO_NEED_NOW (33%) · chuyển đổi nhóm tiếp cận 17%*

AI đề xuất **Thấu chi cá nhân (M-Payroll / M-First)** chủ yếu vì rule "Khách cá nhân có thu nhập ổn định, cần nguồn tiền dự phòng ngắn hạn" và hard-gate: rule fit tổng hợp theo cột 'Khách hàng/Nhu cầu phù hợp'. RM phản hồi: chỉ **42% đồng ý** trên 199 lượt; lý do từ chối chính là **NO_NEED_NOW** (33%); tỉ lệ chuyển đổi của nhóm được RM theo đuổi = 17%. → Chưa đủ tín hiệu nhất quán để kết luận. Tiếp tục thu thập feedback.

### ⚪ không sửa — LEND_OVERDRAFT_PERSONAL · phân khúc MASS
*INSUFFICIENT_SIGNAL · n=172 · đồng ý 41% · lý do chính NO_NEED_NOW (34%) · chuyển đổi nhóm tiếp cận 19%*

AI đề xuất **Thấu chi cá nhân (M-Payroll / M-First)** cho phân khúc MASS chủ yếu vì rule "Khách cá nhân có thu nhập ổn định, cần nguồn tiền dự phòng ngắn hạn" và hard-gate: rule fit tổng hợp theo cột 'Khách hàng/Nhu cầu phù hợp'. RM phản hồi: chỉ **41% đồng ý** trên 172 lượt; lý do từ chối chính là **NO_NEED_NOW** (34%); tỉ lệ chuyển đổi của nhóm được RM theo đuổi = 19%. → Chưa đủ tín hiệu nhất quán để kết luận. Tiếp tục thu thập feedback.

### ⚪ không sửa — CARD_CC_SIGNATURE · phân khúc MASS_AFFLUENT
*RM_SKEPTICISM · n=30 · đồng ý 47% · lý do chính WRONG_TIMING (33%) · chuyển đổi nhóm tiếp cận 50%*

AI đề xuất **Thẻ tín dụng Visa Signature** cho phân khúc MASS_AFFLUENT chủ yếu vì rule "Khách thu nhập khá / cao, ưu tiên hoàn tiền cho chi tiêu thiết yếu và phong cách sống" và hard-gate: rule fit tổng hợp theo cột 'Khách hàng/Nhu cầu phù hợp'. RM phản hồi: chỉ **47% đồng ý** trên 30 lượt; lý do từ chối chính là **WRONG_TIMING** (33%); tỉ lệ chuyển đổi của nhóm được RM theo đuổi = 50%. → Nhóm này **vẫn chuyển đổi tốt** khi RM tiếp cận (> 18%), nên đây là **RM thận trọng / lệch thời điểm**, KHÔNG phải mô hình sai. Giữ nguyên, chỉ nhắc lại về timing.

## 3. Hiệu chỉnh đã áp dụng (`models/rule_overrides.json`)

| Sản phẩm | Loại | Trước | Sau | Đề xuất bị ảnh hưởng |
|---|---|---|---|--:|
| FD_SAV_KIDS | min_fit | `{"min_fit": null, "block_segments": null, "min_income": null}` | `{"min_fit": 0.6, "block_segments": null, "min_income": null}` | 8,121 |
| LEND_AUTO_NEW | min_fit | `{"min_fit": null, "block_segments": null, "min_income": null}` | `{"min_fit": 0.6, "block_segments": null, "min_income": null}` | 6,203 |
| FD_SAV_KIDS | block_segments | `{"min_fit": 0.6, "block_segments": null, "min_income": null}` | `{"min_fit": 0.6, "block_segments": ["MASS"], "min_income": null}` | 2,419 |
| FD_SAV_KIDS | block_segments | `{"min_fit": 0.6, "block_segments": ["MASS"], "min_income": null}` | `{"min_fit": 0.6, "block_segments": ["MASS", "MASS_AFFLUENT"], "min_income": null}` | 1,855 |
| LEND_UNSECURED_PAYROLL | min_income | `{"min_fit": null, "block_segments": null, "min_income": null}` | `{"min_fit": null, "block_segments": null, "min_income": 20000000.0}` | 1,886 |
| LEND_AUTO_NEW | block_segments | `{"min_fit": 0.6, "block_segments": null, "min_income": null}` | `{"min_fit": 0.6, "block_segments": ["MASS_AFFLUENT"], "min_income": null}` | 1,277 |
| LEND_AUTO_NEW | block_segments | `{"min_fit": 0.6, "block_segments": ["MASS_AFFLUENT"], "min_income": null}` | `{"min_fit": 0.6, "block_segments": ["MASS", "MASS_AFFLUENT"], "min_income": null}` | 1,687 |

→ Chạy `py src/product_analysis.py` để chấm lại điểm; `py src/rm_feedback_agent.py` để đo lại agreement (kỳ vọng tăng).