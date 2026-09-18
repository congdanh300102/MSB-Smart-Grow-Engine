# MSB Smart Growth Engine — Dashboard Streamlit

Trực quan hoá công cụ theo **12 Actions** của "Thiết kế hành trình AI", trên danh mục
**35 sản phẩm MSB**. App **chỉ đọc `data/parquet/`** (không cần PostgreSQL).

Demo online: https://huggingface.co/spaces/danh30012002/msb-smart-growth-engine
(bản stlite chạy trong trình duyệt; cách deploy xem [../HUGGINGFACE.md](../HUGGINGFACE.md)).

## Chạy

```bash
pip install -r requirements.txt            # streamlit, pandas, pyarrow, plotly

# cần có dữ liệu + artefact model (xem README gốc, mục 4):
py ../src/generate_data.py --customers 25000 --seed 42 --casa-days 45 --digital-days 45
py ../src/train_models.py --apply
py ../src/product_analysis.py
py ../src/rm_feedback_agent.py --apply && py ../src/product_analysis.py

streamlit run app.py                       # http://localhost:8501
```

| Biến môi trường | Mặc định | Ý nghĩa |
|---|---|---|
| `APP_MAX_CUST` | `25000` | Số khách tối đa được nạp (lấy mẫu đều). Đặt `10000` khi RAM khoảng 1 GB |
| `FORCE_WASM` | — | `1` để giả lập chế độ stlite (không dùng dtype `category`) |

App đọc `*.parquet` khi chạy trên máy và `*.csv.gz` khi chạy bản stlite (không có pyarrow).

## 8 trang

| Trang | Nội dung |
|---|---|
| 🏠 Giới thiệu & Mục đích | Công cụ là gì, 5 bài toán, kiến trúc hành trình AI, sản phẩm giai đoạn đầu |
| 🧭 Hành trình AI — 12 Actions | Đi qua từng Action với dữ liệu thật: signals → SGS → Gate 1 → NBP → NBA → kênh/thời điểm → nội dung → Gate 2/3 → funnel phản hồi → learning loop |
| 🎯 RM Opportunity Desk | Top-N cơ hội theo chi nhánh × nhóm × sản phẩm (Gate 1 + Smart Growth Score), tải CSV |
| 👤 Customer 360 | Hồ sơ một khách: mức phù hợp 35 SP, radar 6 thành phần, NBP/NBA, "Why this customer" (đóng góp logit) |
| 🛍️ Sản phẩm & Nhu cầu | Danh mục 35 SP · dự báo mở mới 30/60/90 ngày · heatmap phân khúc × SP · gợi ý theo khách |
| 🔁 AI Agent · Feedback & Hiệu chỉnh | % RM đồng ý, heatmap đồng thuận, "mô hình sai thật" hay "RM thận trọng", hiệu chỉnh đã áp, form RM gửi feedback |
| 📊 Manager Intelligence | KPI, phân bố priority, SGS theo phân khúc, cơ cấu NBP, dự báo 90 ngày, cơ hội theo nhóm, kết quả RM, xếp hạng chi nhánh |
| 🤖 Mô hình Propensity | Cấu thành propensity hybrid (Top-N), giải thích tham số, **kế hoạch phát triển theo kỳ**, metric/hệ số/calibration của 4 mô hình LR |

## Tính năng tương tác

### 🔎 Bộ lọc & tìm kiếm trên từng biểu đồ
Có ở **Manager Intelligence**, **Sản phẩm & Nhu cầu** (Dự báo cầu, Heatmap) và **AI Agent**
(tab Đồng thuận). Bấm nút **🔎 Lọc** phía trên biểu đồ để:
- chọn nhiều giá trị: phân khúc, nhóm SP, priority, vùng, loại khách, loại hành động RM… (để trống = tất cả);
- tìm kiếm **không dấu, không phân biệt hoa thường** (ví dụ `the tin dung` khớp "Thẻ tín dụng");
- chọn **Top N** ở các biểu đồ xếp hạng;
- **Xoá bộ lọc** để về mặc định.

Bộ lọc chỉ áp cho biểu đồ chứa nó. Nhãn nút hiện số bộ lọc đang bật (`🔎 Lọc · 2`), dưới biểu đồ
có dòng "Đang lọc x / y bản ghi". Muốn thêm bộ lọc cho biểu đồ mới, gọi hàm dùng chung:

```python
f, top_n = chart_filter("khoa_duy_nhat", df,
                        filters={"customer_segment": "Phân khúc"},
                        search=("Tìm sản phẩm", ["product_code", "product_name"]),
                        top=(10, 35), defaults={"priority_level": ["Very High", "High"]})
if not _empty_chart(f):
    ...  # vẽ biểu đồ từ f
```

`st.popover` cần Streamlit ≥ 1.32; bản cũ hơn tự chuyển sang `st.expander`.

### 🤖 Kế hoạch phát triển theo kỳ (trang Mô hình Propensity)
- Chọn **độ dài kỳ** (tháng / quý / nửa năm), **số kỳ** và **sản phẩm trọng tâm** (tối đa 8).
- Chỉnh các tham số: trọng số LR `w`, tỉ lệ chuyển đổi nền, độ phủ tiếp cận, mức tăng uplift mỗi kỳ.
- Nhập **uplift chiến dịch (%)** và **chỉ tiêu** cho từng kỳ × sản phẩm ngay trong bảng.
- Kết quả: dự báo khách mới theo kỳ (có trừ dần tệp đã mở), heatmap % đạt, lũy kế so với chỉ tiêu,
  đánh giá **Đạt (≥100%) / Sát chỉ tiêu (≥85%) / Rủi ro**, và **uplift cần có** để đạt chỉ tiêu.

Công thức dự báo mỗi kỳ:

```
Σpropensity(w) × tỉ lệ chuyển đổi × độ phủ × (số ngày / 90) × (1 + uplift) × tỉ lệ tệp còn lại
```

Mục **"ℹ️ Vì sao chọn các tham số này?"** giải thích lý do chọn `w = 0.45`, 14%, 55%, các ngưỡng…
kèm số liệu đối chiếu (AUC, P_LR so với fit, tỉ lệ chuyển đổi trong RM feedback).

## Ghi chú kỹ thuật

- Dữ liệu được cache bằng `st.cache_data`. Các bảng lớn, ít dùng được nạp lười qua `get_df()`.
- Khi chạy trên máy, cột chuỗi được chuyển sang `category` để tiết kiệm RAM. Bản stlite giữ `object`,
  vì Arrow trên WASM không ổn định với `category`.
- Màu thương hiệu đỏ MSB. Dữ liệu 100% giả lập (nhãn "Synthetic / Masked").
