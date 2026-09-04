# MSB Smart Growth Engine — Streamlit demo

Trực quan hoá công cụ theo **12 Actions** trong "2. Thiết kế hành trình AI".

## Chạy

```bash
pip install -r ../requirements.txt        # cần streamlit, plotly

# đảm bảo đã có dữ liệu Parquet + artefact model:
py ../src/generate_data.py --customers 25000 --seed 42 --casa-days 45 --digital-days 45
py ../src/train_models.py --apply

streamlit run app.py          # mở http://localhost:8501
```

App **chỉ đọc `data/parquet/`** (không cần PostgreSQL).

## 6 trang

| Trang | Nội dung |
|---|---|
| 🏠 Giới thiệu & Mục đích | Công cụ là gì, 2 track (AI FOR CUSTOMERS / AI FOR MY TEAM), 5 bài toán, sơ đồ luồng dữ liệu |
| 🧭 Hành trình AI — 12 Actions | Slider qua từng Action, mỗi bước gắn dữ liệu/chart thật (signals → opportunity → gate 1 → NBP → NBA → kênh/thời điểm → nội dung → gate 2/3 → orchestration → phản hồi → learning loop) |
| 🎯 RM Opportunity Desk | Top-N cơ hội / chi nhánh / sản phẩm (2 tầng lọc: eligibility + Smart Growth Score), tải CSV |
| 👤 Customer 360 | Hồ sơ 1 khách: điểm theo sản phẩm, radar 6 thành phần, Next Best Product/Action, "Why this customer" (đóng góp logit từng feature) |
| 📊 Manager Intelligence | KPI danh mục, phân bố priority, SGS theo phân khúc, cơ cấu NBP, kết quả feedback RM, xếp hạng chi nhánh |
| 🤖 Mô hình Propensity | Hồi quy logistic: bảng metric test, ROC-AUC theo split, hệ số wᵢ + odds ratio, calibration, `model_report.md` |

Màu thương hiệu đỏ MSB. Dữ liệu 100% giả lập (nhãn "Synthetic / Masked").
