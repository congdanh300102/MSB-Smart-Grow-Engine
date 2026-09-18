# Deploy lên Hugging Face Spaces (Static + stlite)

Space hiện tại: **https://huggingface.co/spaces/danh30012002/msb-smart-growth-engine**
(`sdk: static`)

Hugging Face không còn Streamlit Space miễn phí. App được đóng gói bằng
[stlite](https://github.com/whitphx/stlite): Streamlit chạy bằng WebAssembly **trong trình duyệt
người xem**, Space chỉ phục vụ file tĩnh. App dùng đủ 25.000 khách hàng, lần đầu tải khoảng 55 MB,
các lần sau dùng cache.

> ⚠️ **Space KHÔNG đồng bộ với GitHub.** `git commit` / `git push origin` không làm Space thay đổi.
> Mỗi lần sửa `streamlit_app/app.py` hoặc dữ liệu đều phải **build lại và upload** như dưới đây.

---

## Cách 1: một lệnh (khuyên dùng)

Lần đầu trên máy:

```powershell
py -m pip install -U huggingface_hub
hf auth login          # dán Access Token quyền **Write**: https://huggingface.co/settings/tokens
```

Mỗi lần cập nhật:

```powershell
cd "D:\Tài liệu\Work\code\MSB-Smart-Grow-Engine"
py stlite/build.py --upload danh30012002/msb-smart-growth-engine
```

Lệnh này làm hai việc:
1. Sinh `stlite/site/`, gồm `index.html`, `README.md` (header `sdk: static`),
   `streamlit_app/app.py`, `data/parquet/*.csv.gz` và `models/*`.
2. Upload cả thư mục lên Space trong một commit.

Có thể thêm `-m "nội dung commit"`.

## Cách 2: CLI `hf` thủ công

```powershell
py stlite/build.py
hf upload danh30012002/msb-smart-growth-engine stlite/site . --repo-type space --commit-message "update"
```

## Cách 3: git

```powershell
git lfs install
git clone https://huggingface.co/spaces/danh30012002/msb-smart-growth-engine $env:TEMP\msb-space
Copy-Item "D:\Tài liệu\Work\code\MSB-Smart-Grow-Engine\stlite\site\*" $env:TEMP\msb-space -Recurse -Force
cd $env:TEMP\msb-space
git add -A; git commit -m "update"; git push     # user = danh30012002, password = Access Token
```

Hugging Face từ chối file nhị phân (`.csv.gz`) không đi qua LFS/Xet. Nếu push báo
*"contains binary files"*, dùng Cách 1 hoặc Cách 2.

---

## Kiểm tra sau khi deploy

- Xem commit mới nhất ở tab **Files → History** của Space; Space build lại trong khoảng 30 giây.
- Tra nhanh qua API (trường `lastModified`): https://huggingface.co/api/spaces/danh30012002/msb-smart-growth-engine
- Nếu vẫn thấy giao diện cũ, nhấn **Ctrl+Shift+R** (trình duyệt đang giữ `app.py` cũ).

## Lỗi thường gặp

| Triệu chứng | Xử lý |
|---|---|
| `401` / `403` khi upload | Token chưa có quyền Write, hoặc chưa đăng nhập: chạy `hf auth login` |
| `hf` không nhận lệnh | `py -m pip install -U huggingface_hub`, mở lại terminal |
| `pip` / `py` treo không in gì (Python 3.12 trên Windows, thường do WMI) | Dùng **Cách 3 (git)**: chỉ cần `py stlite/build.py` + git, không cần `huggingface_hub` |
| Commit GitHub xong nhưng Space không đổi | Đúng như thiết kế: chạy `py stlite/build.py --upload ...` |
| Loading mãi không xong | F12 → Console. Thường do mạng/CDN (trang tự tải lại 2 lần); lỗi package → sửa `REQUIREMENTS` trong `stlite/build.py` |
| App báo lỗi Python | Chạy `streamlit run streamlit_app/app.py` trên máy để tìm lỗi. stlite dùng Streamlit 1.50 + pandas 2.2, không có pyarrow |
| `short_description` bị từ chối | Hugging Face giới hạn ≤ 60 ký tự (sửa `SPACE_README` trong `stlite/build.py`) |

## Chạy thử site tĩnh trước khi upload

```powershell
py stlite/build.py
cd stlite/site; py -m http.server 8000     # mở http://localhost:8000
```

---

## Phương án khác: Streamlit Community Cloud

Trỏ tới `streamlit_app/app.py` và `streamlit_app/requirements.txt`. RAM chỉ khoảng 1 GB, nên đặt
`APP_MAX_CUST=10000` trong **Settings → Secrets** để app lấy mẫu khách hàng.
