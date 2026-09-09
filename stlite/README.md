# Deploy bản STATIC (stlite) — Hugging Face Static Space (free)

HF đã bỏ Streamlit Space free (chỉ còn **Static** miễn phí). Giải pháp: chạy Streamlit
bằng **WebAssembly trong trình duyệt** ([stlite](https://github.com/whitphx/stlite)) →
site tĩnh, deploy lên Static Space, **full 25.000 khách hàng**, không giới hạn RAM server
(dùng RAM máy người xem). Dữ liệu tải 1 lần rồi cache trong trình duyệt.

## 1. Build site

```bash
py stlite/build.py
#  -> stlite/site/  (index.html + streamlit_app/app.py + data/parquet/*.parquet + models/*)
#     ~16MB dữ liệu, app đọc thẳng qua fetch URL.
```

Chạy lại bước này mỗi khi đổi `streamlit_app/app.py` hoặc regen dữ liệu.

## 2. Tạo Static Space

https://huggingface.co/new-space → Owner = bạn · Space name = `msb-smart-growth-engine`
· **SDK = Static** · Public. Không chọn template (Blank).

## 3. Push nội dung `stlite/site/` lên Space

Access Token (quyền **write**): https://huggingface.co/settings/tokens

```bash
# clone Space rỗng vào thư mục tạm
git clone https://huggingface.co/spaces/<username>/msb-smart-growth-engine /tmp/msb-space
cd /tmp/msb-space

# copy site vào (đè index.html/README.md mặc định của HF)
cp -r "d:/Tài liệu/Work/code/MSB-Smart-Grow-Engine/stlite/site/." .

git add -A
git commit -m "MSB Smart Growth Engine — stlite static build"
git push            # user = username HF, password = Access Token
```

Windows PowerShell thay `cp -r ... .` bằng:
`Copy-Item "d:\Tài liệu\Work\code\MSB-Smart-Grow-Engine\stlite\site\*" . -Recurse -Force`

Space build ~30 giây (chỉ là static files). Mở:
`https://huggingface.co/spaces/<username>/msb-smart-growth-engine`

Lần đầu tải ~55MB (Pyodide + pandas/pyarrow/plotly + data) → 30–90 giây. Sau đó nhanh.

## 4. Cập nhật sau này

```bash
py stlite/build.py
cd /tmp/msb-space
cp -r "d:/.../stlite/site/." .
git add -A && git commit -m "update" && git push
```

## Lỗi thường gặp

| Triệu chứng | Xử lý |
|---|---|
| Màn hình loading mãi không xong | Mở DevTools (F12) → Console. Thường do 1 package trong `requirements` không cài được trên Pyodide — sửa `REQUIREMENTS` trong `stlite/build.py` |
| "ModuleNotFoundError: pyarrow" | Pyodide phiên bản stlite này thiếu pyarrow → nâng `STLITE_VERSION` trong build.py |
| Không tải được `.parquet` | Kiểm tra file có trong `stlite/site/data/parquet/` không; HF serve file tĩnh với CORS mở |
| App lỗi Python | Chạy `streamlit run streamlit_app/app.py` cục bộ để tái hiện; stlite dùng pandas 2.2 |

## Chạy thử site cục bộ trước khi push

```bash
cd stlite/site && py -m http.server 8000
# mở http://localhost:8000
```
