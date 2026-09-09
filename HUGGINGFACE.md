# Deploy lên Hugging Face Spaces (Streamlit SDK)

Free tier HF Spaces: **16 GB RAM / 2 vCPU** → chạy full **25.000 khách hàng** thoải mái
(khác Streamlit Community Cloud chỉ ~1 GB).

Repo đã cấu hình sẵn:
- YAML header trong [`README.md`](README.md): `sdk: streamlit`, `app_file: streamlit_app/app.py`
- [`requirements.txt`](requirements.txt): chỉ deps cho app (streamlit / pandas<2.3 / pyarrow / plotly / numpy)
- `APP_MAX_CUST` mặc định `25000` → không lấy mẫu

---

## Cách 1 — push cả LFS (đơn giản nhất)

```bash
# tạo Space trống: https://huggingface.co/new-space  → Owner = bạn,
#   Space name = msb-smart-growth-engine, SDK = Streamlit, Hardware = CPU basic (free)

git remote add hf https://huggingface.co/spaces/<username>/msb-smart-growth-engine
git push hf main
```

- Lần đầu Git LFS upload ~475 MB (5–10 phút). HF hỏi user/token → dùng
  **Access Token** (https://huggingface.co/settings/tokens, quyền *write*) làm mật khẩu.
- Space tự build (~3 phút) rồi chạy. URL: `https://huggingface.co/spaces/<username>/msb-smart-growth-engine`.

## Cách 2 — bỏ file LFS trước khi push (nhẹ hơn, ~30 MB)

App **không dùng** 4 file LFS (`fact_casa_daily` / `fact_transaction` / `fact_digital_activity`
CSV + `fact_casa_daily.parquet`). Tách nhánh riêng cho HF, gỡ chúng ra:

```bash
git checkout -b hf-deploy
git rm --cached data/csv/fact_casa_daily.csv data/csv/fact_transaction.csv \
                data/csv/fact_digital_activity.csv data/parquet/fact_casa_daily.parquet
printf '' > .gitattributes            # bỏ tracking LFS
git commit -am "hf-deploy: gỡ file LFS không cần cho app"
git remote add hf https://huggingface.co/spaces/<username>/msb-smart-growth-engine
git push hf hf-deploy:main
git checkout main                     # quay lại nhánh chính
```

---

## Cập nhật app sau này

```bash
git checkout main && git pull
# ... sửa code / chạy lại pipeline ...
git push hf main                      # (hoặc: git checkout hf-deploy && git merge main && git push hf hf-deploy:main)
```

## Chỉnh tài nguyên / biến môi trường trên HF

Space → **Settings**:
- **Variables and secrets** → thêm `APP_MAX_CUST` nếu muốn giới hạn (mặc định 25000 = full).
- **Hardware**: CPU basic (free, 16 GB) đủ dùng; có thể nâng nếu cần nhanh hơn.

## Lỗi thường gặp

| Triệu chứng | Xử lý |
|---|---|
| Build fail ở `pip install` | Xem log; thường do pin version — sửa `requirements.txt` rồi push lại |
| "This app has gone over its resource limits" | Hiếm trên HF (16GB). Nếu có: Space Settings → Variables → `APP_MAX_CUST=15000` |
| Trang sản phẩm trống | Reboot Space (Settings → Factory reboot) để xoá cache `@st.cache_data` |
| `git push hf` bị 403 | Dùng Access Token (quyền write) làm password, không phải mật khẩu tài khoản |
