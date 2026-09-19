# Deploy lên GreenNode Cloud (app Streamlit gốc, native)

Đã build & test cục bộ: `docker build` xong (~732MB), `docker run` health-check trả `ok`,
log sạch (không traceback). Image chạy app **gốc** (không phải bản stlite/WASM) — có server
thật nên không cần lách giới hạn RAM, dùng đủ 25.000 khách hàng, đầy đủ tính năng.

> GreenNode không có kiểu "PaaS git-push" như Streamlit Cloud/HF Spaces. Có 2 hướng chính:
> **vServer** (VM thường + Docker — đơn giản nhất, khuyên dùng để bắt đầu) hoặc **VKS**
> (Kubernetes, cho production/scale). Phần thao tác trên Console GreenNode (tạo VM, mở port,
> lấy URL registry) mình không có sẵn thông tin chính xác của tài khoản bạn — bạn tự làm theo
> giao diện Console; các lệnh Docker/Linux dưới đây thì chạy được y hệt trên mọi VM.

---

## 0. Đã có sẵn trong repo

| File | Vai trò |
|---|---|
| `Dockerfile` | Build image app Streamlit gốc (python:3.12-slim + requirements.txt + data/parquet + models) |
| `.dockerignore` | Loại data/csv, stlite/, sql/... khỏi build context |
| `docker-compose.yml` | service `app` (Streamlit, port 8501) + service `db` (Postgres, tuỳ chọn) |

Build/chạy thử cục bộ trước khi deploy:
```bash
docker build -t msb-sge-app:local .
docker run -d -p 8501:8501 --name msb_sge_app msb-sge-app:local
curl http://localhost:8501/_stcore/health     # -> ok
```

---

## 1. Tạo vServer trên GreenNode

Console GreenNode → **vServer** → tạo máy mới:
- OS: Ubuntu 22.04/24.04 (khuyến nghị)
- Cấu hình: 2 vCPU / 4GB RAM trở lên là thoải mái cho app này (image 732MB, RAM lúc chạy ~150–300MB)
- Mở **Security Group / Firewall**: cho phép inbound **TCP 80** (hoặc 8501 nếu chưa dùng Nginx) từ `0.0.0.0/0`
- Ghi lại **IP public** của VM sau khi tạo xong

## 2. Cài Docker trên VM (SSH vào VM trước)

```bash
ssh ubuntu@<IP_VM>     # user tuỳ image GreenNode cung cấp (ubuntu/root...)

curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
docker --version

# Ubuntu tối giản không có sẵn git — cài trước khi clone
sudo apt-get update && sudo apt-get install -y git
git --version
```

## 3. Đưa code lên VM

Cách đơn giản nhất — clone thẳng từ GitHub (repo đã public trên `congdanh300102/MSB-Smart-Grow-Engine`):

```bash
git clone https://github.com/congdanh300102/MSB-Smart-Grow-Engine.git
cd MSB-Smart-Grow-Engine
git lfs install && git lfs pull   # nếu muốn cả data/csv gốc — app không bắt buộc cần
```

(App chỉ đọc `data/parquet/` — không cần LFS để build image.)

## 4. Build & chạy trên VM

```bash
docker build -t msb-sge-app:latest .
docker run -d --name msb_sge_app --restart unless-stopped -p 80:8501 msb-sge-app:latest
```

Hoặc dùng compose (kèm cả Postgres nếu muốn):
```bash
docker compose up -d app
```

Kiểm tra:
```bash
curl http://localhost/_stcore/health     # -> ok  (nếu map port 80)
```

Mở trình duyệt: `http://<IP_VM>` (hoặc `http://<IP_VM>:8501` nếu map thẳng 8501).

## 5. (Tuỳ chọn) Domain + HTTPS bằng Caddy

Nếu có domain trỏ về IP VM, dùng Caddy để tự lấy chứng chỉ HTTPS + reverse-proxy về app:

```bash
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update && sudo apt-get install -y caddy

echo "your-domain.com {
    reverse_proxy localhost:8501
}" | sudo tee /etc/caddy/Caddyfile
sudo systemctl restart caddy
```
Đổi container `app` sang chạy nội bộ (`-p 127.0.0.1:8501:8501`) rồi để Caddy expose port 80/443.

## 6. Cập nhật sau này

```bash
cd MSB-Smart-Grow-Engine && git pull
docker build -t msb-sge-app:latest .
docker stop msb_sge_app && docker rm msb_sge_app
docker run -d --name msb_sge_app --restart unless-stopped -p 80:8501 msb-sge-app:latest
```

---

## Phương án B: VKS (Kubernetes) — khi cần scale/production

1. Đẩy image lên registry của GreenNode (**vCR**) — lấy hostname/thông tin đăng nhập chính xác
   từ Console GreenNode (mục Container Registry), dạng chung:
   ```bash
   docker login <registry-host-cua-ban>
   docker tag msb-sge-app:latest <registry-host-cua-ban>/msb-sge-app:latest
   docker push <registry-host-cua-ban>/msb-sge-app:latest
   ```
2. Áp manifest cơ bản (sửa `image:` cho khớp registry ở trên) — file mẫu: `k8s/deployment.yaml`.
3. `kubectl apply -f k8s/deployment.yaml` (cần `kubeconfig` GreenNode cấp trong Console VKS).

---

## Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `APP_MAX_CUST` | `25000` | Số khách hàng nạp vào app (server thật → để nguyên 25000, đủ RAM) |

## Xử lý sự cố

| Triệu chứng | Xử lý |
|---|---|
| `docker build` treo/lỗi TLS khi pull `python:3.12-slim` | Docker Hub chập chờn — thử lại `docker pull python:3.12-slim` rồi build lại |
| Container thoát ngay sau khi chạy | `docker logs msb_sge_app` xem traceback |
| Không truy cập được từ ngoài | Kiểm tra Security Group/Firewall VM đã mở port 80 (hoặc 8501) inbound |
| `_stcore/health` không trả `ok` | Đợi thêm ~15–20s (Streamlit khởi động), hoặc xem `docker logs` |
