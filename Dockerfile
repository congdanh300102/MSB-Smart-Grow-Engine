# MSB Smart Growth Engine — app Streamlit gốc (native), chạy trên server thật
# (VM/vServer/VKS trên GreenNode, hoặc bất kỳ Docker host nào — không cần lách RAM
# kiểu stlite/WASM vì có compute thật).
FROM python:3.12-slim

WORKDIR /app

# curl chỉ dùng để healthcheck + debug; nhẹ, không cần build-essential (pandas/pyarrow
# cài bằng wheel sẵn cho manylinux, không cần biên dịch).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Chỉ copy đúng phần app cần (không copy data/csv/, stlite/, sql/... cho image gọn)
COPY streamlit_app/ streamlit_app/
COPY data/parquet/ data/parquet/
COPY models/ models/

ENV APP_MAX_CUST=25000 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app/app.py"]
