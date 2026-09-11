"""
Đóng gói app Streamlit thành site TĨNH chạy bằng stlite (Streamlit-in-WASM) — không
cần server, deploy free lên Hugging Face **Static** Space / GitHub Pages / Netlify.

    py stlite/build.py

-> tạo  stlite/site/  gồm: index.html + README.md + streamlit_app/app.py
        + data/parquet/<các bảng app cần> + models/<report>.
Đẩy nội dung stlite/site/ (không phải cả repo) lên Static Space.
"""
from __future__ import annotations

import json
import os
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SITE = os.path.join(HERE, "site")

STLITE_VERSION = "0.85.1"

# Bảng dữ liệu app thực sự đọc (EAGER + LAZY trong app.py). KHÔNG gồm fact_casa_daily/
# fact_transaction... (app không dùng). ~16MB.
PARQUET = [
    "customer_360_feature_mart", "ai_customer_score", "dim_customer", "dim_product",
    "rm_action_feedback", "ai_model_coefficient", "ai_model_metric", "ai_model_registry",
    "dim_product_catalogue", "agg_product_demand", "agg_segment_product_affinity",
    "fact_customer_product_holding", "agg_product_propensity", "ai_agent_review",
    "ai_model_adjustment", "ai_product_recommendation_v2", "rm_feedback_ai",
    "ai_score_reason", "ai_recommendation", "fact_campaign", "ml_propensity_training_set",
]
MODELS = ["model_report.md", "ai_agent_report.md", "rule_overrides.json"]
# pandas/numpy bundled sẵn trong Pyodide; KHÔNG dùng pyarrow (bấp bênh trên WASM)
# -> dữ liệu ship dạng csv.gz, app.py tự đọc (xem _read_table).
REQUIREMENTS = ["plotly"]

INDEX_HTML = """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MSB Smart Growth Engine</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@{ver}/build/stlite.css" />
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; }}
    #boot {{ position: fixed; inset: 0; display: flex; flex-direction: column;
            align-items: center; justify-content: center; gap: 12px; color: #1d2733;
            background: #fafafa; z-index: 9; }}
    #boot .t {{ font-weight: 700; font-size: 1.1rem; }}
    #boot .s {{ color: #5B6770; font-size: .9rem; max-width: 460px; text-align: center; }}
    #boot .bar {{ width: 220px; height: 4px; background: #eee; border-radius: 4px; overflow: hidden; }}
    #boot .bar i {{ display: block; height: 100%; width: 40%; background: #E4002B;
                    animation: mv 1.1s infinite ease-in-out; }}
    @keyframes mv {{ 0% {{ margin-left: -40%; }} 100% {{ margin-left: 100%; }} }}
  </style>
</head>
<body>
  <div id="boot">
    <div class="t">MSB Smart Growth Engine</div>
    <div class="bar"><i></i></div>
    <div class="s">Đang tải Python + dữ liệu (~55MB, chỉ lần đầu, sau đó cache trong trình duyệt).
      Full 25.000 khách hàng — chạy hoàn toàn trên máy bạn, không cần server.</div>
  </div>
  <div id="root"></div>
  <script type="module">
    import {{ mount }} from "https://cdn.jsdelivr.net/npm/@stlite/browser@{ver}/build/stlite.js";
    mount(
      {{
        requirements: {reqs},
        entrypoint: "streamlit_app/app.py",
        files: {files},
      }},
      document.getElementById("root"),
    );
    const iv = setInterval(() => {{
      if (document.querySelector("#root .stApp, #root [data-testid='stAppViewContainer'], #root iframe")) {{
        document.getElementById("boot")?.remove();
        clearInterval(iv);
      }}
    }}, 400);
    setTimeout(() => {{ document.getElementById("boot")?.remove(); clearInterval(iv); }}, 120000);
  </script>
</body>
</html>
"""

SPACE_README = """---
title: MSB Smart Growth Engine
emoji: 📈
colorFrom: red
colorTo: gray
sdk: static
pinned: false
license: mit
short_description: AI Customer Intelligence & Sales Growth (stlite demo)
---

# MSB Smart Growth Engine — bản static (stlite)

Streamlit chạy bằng WebAssembly trong trình duyệt (stlite) — không cần compute, deploy
được lên Hugging Face **Static** Space. Full 25.000 khách hàng, dữ liệu tải 1 lần rồi cache.

Nguồn: https://github.com/congdanh300102/MSB-Smart-Grow-Engine — sinh lại site:
`py stlite/build.py`.
"""


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    if os.path.isdir(SITE):
        shutil.rmtree(SITE)
    os.makedirs(SITE)

    _copy(os.path.join(ROOT, "streamlit_app", "app.py"),
          os.path.join(SITE, "streamlit_app", "app.py"))

    files = {"streamlit_app/app.py": {"url": "./streamlit_app/app.py"}}
    total = 0
    for name in PARQUET:
        src = os.path.join(ROOT, "data", "parquet", f"{name}.parquet")
        if not os.path.exists(src):
            print(f"  ! thiếu {src}")
            continue
        rel = f"data/parquet/{name}.csv.gz"          # -> app.py _read_table đọc csv.gz
        dst = os.path.join(SITE, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        pd.read_parquet(src).to_csv(dst, index=False, compression="gzip")
        files[rel] = {"url": f"./{rel}"}
        total += os.path.getsize(dst)
    for name in MODELS:
        src = os.path.join(ROOT, "models", name)
        if os.path.exists(src):
            rel = f"models/{name}"
            _copy(src, os.path.join(SITE, rel))
            files[rel] = {"url": f"./{rel}"}

    html = INDEX_HTML.format(
        ver=STLITE_VERSION,
        reqs=json.dumps(REQUIREMENTS),
        files=json.dumps(files, indent=8, ensure_ascii=False),
    )
    open(os.path.join(SITE, "index.html"), "w", encoding="utf-8").write(html)
    open(os.path.join(SITE, "README.md"), "w", encoding="utf-8").write(SPACE_README)

    print(f"OK -> {os.path.relpath(SITE, ROOT)}/  "
          f"({len(files)} file, data {total/1e6:.1f} MB)")
    print("Đẩy lên Static Space: xem stlite/README.md")


if __name__ == "__main__":
    main()
