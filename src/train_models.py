"""
MSB SMART GROWTH ENGINE - Product Propensity: logistic regression
================================================================

Implements doc "a. Product Propensity":

    Z = b + w1*X1 + w2*X2 + w3*X3 + w4*X4 + w5*X5 + w6*X6
    P = 1 / (1 + e^-Z)                       (sigmoid)
    Loss = -[ y*log(p) + (1-y)*log(1-p) ]    (binary cross-entropy)
    y = 1 nếu khách đang dùng sản phẩm, 0 nếu chưa

One logistic-regression model per target product (P001/P004/P007/P009), trained
on `data/csv/ml_propensity_training_set.csv` (X1..X6 built by src/scoring.py).

    py src/train_models.py                 # train + test + write report/artefacts
    py src/train_models.py --apply         # + rewrite ai_customer_score / ai_recommendation
                                           #   using model P and the doc's 5 business scores

Outputs:
    models/propensity_<product>.joblib      pickled sklearn Pipeline
    models/metrics.json                     all metrics, all products
    models/model_report.md                  human-readable report
    models/roc_<product>.png, calibration_<product>.png
    data/csv/ai_model_registry.csv|.parquet
    data/csv/ai_model_coefficient.csv|.parquet
    data/csv/ai_model_metric.csv|.parquet
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score, roc_curve,
)

import scoring

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "data", "csv")
PARQUET = os.path.join(ROOT, "data", "parquet")
MODELS = os.path.join(ROOT, "models")

MODEL_ID = "sge_propensity_logreg"
MODEL_VERSION = "v3.0"
FEATURES = scoring.PROPENSITY_FEATURES
SEED = 42


# --------------------------------------------------------------------------
def _ks(y_true, p):
    order = np.argsort(p)
    y = np.asarray(y_true)[order]
    cp = np.cumsum(y) / max(1, y.sum())
    cn = np.cumsum(1 - y) / max(1, (1 - y).sum())
    return float(np.max(np.abs(cp - cn)))


def _calibration_table(y_true, p, bins=10):
    df = pd.DataFrame({"y": np.asarray(y_true, float), "p": np.asarray(p, float)})
    df["bucket"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("bucket").agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
    return g.reset_index()


# doc "a. Product Propensity". Nhãn = KH mở sản phẩm trong 90 ngày tới, train trên
# tệp CHƯA sở hữu sản phẩm (doc: điều kiện "target_product = 0"). Dùng nhãn adoption
# thay cho "đang sở hữu" để tránh X6 rò rỉ nhãn. `y_holds_product` vẫn có trong bảng
# training để phân tích. Đổi qua LABEL="y_holds_product" nếu muốn model lookalike.
LABEL = "y_adopt_next_90d"
TRAIN_ON_NON_HOLDERS = True


def train_one(pid, group, frame):
    d = frame[frame["product_id"] == pid]
    if TRAIN_ON_NON_HOLDERS:
        d = d[d["y_holds_product"] == 0]
    d = d.reset_index(drop=True)
    X = d[FEATURES].to_numpy(dtype=float)
    y = d[LABEL].to_numpy(dtype=int)
    pos_rate = float(y.mean())

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    ])
    pipe.fit(Xtr, ytr)

    # coefficients back on the ORIGINAL X scale (undo StandardScaler) so they map
    # to the doc's  Z = b + w1*X1 + ...  directly
    scaler = pipe.named_steps["scale"]
    clf = pipe.named_steps["clf"]
    w_scaled = clf.coef_.ravel()
    w = w_scaled / scaler.scale_
    b = float(clf.intercept_[0] - np.sum(w_scaled * scaler.mean_ / scaler.scale_))
    coef = {"intercept": b, **{f: float(wi) for f, wi in zip(FEATURES, w)}}
    odds = {k: float(np.exp(v)) for k, v in coef.items()}

    def _metrics(Xs, ys, split):
        p = pipe.predict_proba(Xs)[:, 1]
        pred = (p >= 0.5).astype(int)
        m = {
            "roc_auc": float(roc_auc_score(ys, p)) if len(np.unique(ys)) > 1 else float("nan"),
            "pr_auc": float(average_precision_score(ys, p)),
            "log_loss": float(log_loss(ys, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
            "brier": float(brier_score_loss(ys, p)),
            "accuracy": float(accuracy_score(ys, pred)),
            "precision": float(precision_score(ys, pred, zero_division=0)),
            "recall": float(recall_score(ys, pred, zero_division=0)),
            "f1": float(f1_score(ys, pred, zero_division=0)),
            "ks": _ks(ys, p),
            "positive_rate": float(np.mean(ys)),
            "n": int(len(ys)),
        }
        cm = confusion_matrix(ys, pred, labels=[0, 1])
        m["tn"], m["fp"], m["fn"], m["tp"] = (int(x) for x in cm.ravel())
        return m, p

    m_tr, _ = _metrics(Xtr, ytr, "train")
    m_te, p_te = _metrics(Xte, yte, "test")

    # 5-fold CV AUC on the full set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    p_cv = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    m_cv = {
        "roc_auc": float(roc_auc_score(y, p_cv)),
        "pr_auc": float(average_precision_score(y, p_cv)),
        "log_loss": float(log_loss(y, np.clip(p_cv, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "n": int(len(y)),
    }

    # refit on ALL data for the deployed model
    pipe.fit(X, y)

    calib = _calibration_table(yte, p_te)

    # plots
    os.makedirs(MODELS, exist_ok=True)
    fpr, tpr, _ = roc_curve(yte, p_te)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC={m_te['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {group}"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(MODELS, f"roc_{group}.png"), dpi=110); plt.close()

    plt.figure(figsize=(4, 4))
    plt.plot(calib["pred"], calib["actual"], "o-", label="model")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("Predicted P"); plt.ylabel("Observed rate")
    plt.title(f"Calibration - {group}"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(MODELS, f"calibration_{group}.png"), dpi=110); plt.close()

    import joblib
    joblib.dump(pipe, os.path.join(MODELS, f"propensity_{group}.joblib"))

    return {
        "product_id": pid, "product_group": group,
        "n_total": int(len(y)), "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "positive_rate": pos_rate,
        "coefficients": coef, "odds_ratios": odds,
        "metrics": {"train": m_tr, "test": m_te, "cv": m_cv},
        "calibration_test": calib.to_dict(orient="records"),
        "pipeline": pipe,
    }


# --------------------------------------------------------------------------
def write_artefacts(results):
    trained_at = datetime.now().replace(microsecond=0)
    reg, coef_rows, metric_rows = [], [], []
    for r in results:
        reg.append({
            "model_id": MODEL_ID, "model_version": MODEL_VERSION, "product_id": r["product_id"],
            "algorithm": "logistic_regression", "trained_at": trained_at,
            "n_train": r["n_train"], "n_test": r["n_test"],
            "positive_rate": round(r["positive_rate"], 6),
            "feature_list": ",".join(FEATURES),
        })
        for fname, cval in r["coefficients"].items():
            coef_rows.append({
                "model_id": MODEL_ID, "model_version": MODEL_VERSION, "product_id": r["product_id"],
                "feature_name": fname, "coefficient": round(cval, 6),
                "odds_ratio": round(r["odds_ratios"][fname], 6),
            })
        for split, mm in r["metrics"].items():
            for k, v in mm.items():
                metric_rows.append({
                    "model_id": MODEL_ID, "model_version": MODEL_VERSION, "product_id": r["product_id"],
                    "dataset_split": split, "metric_name": k,
                    "metric_value": None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 6),
                })

    for name, rows in [("ai_model_registry", reg), ("ai_model_coefficient", coef_rows),
                       ("ai_model_metric", metric_rows)]:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(CSV, f"{name}.csv"), index=False, encoding="utf-8")
        try:
            df.to_parquet(os.path.join(PARQUET, f"{name}.parquet"), index=False)
        except Exception as e:
            print(f"  ! parquet skipped for {name}: {e}")

    metrics_json = {
        "model_id": MODEL_ID, "model_version": MODEL_VERSION,
        "trained_at": trained_at.isoformat(), "features": FEATURES,
        "products": {r["product_group"]: {
            "product_id": r["product_id"], "n_train": r["n_train"], "n_test": r["n_test"],
            "positive_rate": r["positive_rate"], "coefficients": r["coefficients"],
            "odds_ratios": r["odds_ratios"], "metrics": r["metrics"],
        } for r in results},
    }
    os.makedirs(MODELS, exist_ok=True)
    with open(os.path.join(MODELS, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    _write_report(results, trained_at)


def _write_report(results, trained_at):
    L = []
    L.append(f"# Product Propensity - Logistic Regression report\n")
    L.append(f"- model_id: `{MODEL_ID}`  ·  version: `{MODEL_VERSION}`  ·  trained: {trained_at}")
    L.append(f"- features (doc a.): {', '.join(FEATURES)}")
    L.append(f"- 1 model / target product  ·  75/25 stratified split  ·  `class_weight=balanced`  ·  5-fold CV\n")

    L.append("## Test-set metrics\n")
    L.append("| product | n_test | pos_rate | ROC-AUC | PR-AUC | LogLoss | Brier | Acc | Precision | Recall | F1 | KS | CV-AUC |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in results:
        t, cv = r["metrics"]["test"], r["metrics"]["cv"]
        L.append(f"| {r['product_group']} ({r['product_id']}) | {t['n']} | {t['positive_rate']:.3f} | "
                 f"{t['roc_auc']:.3f} | {t['pr_auc']:.3f} | {t['log_loss']:.3f} | {t['brier']:.3f} | "
                 f"{t['accuracy']:.3f} | {t['precision']:.3f} | {t['recall']:.3f} | {t['f1']:.3f} | "
                 f"{t['ks']:.3f} | {cv['roc_auc']:.3f} |")

    L.append("\n## Learned coefficients  (Z = b + Σ wᵢ·Xᵢ, on original 0–1 feature scale)\n")
    for r in results:
        L.append(f"### {r['product_group']} ({r['product_id']})\n")
        L.append("| term | coefficient | odds ratio |")
        L.append("|---|--:|--:|")
        for f in ["intercept"] + FEATURES:
            L.append(f"| {f} | {r['coefficients'][f]:+.4f} | {r['odds_ratios'][f]:.3f} |")
        cm = r["metrics"]["test"]
        L.append(f"\nConfusion matrix (test, threshold 0.5): "
                 f"TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}\n")

    L.append("## Calibration (test, deciles of predicted P)\n")
    for r in results:
        L.append(f"### {r['product_group']}\n")
        L.append("| decile | n | mean predicted P | observed rate |")
        L.append("|--:|--:|--:|--:|")
        for row in r["calibration_test"]:
            L.append(f"| {int(row['bucket'])+1} | {int(row['n'])} | {row['pred']:.3f} | {row['actual']:.3f} |")
        L.append("")

    with open(os.path.join(MODELS, "model_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# --------------------------------------------------------------------------
def apply_scores(results):
    """Rewrite ai_customer_score / ai_recommendation using model P + doc business scores."""
    mart = pd.read_csv(os.path.join(CSV, "customer_360_feature_mart.csv"))
    campaign = pd.read_csv(os.path.join(CSV, "fact_campaign.csv"))
    dim_customer = pd.read_csv(os.path.join(CSV, "dim_customer.csv"))

    feats = scoring.build_propensity_dataset(mart, campaign)
    biz = scoring.business_scores(mart, campaign)

    pipes = {r["product_id"]: r["pipeline"] for r in results}
    feats = feats.sort_values(["customer_id", "product_id"]).reset_index(drop=True)
    feats["propensity_probability"] = np.nan
    for pid, pipe in pipes.items():
        mask = feats["product_id"] == pid
        feats.loc[mask, "propensity_probability"] = pipe.predict_proba(
            feats.loc[mask, FEATURES].to_numpy(dtype=float))[:, 1]

    df = feats.merge(biz, on=["customer_id", "product_id", "product_group"], how="left")
    df["product_propensity_score"] = (df["propensity_probability"] * 100).round(2)
    df["smart_growth_score"] = scoring.smart_growth_score(
        df["product_propensity_score"], df["customer_value_score"], df["intent_signal_score"],
        df["engagement_score"], df["timing_score"], df["relationship_score"]).round(2)
    df["priority_level"] = scoring.priority_level(df["smart_growth_score"])

    ts = pd.Timestamp(scoring.AS_OF) + pd.Timedelta(hours=18)
    df = df.sort_values(["customer_id", "product_id"]).reset_index(drop=True)
    df["score_id"] = [f"SC{i:09d}" for i in range(1, len(df) + 1)]
    ai_score = pd.DataFrame({
        "score_id": df["score_id"], "customer_id": df["customer_id"],
        "snapshot_date": scoring.AS_OF, "product_id": df["product_id"],
        "model_id": MODEL_ID, "model_version": MODEL_VERSION,
        "propensity_probability": df["propensity_probability"].round(4),
        "product_propensity_score": df["product_propensity_score"],
        "customer_value_score": df["customer_value_score"].round(2),
        "intent_signal_score": df["intent_signal_score"].round(2),
        "engagement_score": df["engagement_score"].round(2),
        "timing_score": df["timing_score"].round(2),
        "relationship_score": df["relationship_score"].round(2),
        "smart_growth_score": df["smart_growth_score"],
        "priority_level": df["priority_level"],
        "prediction_timestamp": ts,
    })
    ai_score.to_csv(os.path.join(CSV, "ai_customer_score.csv"), index=False, encoding="utf-8")
    try:
        ai_score.to_parquet(os.path.join(PARQUET, "ai_customer_score.parquet"), index=False)
    except Exception as e:
        print(f"  ! parquet skipped: {e}")

    # rebuild ai_recommendation: top-3 products / customer by smart_growth_score
    supp_cols = ["do_not_contact_flag"]
    dc = dim_customer.set_index("customer_id")
    reco_rows = []
    k = 1
    ANGLE = {"CREDIT_CARD": "Cashback + ưu đãi du lịch, phù hợp mức chi tiêu cao và thu nhập ổn định.",
             "LOAN": "Lãi suất cố định ưu đãi, duyệt nhanh, giải ngân trong ngày.",
             "DEPOSIT": "Cộng thêm lãi suất khi gửi online, tối ưu dòng tiền nhàn rỗi sắp đáo hạn.",
             "INVESTMENT": "Danh mục đầu tư cân bằng, miễn phí phí quản lý 6 tháng đầu."}
    mart_i = mart.set_index("customer_id")
    for cid, sub in df.groupby("customer_id"):
        sub = sub.sort_values("smart_growth_score", ascending=False).reset_index(drop=True)
        seg = dc.at[cid, "customer_segment"] if cid in dc.index else "MASS"
        dnc = bool(dc.at[cid, "do_not_contact_flag"]) if cid in dc.index else False
        active = bool(dc.at[cid, "customer_active_flag"]) if cid in dc.index else True
        s15 = bool(mart_i.at[cid, "serious_complaint_15d"]) if cid in mart_i.index else False
        c7 = int(mart_i.at[cid, "contact_count_7d"]) if cid in mart_i.index else 0
        supp = bool(dnc or (not active) or s15 or c7 >= 2)
        for rank in range(min(3, len(sub))):
            row = sub.iloc[rank]
            grp, sgs, pl = row["product_group"], row["smart_growth_score"], row["priority_level"]
            if supp:
                action, status, channel = "NO_CONTACT", "BLOCKED", "NONE"
            elif seg in ("PRIVATE", "AFFLUENT"):
                action = "RM_CALL" if rank == 0 else "RM_ASSISTED_MESSAGE"
                status, channel = "RM_APPROVAL", "RM_CALL"
            elif pl in ("Very High", "High"):
                action, status, channel = "RM_ASSISTED_MESSAGE", "SENT", "RM_CALL"
            else:
                action, status, channel = "NURTURE", "NEW", "EMAIL"
            timing = ("Trong vòng 48 giờ" if rank == 0 and sgs >= 80 else
                      "Trong tuần này" if rank == 0 else
                      "Trong 2 tuần tới" if rank == 1 else "Nurture 30 ngày")
            reco_rows.append({
                "recommendation_id": f"REC{k:09d}", "customer_id": cid, "score_id": row["score_id"],
                "recommended_product_id": row["product_id"], "recommended_action": action,
                "recommended_channel": channel, "recommended_timing": timing,
                "message_angle": ANGLE[grp],
                "expected_conversion": round(float(np.clip(row["propensity_probability"] * 0.85, 0.01, 0.95)), 4),
                "priority": pl, "suppression_flag": supp,
                "recommendation_timestamp": ts, "status": status, "priority_rank": rank + 1,
            })
            k += 1
    reco = pd.DataFrame(reco_rows)
    reco.to_csv(os.path.join(CSV, "ai_recommendation.csv"), index=False, encoding="utf-8")
    try:
        reco.to_parquet(os.path.join(PARQUET, "ai_recommendation.parquet"), index=False)
    except Exception as e:
        print(f"  ! parquet skipped: {e}")

    # ai_score_reason + rm_action_feedback reference score_id/recommendation_id that changed
    # -> rebuild reasons from top product, drop stale feedback rows that no longer resolve
    _rebuild_reasons(df, results)
    _remap_feedback(reco)
    print(f"  ai_customer_score  -> {len(ai_score):,} rows (model propensity)")
    print(f"  ai_recommendation  -> {len(reco):,} rows")


def _rebuild_reasons(df, results):
    """Explainable AI: đóng góp logit của từng feature vào P của sản phẩm #1.
    contribution = wᵢ·(Xᵢ − X̄ᵢ)  (so với KH trung bình) — chuẩn cho hồi quy logistic."""
    coefs = {r["product_id"]: r["coefficients"] for r in results}
    means = {r["product_id"]: {f: df.loc[df.product_id == r["product_id"], f].mean() for f in FEATURES}
             for r in results}
    rows = []
    k = 1
    for cid, sub in df.groupby("customer_id"):
        top = sub.sort_values("smart_growth_score", ascending=False).iloc[0]
        pid = top["product_id"]
        contribs = []
        for f in FEATURES:
            c = coefs[pid][f] * (float(top[f]) - means[pid][f])
            contribs.append((f, float(top[f]), c))
        contribs.sort(key=lambda t: -abs(t[2]))
        for rank, (fn, fv, c) in enumerate(contribs, 1):
            rows.append({"reason_id": f"RS{k:010d}", "score_id": top["score_id"],
                         "feature_name": fn, "feature_value": round(fv, 4),
                         "contribution_score": round(c, 4),
                         "impact_direction": "POSITIVE" if c >= 0 else "NEGATIVE",
                         "reason_rank": rank})
            k += 1
    rr = pd.DataFrame(rows)
    rr.to_csv(os.path.join(CSV, "ai_score_reason.csv"), index=False, encoding="utf-8")
    try:
        rr.to_parquet(os.path.join(PARQUET, "ai_score_reason.parquet"), index=False)
    except Exception:
        pass


def _remap_feedback(reco):
    """Keep RM feedback for rank-1 non-blocked recos, re-point recommendation_id, and
    resample the outcome from the NEW model expected_conversion so the feedback loop
    stays coherent (P cao hơn -> tỉ lệ Converted thực tế cao hơn)."""
    fb_path = os.path.join(CSV, "rm_action_feedback.csv")
    if not os.path.exists(fb_path):
        return
    rng = np.random.default_rng(SEED)
    fb = pd.read_csv(fb_path)
    r1 = reco[(reco.priority_rank == 1) & (reco.status != "BLOCKED")].drop_duplicates("customer_id")
    r1 = r1.set_index("customer_id")[["recommendation_id", "expected_conversion"]]
    fb = fb[fb["customer_id"].isin(r1.index)].copy()
    fb["recommendation_id"] = fb["customer_id"].map(r1["recommendation_id"])
    ec = fb["customer_id"].map(r1["expected_conversion"]).to_numpy()

    u = rng.random(len(fb))
    p_conv = np.clip(ec * 0.9, 0.02, 0.9)
    resp = np.where(u < p_conv, "CONVERTED",
           np.where(u < p_conv + 0.12, "APPLIED",
           np.where(u < p_conv + 0.34, "INTERESTED",
           np.where(u < p_conv + 0.52, "CALLBACK",
           np.where(u < p_conv + 0.80, "NO_RESPONSE", "NOT_INTERESTED")))))
    fb["customer_response"] = resp
    fb["converted_flag"] = resp == "CONVERTED"
    fb["application_flag"] = np.isin(resp, ["APPLIED", "CONVERTED"])
    fb["appointment_flag"] = np.isin(resp, ["INTERESTED", "APPLIED", "CONVERTED"]) & (rng.random(len(fb)) < 0.5)
    fb["result_status"] = np.select(
        [np.isin(resp, ["CONVERTED", "APPLIED"]), resp == "CALLBACK", resp == "NO_RESPONSE"],
        ["DONE", "SCHEDULED", "NO_ANSWER"], default="CONTACTED")
    fb["reason_not_interested"] = np.where(resp == "NOT_INTERESTED", fb["reason_not_interested"].fillna("Chưa có nhu cầu"), None)
    fb.to_csv(fb_path, index=False, encoding="utf-8")
    try:
        fb.to_parquet(os.path.join(PARQUET, "rm_action_feedback.parquet"), index=False)
    except Exception:
        pass


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="rewrite ai_customer_score / ai_recommendation with model output")
    args = ap.parse_args()

    frame = pd.read_csv(os.path.join(CSV, "ml_propensity_training_set.csv"))
    print(f"Training set: {len(frame):,} rows  ({frame['customer_id'].nunique():,} customers x "
          f"{frame['product_id'].nunique()} products)")

    results = []
    for grp in scoring.GROUPS:
        pid = scoring.TARGET_PRODUCTS[grp]
        r = train_one(pid, grp, frame)
        results.append(r)
        t = r["metrics"]["test"]
        print(f"  {grp:<12} ({pid})  pos={r['positive_rate']:.3f}  "
              f"AUC={t['roc_auc']:.3f}  PR-AUC={t['pr_auc']:.3f}  LogLoss={t['log_loss']:.3f}  "
              f"Acc={t['accuracy']:.3f}  F1={t['f1']:.3f}  CV-AUC={r['metrics']['cv']['roc_auc']:.3f}")

    write_artefacts(results)
    print(f"\nArtefacts -> models/  and  data/csv/ai_model_*.csv")

    if args.apply:
        print("\n--apply: rewriting scores with logistic-regression propensity ...")
        apply_scores(results)
        print("Now reload:  psql ... -f sql/04_load_models.sql")


if __name__ == "__main__":
    main()
