"""Train the fraud triage model and evaluate precision at analyst capacity.

Pipeline:
  1. Generate synthetic transactions (src/generate_data.py) if
     data/transactions_raw.csv is missing, else use the existing file.
  2. Add velocity features (src/features.py, prior transactions only).
  3. Split: train 70% / validation 15% / test 15%, stratified, seed 42.
     The cost-based threshold is chosen on VALIDATION only.
  4. Model: LightGBM if importable, else HistGradientBoostingClassifier.
     Categorical features (product, device, email, card) are one-hot
     encoded via pandas get_dummies (fit on train columns, reindexed).
  5. Cost model (documented assumptions, not a real fraud-ops cost):
       missed-fraud cost per missed fraud = transaction amount
       false-alarm cost per false alert   = $20 (analyst review)
     Threshold = the threshold on validation that minimises
     total expected cost = missed_cost + false_alarm_cost.
  6. Precision at fixed analyst capacity, evaluated on TEST:
       capacity = 100 alerts/day (documented assumption).
       test_days = (max test timestamp - min test timestamp) in days.
       K = round(capacity * test_days), at least 1.
       Take the top-K test transactions by model score, compute
       precision@K and recall@K.
  7. Charts: ROC, precision-recall, precision@K curve, feature importance.
  8. Alerts sample: top 200 test transactions by score, with template
     narratives from src/narrative.py, saved to outputs/alerts_sample.csv.

Run from the project root:
    python src/train.py
Outputs: outputs/metrics.json, outputs/charts/*.png, outputs/alerts_sample.csv
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import add_velocity_features
from generate_data import main as generate_main
from narrative import generate_narrative

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "transactions_raw.csv"
DATA_FEAT = ROOT / "data" / "transactions.csv"
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"
SEED = 42
CAPACITY_PER_DAY = 100
FALSE_ALARM_COST = 20.0

NUM_FEATURES = [
    "amount", "amount_log", "hour", "is_night", "is_new_device",
    "count_1h", "sum_1h", "count_24h", "sum_24h", "count_7d",
    "time_since_last_hours", "amount_vs_user_mean",
]
CAT_FEATURES = ["product_cd", "device_type", "email_domain", "card_type"]


def get_model():
    try:
        from lightgbm import LGBMClassifier

        return "LightGBM", LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=SEED, n_jobs=4, verbose=-1,
        )
    except Exception as exc:
        from sklearn.ensemble import HistGradientBoostingClassifier

        print(f"LightGBM unavailable ({exc}); using HistGradientBoosting.")
        return "HistGradientBoosting", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=SEED
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    if not DATA_RAW.exists():
        generate_main()
    raw = pd.read_csv(DATA_RAW, parse_dates=["transaction_dt"])
    feat = add_velocity_features(raw)
    feat.to_csv(DATA_FEAT, index=False)

    X = pd.get_dummies(feat[NUM_FEATURES + CAT_FEATURES], columns=CAT_FEATURES, dtype=float)
    y = feat["is_fraud"].values
    # CHRONOLOGICAL split (not random): a random stratified split would
    # make the test set span the whole 90-day period, so "alerts per day"
    # over the test period would be meaningless (K would equal almost
    # the whole test set). Chronological order is already the order of
    # feat (features.py returns it sorted by transaction_dt).
    # Train = first 70%, validation = next 15%, test = last 15%.
    n = len(feat)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    idx_tr = feat.index.values[:n_train]
    idx_val = feat.index.values[n_train:n_train + n_val]
    idx_te = feat.index.values[n_train + n_val:]
    X_tr, y_tr = X.loc[idx_tr], y[:n_train]
    X_val, y_val = X.loc[idx_val], y[n_train:n_train + n_val]
    X_te, y_te = X.loc[idx_te], y[n_train + n_val:]
    model_name, model = get_model()
    model.fit(X_tr, y_tr)
    p_val = model.predict_proba(X_val)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]

    # Cost-based threshold on validation only
    val_amounts = feat.loc[idx_val, "amount"].values
    thresholds = np.unique(np.quantile(p_val, np.linspace(0.5, 0.999, 200)))
    best_thr, best_cost = 0.5, float("inf")
    for thr in thresholds:
        pred = p_val >= thr
        missed = ((~pred) & (y_val == 1))
        false_al = (pred & (y_val == 0))
        cost = float((val_amounts[missed]).sum() + FALSE_ALARM_COST * int(false_al.sum()))
        if cost < best_cost:
            best_cost, best_thr = cost, float(thr)

    # Test metrics
    auc = float(roc_auc_score(y_te, p_te))
    ap = float(average_precision_score(y_te, p_te))
    pred_thr = p_te >= best_thr
    tp = int(((pred_thr) & (y_te == 1)).sum())
    fp = int(((pred_thr) & (y_te == 0)).sum())
    fn = int(((~pred_thr) & (y_te == 1)).sum())
    prec_thr = tp / (tp + fp) if (tp + fp) else 0.0
    rec_thr = tp / (tp + fn) if (tp + fn) else 0.0

    # Precision at fixed analyst capacity on TEST
    test_times = feat.loc[idx_te, "transaction_dt"]
    test_days = float((test_times.max() - test_times.min()).total_seconds() / 86400.0)
    K = max(1, int(round(CAPACITY_PER_DAY * test_days)))
    K = min(K, len(y_te))
    order = np.argsort(p_te)[::-1]
    topk = order[:K]
    prec_k = float(y_te[topk].mean())
    rec_k = float(y_te[topk].sum() / max(1, int(y_te.sum())))

    # Charts
    fpr, tpr, _ = roc_curve(y_te, p_te)
    plt.figure(figsize=(7, 5)); plt.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}"); plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC — fraud (held-out test)"); plt.legend(); plt.tight_layout()
    plt.savefig(CHARTS / "roc.png", dpi=150); plt.close()

    prec_curve, rec_curve, _ = precision_recall_curve(y_te, p_te)
    plt.figure(figsize=(7, 5)); plt.plot(rec_curve, prec_curve, label=f"PR AUC (AP)={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall — fraud (held-out test)"); plt.legend(); plt.tight_layout()
    plt.savefig(CHARTS / "pr_curve.png", dpi=150); plt.close()

    ks = [50, 100, 200, 500, 1000, 2000]
    ks = [k for k in ks if k <= len(y_te)]
    precs = [float(y_te[np.argsort(p_te)[::-1][:k]].mean()) for k in ks]
    plt.figure(figsize=(7, 5)); plt.plot(ks, precs, "o-"); plt.xlabel("Top-K alerts"); plt.ylabel("Precision@K")
    plt.title("Precision at K — fraud (held-out test)"); plt.tight_layout()
    plt.savefig(CHARTS / "precision_at_k.png", dpi=150); plt.close()

    try:
        importances = model.feature_importances_
        feat_names = list(X.columns)
    except Exception:
        from sklearn.inspection import permutation_importance

        perm = permutation_importance(model, X_te[:2000], y_te[:2000], n_repeats=3, random_state=SEED, n_jobs=4)
        importances = perm.importances_mean; feat_names = list(X.columns)
    order_imp = np.argsort(importances)[::-1][:12]
    plt.figure(figsize=(8, 5)); plt.barh([feat_names[i] for i in order_imp][::-1], importances[order_imp][::-1])
    plt.xlabel("Importance"); plt.title("Top features"); plt.tight_layout()
    plt.savefig(CHARTS / "feature_importance.png", dpi=150); plt.close()

    # Alerts sample with template narratives (top 200 by test score)
    test_feat = feat.loc[idx_te].reset_index(drop=True)
    test_feat = test_feat.assign(score=p_te)
    top = test_feat.sort_values("score", ascending=False).head(200).copy()
    narratives = []
    for _, row in top.iterrows():
        narr = generate_narrative(row, float(row["score"]))
        narratives.append(narr["narrative"])
    top["template_narrative"] = narratives
    top.to_csv(OUT / "alerts_sample.csv", index=False)

    metrics = {
        "run_date": "2026-11-04",
        "data_source": "Synthetic transactions mimicking IEEE-CIS schema (see DATA_SOURCES.md) — IEEE-CIS was NOT used (Kaggle auth required)",
        "data_file": "data/transactions.csv",
        "n_total": int(len(feat)),
        "n_train": int(len(y_tr)),
        "n_validation": int(len(y_val)),
        "n_test": int(len(y_te)),
        "fraud_rate_total": float(feat["is_fraud"].mean()),
        "fraud_rate_test": float(y_te.mean()),
        "seed": SEED,
        "model": model_name,
        "numeric_features": NUM_FEATURES,
        "categorical_features_one_hot": CAT_FEATURES,
        "roc_auc_test": auc,
        "pr_auc_average_precision_test": ap,
        "cost_model": {"missed_fraud_cost": "transaction amount", "false_alarm_cost_usd": FALSE_ALARM_COST},
        "cost_threshold_validation": float(best_thr),
        "precision_at_cost_threshold_test": float(prec_thr),
        "recall_at_cost_threshold_test": float(rec_thr),
        "analyst_capacity_per_day": CAPACITY_PER_DAY,
        "test_period_days": test_days,
        "precision_at_capacity_K": K,
        "precision_at_capacity": prec_k,
        "recall_at_capacity": rec_k,
        "narrative_method": "template-based, grounded in transaction velocity/amount/device factors (no LLM call)",
        "llm_enabled": False,
        "human_review_note": "No human review was conducted and no review accuracy percentage is claimed. See docs/human_review_protocol.md for how a review would be rated.",
    }
    with open(OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
