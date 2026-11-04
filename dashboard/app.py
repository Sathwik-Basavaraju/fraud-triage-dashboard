"""Streamlit Fraud Triage Dashboard (template narratives only).

Run from the project root:
    streamlit run dashboard/app.py

The dashboard reads outputs/metrics.json and outputs/alerts_sample.csv
(if present — run `python src/train.py` first to generate them).
Alert narratives shown here are TEMPLATE-generated (see src/narrative.py);
no LLM is called by this app.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "outputs" / "metrics.json"
ALERTS = ROOT / "outputs" / "alerts_sample.csv"


def main() -> None:
    st.title("Fraud Triage Dashboard — Template Narratives")
    st.caption("Synthetic data for demonstration only. Narratives are template-generated, not LLM-written.")
    if METRICS.exists():
        metrics = json.loads(METRICS.read_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC AUC (test)", f"{metrics['roc_auc_test']:.3f}")
        c2.metric("PR AUC (test)", f"{metrics['pr_auc_average_precision_test']:.3f}")
        c3.metric("Precision @ capacity", f"{metrics['precision_at_capacity']:.3f}")
        c4.metric("Recall @ capacity", f"{metrics['recall_at_capacity']:.3f}")
        st.write(f"Model: {metrics['model']} | Capacity: {metrics['analyst_capacity_per_day']} alerts/day | "
                 f"Top-K: {metrics['precision_at_capacity_K']} | Test period: {metrics['test_period_days']:.1f} days")
    else:
        st.warning("Run `python src/train.py` first to generate outputs/metrics.json.")
        return
    if not ALERTS.exists():
        st.warning("Run `python src/train.py` first to generate outputs/alerts_sample.csv.")
        return
    alerts = pd.read_csv(ALERTS)
    st.subheader("Top alerts (by model score)")
    st.dataframe(alerts[["transaction_id", "transaction_dt", "user_id", "amount", "product_cd", "device_type", "is_fraud", "score"]].head(100))
    st.subheader("Case narrative")
    sel = st.selectbox("Select a transaction", alerts["transaction_id"].tolist())
    row = alerts[alerts["transaction_id"] == sel].iloc[0]
    st.write(f"**Transaction {row['transaction_id']}** — user {row['user_id']}, ${row['amount']:.2f}, "
             f"score {row['score']:.3f}, actual fraud label (test only): {int(row['is_fraud'])}")
    st.write(row["template_narrative"])


if __name__ == "__main__":
    main()
