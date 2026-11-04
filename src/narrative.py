"""Template-based alert narratives (NO LLM call by default).

Each narrative is grounded in the transaction's own values and in
its velocity features, and lists the top contributing factors
explicitly. No narrative is presented as LLM-written: the file
name / dashboard label is "template narrative" unless an LLM was
actually called, which does not happen in this project (see
src/llm_client.py, disabled by default).

Top-factor logic (rule-based, documented):
  - amount > 250, is_night, is_new_device, count_1h >= 3,
    count_24h >= 5, amount_vs_user_mean > 2, product in {C, H}
  Factors present for a transaction are ranked by a fixed weight
  order (amount, velocity, new device, night, product) and the
  top three are listed. Weights are for explanation ranking only,
  not model SHAP values.
"""

from typing import Dict, List


def top_factors(row) -> List[str]:
    factors = []
    if row["amount"] > 250:
        factors.append(f"high amount (${row['amount']:.2f})")
    if row.get("count_1h", 0) >= 3:
        factors.append(f"{int(row['count_1h'])} prior transactions from this user in the last hour")
    if row.get("count_24h", 0) >= 5:
        factors.append(f"{int(row['count_24h'])} prior transactions from this user in the last 24 hours")
    if row.get("is_new_device", 0) == 1:
        factors.append(f"new device type for this user ({row['device_type']})")
    if row.get("amount_vs_user_mean", 0) > 2:
        factors.append(f"amount is {row['amount_vs_user_mean']:.1f} standard deviations above this user's prior mean")
    if row.get("is_night", 0) == 1:
        factors.append(f"transaction at night (hour {int(row['hour'])})")
    if row.get("product_cd") in ("C", "H"):
        factors.append(f"product code {row['product_cd']}")
    return factors[:3]


def generate_narrative(row, predicted_probability: float) -> Dict[str, object]:
    factors = top_factors(row)
    if factors:
        factor_text = "; ".join(factors)
        text = (
            f"Template narrative — Transaction {row['transaction_id']} "
            f"for user {row['user_id']} (${row['amount']:.2f}, {row['product_cd']}, "
            f"{row['device_type']}) was flagged with a model score of "
            f"{predicted_probability:.3f}. Key factors: {factor_text}. "
            f"Review the user's recent velocity (1h count: {int(row.get('count_1h', 0))}, "
            f"24h count: {int(row.get('count_24h', 0))}) before deciding."
        )
    else:
        text = (
            f"Template narrative — Transaction {row['transaction_id']} "
            f"for user {row['user_id']} (${row['amount']:.2f}) was flagged with "
            f"a model score of {predicted_probability:.3f}, but no single "
            f"rule factor stood out; review amount, device and recent velocity."
        )
    return {"narrative": text, "top_factors": factors, "generated_by": "template (no LLM call)"}
