"""Generate SYNTHETIC fraud transactions mimicking IEEE-CIS schema.

IEEE-CIS Fraud Detection (590K+ transactions) is distributed via
Kaggle, which requires Kaggle authentication. No Kaggle credentials
were available, so IEEE-CIS was NOT downloaded and is NOT claimed
here. Instead, this script generates a synthetic transaction set
with a similar shape: transaction amount, product code, card/user
identifiers, device and email-domain fields, and a timestamp.

Fraud is generated from a known, documented risk rule (see below),
so the task is learnable but the labels are synthetic — they are
NOT real fraud labels, and no IEEE-CIS figure is reproduced.

Risk rule (documented, not hidden):
  logit = -4.5
        + 1.2 * (amount > 250)
        + 0.8 * (hour in 0..4)
        + 1.0 * (device is new for this user)
        + 0.7 * (velocity_1h >= 3)
        + 0.6 * (product in {C, H})
  fraud ~ Bernoulli(sigmoid(logit)), with a small random component.

Usage:
    python src/generate_data.py
    # writes data/transactions_raw.csv (no velocity features yet;
    # run src/features.py to add them, or src/train.py which does both)
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "transactions_raw.csv"
SEED = 42
N = 60000
N_USERS = 8000


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    user_ids = rng.integers(0, N_USERS, size=N)
    # Timestamps over 90 days, in seconds since start
    start = pd.Timestamp("2024-01-01")
    days = rng.uniform(0, 90, size=N)
    times = start + pd.to_timedelta(days, unit="D")
    times = np.sort(times)  # chronological order helps velocity features
    # Re-shuffle user assignment after sorting times? Keep times sorted,
    # users random per row (velocity is computed per user in time order).
    amounts = rng.lognormal(mean=np.log(80), sigma=1.0, size=N).round(2)
    amounts = np.clip(amounts, 1, 5000)
    products = rng.choice(["W", "C", "R", "H", "S"], size=N, p=[0.35, 0.25, 0.15, 0.15, 0.10])
    devices = rng.choice(["mobile", "desktop", "tablet"], size=N, p=[0.55, 0.35, 0.10])
    email_domains = rng.choice(["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "other"], size=N, p=[0.4, 0.25, 0.15, 0.1, 0.1])
    card_types = rng.choice(["visa", "mastercard", "amex", "discover"], size=N, p=[0.45, 0.35, 0.12, 0.08])
    hours = pd.DatetimeIndex(times).hour.values

    # New-device flag: first transaction for a user-device pair in this
    # synthetic history is treated as "new" with some probability, to
    # create a learnable signal without leaking the label.
    df = pd.DataFrame({
        "transaction_id": np.arange(1, N + 1),
        "transaction_dt": times,
        "user_id": user_ids,
        "amount": amounts,
        "product_cd": products,
        "device_type": devices,
        "email_domain": email_domains,
        "card_type": card_types,
        "hour": hours,
    })
    # Velocity proxy for generation only (recomputed properly in features.py)
    df = df.sort_values("transaction_dt").reset_index(drop=True)
    # Approximate 1h count per user via a rolling count on sorted data
    df["velocity_1h_gen"] = 0
    # Simple generation-time signal: users with many transactions in a short window
    # are marked by counting prior rows for same user within 1h using groupby + time diff
    # (vectorised approximation: count of prior user rows, capped)
    df["prior_user_count"] = df.groupby("user_id").cumcount()
    is_new_device = np.zeros(N, dtype=bool)
    seen = set()
    for i, (u, d) in enumerate(zip(df["user_id"].values, df["device_type"].values)):
        key = (int(u), str(d))
        if key not in seen:
            seen.add(key)
            is_new_device[i] = rng.random() < 0.7
    df["is_new_device"] = is_new_device.astype(int)
    velocity_flag = (df["prior_user_count"] >= 3).astype(int).values
    logit = (
        -4.5
        + 1.2 * (df["amount"].values > 250)
        + 0.8 * ((df["hour"].values >= 0) & (df["hour"].values <= 4))
        + 1.0 * df["is_new_device"].values
        + 0.7 * velocity_flag
        + 0.6 * df["product_cd"].isin(["C", "H"]).values.astype(float)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["is_fraud"] = (rng.random(N) < prob).astype(int)
    df = df.drop(columns=["velocity_1h_gen", "prior_user_count"])
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} shape={df.shape} fraud_rate={df['is_fraud'].mean():.4f}")


if __name__ == "__main__":
    main()
