"""Transaction-velocity features (computed from raw transactions only).

For each transaction, in chronological order per user:
  - count_1h, sum_1h, count_24h, sum_24h, count_7d: prior transactions
    for the same user in the preceding 1 hour / 24 hours / 7 days.
  - time_since_last_hours: hours since the user's previous transaction.
  - amount_vs_user_mean: amount minus the user's prior mean amount,
    divided by (prior std + 1) — a z-like deviation, computed only
    from prior transactions to avoid leakage.
  - is_night: hour in 0..4.
  - amount_log: log1p(amount).

Implementation is a straightforward per-user loop over the sorted
frame; with 60,000 rows it runs in seconds. No label information
is used in any feature.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["transaction_dt"] = pd.to_datetime(df["transaction_dt"])
    df = df.sort_values(["user_id", "transaction_dt"]).reset_index(drop=True)
    n = len(df)
    count_1h = np.zeros(n, dtype=int)
    sum_1h = np.zeros(n, dtype=float)
    count_24h = np.zeros(n, dtype=int)
    sum_24h = np.zeros(n, dtype=float)
    count_7d = np.zeros(n, dtype=int)
    time_since_last = np.full(n, np.nan)
    amount_vs_user_mean = np.zeros(n, dtype=float)

    for user, idx in df.groupby("user_id").groups.items():
        idx = np.array(sorted(idx))
        times = df.loc[idx, "transaction_dt"].values.astype("datetime64[s]").astype(np.int64)
        amounts = df.loc[idx, "amount"].values.astype(float)
        for j, pos in enumerate(idx):
            t = times[j]
            # prior transactions only (strictly before, plus same-second prior rows are excluded by < t)
            prior = times[:j] < t
            if not prior.any():
                continue
            prior_times = times[:j][prior]
            prior_amounts = amounts[:j][prior]
            dt_hours = (t - prior_times) / 3600.0
            m1 = dt_hours <= 1
            m24 = dt_hours <= 24
            m7 = dt_hours <= 24 * 7
            count_1h[pos] = int(m1.sum()); sum_1h[pos] = float(prior_amounts[m1].sum())
            count_24h[pos] = int(m24.sum()); sum_24h[pos] = float(prior_amounts[m24].sum())
            count_7d[pos] = int(m7.sum())
            time_since_last[pos] = float(dt_hours.min())
            mean_prior = prior_amounts.mean(); std_prior = prior_amounts.std() if len(prior_amounts) > 1 else 1.0
            amount_vs_user_mean[pos] = float((amounts[j] - mean_prior) / (std_prior + 1.0))

    df["count_1h"] = count_1h
    df["sum_1h"] = sum_1h
    df["count_24h"] = count_24h
    df["sum_24h"] = sum_24h
    df["count_7d"] = count_7d
    df["time_since_last_hours"] = time_since_last
    df["time_since_last_hours"] = df["time_since_last_hours"].fillna(24.0)
    # Clip the deviation: with only a few prior transactions the
    # denominator can be near 1, producing extreme values that would
    # dominate a model. Clipping to [-10, 10] keeps the signal usable.
    df["amount_vs_user_mean"] = np.clip(amount_vs_user_mean, -10, 10)
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] <= 4)).astype(int)
    df["amount_log"] = np.log1p(df["amount"])
    return df.sort_values("transaction_dt").reset_index(drop=True)
