# Human Review Protocol — Fraud Alert Summaries (Protocol Only)

**No human review was conducted for this project, and no review
accuracy percentage is claimed anywhere in this repo.**

This protocol describes how a review WOULD be run, so that a future
reviewer can reproduce it and report a real, audited number.

## Sample
- Take the top alerts from outputs/alerts_sample.csv (or a fresh
  test-set sample), blinded to the actual fraud label.
- Recommended sample size for a first review: 100 alerts, sampled
  across score bands (not only the highest scores).

## Rating scale (per alert)
- `correct` — the narrative's top factors are all true for the
  transaction, and no material factor is invented.
- `partially_correct` — at least one top factor is true, but one
  factor is wrong, vague, or missing a material factor.
- `incorrect` — the narrative's main factor is false, or the
  narrative invents a factor not present in the transaction data.

## What to check (checklist)
1. Amount, product, device, user and timestamp match the transaction row.
2. Each listed velocity factor (1h/24h counts) matches the computed
   features in the data file.
3. "New device" matches is_new_device in the data file.
4. No factor is stated that is not in the data or features.

## Reporting (only after a real review)
Report: n reviewed, % correct / partially_correct / incorrect,
with the reviewer name/date and the review file (see review_template.csv).
Do NOT report a percentage until the review file is filled in.
