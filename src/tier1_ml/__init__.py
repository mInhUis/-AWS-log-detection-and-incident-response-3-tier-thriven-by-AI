"""Tier 1 — Isolation Forest / PyOD on entity-window feature vectors.

Operates on the scalar feature vectors produced by `src.preprocessing.preprocess`
(NOT on raw events or Drain3 sequences). Trained semi-supervised on synthetic
normal entity-windows only — the model learns the shape of legitimate behaviour
and flags anything that deviates.

Why Isolation Forest here:
  - Unsupervised: we have no labelled normal/abnormal at the volume we need.
  - Tabular feature input matches our aggregation step naturally.
  - Single interpretable hyperparameter (contamination) — easy to tune on val.
  - CPU-only, fractions of a cent per million events, satisfying the cost
    contract for the cheapest tier.

Eliminates ~95% of event volume before any GPU computation. Output: a list of
suspicious entity-windows whose member sessions are forwarded to Tier 2.
"""
