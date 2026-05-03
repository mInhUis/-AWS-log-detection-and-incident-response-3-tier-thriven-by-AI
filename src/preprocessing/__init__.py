"""Preprocessing: feature aggregation + Drain3 templating + identity-based splits.

Two outputs feed the two ML tiers:

  preprocess.py   -> Tier 1 features
      Aggregates events into per-(IAM-principal, time-window) feature vectors:
      distinct API count, sensitive-action count, error rate, geographic spread,
      time-of-day deviation. Scalar features only — Tier 1 does not see sequences.

  drain_parser.py -> Tier 2 sequences
      Drain3 over event names produces integer log-key sequences per session,
      plus a serialised vocabulary (key_id <-> event_name).

Splits are identity-based (CLAUDE.md §9): no IAM principal appears in more than
one of {train, val, test}. This is the most common leakage source in log-anomaly
papers; we enforce it here once so downstream code cannot reintroduce it.
"""
