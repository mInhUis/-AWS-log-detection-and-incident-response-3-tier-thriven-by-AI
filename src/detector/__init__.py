"""Tier 2 — DeepLog LSTM on Drain3 integer log-key sequences.

Implements Du et al. (2017): an LSTM trained on normal sequences predicts the
next log key given a sliding window of length `window_size`. At inference,
if the observed next key is NOT in the model's top-k predictions, the window
is flagged as anomalous.

Strictly mathematical — no semantic reasoning. The LSTM operates on integer
key IDs only; never embed event names, principals, IPs, or message strings
here (CLAUDE.md §3 locked decision). Semantics belong in Tier 3.

Trained semi-supervised on normal sequences only. Output: confirmed-anomalous
sequences (~0.1% of original event volume) forwarded to Tier 3.
"""
