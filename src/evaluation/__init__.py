"""Evaluation: per-tier P/R/F1, confusion matrices, ablations, cost table.

Discipline (CLAUDE.md §7):
  - Three splits, not two. Train -> Validation (tune) -> Test (touched once).
  - Always report Precision, Recall, F1; accuracy is footnote-only.
  - Always show confusion matrix per tier and end-to-end.
  - Per-attack-type breakdown for flaws.cloud levels 1-6 where labelled.
  - Per-tier metrics + end-to-end metrics — both are mandatory.
  - Bootstrapped 95% CIs on F1 (flaws.cloud is small; honest CIs matter).

Ablations to run:
  - Cascade vs. Tier 2 alone (no Isolation Forest filter).
  - Cascade vs. Tier 3 alone (LLM sees everything Tier 1 flags).
  - top_k sweep over {3, 5, 7, 9} for Tier 2.

Cost analysis: measure events/sec and latency at each tier on local hardware,
fill in the §3 table. This measurement is the thesis contribution.
"""
