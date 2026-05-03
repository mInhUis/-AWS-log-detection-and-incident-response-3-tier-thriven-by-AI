# `data/` — Datasets and the train / val / test contract

> Status as of **2026-04-26** (Day 1 of the 13-day plan).

## What is in this directory today

| Path | Status | Purpose |
|------|--------|---------|
| `flaws_merged.jsonl` | ✅ present (~2.4 GB) | Raw flaws.cloud CloudTrail attack corpus. Source of the **labelled attack slice** for val + test. |
| `raw/` | empty | Drop zone for synthetic normal + any new raw logs. |
| `processed/` | empty | Drain3 templates, integer key sequences, per-entity-window feature vectors, train/val/test split files. |
| `rag/` | empty | MITRE ATT&CK STIX chunks + flaws.cloud writeups (built Day 4). |

## ⚠️ Synthetic normal data — NOT YET GENERATED

The pipeline **cannot run end-to-end** until synthetic normal CloudTrail
exists in `data/raw/synthetic_normal.jsonl`. Both Tier 1 (Isolation Forest
on entity-window features) and Tier 2 (DeepLog on Drain3 sequences) train
on this corpus.

Generation is the **first task of Day 2**. Required design choices, all to
be confirmed before code is written:

- Number of distinct IAM principals (target: ~50–200; enough for an honest
  identity-based split).
- Event-name distribution (sample from common AWS read APIs: STS, IAM list/get,
  S3 list/get, EC2 describe).
- Time-of-day priors (business-hours bias).
- Session length distribution.
- Total event volume (target: 1–3 M events; gives Tier 1 enough
  entity-windows for stable contamination tuning).
- Output path: `data/raw/synthetic_normal.jsonl`.

## Train / val / test contract (CLAUDE.md §9 — non-negotiable)

| Split | Composition | Purpose |
|-------|-------------|---------|
| **train** | synthetic normal only | Fit Tier 1 IF and Tier 2 DeepLog. Both tiers see this exact set. |
| **val** | held-out synthetic normal **+** small labelled slice of flaws.cloud | Tune Tier 1 `contamination` and Tier 2 `top_k ∈ {3,5,7,9}`. |
| **test** | held-out flaws.cloud attacks **+** held-out synthetic normal | Reported numbers. Touched **exactly once**. |

### Identity-based splitting

No IAM principal appears in more than one of `{train, val, test}` — for both
Tier 1 feature vectors and Tier 2 log-key sequences. This is the most common
leakage source in log-anomaly papers and is enforced once in
`src.preprocessing` so downstream code cannot reintroduce it.

### What we **never** do

- Re-split because results look bad (test-set fitting).
- Tune any hyperparameter on the test set.
- Mix train and test IAM principals via the synthetic normal pool.
- Report only accuracy on this imbalanced data — always P/R/F1.

## Reproducibility

Everything in this directory except this README is gitignored. To rebuild
on a fresh checkout, run (once Day 2 is complete):

```bash
python scripts/generate_synthetic_normal.py --events 1000000 --principals 100
python scripts/preprocess.py --mode full
```

Both scripts will read paths from `configs/default.yaml`, which env-vars
override per CLAUDE.md §5.5.
