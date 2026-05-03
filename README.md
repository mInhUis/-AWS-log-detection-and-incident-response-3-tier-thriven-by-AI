# Cost-Aware Cascaded Anomaly Detection on AWS CloudTrail

Bachelor's thesis prototype: a 3-tier cascade combining Isolation Forest
(Tier 1, CPU), DeepLog LSTM (Tier 2, GPU), and Llama 3.1 8B 4-bit + RAG
(Tier 3, local LLM) for security incident analysis on CloudTrail logs.

The thesis contribution is the **integrated cost-aware cascade design**, not
any individual algorithm — each tier processes a strictly smaller fraction of
events than the one before it.

> **`CLAUDE.md` is the source of truth** for architecture, scope, and
> coding standards. Read it before contributing.

## Pipeline at a glance

```
Raw CloudTrail JSON
   ─► Tier 1  Isolation Forest on entity-window features      (~95% rejected)
   ─► Tier 2  DeepLog LSTM on Drain3 key sequences            (~99.9% rejected cumulative)
   ─► Tier 3  Llama 3.1 8B 4-bit + RAG over MITRE ATT&CK      → incident report
```

## Hardware assumptions

- **Heavy training / inference:** Google Colab Pro (single GPU, 12 h sessions).
- **Local development / demo:** Dell Inspiron 5425, Ryzen 7 5825U, integrated
  Radeon graphics, 16 GB RAM. Treated as **CPU-only** for ML.

## Layout

See [`CLAUDE.md` §8](CLAUDE.md) for the authoritative directory layout.

## Quick start (after Day 2 — once synthetic normal exists)

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                            # smoke test
```

## Status

Day 1 (2026-04-26): repository scaffolded, scope locked, synthetic-normal
generation queued for Day 2. See [`data/README.md`](data/README.md).
