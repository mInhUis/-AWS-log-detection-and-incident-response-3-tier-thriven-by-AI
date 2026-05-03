"""Top-level package for the cost-aware cascaded anomaly detection thesis.

Pipeline (CLAUDE.md §3):
    Raw CloudTrail JSON
      -> src.preprocessing  (parse, aggregate, split)
      -> src.tier1_ml       (Isolation Forest on entity-window features)
      -> src.detector       (DeepLog LSTM on Drain3 key sequences)
      -> src.llm            (Llama 3.1 8B 4-bit + RAG, semantic analysis)
      -> src.pipeline       (single cascade.process() that wires Tier1->Tier2->Tier3)
      -> src.evaluation     (per-tier P/R/F1, ablations, cost table)

The cost-aware cascade is the thesis contribution: each tier processes a strictly
smaller fraction of events than the previous one, keeping per-event compute cost
sub-linear in attack rarity.
"""
