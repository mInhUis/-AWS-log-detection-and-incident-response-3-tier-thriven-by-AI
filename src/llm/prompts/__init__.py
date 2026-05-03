"""Versioned prompt templates for Tier 3 LLM incident analysis (Days 5-6).

**Why versioned prompt files?**
Prompt templates are first-class experimental artifacts (CLAUDE.md §10).
Every evaluation run must record which prompt version produced its results
so the thesis ablation section can compare v1 (zero-shot) vs v2 (RAG-augmented)
under controlled conditions.  Storing prompts in separate versioned Python
modules makes this traceable via ``git log`` and prevents silent template drift
between runs.

**Versioning contract:**
  - Never modify an existing version file once evaluation has started.
  - Bump the version number (v3.py, etc.) for any change that affects output.
  - The ``prompt_version`` key in ``configs/default.yaml`` selects the active
    template; ``llama_inference.py`` reads that key at runtime.

**Available versions:**
    v1: Zero-shot, no ATT&CK context.  Baseline for ablation.
    v2: RAG-augmented — injects retrieved MITRE technique descriptions before
        the flagged event list.

**Usage:**
    from src.llm.prompts import v1, v2
    prompt = v2.build_prompt(anomalous_logs, context, retrieved_docs)
"""
