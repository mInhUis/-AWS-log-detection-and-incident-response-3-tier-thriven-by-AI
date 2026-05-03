"""Tier 3 — Llama 3.1 8B 4-bit semantic analysis with RAG over MITRE ATT&CK.

Runs only on sequences Tier 2 has already flagged (~0.001% of original event
volume). Receives:
  (a) the flagged event sequence in human-readable form (key IDs reversed
      back to CloudTrail event names + enrichment fields),
  (b) MITRE ATT&CK / flaws.cloud snippets retrieved from ChromaDB by
      `src.llm.rag`,
  (c) a versioned prompt template from `src.llm.prompts/`.

Produces an analyst-facing incident report (suggested ATT&CK technique,
narrative explanation, mitigation recommendations).

Inference backend: Ollama (llama.cpp under the hood). Never use
`transformers.pipeline` for inference on the demo laptop — too slow on
integrated graphics (CLAUDE.md §10).

Hallucination guardrails: the dashboard labels Tier 3 output as
"suggested analysis — requires analyst review."
"""
