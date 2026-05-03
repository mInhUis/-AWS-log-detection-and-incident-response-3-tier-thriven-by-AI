"""RAG support for Tier 3: ChromaDB index over MITRE ATT&CK + flaws.cloud writeups.

**Why retrieval-augmented generation here?**
The base Llama-3.1-8B model's parametric knowledge of ATT&CK is frozen at
training time and is shallow at 8B parameters for specialised security taxonomy.
RAG (Lewis et al., 2020) injects current, exact technique descriptions at
inference time, which improves the faithfulness of the MITRE mapping and lets
us update the corpus without retraining.

**Corpus (built on Day 4):**
  - MITRE ATT&CK Enterprise STIX bundle, chunked per technique (~700 docs).
  - flaws.cloud level 1–6 writeups (Scott Piper's attack descriptions).

**Embedding model:** Local, CPU-only (BGE-small-en-v1.5, ~130 MB).
No external embedding APIs — CLAUDE.md §10.

**Public API:**
    from src.llm.rag.corpus import load_attack_stix, load_flaws_writeups
    from src.llm.rag.embedder import LocalEmbedder
    from src.llm.rag.retriever import RAGRetriever
"""
