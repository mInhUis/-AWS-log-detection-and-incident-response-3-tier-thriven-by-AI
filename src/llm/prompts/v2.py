"""Prompt template v2 — RAG-augmented Llama-3 incident analysis (Days 5-6).

**Version:** v2  (do not modify once evaluation has started)
**Prompt style:** RAG-augmented structured-output instruction
**RAG:** injects top-k MITRE ATT&CK technique descriptions retrieved from
         ChromaDB before the flagged event list

**Why RAG improves faithfulness (the key thesis argument):**
The base Llama-3-8B model has seen MITRE ATT&CK in pre-training only
approximately and at 8B parameters its parametric knowledge of specific
technique IDs (e.g. T1552.005) and their exact AWS API indicators is thin.
Lewis et al. (2020) introduced RAG as a principled way to inject factual
external knowledge at inference time without retraining.  For our use case:

  - Without RAG (v1): the model may hallucinate T-IDs or describe techniques
    that are not supported by the observed CloudTrail API sequence.
  - With RAG (v2): the retrieved technique descriptions constrain the model
    to techniques that actually match the semantic content of the query,
    reducing hallucination and improving precision of the MITRE mapping.

The ablation study (Day 8) measures the precision improvement of v2 over v1
on the 6 flaws.cloud test scenarios.

**Context block placement:**
The retrieved technique descriptions are placed BEFORE the flagged event list
in the user turn.  This mirrors the analyst workflow: read the relevant
framework context first, then examine the evidence.  Placing the RAG context
after the events (post-hoc framing) empirically degrades adherence to the
provided context at this parameter count.

**Inheritance:**
v2 reuses ``SYSTEM_PROMPT`` from v1.  The system instruction does not change;
only the user turn is augmented with the RAG block.

**Evaluation use:**
    Set ``prompt_version: v2`` in configs/default.yaml.  Compare v2 MITRE
    mapping precision against v1 in the ablation table.
"""
from __future__ import annotations

from typing import Any, Final

from .v1 import SYSTEM_PROMPT, build_user_message as _v1_build_user_message

# ---------------------------------------------------------------------------
#  Version identifier — recorded in every evaluation run
# ---------------------------------------------------------------------------

PROMPT_VERSION: Final[str] = "v2"

# ---------------------------------------------------------------------------
#  RAG context formatter
# ---------------------------------------------------------------------------


def _format_rag_context(retrieved_docs: list[str]) -> str:
    """Format retrieved ATT&CK documents into a numbered context block.

    Each document is numbered and lightly truncated so the total context block
    stays within Llama-3's 2 048-token window alongside the event list.  At
    4 documents × 600 chars ≈ 2 400 chars ≈ 600 tokens; the event list typically
    adds another 300–500 tokens, leaving headroom for the 1 024-token response.

    Args:
        retrieved_docs: List of document texts from ``RAGRetriever.query()``.

    Returns:
        Formatted string to prepend to the user message.  Empty string if
        ``retrieved_docs`` is empty (degrades gracefully to v1 behaviour).
    """
    if not retrieved_docs:
        return ""

    lines: list[str] = [
        "Relevant MITRE ATT&CK techniques retrieved by semantic search "
        "(use these to inform your classification, do not invent others):\n"
    ]
    for i, doc in enumerate(retrieved_docs, start=1):
        # Truncate at 600 chars; BGE already embedded the full text, so the
        # truncation here only affects the LLM prompt, not the retrieval result
        truncated: str = doc[:600] + ("..." if len(doc) > 600 else "")
        lines.append(f"[{i}] {truncated}\n")

    lines.append(
        "Only cite technique IDs that appear in the above context AND are "
        "directly evidenced by the flagged events below.\n"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  User message builder
# ---------------------------------------------------------------------------


def build_user_message(
    anomalous_logs: list[str],
    context: dict[str, Any],
    retrieved_docs: list[str] | None = None,
) -> str:
    """Format the user-turn with an optional RAG context block.

    The RAG block is prepended before the flagged event list so the model
    reads the technique reference before the evidence — this order empirically
    improves adherence to the retrieved context at 8B parameters.

    Args:
        anomalous_logs: Drain template strings for each flagged event.
        context:        Metadata dict (see v1.build_user_message for keys).
        retrieved_docs: Top-k ATT&CK document texts from ``RAGRetriever.query()``.
                        Pass ``None`` or ``[]`` to skip RAG injection (falls back
                        to the same user message as v1).

    Returns:
        User-turn string: [ATT&CK context block] + [flagged events block].
    """
    events_block: str = _v1_build_user_message(anomalous_logs, context)

    if not retrieved_docs:
        # Graceful fallback — produces the same message as v1
        return events_block

    rag_block: str = _format_rag_context(retrieved_docs)
    # Separator line between the RAG context and the event list
    return f"{rag_block}\n---\n{events_block}"


# ---------------------------------------------------------------------------
#  Full prompt builder
# ---------------------------------------------------------------------------


def build_prompt(
    anomalous_logs: list[str],
    context: dict[str, Any],
    retrieved_docs: list[str] | None = None,
) -> str:
    """Construct the Llama-3 Instruct prompt with injected ATT&CK context (v2).

    The system prompt is identical to v1.  The user turn is augmented with the
    retrieved technique descriptions.  The Llama-3 special-token format is
    preserved.

    Args:
        anomalous_logs: Drain template strings for each flagged event.
        context:        Metadata dict (see v1.build_user_message).
        retrieved_docs: Top-k retrieved ATT&CK technique texts.
                        Pass ``None`` to degrade to v1 behaviour.

    Returns:
        Fully formatted prompt string for tokenisation or Ollama chat API.
    """
    user_msg: str = build_user_message(anomalous_logs, context, retrieved_docs)
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
