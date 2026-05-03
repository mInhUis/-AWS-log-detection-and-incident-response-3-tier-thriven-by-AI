"""Prompt template v1 — zero-shot Llama-3 incident analysis (Day 5-6).

**Version:** v1  (do not modify once evaluation has started)
**Prompt style:** zero-shot structured-output instruction
**RAG:** none — the model relies entirely on its parametric knowledge of ATT&CK

**Design rationale:**
The system prompt specifies EXACTLY the five output sections the LLM must
produce.  Explicit section headers reduce the probability that the model
generates unstructured prose or omits a required component.  At 8B parameters
Llama-3 reliably follows structural instructions when they are stated in the
system turn rather than the user turn.

The "Do not speculate beyond what the evidence supports" constraint is
critical for a thesis evaluation: it discourages ATT&CK mapping hallucination
(e.g. mapping ``ListBuckets`` to T1046 Network Service Scanning, which is
technically incorrect for CloudTrail semantics).

**Few-shot examples:** None in v1.  Adding 1–2 examples is the main planned
difference in v3 (post-thesis scope).  v2 adds RAG context instead.

**Llama-3 special tokens:**
    <|begin_of_text|>          — start of sequence
    <|start_header_id|>role<|end_header_id|>  — role header
    <|eot_id|>                 — end of turn

These tokens are required by the Llama-3 Instruct chat template and are
different from the Llama-2 ``[INST]`` format.  Using the wrong token format
causes the model to ignore the system prompt entirely (a common student error).

**Evaluation use:**
    Set ``prompt_version: v1`` in configs/default.yaml.  Use this as the
    baseline in the ablation: v1 (no RAG) vs v2 (with RAG).
"""
from __future__ import annotations

from typing import Any, Final

# ---------------------------------------------------------------------------
#  Version identifier — recorded in every evaluation run
# ---------------------------------------------------------------------------

PROMPT_VERSION: Final[str] = "v1"

# ---------------------------------------------------------------------------
#  System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: Final[str] = (
    "You are a senior AWS Cloud Security Analyst performing autonomous "
    "incident response.  A DeepLog LSTM anomaly-detection model has flagged "
    "the following CloudTrail events as statistically anomalous — their "
    "log-key transitions deviate significantly from the learned normal baseline.\n\n"
    "Analyse the flagged events and produce a structured incident report "
    "with EXACTLY the following five sections:\n\n"
    "1. **Incident Summary** — 1–2 sentences describing the overall incident.\n"
    "2. **Anomaly Analysis** — For each flagged event, explain why it is "
    "suspicious in the context of AWS security and CloudTrail semantics.\n"
    "3. **Severity Assessment** — Rate as Critical / High / Medium / Low "
    "with a one-sentence justification referencing the anomaly ratio and "
    "the nature of the flagged API calls.\n"
    "4. **Attack Pattern Classification** — Map to MITRE ATT&CK tactics "
    "and technique IDs (e.g. T1136, T1530) where directly supported by the "
    "flagged events.  Do not invent mappings not supported by the evidence.\n"
    "5. **Recommended Mitigation** — Numbered, actionable remediation steps "
    "an incident responder should execute immediately.\n\n"
    "Be concise, precise, and actionable.  "
    "Do not speculate beyond what the evidence supports.  "
    "Label this report as requiring analyst review before any automated action."
)

# ---------------------------------------------------------------------------
#  User message builder
# ---------------------------------------------------------------------------


def build_user_message(
    anomalous_logs: list[str],
    context: dict[str, Any],
) -> str:
    """Format the user-turn content from reverse-mapped anomaly data.

    ``anomalous_logs`` contains the Drain3 template strings for each flagged
    event — these are the human-readable event names that Tier 2 flagged, not
    raw JSON (the cascade reverse-maps log-key IDs to names before calling Tier 3,
    per CLAUDE.md §3).

    Args:
        anomalous_logs: Drain template strings, e.g. ``["CreateUser <*>"]``.
        context:        Metadata dict with optional keys:
                        ``session_id``, ``total_events``, ``anomaly_ratio``,
                        ``timestamps``, ``source_ips``, ``user_arns``,
                        ``event_names``.

    Returns:
        Formatted user-turn string ready for the Llama-3 chat template.
    """
    session_id: int | str = context.get("session_id", "N/A")
    total: int = context.get("total_events", len(anomalous_logs))
    ratio: float = context.get("anomaly_ratio", 1.0)
    timestamps: list[str] = context.get("timestamps", [])
    source_ips: list[str] = context.get("source_ips", [])
    user_arns: list[str] = context.get("user_arns", [])
    event_names: list[str] = context.get("event_names", [])

    header: str = (
        f"Session ID: {session_id}\n"
        f"Total events in session: {total}  |  "
        f"Flagged as anomalous: {len(anomalous_logs)} ({ratio:.1%})\n\n"
        "Flagged events (DeepLog top-k anomaly detection):\n"
    )

    lines: list[str] = []
    for idx, log_text in enumerate(anomalous_logs):
        ts: str = timestamps[idx] if idx < len(timestamps) else "N/A"
        ip: str = source_ips[idx] if idx < len(source_ips) else "N/A"
        arn: str = user_arns[idx] if idx < len(user_arns) else "N/A"
        evt: str = event_names[idx] if idx < len(event_names) else "N/A"
        lines.append(
            f"  {idx + 1}. [{ts}] {evt}\n"
            f"     Drain template : {log_text}\n"
            f"     Source IP      : {ip}\n"
            f"     Identity (ARN) : {arn}"
        )

    return header + "\n".join(lines)


# ---------------------------------------------------------------------------
#  Full prompt builder
# ---------------------------------------------------------------------------


def build_prompt(
    anomalous_logs: list[str],
    context: dict[str, Any],
) -> str:
    """Construct the full Llama-3 Instruct chat-format prompt (v1, no RAG).

    Uses the Llama-3 special-token chat template.  This exact token format
    is required for the Instruct variant; using the pre-training format
    (no special tokens) will cause the model to ignore the system turn.

    Args:
        anomalous_logs: Drain template strings for each flagged event.
        context:        Metadata dict (see ``build_user_message``).

    Returns:
        Fully formatted prompt string, ready for tokenisation or for passing
        directly to Ollama's ``messages`` API as a pre-formatted string.
    """
    user_msg: str = build_user_message(anomalous_logs, context)
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
