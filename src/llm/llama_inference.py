"""Llama-3 incident-response report generator — Tier 3 (Days 5-6).

**Why an LLM after DeepLog? (thesis cost-awareness argument)**
DeepLog's LSTM detects *that* an anomaly occurred — a mathematical deviation
in the log-key sequence.  It cannot explain *what* the anomaly means in a
security context or *how* to remediate it.  Llama-3-8B bridges this semantic
gap: given the human-readable Drain3 templates of flagged events plus retrieved
MITRE ATT&CK context, it produces a structured incident report.

**Two inference backends:**
    ``ollama``   — HTTP calls to a local Ollama daemon (llama.cpp under the
                   hood).  This is the primary backend for the laptop demo and
                   for both Path A (fine-tuned GGUF adapter) and Path B
                   (base model + RAG).  Estimate 3–10 tokens/sec on the
                   Dell Ryzen 7 5825U.  CLAUDE.md §10 mandates this backend
                   for laptop inference — never ``transformers.pipeline()``.
    ``unsloth``  — Unsloth FastLanguageModel, for Colab inference after QLoRA
                   fine-tuning (Path A, Day 5).  Only run on GPU.

**RAG integration (Day 4 + Day 5):**
    ``generate_report()`` accepts an optional ``rag_context: list[str]``
    parameter.  When provided and ``prompt_version='v2'``, the retrieved
    MITRE ATT&CK technique descriptions are injected into the user turn
    before the flagged event list (see prompts/v2.py for the rationale).

**Prompt versioning:**
    ``ResponderConfig.prompt_version`` selects the template module:
      'v1' → zero-shot (no RAG context, ablation baseline)
      'v2' → RAG-augmented (primary evaluation config)

**Data flow shapes:**
    anomalous_logs : list[str]       — Drain template strings, n_flagged events
    context        : dict[str, Any]  — session metadata
    rag_context    : list[str] | None — top-k retrieved ATT&CK doc texts
    prompt         : str             — full Llama-3 chat-format string
    input_ids      : (1, seq_len) int64   [real mode, Unsloth path]
    output_ids     : (1, seq_len + max_new_tokens) int64
    report         : str             — structured 5-section incident report

**Mock mode (default):**
    ``ResponderConfig(mock_mode=True)`` (default) returns a deterministic
    template report without loading any model.  Required for CI and for
    laptop-side development where no GPU is available.

**Environment variables:**
    RESPONDER_MODEL      — Ollama model tag or Unsloth model ID
    RESPONDER_MOCK       — 'true' to force mock mode
    RESPONDER_MAX_TOKENS — Max new tokens (default: 1024)
    RESPONDER_BACKEND    — 'ollama' or 'unsloth' (default: 'ollama')
    RESPONDER_VERSION    — Prompt version: 'v1' or 'v2' (default: 'v2')
    OLLAMA_HOST          — Ollama daemon URL (default: http://localhost:11434)
    CUDA_MEMORY_MAX      — VRAM fraction cap for Unsloth path (default: 0.5)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Final

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResponderConfig:
    """Immutable configuration for the Llama-3 responder.

    Attributes:
        backend:           Inference backend: ``'ollama'`` (laptop/demo) or
                           ``'unsloth'`` (Colab after fine-tuning).
        ollama_model:      Ollama model tag, e.g.
                           ``'llama3.1:8b-instruct-q4_K_M'``.  Matches
                           ``tier3.model`` in configs/default.yaml.
        ollama_host:       Ollama daemon base URL.
        model_name:        HuggingFace model ID used by the Unsloth backend.
        max_new_tokens:    Maximum tokens the model may generate per report.
        temperature:       Sampling temperature.  0.3 keeps output stable
                           across repeated runs for reproducible thesis results.
        top_p:             Nucleus-sampling probability mass.
        repetition_penalty: Penalises repeated token sequences.
        max_seq_length:    Context window limit (2 048 reduces KV-cache VRAM).
        prompt_version:    Template version to use: ``'v1'`` or ``'v2'``.
                           'v2' injects RAG context; 'v1' is the ablation
                           baseline without RAG.
        mock_mode:         When True, ``generate_report()`` returns a template
                           report without loading any model — safe for CI and
                           laptop-side development.
    """

    backend: str = "ollama"
    ollama_model: str = "llama3.1:8b-instruct-q4_K_M"
    ollama_host: str = "http://localhost:11434"
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_new_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    max_seq_length: int = 2048
    prompt_version: str = "v2"
    mock_mode: bool = True  # default mock per CLAUDE.md §5.6 and §13


def config_from_env() -> ResponderConfig:
    """Build a ``ResponderConfig`` from environment variables.

    Useful for ``scripts/`` entry points that want to respect env vars without
    requiring an argparse layer.

    Returns:
        ResponderConfig populated from environment, with defaults for any
        unset variable.
    """
    return ResponderConfig(
        backend=os.environ.get("RESPONDER_BACKEND", "ollama"),
        ollama_model=os.environ.get("RESPONDER_MODEL", "llama3.1:8b-instruct-q4_K_M"),
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        model_name=os.environ.get("RESPONDER_MODEL", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"),
        max_new_tokens=int(os.environ.get("RESPONDER_MAX_TOKENS", "1024")),
        temperature=float(os.environ.get("RESPONDER_TEMPERATURE", "0.3")),
        top_p=float(os.environ.get("RESPONDER_TOP_P", "0.9")),
        prompt_version=os.environ.get("RESPONDER_VERSION", "v2"),
        mock_mode=os.environ.get("RESPONDER_MOCK", "true").lower() == "true",
    )


# ---------------------------------------------------------------------------
#  Mock report generator (no model required)
# ---------------------------------------------------------------------------

def _generate_mock_report(
    anomalous_logs: list[str],
    context: dict[str, Any],
) -> str:
    """Produce a deterministic template-based mock report with the 5-section structure.

    The mock uses lightweight heuristics on event names and anomaly ratio to
    populate plausible content so the full pipeline produces visually meaningful
    output during development, CI, and the early demo stages.  It does NOT call
    any model.
    """
    session_id: int | str = context.get("session_id", "N/A")
    ratio: float = context.get("anomaly_ratio", 0.0)
    event_names: list[str] = context.get("event_names", [])
    timestamps: list[str] = context.get("timestamps", [])
    source_ips: list[str] = context.get("source_ips", [])
    user_arns: list[str] = context.get("user_arns", [])

    # ---- Severity heuristic ----
    if ratio > 0.3:
        severity = "High"
        severity_reason = (
            f"{ratio:.0%} of session events are anomalous, indicating a "
            "sustained deviation from the normal behaviour baseline."
        )
    elif ratio > 0.15:
        severity = "Medium"
        severity_reason = (
            f"{ratio:.0%} anomaly ratio suggests intermittent suspicious "
            "activity within an otherwise normal session."
        )
    else:
        severity = "Low"
        severity_reason = (
            f"Only {ratio:.0%} of events are anomalous; this may be benign "
            "noise but warrants monitoring."
        )

    # ---- MITRE mapping heuristic (name-based, not semantic) ----
    name_set: set[str] = {n.lower() for n in event_names}
    if {"createuser", "attachuserpolicy"} & name_set:
        attack_pattern = (
            "MITRE ATT&CK: T1136 (Create Account), T1098 (Account "
            "Manipulation) — Privilege Escalation / Persistence."
        )
    elif {"putbucketpolicy", "putobject", "getobject"} & name_set:
        attack_pattern = (
            "MITRE ATT&CK: T1530 (Data from Cloud Storage Object) — "
            "Collection / Data Exfiltration."
        )
    elif {"assumerole", "getcalleridentity"} & name_set:
        attack_pattern = (
            "MITRE ATT&CK: T1550.001 (Application Access Token), "
            "T1078.004 (Valid Cloud Accounts) — Lateral Movement / Defence Evasion."
        )
    elif {"registertaskdefinition", "runtask", "putimage"} & name_set:
        attack_pattern = (
            "MITRE ATT&CK: T1525 (Implant Internal Image), "
            "T1610 (Deploy Container) — Persistence / Execution."
        )
    else:
        attack_pattern = (
            "No direct MITRE ATT&CK mapping identified from event names alone.  "
            "Manual review recommended."
        )

    # ---- Per-event lines ----
    event_lines: list[str] = []
    for i, log in enumerate(anomalous_logs):
        ts = timestamps[i] if i < len(timestamps) else "N/A"
        ip = source_ips[i] if i < len(source_ips) else "N/A"
        arn = user_arns[i] if i < len(user_arns) else "N/A"
        evt = event_names[i] if i < len(event_names) else "N/A"
        event_lines.append(
            f"  - [{ts}] **{evt}** from {ip} (identity: {arn})\n"
            f"    Template: `{log}`\n"
            f"    Deviation: this event was not among the top-k predicted "
            f"next events by the DeepLog LSTM, indicating it falls outside "
            f"the learned normal sequence pattern."
        )

    events_block: str = "\n".join(event_lines) or "  No detailed event data available."

    return (
        f"## 1. Incident Summary\n\n"
        f"DeepLog flagged {len(anomalous_logs)} anomalous event(s) in "
        f"session {session_id}.  The flagged API calls deviate from the "
        f"normal CloudTrail activity baseline learned during training on "
        f"synthetic normal data.\n\n"
        f"## 2. Anomaly Analysis\n\n"
        f"{events_block}\n\n"
        f"## 3. Severity Assessment\n\n"
        f"**{severity}** — {severity_reason}\n\n"
        f"## 4. Attack Pattern Classification\n\n"
        f"{attack_pattern}\n\n"
        f"## 5. Recommended Mitigation\n\n"
        f"1. Immediately review IAM activity for the affected identities "
        f"in the AWS Console.\n"
        f"2. Rotate or revoke access keys associated with the flagged ARNs.\n"
        f"3. Enable CloudTrail log-file validation if not already active.\n"
        f"4. Restrict overly permissive IAM policies identified in the "
        f"flagged events.\n"
        f"5. Escalate to the security operations team for deeper forensic "
        f"investigation.\n\n"
        f"---\n"
        f"**[MOCK MODE — report generated without Llama-3 inference.  "
        f"Requires analyst review before any action.]**"
    )


# ---------------------------------------------------------------------------
#  LlamaResponder
# ---------------------------------------------------------------------------


class LlamaResponder:
    """Generates structured incident reports from anomalous CloudTrail events.

    Supports three operating modes controlled by ``ResponderConfig``:
    1. **Mock mode** (``mock_mode=True``, default): template-based report,
       no model loading, safe for CI and laptop-side development.
    2. **Ollama mode** (``backend='ollama'``): HTTP calls to a running Ollama
       daemon.  Primary mode for the laptop demo and end-to-end evaluation.
    3. **Unsloth mode** (``backend='unsloth'``): Unsloth 4-bit model on a
       CUDA GPU.  Used on Colab after QLoRA fine-tuning (Path A, Day 5).

    **RAG injection:**
    Pass ``rag_context`` (list of retrieved ATT&CK doc texts) to
    ``generate_report()``.  When ``prompt_version='v2'``, these are injected
    into the user turn before the flagged event list.

    Usage::

        # Mock (default — no model needed):
        r = LlamaResponder()
        report = r.generate_report(["CreateUser <*>"], {"session_id": 1})

        # Ollama with RAG (production demo):
        r = LlamaResponder(ResponderConfig(mock_mode=False, backend="ollama"))
        report = r.generate_report(
            anomalous_logs=["CreateUser <*>"],
            context={"session_id": 1, "anomaly_ratio": 0.25},
            rag_context=retrieved_docs,      # from RAGRetriever.query()
        )
    """

    def __init__(self, config: ResponderConfig | None = None) -> None:
        self._config: ResponderConfig = config or ResponderConfig()
        # Unsloth backend state (lazy-loaded)
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded: bool = False

    # ------------------------------------------------------------------ #
    #  Ollama backend                                                      #
    # ------------------------------------------------------------------ #

    def _generate_ollama_report(
        self,
        anomalous_logs: list[str],
        context: dict[str, Any],
        rag_context: list[str] | None = None,
    ) -> str:
        """Call the local Ollama daemon and return the generated report.

        Uses the ``ollama`` Python package (``requirements.txt: ollama~=0.3``).
        Passes the system and user messages as separate ``messages`` entries so
        Ollama handles the Llama-3 chat template internally — this avoids
        double-formatting issues that occur when passing a pre-formatted string
        to Ollama's raw completion endpoint.

        Args:
            anomalous_logs: Drain template strings for each flagged event.
            context:        Session metadata dict.
            rag_context:    Retrieved ATT&CK doc texts for v2 injection.

        Returns:
            Generated incident report text.

        Raises:
            RuntimeError: If Ollama is not running or the model is not pulled.
        """
        # Lazy import: ollama is only needed when backend='ollama' and not mock
        try:
            import ollama as _ollama  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "The 'ollama' package is required for Ollama-backend inference.\n"
                "Install: pip install ollama~=0.3\n"
                "And ensure the Ollama daemon is running: ollama serve"
            ) from exc

        # Select prompt version for user message construction
        if self._config.prompt_version == "v2":
            from .prompts import v2 as _pv
            user_msg: str = _pv.build_user_message(
                anomalous_logs, context, rag_context
            )
        else:
            from .prompts import v1 as _pv  # type: ignore[assignment]
            user_msg = _pv.build_user_message(anomalous_logs, context)

        from .prompts.v1 import SYSTEM_PROMPT

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        try:
            response = _ollama.chat(
                model=self._config.ollama_model,
                messages=messages,
                options={
                    "temperature": self._config.temperature,
                    "top_p": self._config.top_p,
                    "repeat_penalty": self._config.repetition_penalty,
                    "num_predict": self._config.max_new_tokens,
                },
            )
        except Exception as exc:
            # Surface a clear error message — the most common failure is
            # Ollama not running, which manifests as a connection-refused error
            raise RuntimeError(
                f"Ollama inference failed: {exc}\n"
                "Troubleshooting:\n"
                "  1. Start Ollama daemon:  ollama serve\n"
                f"  2. Pull the model:       ollama pull {self._config.ollama_model}\n"
                "  3. Verify connectivity:  curl http://localhost:11434/api/tags\n"
                "  4. Fall back to mock:    set mock_mode=True in ResponderConfig"
            ) from exc

        # response["message"]["content"] is the assistant's turn text
        return str(response["message"]["content"])

    # ------------------------------------------------------------------ #
    #  Unsloth backend (Colab only)                                        #
    # ------------------------------------------------------------------ #

    def _load_unsloth_model(self) -> None:
        """Load the quantised Llama-3 model via Unsloth (lazy, idempotent).

        This method is only called when ``backend='unsloth'``.  It applies the
        CUDA_MEMORY_MAX VRAM cap before allocation.

        Raises:
            RuntimeError: If unsloth is not installed or CUDA is out of memory.
        """
        if self._loaded:
            return

        try:
            from unsloth import FastLanguageModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "Unsloth is required for the 'unsloth' backend but is not "
                "installed.  On Colab:\n"
                '  pip install "unsloth[colab-new] @ '
                'git+https://github.com/unslothai/unsloth.git"\n'
                "Alternatively use the Ollama backend:\n"
                "  config = ResponderConfig(backend='ollama')"
            ) from exc

        import torch

        if torch.cuda.is_available():
            # Cap VRAM usage so we don't starve other GPU processes on shared Colab
            vram_frac: float = float(os.environ.get("CUDA_MEMORY_MAX", "0.5"))
            torch.cuda.set_per_process_memory_fraction(vram_frac)

        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self._config.model_name,
                max_seq_length=self._config.max_seq_length,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
        except (RuntimeError, Exception) as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "CUDA OOM while loading Llama-3.  Options:\n"
                    "  1. Close other GPU processes.\n"
                    f"  2. Lower CUDA_MEMORY_MAX (currently "
                    f"{os.environ.get('CUDA_MEMORY_MAX', '0.5')}).\n"
                    "  3. Use the Ollama backend instead."
                ) from exc
            raise

        self._model = model
        self._tokenizer = tokenizer
        self._loaded = True

    def _generate_unsloth_report(
        self,
        anomalous_logs: list[str],
        context: dict[str, Any],
        rag_context: list[str] | None = None,
    ) -> str:
        """Run Unsloth 4-bit inference and extract the assistant turn.

        Args:
            anomalous_logs: Drain template strings.
            context:        Session metadata dict.
            rag_context:    Retrieved ATT&CK docs for v2 prompt.

        Returns:
            Assistant response text (stripped of special tokens).
        """
        self._load_unsloth_model()

        import torch

        # Build prompt using the versioned template
        if self._config.prompt_version == "v2":
            from .prompts import v2 as _pv
            prompt: str = _pv.build_prompt(anomalous_logs, context, rag_context)
        else:
            from .prompts import v1 as _pv  # type: ignore[assignment]
            prompt = _pv.build_prompt(anomalous_logs, context)

        # input_ids: (1, seq_len) int64
        inputs = self._tokenizer(prompt, return_tensors="pt").to(
            self._model.device
        )

        with torch.no_grad():
            # output_ids: (1, seq_len + max_new_tokens) int64
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                repetition_penalty=self._config.repetition_penalty,
                do_sample=True,
            )

        full_text: str = self._tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )

        # Extract only the assistant's response (after the last 'assistant' header)
        marker: str = "assistant"
        marker_idx: int = full_text.rfind(marker)
        if marker_idx != -1:
            return full_text[marker_idx + len(marker):].strip()
        return full_text.strip()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate_report(
        self,
        anomalous_logs: list[str],
        context: dict[str, Any],
        rag_context: list[str] | None = None,
    ) -> str:
        """Generate a structured incident report from anomalous CloudTrail events.

        This is the single entry point called by ``pipeline/cascade.py`` after
        Tier 2 has confirmed anomalous sequences and the cascade has reverse-
        mapped log-key IDs back to Drain3 template strings.

        Args:
            anomalous_logs: Drain template strings for each flagged event, e.g.
                            ``["CreateUser <*>", "AttachUserPolicy <*>"]``.
                            These come from the log-key → template reverse map.
            context:        Session metadata dict with optional keys:
                            ``session_id``, ``total_events``, ``anomaly_ratio``,
                            ``timestamps``, ``source_ips``, ``user_arns``,
                            ``event_names``.
            rag_context:    Top-k retrieved MITRE ATT&CK technique texts from
                            ``RAGRetriever.query()``.  Pass ``None`` to skip
                            RAG injection (falls back to v1 prompt behaviour
                            regardless of ``prompt_version``).

        Returns:
            Structured incident report string with five sections:
            Incident Summary, Anomaly Analysis, Severity Assessment,
            Attack Pattern Classification, Recommended Mitigation.

            When ``mock_mode=True``, ends with a [MOCK MODE] disclaimer.

        Note:
            LLM outputs may contain hallucinated ATT&CK IDs or CloudTrail
            API names.  The thesis dashboard labels all LLM output as
            "suggested analysis — requires analyst review" (CLAUDE.md §10).
        """
        if self._config.mock_mode:
            return _generate_mock_report(anomalous_logs, context)

        if self._config.backend == "ollama":
            return self._generate_ollama_report(anomalous_logs, context, rag_context)

        if self._config.backend == "unsloth":
            return self._generate_unsloth_report(anomalous_logs, context, rag_context)

        raise ValueError(
            f"Unknown backend: '{self._config.backend}'.  "
            "Valid options: 'ollama', 'unsloth'."
        )
