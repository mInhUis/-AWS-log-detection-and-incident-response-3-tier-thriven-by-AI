"""3-tier cost-aware cascade orchestrator: Tier 1 → Tier 2 → Tier 3.

WHY THIS CASCADE IS THE THESIS CONTRIBUTION (CLAUDE.md §3):
    The cascade's core principle is volume reduction at each tier.  Each tier
    handles a *strictly smaller* subset of events than the previous one:

      Tier 1 — Isolation Forest (CPU-only, < $0.01 / 1M events):
        Aggregates per-entity-per-window scalar features and discards ~95% of
        events.  Only entity-windows whose feature vectors are anomalous by
        Isolation Forest standards pass to Tier 2.

      Tier 2 — DeepLog LSTM (GPU, ~$0.10 / 1M events on the original volume):
        Runs ONLY on the events of Tier-1-flagged entities.  Applies the top-k
        next-log-key anomaly check on Drain3 integer sequences.  Discards ~99%
        of the Tier-1 output.

      Tier 3 — Llama 3.1 8B 4-bit via Ollama (local, free):
        Runs ONLY on confirmed-anomalous sequences.  Produces structured
        5-section incident reports with MITRE ATT&CK mapping and RAG-retrieved
        technique context.  Sees ~0.001% of original event volume.

    Without Tier 1, DeepLog would run on every entity's events — economically
    identical to naive deep learning with no first-pass filter.  The cascade
    is the architecture; the individual algorithms are well-established choices.

PIPELINE POSITION (CLAUDE.md §8):
    src/pipeline/cascade.py — single process() entry point.
    Input : raw CloudTrail events (list[dict]).
    Output: CascadeResult (incident reports + per-tier cost metrics).

DATA FLOW:
    raw events (list[dict])
        ↓  _build_events_dataframe()          normalize + typed DataFrame
        ↓  aggregate_features(df, baseline)   per-entity-window feature vectors
        ↓  if_model.predict(X)                Isolation Forest scoring
        ↓  filter to flagged principals        ~5% pass through
        ↓  drain_parser.transform(messages)   log-key sequence extraction
        ↓  _detect_anomalies_safe()           DeepLog OOV-safe top-k check
        ↓  reverse_map_events()               int IDs → human-readable strings
        ↓  retriever.query() + responder.generate_report()   Llama + RAG
        ↓  CascadeResult
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final

import pandas as pd
import torch

# ---------------------------------------------------------------------------
#  sys.path guard — enables "python scripts/run_cascade.py" from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.detector.deeplog import DeepLogModel
from src.detector.drain_parser import DrainParser
from src.llm.llama_inference import LlamaResponder
from src.llm.rag.retriever import RAGRetriever
from src.pipeline.reverse_map import reverse_map_events
from src.preprocessing.features import FEATURE_COLUMNS, aggregate_features
from src.preprocessing.normalize import normalize_event
from src.tier1_ml.baseline import HourBaseline

# ---------------------------------------------------------------------------
#  CascadeResult — structured output of a complete cascade run
# ---------------------------------------------------------------------------


@dataclass
class CascadeResult:
    """Structured output of a complete 3-tier cascade run.

    Attributes:
        reports: One dict per session that survived all three tiers.
                 Each dict: {session_id, report_text, flagged_events, anomaly_ratio}.
                 flagged_events is the list of enriched event dicts from
                 reverse_map_events() — full metadata for downstream evaluation.
        metrics: Per-tier volume counts and latencies.  Designed to fill the
                 cost-analysis table in the thesis evaluation chapter (CLAUDE.md §3).
                 Keys: tier1_input_events, tier1_input_windows, tier1_flagged_windows,
                 tier1_flagged_principals, tier2_input_events, tier2_flagged_events,
                 tier3_input_sessions, tier{1,2,3}_latency_s, total_latency_s.
    """

    __slots__ = ("reports", "metrics")

    reports: list[dict[str, Any]]
    metrics: dict[str, Any]


# ---------------------------------------------------------------------------
#  OOV-safe inference engine — inlined to avoid the broken relative imports
#  in src/preprocessing/run_eval.py  (that file uses "from deeplog import ...")
# ---------------------------------------------------------------------------


def _detect_anomalies_safe(
    model: DeepLogModel,
    sequences: list[list[int]],
    window_size: int,
    top_k: int,
    device: torch.device,
) -> list[list[bool]]:
    """Top-k anomaly detection with correct OOV masking (the fixed version).

    WHY NOT detect.detect_anomalies()?
        The current src/detector/detect.py has a latent bug: it computes
        safe_window (OOV keys clamped to 0) but then passes the *original*
        unsanitised window to torch.tensor.  On Colab this silently produces
        garbage logits; on some PyTorch builds it raises a CUDA index error.

        This function matches src/preprocessing/run_eval.detect_anomalies_safe:
        it consistently uses safe_window for the tensor creation.

    OOV strategy:
        1. If current event key >= model.num_keys → flag True (unknown event).
        2. Clamp past OOV keys in the window to 0 before the LSTM forward pass.
           Key 0 is always valid for a trained model.
        3. Only then check whether the current (in-vocabulary) key is in the
           model's top-k predicted next keys.

    Args:
        model:       Trained DeepLogModel, already set to eval mode is fine;
                     this function calls model.eval() internally.
        sequences:   List of per-session integer log-key sequences.
        window_size: Sliding window width used during training.
        top_k:       Number of top predictions to accept as "normal".
                     Tune on validation only (CLAUDE.md §7).
        device:      Torch device matching the model's current device.

    Returns:
        list[list[bool]] — same shape as sequences; True = anomalous event.

    Tensor shapes (per step):
        safe_window_tensor : (1, window_size) int64
        logits             : (1, num_keys)    float32
        top_indices        : (1, top_k)       int64
    """
    model_num_keys: int = model.num_keys
    model.eval()
    all_flags: list[list[bool]] = []

    with torch.no_grad():
        for seq in sequences:
            seq_len: int = len(seq)
            flags: list[bool] = [False] * seq_len

            if seq_len <= window_size:
                # Too short to form even one complete prediction window.
                all_flags.append(flags)
                continue

            for i in range(window_size, seq_len):
                true_key: int = seq[i]

                # Step 1: current event is OOV → anomaly by definition.
                # The model has no embedding for this key; no inference needed.
                if true_key >= model_num_keys:
                    flags[i] = True
                    continue

                # Step 2: clamp past OOV keys so the embedding table never
                # receives an out-of-range index.  Substituting 0 is safe
                # because key 0 is always present in a trained vocabulary.
                raw_window: list[int] = seq[i - window_size : i]
                safe_window: list[int] = [
                    k if k < model_num_keys else 0 for k in raw_window
                ]

                # safe_window_tensor: (1, window_size) int64
                safe_window_tensor: torch.Tensor = torch.tensor(
                    [safe_window], dtype=torch.long, device=device
                )
                # logits: (1, num_keys) float32
                logits, _ = model(safe_window_tensor)

                # top_indices: (1, effective_k) → squeeze to (effective_k,) int64
                # min() guards against top_k > vocabulary (small test models).
                effective_k: int = min(top_k, model_num_keys)
                _, top_indices = torch.topk(logits, k=effective_k, dim=-1)
                top_set: set[int] = set(top_indices.squeeze(0).tolist())

                if true_key not in top_set:
                    flags[i] = True  # event is anomalous

            all_flags.append(flags)

    return all_flags


# ---------------------------------------------------------------------------
#  CascadePipeline
# ---------------------------------------------------------------------------


class CascadePipeline:
    """Wires Tier 1 (Isolation Forest) → Tier 2 (DeepLog) → Tier 3 (Llama).

    The caller is responsible for loading all models before instantiation.
    This keeps I/O and checkpoint-loading in the scripts layer (run_cascade.py)
    and keeps the pipeline layer as pure computation — easier to unit-test.

    Args:
        if_bundle:      Joblib bundle dict from the Isolation Forest .pkl.
                        Required keys: "model" (IsolationForest instance),
                        "feature_columns" (list[str]).  Validated at init.
        hour_baseline:  HourBaseline fitted on training data ONLY.  Loaded via
                        src.tier1_ml.baseline.load_baseline().  Used by
                        aggregate_features() for time_of_day_deviation.
        deeplog_model:  Trained DeepLogModel.  Can be on CPU or GPU — the
                        cascade will use whatever device it is already on.
        window_size:    Sliding window width matching the training run.
                        Stored in the DeepLog checkpoint as ckpt["window_size"].
        drain_parser:   DrainParser with loaded state (after load_state()).
                        Must be in read-only transform mode — not fitted here.
        responder:      LlamaResponder instance.  Use mock_mode=True (default)
                        for development; set mock_mode=False for final demo.
        retriever:      RAGRetriever over the ATT&CK corpus.  Pass None to skip
                        RAG (falls back to v1 prompt behaviour automatically).
        top_k:          DeepLog top-k anomaly threshold.  Tune on validation
                        set (k ∈ {3, 5, 7, 9}) per CLAUDE.md §7.  Default 9.
        skip_tier1:     If True, bypass the Isolation Forest and send ALL
                        events to Tier 2.  Used for the ablation study:
                        "full cascade vs Tier 2 alone" (CLAUDE.md §7).
    """

    def __init__(
        self,
        if_bundle: dict[str, Any],
        hour_baseline: HourBaseline,
        deeplog_model: DeepLogModel,
        window_size: int,
        drain_parser: DrainParser,
        responder: LlamaResponder,
        retriever: RAGRetriever | None = None,
        top_k: int = 9,
        skip_tier1: bool = False,
    ) -> None:
        # ---- Validate IF bundle before storing ----------------------------
        # A feature-set mismatch here means the pipeline code changed since
        # training.  Better to fail loudly at construction than produce
        # garbage anomaly scores silently.
        _missing: set[str] = {"model", "feature_columns"} - set(if_bundle.keys())
        if _missing:
            raise ValueError(
                f"Isolation Forest bundle missing required keys: {_missing}. "
                "Retrain: python -m src.tier1_ml.train"
            )
        if if_bundle["feature_columns"] != FEATURE_COLUMNS:
            raise ValueError(
                f"IF bundle feature_columns mismatch.\n"
                f"  Bundle:  {if_bundle['feature_columns']}\n"
                f"  Current: {FEATURE_COLUMNS}\n"
                "Retrain the Isolation Forest: python -m src.tier1_ml.train"
            )

        self._if_model: Any = if_bundle["model"]
        self._hour_baseline: HourBaseline = hour_baseline
        self._deeplog: DeepLogModel = deeplog_model
        self._window_size: int = window_size
        self._drain: DrainParser = drain_parser
        self._responder: LlamaResponder = responder
        self._retriever: RAGRetriever | None = retriever
        self._top_k: int = top_k
        self._skip_tier1: bool = skip_tier1

        # Infer device from model parameters so callers don't pass it explicitly.
        self._device: torch.device = next(deeplog_model.parameters()).device

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process(
        self,
        events: list[dict[str, Any]],
        message_fn: Callable[[dict[str, Any]], str] | None = None,
    ) -> CascadeResult:
        """Run the complete 3-tier cascade on a batch of raw CloudTrail events.

        IMPORTANT — message_fn format must match Drain training:
            DrainParser.transform() is read-only at inference time.  If the
            string format passed to transform() differs from what was used in
            fit_transform() during training, virtually all events will be OOV.
            The default (just eventName) is correct when Drain was trained on
            eventName-only strings.

        Args:
            events:     Raw CloudTrail event dicts.  Supports both the flat
                        synthetic schema and the nested AWS official schema.
                        Each dict needs at minimum: eventTime, eventName,
                        eventSource, awsRegion, and a principal identity.
            message_fn: Callable(raw_event_dict) → str.  Converts an event
                        to the log-message string passed to DrainParser.
                        Default: lambda e: e.get("eventName", "<unknown>").
                        Override if Drain was trained on a richer string format.

        Returns:
            CascadeResult with incident reports and per-tier cost metrics.
        """
        if message_fn is None:
            # Drain was trained on "{event_name} {event_source}" (2-token) or
            # "{event_name} {event_source} {error_code}" (3-token) messages.
            # Both schemas (snake_case processed + camelCase raw) are covered.
            # Sending only event_name (1 token) causes 100% OOV — DO NOT shorten.
            def message_fn(e: dict) -> str:
                name   = e.get("event_name")   or e.get("eventName")   or "<unknown>"
                source = e.get("event_source")  or e.get("eventSource") or ""
                error  = e.get("error_code")    or e.get("errorCode")   or ""
                parts  = [p for p in [name, source, error] if p]
                return " ".join(parts)

        total_start: float = time.perf_counter()

        # ── Tier 1: Isolation Forest ────────────────────────────────────────
        t1_start: float = time.perf_counter()

        if self._skip_tier1:
            # Ablation: treat every distinct principal as "flagged" so Tier 2
            # sees the full event set.  Metrics record skip_tier1=True.
            flagged_principals: set[str] = _all_principals(events)
            t1_metrics: dict[str, Any] = {
                "tier1_input_events":     len(events),
                "tier1_input_windows":    0,   # N/A when bypassed
                "tier1_flagged_windows":  0,
                "tier1_flagged_principals": len(flagged_principals),
                "skip_tier1":             True,
            }
        else:
            flagged_principals, t1_metrics = self._run_tier1(events)

        t1_latency: float = time.perf_counter() - t1_start

        # ── Tier 2: DeepLog ─────────────────────────────────────────────────
        t2_start: float = time.perf_counter()

        # Volume contract: Tier 2 NEVER sees events Tier 1 dismissed.
        t2_events: list[dict[str, Any]] = [
            e for e in events if _principal_id(e) in flagged_principals
        ]
        session_data, t2_metrics = self._run_tier2(t2_events, message_fn)

        t2_latency: float = time.perf_counter() - t2_start

        # ── Tier 3: Llama ───────────────────────────────────────────────────
        t3_start: float = time.perf_counter()

        # Volume contract: Tier 3 NEVER sees events Tier 2 cleared as normal.
        reports, t3_metrics = self._run_tier3(session_data)

        t3_latency: float = time.perf_counter() - t3_start
        total_latency: float = time.perf_counter() - total_start

        metrics: dict[str, Any] = {
            **t1_metrics,
            **t2_metrics,
            **t3_metrics,
            "tier1_latency_s": round(t1_latency, 4),
            "tier2_latency_s": round(t2_latency, 4),
            "tier3_latency_s": round(t3_latency, 4),
            "total_latency_s": round(total_latency, 4),
        }

        return CascadeResult(reports=reports, metrics=metrics)

    # ------------------------------------------------------------------ #
    #  Tier 1: Isolation Forest on aggregated feature vectors              #
    # ------------------------------------------------------------------ #

    def _run_tier1(
        self,
        events: list[dict[str, Any]],
    ) -> tuple[set[str], dict[str, Any]]:
        """Score entity-windows with the Isolation Forest; return flagged principals.

        Steps:
          1. Normalise raw events → typed DataFrame (normalize_event).
          2. Aggregate per-entity-window features for both window sizes
             (5 min and 60 min, per WINDOW_SIZES_MINUTES in features.py).
          3. Score the feature matrix with IsolationForest.predict().
          4. Collect principal_ids with at least one flagged window.

        A principal is flagged if ANY of its windows (either window size)
        scores -1.  This is conservative by design: false positives at Tier 1
        are cheap (a short DeepLog run), false negatives miss attacks.

        Args:
            events: All raw CloudTrail events in the batch.

        Returns:
            (flagged_principals, tier1_metrics)
        """
        if not events:
            return set(), _empty_t1_metrics(0)

        df: pd.DataFrame = _build_events_dataframe(events)
        if df.empty:
            return set(), _empty_t1_metrics(len(events))

        # aggregate_features produces rows for BOTH window sizes in one call.
        try:
            feature_df: pd.DataFrame = aggregate_features(df, self._hour_baseline)
        except Exception as exc:
            raise RuntimeError(
                f"Tier 1 feature aggregation failed: {exc}\n"
                "Ensure events have valid eventTime, eventName, and "
                "eventSource fields."
            ) from exc

        if feature_df.empty:
            # All windows are sparse (event_count < 3) — nothing to score.
            # Sparse windows are noise-prone; skipping them is intentional.
            return set(), _empty_t1_metrics(len(events))

        # X: (n_windows, len(FEATURE_COLUMNS)) float32
        X = feature_df[FEATURE_COLUMNS].values.astype("float32")
        n_windows: int = int(X.shape[0])

        # IsolationForest.predict: +1 = normal, -1 = anomalous
        predictions = self._if_model.predict(X)  # shape: (n_windows,)
        flagged_mask = predictions == -1

        n_flagged_windows: int = int(flagged_mask.sum())
        flagged_df = feature_df[flagged_mask]
        flagged_principals: set[str] = set(
            flagged_df["principal_id"].astype(str).tolist()
        )

        return flagged_principals, {
            "tier1_input_events":       len(events),
            "tier1_input_windows":      n_windows,
            "tier1_flagged_windows":    n_flagged_windows,
            "tier1_flagged_principals": len(flagged_principals),
        }

    # ------------------------------------------------------------------ #
    #  Tier 2: DeepLog on Drain3 sequences of Tier-1-flagged entities     #
    # ------------------------------------------------------------------ #

    def _run_tier2(
        self,
        events: list[dict[str, Any]],
        message_fn: Callable[[dict[str, Any]], str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run DeepLog on per-principal event sequences; collect anomalous sessions.

        Sessions are defined per principal: all events for a flagged principal
        form one session, sorted by eventTime.  This is simpler than splitting
        on inactivity gaps and is documented as a design choice in the thesis
        methodology (post-thesis enhancement: session segmentation by gap).

        Args:
            events:     Events for Tier-1-flagged principals ONLY.
                        Volume contract: Tier 2 never sees what Tier 1 dismissed.
            message_fn: Callable that renders a raw event to a Drain log-message
                        string.  Must match the format used during Drain training.

        Returns:
            (session_data, tier2_metrics)
            session_data maps principal_id → {flags, log_keys, events,
            templates, oov_key} for sessions with at least one anomalous event.
        """
        if not events:
            return {}, {"tier2_input_events": 0, "tier2_flagged_events": 0}

        # ---- Group by principal, sort by time ----------------------------
        by_principal: dict[str, list[dict[str, Any]]] = {}
        for e in events:
            pid: str = _principal_id(e)
            by_principal.setdefault(pid, []).append(e)

        for pid in by_principal:
            by_principal[pid].sort(
                key=lambda e: e.get("event_time") or e.get("eventTime") or ""
            )

        # ---- Drain vocabulary (read-only, same state as training) --------
        templates: dict[int, str] = self._drain.get_templates()
        oov_key: int = self._drain.oov_key

        total_flagged: int = 0
        session_data: dict[str, dict[str, Any]] = {}

        for pid, sess_events in by_principal.items():
            # Convert raw events → log-message strings → integer log-key IDs.
            # transform() is read-only (never fit_transform on non-training data).
            messages: list[str] = [message_fn(e) for e in sess_events]
            log_keys: list[int] = self._drain.transform(messages)
            # log_keys: list[int] of length len(sess_events)

            # Wrap in a list because _detect_anomalies_safe is batch-oriented.
            all_flags: list[list[bool]] = _detect_anomalies_safe(
                model=self._deeplog,
                sequences=[log_keys],
                window_size=self._window_size,
                top_k=self._top_k,
                device=self._device,
            )
            flags: list[bool] = all_flags[0]
            # flags: list[bool] of length len(sess_events)

            n_anomalous: int = sum(flags)
            if n_anomalous == 0:
                continue  # session is normal — Tier 3 never sees it

            total_flagged += n_anomalous
            session_data[pid] = {
                "flags":     flags,
                "log_keys":  log_keys,
                "events":    sess_events,
                "templates": templates,
                "oov_key":   oov_key,
            }

        return session_data, {
            "tier2_input_events":  len(events),
            "tier2_flagged_events": total_flagged,
        }

    # ------------------------------------------------------------------ #
    #  Tier 3: Llama semantic analysis on confirmed anomalous sessions     #
    # ------------------------------------------------------------------ #

    def _run_tier3(
        self,
        session_data: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Generate Llama incident reports for each session that cleared Tier 2.

        For each anomalous session:
          1. Reverse-map integer log-key IDs → human-readable event strings.
          2. Retrieve MITRE ATT&CK context from ChromaDB (if retriever is set).
          3. Generate a structured 5-section incident report via LlamaResponder.

        CLAUDE.md §10 contract: Tier 3 sees ONLY sequences DeepLog confirmed as
        anomalous.  It NEVER receives raw events or Tier-1-dismissed events.

        Args:
            session_data: Mapping from principal_id to anomaly data from _run_tier2.
                          Each value: {flags, log_keys, events, templates, oov_key}.

        Returns:
            (reports, tier3_metrics)
        """
        reports: list[dict[str, Any]] = []

        for session_id, data in session_data.items():
            flags: list[bool]          = data["flags"]
            log_keys: list[int]        = data["log_keys"]
            sess_events: list[dict]    = data["events"]
            templates: dict[int, str]  = data["templates"]
            oov_key: int               = data["oov_key"]

            # ---- Reverse-map: int keys → enriched event dicts ------------
            enriched: list[dict[str, Any]] = reverse_map_events(
                templates=templates,
                log_keys=log_keys,
                flags=flags,
                raw_events=sess_events,
                oov_key=oov_key,
            )

            # ---- Assemble LlamaResponder context -------------------------
            drain_templates: list[str] = [e["drain_template"] for e in enriched]
            event_names: list[str]     = [e["event_name"]     for e in enriched]
            timestamps: list[str]      = [e["timestamp"]      for e in enriched]
            source_ips: list[str]      = [e["source_ip"]      for e in enriched]
            user_arns: list[str]       = [e["user_arn"]       for e in enriched]

            n_total: int = len(sess_events)
            n_flagged: int = len(enriched)
            anomaly_ratio: float = n_flagged / n_total if n_total > 0 else 0.0

            context: dict[str, Any] = {
                "session_id":    session_id,
                "total_events":  n_total,
                "anomaly_ratio": anomaly_ratio,
                "timestamps":    timestamps,
                "source_ips":    source_ips,
                "user_arns":     user_arns,
                "event_names":   event_names,
            }

            # ---- RAG retrieval (graceful fallback if unavailable) --------
            # Build query from the first ≤6 distinct event names — enough
            # semantic signal for the BGE retriever without token-budget waste.
            rag_context: list[str] | None = None
            if self._retriever is not None:
                rag_query: str = " ".join(dict.fromkeys(event_names[:6]))
                try:
                    rag_context = self._retriever.query(rag_query, top_k=4)
                except RuntimeError:
                    # Index empty or not built — fall back to v1 (no RAG).
                    rag_context = None

            # ---- LLM report generation ------------------------------------
            # CLAUDE.md §10: dashboard must label this as "suggested analysis —
            # requires analyst review".  The responder adds a [MOCK MODE]
            # disclaimer automatically in mock_mode=True.
            report_text: str = self._responder.generate_report(
                anomalous_logs=drain_templates,
                context=context,
                rag_context=rag_context,
            )

            reports.append({
                "session_id":    session_id,
                "report_text":   report_text,
                "flagged_events": enriched,
                "anomaly_ratio": anomaly_ratio,
            })

        return reports, {"tier3_input_sessions": len(session_data)}


# ---------------------------------------------------------------------------
#  Module-level helper functions (not methods — easier to test in isolation)
# ---------------------------------------------------------------------------


def _all_principals(events: list[dict[str, Any]]) -> set[str]:
    """Return the set of all distinct principal_ids in the event list."""
    return {_principal_id(e) for e in events}


def _principal_id(event: dict[str, Any]) -> str:
    """Extract principal_id from any supported schema.

    Preference order:
      1. userIdentity.principalId  (nested AWS raw CloudTrail schema)
      2. userIdentity.userName     (nested fallback for root/IAM users)
      3. userIdentity.arn          (last resort in nested schema)
      4. principalId               (flat camelCase synthetic schema)
      5. principal_id              (processed snake_case schema from ingest.py)
    """
    uid = event.get("userIdentity")
    if isinstance(uid, dict):
        pid = uid.get("principalId") or uid.get("userName") or uid.get("arn")
        if pid:
            return str(pid)
    # Flat camelCase synthetic schema
    pid_camel = event.get("principalId")
    if pid_camel:
        return str(pid_camel)
    # Processed snake_case schema (output of ingest.load_events())
    pid_snake = event.get("principal_id")
    if pid_snake:
        return str(pid_snake)
    return "<unknown>"


def _build_events_dataframe(events: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalise raw CloudTrail events into the DataFrame shape aggregate_features() expects.

    Required output columns:
        event_time (datetime64[ns, UTC]), event_name, event_source,
        principal_id, entity_type, error_code, aws_region, _source.

    Supports three input schemas detected automatically from the first event:
      - Processed snake_case (ingest.load_events() output): event_name,
        event_time, principal_id, source_ip, error_code, aws_region, _source.
        This is the common case when running from processed JSONL files.
      - Flat camelCase synthetic: eventName, eventTime, principalId, entityType.
      - Nested AWS official: userIdentity dict, eventName, awsRegion.

    Rows with unparseable timestamps are kept (current UTC substituted) rather
    than dropped — silently losing events would undercount sensitive_action_count.

    Args:
        events: Event dicts in any supported schema, any order.

    Returns:
        Typed DataFrame.  Empty if events is empty.
    """
    if not events:
        return pd.DataFrame()

    # ---- Schema detection ------------------------------------------------
    # The processed schema (output of ingest.load_events / run_day2.py) uses
    # snake_case keys.  Detecting from the first event is sufficient — the
    # pipeline never mixes schemas within a single JSONL file.
    _is_processed: bool = "event_name" in events[0]

    rows: list[dict[str, Any]] = []
    n_bad_ts: int = 0

    for raw in events:
        # ---- Field extraction (schema-specific) --------------------------
        if _is_processed:
            # Snake_case processed schema — fields are ready to use directly.
            # event_time may be "2023-01-01 13:19:35+00:00" (space separator)
            # or "2023-01-01T13:19:35Z"; pd.Timestamp handles both.
            ts_str: str = str(raw.get("event_time", ""))
            event_name: str  = str(raw.get("event_name",   "<unknown>"))
            event_source: str = str(raw.get("event_source", "<unknown>"))
            principal_id: str = str(raw.get("principal_id", "<unknown>"))
            entity_type: str  = str(raw.get("entity_type",  "<unknown>"))
            error_code: str   = str(raw.get("error_code",   ""))
            aws_region: str   = str(raw.get("aws_region",   "<unknown>"))
            # _source already set correctly in processed data
            source: str = str(raw.get("_source", "unknown"))
        else:
            # Raw CloudTrail (camelCase flat or nested) — use normalize_event.
            norm = normalize_event(raw)
            ts_str = norm["eventTime"]
            event_name  = norm["eventName"]
            event_source = norm["eventSource"]
            principal_id = norm["principalId"]
            entity_type  = norm["entityType"]
            error_code   = norm["errorCode"]
            aws_region   = str(raw.get("awsRegion", "<unknown>"))
            source = "unknown"  # label unknown at inference time

        # ---- Timestamp parse --------------------------------------------
        try:
            event_time = pd.Timestamp(ts_str, tz="UTC")
        except Exception:
            event_time = pd.Timestamp.now(tz="UTC")
            n_bad_ts += 1

        rows.append({
            "event_time":   event_time,
            "event_name":   event_name,
            "event_source": event_source,
            "principal_id": principal_id,
            "entity_type":  entity_type,
            "error_code":   error_code,
            "aws_region":   aws_region,
            # _source is required by aggregate_features() for window labelling.
            # For inference on processed data, preserve the original label so
            # the metrics reflect reality; for raw inference use "unknown".
            "_source":      source,
        })

    if n_bad_ts > 0:
        print(
            f"[cascade] WARNING: {n_bad_ts}/{len(events)} events had "
            f"unparseable event_time — substituted with current UTC timestamp."
        )

    df: pd.DataFrame = pd.DataFrame(rows)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df


def _empty_t1_metrics(n_events: int) -> dict[str, Any]:
    """Return a zero-filled Tier 1 metrics dict for early-exit cases."""
    return {
        "tier1_input_events":       n_events,
        "tier1_input_windows":      0,
        "tier1_flagged_windows":    0,
        "tier1_flagged_principals": 0,
    }
