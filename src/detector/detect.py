"""DeepLog anomaly detection via top-k next-key prediction.

**Why top-k instead of a fixed threshold?**
DeepLog's core insight is that the LSTM learns a *probability distribution*
over the next log key.  Rather than thresholding a continuous score (which
requires careful calibration), the top-k approach asks a simpler question:
"Is the observed next key among the k most likely keys?"  This is robust
because:
    1. Normal sequences have low entropy — the true key almost always
       lands in a small candidate set (k = 5–10 suffices empirically).
    2. Anomalous events (e.g., a ``CreateUser`` after a ``ListBuckets``)
       produce transitions the model has never seen, pushing them far
       outside the top-k set.
    3. ``k`` is the *only* tuning knob, and it has an intuitive
       interpretation: higher k = fewer false positives, lower recall.

Tuning discipline (CLAUDE.md §7):
    Sweep ``top_k`` over {3, 5, 7, 9} on the *validation* split only.
    Never re-tune after consulting test metrics — that is test-set fitting
    and invalidates the evaluation chapter.

Evaluation (Advisor Skill 4 — Imbalanced Metric Focus):
    Cloud anomalies are extremely rare events.  Raw accuracy is misleading
    (a model predicting "normal" for everything achieves >99% accuracy).
    ``evaluate_predictions`` therefore reports **Precision, Recall, and F1**
    as the primary metrics, with accuracy included only for completeness.

Data Flow:
    Trained DeepLogModel + test log-key sessions
        → detect_anomalies()    → per-event bool flags
        → evaluate_predictions() → Precision / Recall / F1 dict
"""

from __future__ import annotations

import torch

from src.detector.deeplog import DeepLogModel


def detect_anomalies(
    model: DeepLogModel,
    sequences: list[list[int]],
    window_size: int,
    top_k: int = 9,
    device: torch.device | None = None,
) -> list[list[bool]]:
    """Flag each event in each session as normal (False) or anomalous (True).

    For each session the first ``window_size`` events are unpredictable
    (no full window exists yet) and are always marked **False** (normal).
    From position ``window_size`` onward, the model predicts the next key
    from the preceding window; if the true key is *not* in the top-k
    predictions, the event is flagged as anomalous.

    Device handling:
        If *device* is ``None`` the function infers the device from the
        model's own parameters (``next(model.parameters()).device``).  This
        guarantees the input tensors always land on the same device as the
        model, avoiding a device-mismatch RuntimeError when the caller omits
        the argument on a CUDA model.

    Args:
        model:       Trained :class:`DeepLogModel` in eval mode.
        sequences:   Per-session log-key sequences (same format as
                     ``LogKeyDataset`` input).
        window_size: Sliding window width (must match training).
        top_k:       Number of top predictions to consider.  Lower k
                     increases sensitivity (more anomalies detected) but
                     also increases false-positive rate.  Must be ≤
                     ``model.num_keys``.
        device:      Torch device for inference.  Defaults to the device
                     the model is already on.

    Returns:
        Nested list mirroring the structure of *sequences*: for each
        session ``s``, ``result[s][j]`` is ``True`` if event ``j`` is
        anomalous.

    Raises:
        ValueError: If ``top_k`` exceeds the model vocabulary size, or if
                    the sequences contain a key ID beyond the model's
                    embedding table (stale model).

    Tensor shapes (per window):
        window_tensor : (1, window_size) int64
        logits        : (1, num_keys)    float32
        top_indices   : (1, top_k)       int64
    """
    # ---- Device: match the model rather than defaulting to CPU ----
    # If the caller passes device=None and the model is on CUDA, creating
    # window_tensor on CPU would trigger a device-mismatch RuntimeError deep
    # inside the forward pass.  Inferring from the model is the idiomatic
    # PyTorch pattern.
    if device is None:
        device = next(model.parameters()).device

    model_num_keys: int = model.num_keys  # uses the public property

    # ---- top_k range guard ----
    # torch.topk raises a cryptic RuntimeError if k > vocabulary size.
    # Catch it early with a clear message.
    if top_k > model_num_keys:
        raise ValueError(
            f"top_k={top_k} exceeds the model vocabulary size "
            f"num_keys={model_num_keys}.  Lower top_k or retrain on a "
            f"richer corpus."
        )

    # ---- Stale-model guard: embedding size vs current vocabulary ----
    # After Drain is rerun the vocabulary size (num_keys) may change.
    # If the model's embedding table doesn't cover every key in the
    # sequences, inference will silently produce garbage or crash with
    # an index-out-of-range error.  Catch this early with a clear
    # message pointing the user to retrain.
    max_key_in_data: int = max(
        (key for seq in sequences for key in seq), default=-1,
    )
    if max_key_in_data >= model_num_keys:
        raise ValueError(
            f"Stale model: the sequences contain log key "
            f"{max_key_in_data} but the model's embedding table only "
            f"covers keys 0..{model_num_keys - 1} "
            f"(num_keys={model_num_keys}). Drain likely produced "
            f"new templates since the model was trained. Delete the "
            f"checkpoint and retrain:\n"
            f"    rm models/deeplog.pt\n"
            f"    python src/models/train_deeplog.py"
        )

    model.eval()
    all_flags: list[list[bool]] = []

    with torch.no_grad():
        for seq in sequences:
            seq_len: int = len(seq)
            flags: list[bool] = [False] * seq_len  # default: normal

            if seq_len <= window_size:
                # Too short to predict anything — mark all normal
                all_flags.append(flags)
                continue

            for i in range(window_size, seq_len):
                window: list[int] = seq[i - window_size : i]
                true_key: int = seq[i]

                # window_tensor: (1, window_size) int64
                window_tensor: torch.Tensor = torch.tensor(
                    [window], dtype=torch.long, device=device,
                )
                # logits: (1, num_keys) float32
                logits, _ = model(window_tensor)

                # top_indices: (1, top_k) → squeeze to (top_k,) int64
                _, top_indices = torch.topk(logits, k=top_k, dim=-1)
                top_set: set[int] = set(top_indices.squeeze(0).tolist())

                if true_key not in top_set:
                    flags[i] = True  # anomalous

            all_flags.append(flags)

    return all_flags


def evaluate_predictions(
    predicted_flags: list[list[bool]],
    ground_truth_flags: list[list[bool]],
) -> dict[str, float]:
    """Compute Precision, Recall, F1, and Accuracy from per-event flags.

    **Imbalanced Metric Focus (Advisor Skill 4):**
    Cloud anomalies are rare.  This function foregrounds Precision, Recall,
    and F1 — the metrics that matter when the positive class is a minority.
    Accuracy is included for completeness but should NOT be used as the
    primary evaluation criterion.

    Definitions (anomaly = positive class):
        TP — model flags anomaly, ground truth is anomaly.
        FP — model flags anomaly, ground truth is normal.
        FN — model says normal, ground truth is anomaly.
        TN — model says normal, ground truth is normal.

    Args:
        predicted_flags:    Output of :func:`detect_anomalies`.
        ground_truth_flags: Same shape, with ``True`` = known anomaly.

    Returns:
        Dict with keys ``precision``, ``recall``, ``f1``, ``accuracy``,
        ``tp``, ``fp``, ``fn``, ``tn``.  Precision and recall default to
        ``0.0`` when their denominators are zero (no positive predictions
        or no positive ground-truth labels respectively).
    """
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    for pred_seq, true_seq in zip(predicted_flags, ground_truth_flags):
        for p, t in zip(pred_seq, true_seq):
            if p and t:
                tp += 1
            elif p and not t:
                fp += 1
            elif not p and t:
                fn += 1
            else:
                tn += 1

    precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall: float = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1: float = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    total: int = tp + fp + fn + tn
    accuracy: float = (tp + tn) / total if total > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }