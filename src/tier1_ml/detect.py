"""
src/tier1_ml/detect.py
─────────────────────────────────────────────────────────────────────────────
Pipeline position: Tier 1 inference — scores a feature matrix against a
fitted Isolation Forest and emits per-window alert records for suspicious
entity-windows.

What happens here?
  IsolationForest.decision_function(X) returns a "normality score" per row:
  higher = more normal, lower = more anomalous.  predict(X) binarises this
  into +1 (normal) / -1 (anomaly) using the contamination threshold set at
  training time.

  Every row where predict == -1 becomes an alert record in the output JSONL.
  That JSONL is the Tier 2 handoff on Day 7: cascade.py reads it and feeds
  only the flagged entities' sequences into the DeepLog LSTM.

Why validate feature_set_hash?  (CLAUDE.md §5.8)
  If someone adds a feature to FEATURE_COLUMNS in features.py after training,
  the parquet files have the new column but the saved model was trained on the
  old 7-column matrix.  Silently scoring with a mismatched matrix would produce
  garbage anomaly scores.  The hash check catches this before any damage.

Why decision_function AND predict?
  predict() alone loses the ranking information: two rows both scored -1 could
  have very different normality scores (one barely anomalous, one extreme).
  Storing the raw decision_function score in the alert lets Day 8 evaluation
  plot ROC curves and tune the threshold rather than using the hard binary.

Data-flow:
  data/processed/val_features.parquet
  models/isolation_forest.pkl
      └─ score_features()  →  data/processed/tier1_alerts_val.jsonl
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Final

import joblib
import pandas as pd

from src.preprocessing.features import FEATURE_COLUMNS

# ── Path constants ────────────────────────────────────────────────────────────
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

ISOLATION_FOREST_PATH: Final[Path] = Path(
    os.environ.get(
        "ISOLATION_FOREST_PATH",
        str(_PROJECT_ROOT / "models" / "isolation_forest.pkl"),
    )
)

VAL_FEATURES_PATH: Final[Path] = Path(
    os.environ.get(
        "VAL_FEATURES_PATH",
        str(_PROJECT_ROOT / "data" / "processed" / "val_features.parquet"),
    )
)

ALERTS_OUTPUT_PATH: Final[Path] = Path(
    os.environ.get(
        "TIER1_ALERTS_PATH",
        str(_PROJECT_ROOT / "data" / "processed" / "tier1_alerts_val.jsonl"),
    )
)


# ═══════════════════════════════════════════════════════════════════════════════
# Public functions
# ═══════════════════════════════════════════════════════════════════════════════


def score_features(
    model_path: Path,
    features_path: Path,
    alerts_output_path: Path,
) -> dict:
    """Score a feature matrix with the fitted Isolation Forest and emit alerts.

    Args:
        model_path:         Path to the joblib bundle from train.py.
        features_path:      Path to a *_features.parquet file from features.py.
        alerts_output_path: Where to write per-window alert records as JSONL.

    Returns:
        Summary dict: {n_rows, n_alerts, alert_rate}.

    Raises:
        FileNotFoundError: If model_path or features_path do not exist.
        ValueError:        If feature_set_hash or column order doesn't match.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {model_path}. "
            "Run: python scripts/run_day2.py --mode mock --force"
        )
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature file not found at {features_path}. "
            "Run: python scripts/run_day2.py --mode mock --force"
        )

    # ── Load and validate bundle ──────────────────────────────────────────────
    bundle: dict = joblib.load(model_path)
    _validate_bundle(bundle)

    # ── Load features ─────────────────────────────────────────────────────────
    df = pd.read_parquet(features_path, engine="pyarrow")

    # Verify column order matches what the model was trained on (not just set)
    if bundle["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError(
            f"Feature column order mismatch.\n"
            f"  Model trained on: {bundle['feature_columns']}\n"
            f"  Current FEATURE_COLUMNS: {FEATURE_COLUMNS}\n"
            "Retrain the model: python -m src.tier1_ml.train"
        )

    # X: (n_windows, 7) float32
    X = df[FEATURE_COLUMNS].values.astype("float32")

    model = bundle["model"]

    # ── Score ─────────────────────────────────────────────────────────────────
    # decision_function: higher = more normal (IsolationForest convention)
    scores = model.decision_function(X)   # shape: (n_windows,) float64
    predictions = model.predict(X)        # shape: (n_windows,) int; -1 or +1

    # ── Build alert records ──────────────────────────────────────────────────
    alerts: list[dict] = []
    for i, pred in enumerate(predictions):
        if pred != -1:
            continue  # normal window — not forwarded to Tier 2

        row = df.iloc[i]
        alert: dict = {
            "principal_id": str(row["principal_id"]),
            "window_start": row["window_start"].isoformat()
            if hasattr(row["window_start"], "isoformat")
            else str(row["window_start"]),
            "window_size_minutes": int(row["window_size_minutes"]),
            "entity_type": str(row.get("entity_type", "")),
            "anomaly_score": float(scores[i]),
            "feature_values": {
                col: float(row[col]) for col in FEATURE_COLUMNS
            },
            "model_version": bundle["sklearn_version"],
        }
        alerts.append(alert)

    # ── Write JSONL ───────────────────────────────────────────────────────────
    alerts_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(alerts_output_path, "w", encoding="utf-8") as fh:
        for alert in alerts:
            fh.write(json.dumps(alert) + "\n")

    n_rows = len(df)
    n_alerts = len(alerts)
    alert_rate = n_alerts / n_rows if n_rows > 0 else 0.0

    print(
        f"Scored {n_rows:,} windows: {n_alerts:,} alerts"
        f" (alert_rate={alert_rate:.4f})"
        f" -> {alerts_output_path}"
    )

    return {
        "n_rows": n_rows,
        "n_alerts": n_alerts,
        "alert_rate": alert_rate,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _validate_bundle(bundle: dict) -> None:
    """Check that the bundle's feature_set_hash matches the current FEATURE_COLUMNS.

    Raises:
        ValueError: If the hash doesn't match (features changed since training).
    """
    expected_hash = hashlib.sha256(
        json.dumps(FEATURE_COLUMNS).encode("utf-8")
    ).hexdigest()

    stored_hash = bundle.get("feature_set_hash", "")
    if stored_hash != expected_hash:
        raise ValueError(
            f"Feature set hash mismatch.\n"
            f"  Stored hash: {stored_hash}\n"
            f"  Current hash: {expected_hash}\n"
            f"  Current FEATURE_COLUMNS: {FEATURE_COLUMNS}\n"
            "Retrain the model: python -m src.tier1_ml.train"
        )

    required_keys = {
        "model", "feature_columns", "contamination", "n_estimators",
        "random_state", "sklearn_version", "train_n_rows", "train_fpr",
        "feature_set_hash",
    }
    missing = required_keys - set(bundle.keys())
    if missing:
        raise ValueError(
            f"Model bundle is missing required keys: {missing}. "
            "Retrain the model: python -m src.tier1_ml.train"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Score features from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Score Tier 1 Isolation Forest")
    parser.add_argument("--model", type=Path, default=ISOLATION_FOREST_PATH)
    parser.add_argument("--features", type=Path, default=VAL_FEATURES_PATH)
    parser.add_argument("--output", type=Path, default=ALERTS_OUTPUT_PATH)
    args = parser.parse_args()

    summary = score_features(
        model_path=args.model,
        features_path=args.features,
        alerts_output_path=args.output,
    )
    print("\n── Scoring summary ──")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
