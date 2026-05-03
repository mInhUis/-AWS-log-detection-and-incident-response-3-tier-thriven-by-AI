"""
src/tier1_ml/train.py
─────────────────────────────────────────────────────────────────────────────
Pipeline position: Tier 1 model fitting — takes the aggregated feature matrix
from preprocessing/features.py and trains a scikit-learn Isolation Forest.

Why Isolation Forest?
  The core problem is semi-supervised anomaly detection: we have abundant
  labelled NORMAL data but no labelled attack examples at training time.
  Isolation Forest (Liu et al., 2008) is the canonical choice here for four
  reasons:
    1. Unsupervised — no attack labels required.
    2. Scales well to tabular feature vectors (our 7-feature space is trivial).
    3. Single interpretable hyperparameter: contamination (expected fraction of
       anomalies in the training set).  For synthetic-only training data,
       contamination="auto" is appropriate because the data is attack-free.
    4. Runs on CPU in seconds for <1M windows — no GPU needed on the laptop.

  Alternatives rejected:
    - LOF (Local Outlier Factor): sensitive to density in high-dimensional
      spaces; our 7D feature space is fine but LOF is O(n²) at scoring time.
    - One-Class SVM: requires kernel selection and is slower to score.
    - PyOD ECOD: theoretically stronger on tabular data but less interpretable
      for thesis defense.  ECOD is the Day 8 ablation comparison target.

Why save a joblib bundle (not just the model)?
  At scoring time (detect.py) we need to verify that the model was trained on
  the same feature set as the current code.  The bundle stores feature_columns
  and a SHA-256 hash of the feature list so detect.py can refuse to score if
  the features have been changed since training — preventing silent mismatch
  bugs (CLAUDE.md §5.8).

Data-flow:
  data/processed/train_features.parquet
      └─ fit_isolation_forest()  →  models/isolation_forest.pkl (joblib bundle)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Final

import joblib
import sklearn
import sklearn.ensemble

from src.preprocessing.features import FEATURE_COLUMNS

# ── Path constants ────────────────────────────────────────────────────────────
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

TRAIN_FEATURES_PATH: Final[Path] = Path(
    os.environ.get(
        "TRAIN_FEATURES_PATH",
        str(_PROJECT_ROOT / "data" / "processed" / "train_features.parquet"),
    )
)

MODEL_OUTPUT_PATH: Final[Path] = Path(
    os.environ.get(
        "ISOLATION_FOREST_PATH",
        str(_PROJECT_ROOT / "models" / "isolation_forest.pkl"),
    )
)

# ── Default hyperparameters ───────────────────────────────────────────────────
# contamination="auto" means the decision boundary is set at the score of
# the most anomalous training sample.  For clean synthetic-only training data
# this is appropriate; set to a float (e.g. 0.01) when mixing in known attacks.
_DEFAULT_CONTAMINATION: Final[str] = "auto"
_DEFAULT_N_ESTIMATORS: Final[int] = 100
_DEFAULT_RANDOM_STATE: Final[int] = 42


# ═══════════════════════════════════════════════════════════════════════════════
# Public functions
# ═══════════════════════════════════════════════════════════════════════════════


def fit_isolation_forest(
    train_features_path: Path,
    model_output_path: Path,
    contamination: str | float = _DEFAULT_CONTAMINATION,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
    random_state: int = _DEFAULT_RANDOM_STATE,
) -> dict:
    """Fit an Isolation Forest on the training feature matrix and save a bundle.

    Args:
        train_features_path: Path to train_features.parquet from features.py.
        model_output_path:   Where to write the joblib bundle (.pkl).
        contamination:       Expected anomaly fraction; "auto" for clean data.
        n_estimators:        Number of isolation trees (more = slower + more stable).
        random_state:        RNG seed for reproducibility.

    Returns:
        Summary dict with train_n_rows, train_fpr, model path.

    Raises:
        FileNotFoundError: If train_features_path does not exist.
        ValueError:        If column count or names don't match FEATURE_COLUMNS.
    """
    if not train_features_path.exists():
        raise FileNotFoundError(
            f"Training features not found at {train_features_path}. "
            "Run: python scripts/run_day2.py --mode mock --force"
        )

    import pandas as pd

    df = pd.read_parquet(train_features_path, engine="pyarrow")

    # ── Validate feature columns ──────────────────────────────────────────────
    # Fail loudly if someone has added/removed a column in features.py without
    # retraining.  A shape check alone would miss column order mismatches.
    actual_cols = [c for c in df.columns if c in set(FEATURE_COLUMNS)]
    if sorted(actual_cols) != sorted(FEATURE_COLUMNS):
        raise ValueError(
            f"Feature column mismatch.\n"
            f"  Expected: {FEATURE_COLUMNS}\n"
            f"  Found in parquet: {list(df.columns)}\n"
            "Re-run features.py to regenerate the feature matrix."
        )

    # X: (n_windows, 7) float32 — the Isolation Forest input matrix
    X = df[FEATURE_COLUMNS].values.astype("float32")
    # Verify shape explicitly so errors are clear, not buried in sklearn stack traces
    if X.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} feature columns, got {X.shape[1]}."
        )

    print(
        f"Training Isolation Forest on {X.shape[0]:,} windows"
        f" × {X.shape[1]} features"
        f" (contamination={contamination}, n_estimators={n_estimators}) ..."
    )

    # ── Fit ──────────────────────────────────────────────────────────────────
    model = sklearn.ensemble.IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,  # use all available CPU cores on the laptop
    )
    model.fit(X)

    # ── Compute train false-positive rate ────────────────────────────────────
    # Since X is pure synthetic normal data, any -1 prediction is a false alarm.
    # This number is the "Tier 1 noise floor" and should appear in the thesis
    # cost table.  contamination="auto" typically gives ~0% on clean data.
    predictions = model.predict(X)
    train_fpr = float((predictions == -1).mean())
    print(f"Train FPR: {train_fpr:.4f} ({(predictions == -1).sum():,} / {len(X):,} windows flagged)")

    # ── Feature-set hash ─────────────────────────────────────────────────────
    # SHA-256 of the sorted JSON representation of FEATURE_COLUMNS.
    # detect.py recomputes this hash at scoring time and refuses to proceed if
    # it doesn't match — prevents silent column-mismatch scoring errors.
    feature_set_hash = hashlib.sha256(
        json.dumps(FEATURE_COLUMNS).encode("utf-8")
    ).hexdigest()

    # ── Save bundle ───────────────────────────────────────────────────────────
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "contamination": contamination,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "sklearn_version": sklearn.__version__,
        "train_n_rows": int(X.shape[0]),
        "train_fpr": train_fpr,
        "feature_set_hash": feature_set_hash,
    }
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_output_path)
    print(f"Saved model bundle -> {model_output_path}")

    return {
        "train_n_rows": bundle["train_n_rows"],
        "train_fpr": train_fpr,
        "model_path": str(model_output_path),
        "sklearn_version": sklearn.__version__,
        "feature_set_hash": feature_set_hash,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Fit Isolation Forest from the command line."""
    parser = argparse.ArgumentParser(description="Train Tier 1 Isolation Forest")
    parser.add_argument(
        "--train-features",
        type=Path,
        default=TRAIN_FEATURES_PATH,
        help="Path to train_features.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_OUTPUT_PATH,
        help="Output path for the model bundle (.pkl)",
    )
    parser.add_argument(
        "--contamination",
        default=_DEFAULT_CONTAMINATION,
        help='Anomaly fraction; "auto" or float 0–0.5',
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=_DEFAULT_N_ESTIMATORS,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=_DEFAULT_RANDOM_STATE,
    )
    args = parser.parse_args()

    # contamination can be "auto" (str) or a float — parse accordingly
    contamination: str | float = args.contamination
    try:
        contamination = float(args.contamination)
    except ValueError:
        pass  # keep as string "auto"

    summary = fit_isolation_forest(
        train_features_path=args.train_features,
        model_output_path=args.output,
        contamination=contamination,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    print("\n── Training summary ──")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
