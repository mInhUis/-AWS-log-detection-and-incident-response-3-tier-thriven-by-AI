#!/usr/bin/env python3
"""
tier1_ml/tune_contamination.py
───────────────────────────────────────────────────────────────────────────────
Step E: Sweep contamination ∈ {0.01, 0.02, 0.05, 0.10}, evaluate each model
on the labelled mixed validation set, and select the best value.

SELECTION CRITERION  (§3 cost-cascade contract):
  max F1  subject to  alert_rate ≤ 0.06

  Rationale: the 5% alert-rate target in §3 allows 1% slack (0.06) because
  contamination is a continuous knob that won't land exactly on 5%.  F1 is
  the primary objective because precision and recall trade off — at low attack
  prevalence (~6.5%) maximising recall alone would flag half the normal windows
  and violate the cost contract.

WHAT THIS DOES:
  For each contamination value:
    1. Retrain Isolation Forest on train_features.parquet (pure synth normal).
    2. Score val_features.parquet (mixed, labelled).
    3. Compute: precision, recall, F1, FPR on normal subset, alert_rate overall,
       confusion matrix, ROC AUC, PR AUC.

  Pick best by criterion above.
  Save winner to models/isolation_forest_tuned.pkl.
  Write full sweep results to data/processed/tier1_metrics_val.json.

  The contamination='auto' model (isolation_forest.pkl) is NOT overwritten —
  it is kept for the Day 8 ablation comparison.

EXPECTED OUTCOME  (from Day 2 status report):
  contamination ≈ 0.02 chosen.
  FPR on normal ≈ 2%.
  Recall on attack ≈ 30–80%.
  Alert rate overall ≈ 3–6%.
  Precision low-ish (expected at this class ratio — documented, not a bug).

USAGE:
  python scripts/tier1_tune_contamination.py
  python scripts/tier1_tune_contamination.py --contamination-values 0.01 0.02 0.05
  python scripts/tier1_tune_contamination.py --dry-run   # skip writes, print only

PIPELINE POSITION:
  Depends on: train_features.parquet, val_features.parquet (labelled, Step D)
  Produces:   models/isolation_forest_tuned.pkl
              data/processed/tier1_metrics_val.json
  Consumed by: Day 3 (DeepLog training), Day 8 (final evaluation)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

# ── Project root on path ──────────────────────────────────────────────────────
_SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.tier1_ml.train import fit_isolation_forest
from src.tier1_ml.detect import score_features
from src.preprocessing.features import FEATURE_COLUMNS

# ── I/O paths ─────────────────────────────────────────────────────────────────
DATA_PROCESSED: Final[Path] = _PROJECT_ROOT / "data" / "processeded1"
MODELS_DIR: Final[Path] = _PROJECT_ROOT / "models1"

TRAIN_FEATURES_PATH: Final[Path] = DATA_PROCESSED / "train_features.parquet"
VAL_FEATURES_PATH: Final[Path] = DATA_PROCESSED / "val_features.parquet"

TUNED_MODEL_PATH: Final[Path] = MODELS_DIR / "isolation_forest_tuned.pkl"
METRICS_PATH: Final[Path] = DATA_PROCESSED / "tier1_metrics_val.json"

# ── Sweep defaults ────────────────────────────────────────────────────────────
DEFAULT_CONTAMINATION_VALUES: Final[tuple[float, ...]] = (0.01, 0.02, 0.05, 0.10)
ALERT_RATE_CAP: Final[float] = 0.06       # §3 cost-cascade contract + 1% slack
N_ESTIMATORS: Final[int] = 100
RANDOM_STATE: Final[int] = 42


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    val_df: pd.DataFrame,
    model_path: Path,
) -> dict:
    """Score val_features and compute full classification metrics.

    Uses a temp file for the alerts JSONL so the sweep doesn't leave
    intermediate files on disk.

    Args:
        val_df:     Loaded val_features.parquet DataFrame (with is_attack_window).
        model_path: Path to the .pkl bundle to score with.

    Returns:
        Dict with precision, recall, f1, fpr_normal, alert_rate, confusion
        matrix, roc_auc, pr_auc, n_attack, n_normal, n_windows.
    """
    import sklearn.metrics as skm
    import joblib

    # Load model bundle directly to call decision_function
    bundle = joblib.load(model_path)
    model = bundle["model"]

    X = val_df[FEATURE_COLUMNS].values.astype("float32")
    y_true = val_df["is_attack_window"].astype(int).values  # 1=attack, 0=normal

    # decision_function: higher = more normal → negate for anomaly scores
    raw_scores = model.decision_function(X)          # higher = more normal
    anomaly_scores = -raw_scores                     # higher = more anomalous
    y_pred = (model.predict(X) == -1).astype(int)   # 1=flagged, 0=normal

    n_windows = len(y_true)
    n_attack = int(y_true.sum())
    n_normal = int((y_true == 0).sum())

    # ── Alert rate (overall) ──────────────────────────────────────────────────
    alert_rate = float(y_pred.mean())

    # ── Precision / Recall / F1 on attack class ───────────────────────────────
    # zero_division=0: if model flags nothing, precision=0 not warning
    precision = float(skm.precision_score(y_true, y_pred, zero_division=0))
    recall    = float(skm.recall_score(y_true, y_pred, zero_division=0))
    f1        = float(skm.f1_score(y_true, y_pred, zero_division=0))

    # ── FPR on normal-only subset ─────────────────────────────────────────────
    # FPR = FP / (FP + TN)  measured only on windows we know are normal
    normal_mask = y_true == 0
    fpr_normal = float(y_pred[normal_mask].mean()) if n_normal > 0 else 0.0

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = skm.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # ── ROC AUC ───────────────────────────────────────────────────────────────
    try:
        roc_auc = float(skm.roc_auc_score(y_true, anomaly_scores))
    except ValueError:
        roc_auc = float("nan")   # only one class present (shouldn't happen)

    # ── PR AUC ───────────────────────────────────────────────────────────────
    try:
        pr_auc = float(skm.average_precision_score(y_true, anomaly_scores))
    except ValueError:
        pr_auc = float("nan")

    return {
        "n_windows":  n_windows,
        "n_attack":   n_attack,
        "n_normal":   n_normal,
        "alert_rate": alert_rate,
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
        "fpr_normal": fpr_normal,
        "roc_auc":    roc_auc,
        "pr_auc":     pr_auc,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Selection criterion
# ─────────────────────────────────────────────────────────────────────────────

def select_best(sweep_results: list[dict]) -> dict | None:
    """Pick the best contamination value by: max F1 s.t. alert_rate ≤ 0.06.

    Falls back to max recall if no candidate satisfies the alert-rate cap
    (documented as a warning, not a hard failure — lets the pipeline proceed
    with a note for the thesis).

    Args:
        sweep_results: List of per-contamination result dicts.

    Returns:
        The best result dict, or None if sweep_results is empty.
    """
    if not sweep_results:
        return None

    # Primary: F1-maximising candidates that satisfy alert-rate cap
    eligible = [r for r in sweep_results if r["metrics"]["alert_rate"] <= ALERT_RATE_CAP]

    if eligible:
        return max(eligible, key=lambda r: r["metrics"]["f1"])

    # Fallback: no candidate within cap — pick max recall (least-bad option)
    return max(sweep_results, key=lambda r: r["metrics"]["recall"])


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_step_e(
    contamination_values: tuple[float, ...] = DEFAULT_CONTAMINATION_VALUES,
    dry_run: bool = False,
) -> dict:
    """Execute the contamination sweep and save the best model.

    Args:
        contamination_values: Float contamination values to sweep.
        dry_run: If True, print results without writing files.

    Returns:
        Full report dict written to tier1_metrics_val.json.
    """
    print("═" * 72)
    print("  Step E: Tier 1 Contamination Sweep & Model Selection")
    print("═" * 72)
    print()

    # ── Input checks ─────────────────────────────────────────────────────────
    for label, path in [
        ("train_features.parquet", TRAIN_FEATURES_PATH),
        ("val_features.parquet",   VAL_FEATURES_PATH),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}")
            sys.exit(1)

    val_df = pd.read_parquet(VAL_FEATURES_PATH)

    if "is_attack_window" not in val_df.columns:
        print("ERROR: val_features.parquet has no 'is_attack_window' column.")
        print("  Run Step D first: python scripts/mix_val_and_regenerate_features.py")
        sys.exit(1)

    n_attack = val_df["is_attack_window"].sum()
    n_normal = (~val_df["is_attack_window"]).sum()
    print(f"Validation set: {len(val_df):,} windows  "
          f"({n_attack} attack / {n_normal} normal)")
    print(f"Contamination values to sweep: {list(contamination_values)}")
    print(f"Selection criterion: max F1  s.t.  alert_rate ≤ {ALERT_RATE_CAP}")
    print()

    if n_attack == 0:
        print("ERROR: No attack windows in val_features.parquet.")
        print("  Step E cannot compute recall or F1. Fix Step D first.")
        sys.exit(1)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    sweep_results: list[dict] = []

    for contamination in contamination_values:
        print(f"── contamination={contamination} {'─' * (50 - len(str(contamination)))}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model = Path(tmpdir) / "if_temp.pkl"

            # Retrain on pure synth-normal train set
            train_summary = fit_isolation_forest(
                train_features_path=TRAIN_FEATURES_PATH,
                model_output_path=tmp_model,
                contamination=contamination,
                n_estimators=N_ESTIMATORS,
                random_state=RANDOM_STATE,
            )

            # Score labelled val set
            metrics = compute_metrics(val_df, tmp_model)

        # Print one-line summary
        eligible_mark = "✓" if metrics["alert_rate"] <= ALERT_RATE_CAP else "✗ (cap)"
        print(
            f"  alert_rate={metrics['alert_rate']:.4f} {eligible_mark}  "
            f"precision={metrics['precision']:.4f}  "
            f"recall={metrics['recall']:.4f}  "
            f"f1={metrics['f1']:.4f}  "
            f"fpr_normal={metrics['fpr_normal']:.4f}  "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )
        cm = metrics["confusion_matrix"]
        print(
            f"  confusion: TP={cm['tp']}  FP={cm['fp']}  "
            f"TN={cm['tn']}  FN={cm['fn']}"
        )
        print()

        sweep_results.append({
            "contamination": contamination,
            "train_fpr": train_summary["train_fpr"],
            "metrics": metrics,
        })

    # ── Select best ───────────────────────────────────────────────────────────
    best = select_best(sweep_results)
    best_contamination = best["contamination"]
    best_metrics = best["metrics"]

    eligible_candidates = [
        r["contamination"] for r in sweep_results
        if r["metrics"]["alert_rate"] <= ALERT_RATE_CAP
    ]

    print("═" * 72)
    print("  Selection Result")
    print("═" * 72)

    if not eligible_candidates:
        print(f"  ⚠  No contamination value satisfied alert_rate ≤ {ALERT_RATE_CAP}")
        print(f"  ⚠  Falling back to max-recall selection: contamination={best_contamination}")
        print(f"  ⚠  Document this in thesis methodology — §3 cost contract not met")
        selection_method = "fallback_max_recall"
    else:
        print(f"  ✓  Best contamination: {best_contamination}")
        selection_method = "max_f1_within_alert_rate_cap"

    print(f"  alert_rate : {best_metrics['alert_rate']:.4f}")
    print(f"  precision  : {best_metrics['precision']:.4f}")
    print(f"  recall     : {best_metrics['recall']:.4f}")
    print(f"  f1         : {best_metrics['f1']:.4f}")
    print(f"  fpr_normal : {best_metrics['fpr_normal']:.4f}")
    print(f"  roc_auc    : {best_metrics['roc_auc']:.4f}")
    print(f"  pr_auc     : {best_metrics['pr_auc']:.4f}")
    cm = best_metrics["confusion_matrix"]
    print(f"  TP={cm['tp']}  FP={cm['fp']}  TN={cm['tn']}  FN={cm['fn']}")
    print()

    # ── Retrain best model to its final save path ─────────────────────────────
    if not dry_run:
        print(f"Saving best model → {TUNED_MODEL_PATH.name} ...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        final_train_summary = fit_isolation_forest(
            train_features_path=TRAIN_FEATURES_PATH,
            model_output_path=TUNED_MODEL_PATH,
            contamination=best_contamination,
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
        )
        print(f"  ✓  Saved to {TUNED_MODEL_PATH}")
        print(f"  (isolation_forest.pkl with contamination='auto' is unchanged)")
    else:
        final_train_summary = {"train_fpr": best["train_fpr"]}
        print(f"DRY-RUN: would save best model to {TUNED_MODEL_PATH}")
    print()

    # ── Build and write report ────────────────────────────────────────────────
    report = {
        "step": "E",
        "description": "Tier 1 Isolation Forest contamination sweep",
        "sweep": sweep_results,
        "selection": {
            "method": selection_method,
            "alert_rate_cap": ALERT_RATE_CAP,
            "best_contamination": best_contamination,
            "best_metrics": best_metrics,
            "eligible_contamination_values": eligible_candidates,
        },
        "model_paths": {
            "auto_baseline": str(MODELS_DIR / "isolation_forest.pkl"),
            "tuned": str(TUNED_MODEL_PATH),
        },
        "val_dataset": {
            "path": str(VAL_FEATURES_PATH),
            "n_windows": int(len(val_df)),
            "n_attack": int(n_attack),
            "n_normal": int(n_normal),
        },
    }

    if not dry_run:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        with open(METRICS_PATH, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"Metrics written → {METRICS_PATH.name}")
    else:
        print(f"DRY-RUN: would write metrics to {METRICS_PATH.name}")

    print()
    print("═" * 72)
    print("  Step E Complete — Day 2 Done")
    print("═" * 72)
    print()
    print("Artifacts produced:")
    print(f"  models/isolation_forest_tuned.pkl  (contamination={best_contamination})")
    print(f"  data/processed/tier1_metrics_val.json")
    print()
    print("Test set (levels 2+5 + synth_reserved) remains untouched.")
    print()
    print("Next: Day 3 — Drain3 parsing → DeepLog LSTM training")
    print()

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step E: Sweep Isolation Forest contamination values on the labelled "
            "mixed validation set and select the best model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/tier1_tune_contamination.py\n"
            "  python scripts/tier1_tune_contamination.py --contamination-values 0.01 0.02 0.05\n"
            "  python scripts/tier1_tune_contamination.py --dry-run\n"
        ),
    )
    p.add_argument(
        "--contamination-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_CONTAMINATION_VALUES),
        metavar="FLOAT",
        help=(
            f"Contamination values to sweep "
            f"(default: {list(DEFAULT_CONTAMINATION_VALUES)})"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sweep results without writing model or metrics files.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_step_e(
            contamination_values=tuple(args.contamination_values),
            dry_run=args.dry_run,
        )
        return 0
    except KeyboardInterrupt:
        print("\n[E] Interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n[E] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())