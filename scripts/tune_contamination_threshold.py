#!/usr/bin/env python3
"""
scripts/tier1_tune_contamination.py
───────────────────────────────────────────────────────────────────────────────
Step E: Calibrate the Isolation Forest alert threshold directly from the
val-set score distribution, satisfying the §3 cost-cascade alert-rate cap.

WHY THRESHOLD-BASED SELECTION INSTEAD OF CONTAMINATION SWEEPING:
  contamination in sklearn.IsolationForest calibrates the decision boundary
  against the *training* score distribution.  When val data comes from a
  different source (flaws.cloud vs synthetic-normal), the threshold learned on
  train transfers poorly — every contamination value we swept produced
  alert_rate > 0.06 on val even when train_fpr was only 1–2%.

  The correct approach is to set the threshold directly on the val anomaly
  score distribution: pick the percentile whose alert_rate satisfies §3, then
  freeze that threshold.  No model parameter is fit on val — only a scalar
  cut-point is selected from val's empirical score distribution.  This is
  standard post-hoc threshold calibration and does not constitute label leakage.

  The underlying model is trained with contamination=0.02 (the most defensible
  assumption for ~2% background anomalies in production).  The sklearn threshold
  baked into that model is ignored at inference time; val_threshold is used
  instead.

SELECTION CRITERION  (§3 cost-cascade contract):
  Highest recall  subject to  alert_rate ≤ ALERT_RATE_CAP (0.06)

  Rationale: with ROC AUC = 0.947 the ranking is fixed — the only degree of
  freedom is where to cut.  F1 is not the right objective here because at ~6.5%
  attack prevalence it conflates recall (what we care about for Tier 1) with
  precision (structurally low at this imbalance; improved by Tier 2 re-ranking).

WHAT THIS PRODUCES:
  models/isolation_forest_tuned.pkl
    — Same bundle as train.py but with two extra keys:
        "val_threshold"            float  anomaly score cut-point
        "val_threshold_percentile" float  e.g. 0.940
        "val_alert_rate_cap"       float  e.g. 0.06
    detect.py should use anomaly_score >= val_threshold to flag windows.

  data/processed/tier1_metrics_val.json
    — Full percentile table + selected operating point + methodology note.

  models/isolation_forest.pkl (contamination='auto') is NOT overwritten.

USAGE:
  python scripts/tier1_tune_contamination.py
  python scripts/tier1_tune_contamination.py --alert-rate-cap 0.05
  python scripts/tier1_tune_contamination.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd

# ── Project root ──────────────────────────────────────────────────────────────
_SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.tier1_ml.train import fit_isolation_forest
from src.preprocessing.features import FEATURE_COLUMNS

# ── I/O paths ─────────────────────────────────────────────────────────────────
DATA_PROCESSED: Final[Path] = _PROJECT_ROOT / "data" / "processeded1"
MODELS_DIR:     Final[Path] = _PROJECT_ROOT / "models1"

TRAIN_FEATURES_PATH: Final[Path] = DATA_PROCESSED / "train_features.parquet"
VAL_FEATURES_PATH:   Final[Path] = DATA_PROCESSED / "val_features.parquet"
AUTO_MODEL_PATH:     Final[Path] = MODELS_DIR / "isolation_forest.pkl"
TUNED_MODEL_PATH:    Final[Path] = MODELS_DIR / "isolation_forest_tuned_threshold.pkl"
METRICS_PATH:        Final[Path] = DATA_PROCESSED / "tier1_metrics_val.json"

# ── Hyperparameters ───────────────────────────────────────────────────────────
ALERT_RATE_CAP: Final[float] = 0.06   # §3 cost-cascade contract + 1% slack
CONTAMINATION:  Final[float] = 0.02   # training-time assumption; threshold overridden post-hoc
N_ESTIMATORS:   Final[int]   = 100
RANDOM_STATE:   Final[int]   = 42


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep_thresholds(
    anomaly_scores: np.ndarray,
    y_true: np.ndarray,
    alert_rate_cap: float,
) -> tuple[list[dict], dict | None]:
    """Evaluate every 0.1-percentile step from p90 to p99.9.

    Args:
        anomaly_scores: Per-window anomaly score (higher = more anomalous).
        y_true:         Binary labels (1=attack, 0=normal).
        alert_rate_cap: Max fraction of windows that may be flagged.

    Returns:
        (table, best) — full per-threshold table and the best eligible row.
        best is None if no threshold satisfies the cap.
    """
    n_attack = int(y_true.sum())
    n_normal = int((y_true == 0).sum())

    percentiles = np.round(np.arange(0.900, 1.000, 0.001), 4)
    table: list[dict] = []

    for pct in percentiles:
        thresh = float(np.percentile(anomaly_scores, pct * 100))
        y_pred = (anomaly_scores >= thresh).astype(int)

        alert_rate = float(y_pred.mean())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        recall    = tp / n_attack if n_attack > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        fpr_normal = fp / n_normal if n_normal > 0 else 0.0

        table.append({
            "percentile":  float(round(pct, 4)),
            "threshold":   float(round(thresh, 6)),
            "alert_rate":  float(round(alert_rate, 6)),
            "recall":      float(round(recall, 6)),
            "precision":   float(round(precision, 6)),
            "f1":          float(round(f1, 6)),
            "fpr_normal":  float(round(fpr_normal, 6)),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })

    eligible = [r for r in table if r["alert_rate"] <= alert_rate_cap]
    best = max(eligible, key=lambda r: r["recall"]) if eligible else None
    return table, best


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_step_e(alert_rate_cap: float = ALERT_RATE_CAP, dry_run: bool = False) -> dict:
    print("═" * 72)
    print("  Step E: Tier 1 Threshold Calibration")
    print("═" * 72)
    print()

    # ── Input checks ─────────────────────────────────────────────────────────
    for label, p in [("train_features", TRAIN_FEATURES_PATH),
                     ("val_features",   VAL_FEATURES_PATH)]:
        if not p.exists():
            print(f"ERROR: {label} not found at {p}")
            sys.exit(1)

    val_df = pd.read_parquet(VAL_FEATURES_PATH)
    if "is_attack_window" not in val_df.columns:
        print("ERROR: val_features.parquet missing is_attack_window — run Step D first.")
        sys.exit(1)

    n_attack = int(val_df["is_attack_window"].sum())
    n_normal = int((~val_df["is_attack_window"]).sum())
    n_total  = len(val_df)

    print(f"Val set  : {n_total:,} windows  ({n_attack} attack / {n_normal} normal)")
    print(f"Cap      : alert_rate ≤ {alert_rate_cap}  →  ≤ {int(alert_rate_cap * n_total)} windows flagged")
    print(f"Model    : contamination={CONTAMINATION} (threshold recalibrated post-hoc on val)")
    print()

    if n_attack == 0:
        print("ERROR: no attack windows in val — fix Step D first.")
        sys.exit(1)

    # ── E-1: Train with fixed contamination ──────────────────────────────────
    print(f"[E-1] Training Isolation Forest (contamination={CONTAMINATION}) ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model = Path(tmpdir) / "if_tmp.pkl"
        train_summary = fit_isolation_forest(
            train_features_path=TRAIN_FEATURES_PATH,
            model_output_path=tmp_model,
            contamination=CONTAMINATION,
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
        )
        bundle = joblib.load(tmp_model)
    print()

    # ── E-2: Score val set ────────────────────────────────────────────────────
    print("[E-2] Scoring val set ...")
    X = val_df[FEATURE_COLUMNS].values.astype("float32")
    anomaly_scores = -bundle["model"].decision_function(X)  # higher = more anomalous
    y_true = val_df["is_attack_window"].astype(int).values

    import sklearn.metrics as skm
    roc_auc = float(skm.roc_auc_score(y_true, anomaly_scores))
    pr_auc  = float(skm.average_precision_score(y_true, anomaly_scores))
    print(f"  ROC AUC = {roc_auc:.4f}   PR AUC = {pr_auc:.4f}  (threshold-independent)")
    print()

    # Score distribution summary
    a_scores = anomaly_scores[y_true == 1]
    n_scores = anomaly_scores[y_true == 0]
    print(f"  Attack  score: mean={a_scores.mean():.4f}  "
          f"p10={np.percentile(a_scores,10):.4f}  "
          f"p50={np.percentile(a_scores,50):.4f}  "
          f"p90={np.percentile(a_scores,90):.4f}")
    print(f"  Normal  score: mean={n_scores.mean():.4f}  "
          f"p10={np.percentile(n_scores,10):.4f}  "
          f"p50={np.percentile(n_scores,50):.4f}  "
          f"p90={np.percentile(n_scores,90):.4f}")
    print()

    # ── E-3: Threshold sweep ──────────────────────────────────────────────────
    print("[E-3] Sweeping thresholds (p90.0 → p99.9) ...")
    table, best = sweep_thresholds(anomaly_scores, y_true, alert_rate_cap)

    # Print rows straddling the alert-rate cap boundary
    cap_idx = next(
        (i for i, r in enumerate(table) if r["alert_rate"] <= alert_rate_cap), None
    )
    show_start = max(0, (cap_idx or 0) - 3)
    show_end   = min(len(table), (cap_idx or 0) + 7)

    print(f"\n  {'pct':>6}  {'thresh':>8}  {'alert_rt':>9}  "
          f"{'recall':>7}  {'prec':>7}  {'f1':>7}  "
          f"{'fpr_n':>7}  {'TP':>4}  {'FP':>4}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*9}  "
          f"{'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*4}  {'─'*4}")

    for r in table[show_start:show_end]:
        cap_ok = "✓" if r["alert_rate"] <= alert_rate_cap else "✗"
        marker = " ◄ SELECTED" if (best and r["percentile"] == best["percentile"]) else ""
        print(
            f"  {r['percentile']:>6.3f}  {r['threshold']:>8.4f}  "
            f"{r['alert_rate']:>8.4f}{cap_ok}  "
            f"{r['recall']:>7.4f}  {r['precision']:>7.4f}  {r['f1']:>7.4f}  "
            f"{r['fpr_normal']:>7.4f}  {r['tp']:>4}  {r['fp']:>4}"
            f"{marker}"
        )
    print()

    # ── Selection result ──────────────────────────────────────────────────────
    print("═" * 72)
    print("  Selection Result")
    print("═" * 72)

    if best is None:
        print(f"  ⚠  No threshold satisfied alert_rate ≤ {alert_rate_cap}")
        print(f"  ⚠  Try --alert-rate-cap 0.08 or investigate val composition")
        selection_method = "none_eligible"
    else:
        print(f"  ✓  Threshold           : {best['threshold']:.6f}  "
              f"(p{best['percentile']*100:.1f})")
        print(f"  alert_rate            : {best['alert_rate']:.4f}  (cap={alert_rate_cap} ✓)")
        print(f"  recall                : {best['recall']:.4f}  "
              f"({best['tp']}/{n_attack} attack windows caught)")
        print(f"  precision             : {best['precision']:.4f}")
        print(f"  f1                    : {best['f1']:.4f}")
        print(f"  fpr_normal            : {best['fpr_normal']:.4f}  "
              f"({best['fp']}/{n_normal} normal falsely flagged)")
        print(f"  TP={best['tp']}  FP={best['fp']}  "
              f"TN={best['tn']}  FN={best['fn']}")
        print(f"  ROC AUC               : {roc_auc:.4f}")
        print(f"  PR AUC                : {pr_auc:.4f}")
        selection_method = "max_recall_within_alert_rate_cap"
    print()

    # ── E-4: Save tuned bundle ────────────────────────────────────────────────
    if not dry_run and best is not None:
        print(f"[E-4] Saving tuned model → {TUNED_MODEL_PATH.name} ...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        fit_isolation_forest(
            train_features_path=TRAIN_FEATURES_PATH,
            model_output_path=TUNED_MODEL_PATH,
            contamination=CONTAMINATION,
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
        )
        # Inject the val-calibrated threshold into the bundle
        tuned_bundle = joblib.load(TUNED_MODEL_PATH)
        tuned_bundle["val_threshold"]            = best["threshold"]
        tuned_bundle["val_threshold_percentile"] = best["percentile"]
        tuned_bundle["val_alert_rate_cap"]       = alert_rate_cap
        joblib.dump(tuned_bundle, TUNED_MODEL_PATH)
        print(f"  ✓  val_threshold = {best['threshold']:.6f} injected into bundle")
        print(f"  (isolation_forest.pkl with contamination='auto' is unchanged)")
    elif dry_run:
        print(f"[E-4] DRY-RUN: would save tuned model to {TUNED_MODEL_PATH.name}")
    print()

    # ── E-5: Write metrics ────────────────────────────────────────────────────
    report = {
        "step": "E",
        "description": "Tier 1 threshold calibration via val-set percentile sweep",
        "methodology_note": (
            "contamination calibrates sklearn's internal threshold against the training "
            "distribution. Because val data (flaws.cloud) has different density from "
            "synthetic-normal train data, the threshold is recalibrated post-hoc by "
            "selecting the val-score percentile whose alert_rate satisfies the §3 "
            "cost-cascade cap. No model parameter is fit on val — only a scalar "
            "cut-point is selected from val's empirical score distribution."
        ),
        "model": {
            "contamination_train": CONTAMINATION,
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "train_fpr": train_summary["train_fpr"],
            "train_n_rows": train_summary["train_n_rows"],
            "feature_set_hash": train_summary["feature_set_hash"],
        },
        "val_dataset": {
            "n_windows": n_total,
            "n_attack": n_attack,
            "n_normal": n_normal,
            "attack_pct": round(100 * n_attack / n_total, 4),
        },
        "threshold_independent": {
            "roc_auc": roc_auc,
            "pr_auc":  pr_auc,
        },
        "selection": {
            "method": selection_method,
            "alert_rate_cap": alert_rate_cap,
            "best": best,
        },
        "threshold_table": table,
        "model_paths": {
            "auto_baseline": str(AUTO_MODEL_PATH),
            "tuned":         str(TUNED_MODEL_PATH),
        },
    }

    if not dry_run:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        with open(METRICS_PATH, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"[E-5] Metrics written → {METRICS_PATH.name}")
    else:
        print(f"[E-5] DRY-RUN: would write metrics to {METRICS_PATH.name}")

    print()
    print("═" * 72)
    print("  Step E Complete — Day 2 Done")
    print("═" * 72)
    print()
    if best:
        print(f"Deliverables:")
        print(f"  models/isolation_forest_tuned.pkl  "
              f"(contamination={CONTAMINATION}, val_threshold={best['threshold']:.6f})")
        print(f"  data/processed/tier1_metrics_val.json")
    print()
    print("Test set (levels 2+5 + synth_reserved) remains untouched.")
    print("Next: Day 3 — Drain3 parsing → DeepLog LSTM training")
    print()

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Step E: calibrate Tier 1 alert threshold on val set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--alert-rate-cap", type=float, default=ALERT_RATE_CAP,
        help=f"Max fraction of windows to flag (default: {ALERT_RATE_CAP})",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print results without writing files.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        run_step_e(alert_rate_cap=args.alert_rate_cap, dry_run=args.dry_run)
        return 0
    except KeyboardInterrupt:
        print("\n[E] Interrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n[E] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())