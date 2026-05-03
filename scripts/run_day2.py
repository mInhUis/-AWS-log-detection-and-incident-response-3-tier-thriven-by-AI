"""
scripts/run_day2.py
---------------------------------──────────────────────────────────
End-to-end Day 2 orchestrator: load synthetic events, split by principal,
aggregate features, fit Isolation Forest, score validation set.

Usage:
  python scripts/run_day2.py --mode mock --force   # ~10 sec; sanity check
  python scripts/run_day2.py --mode full --force   # ~5-10 min; all 500K events

Modes:
  mock  (default) — caps at 5 000 events; fast smoke test before committing
                    to a full run.  DEFAULT because CLAUDE.md §5.6 requires
                    mock as the default to protect the laptop from accidental
                    heavy runs.
  full             — loads all events from synthetic_normal.jsonl.

Flags:
  --force          — overwrite existing outputs; fail loudly otherwise.

Step sequence (matches spec §run_day2.py):
  1. Load events  (ingest.load_events)
  2. Split by principal 90/5/5  (ingest.split_by_principal)
  3. Write JSONL splits + manifest  (ingest.write_split_artifacts)
  4. Fit hour baseline on train  (features.fit_hour_baseline)
     └─ Save  (baseline.save_baseline)
  5. Aggregate train features  (features.aggregate_features)
     └─ Write  (features.write_features)
  6. Aggregate val features using SAME frozen baseline  (features.aggregate_features)
     └─ Write  (features.write_features)
  7. Fit Isolation Forest  (train.fit_isolation_forest)
  8. Score val features  (detect.score_features)
  9. Print final summary table
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Final

# ── Add project root to sys.path so `src.*` imports work when running as a
#    script (not installed as a package).
_SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.features import (
    FEATURE_COLUMNS,
    aggregate_features,
    fit_hour_baseline,
    write_features,
)
from src.preprocessing.ingest import load_events, split_by_principal, write_split_artifacts
from src.tier1_ml.baseline import load_baseline, save_baseline
from src.tier1_ml.detect import score_features
from src.tier1_ml.train import fit_isolation_forest

# ── Path constants (all env-var-overridable)---------------------───────────────────────
SYNTHETIC_NORMAL_PATH: Final[Path] = Path(
    os.environ.get(
        "SYNTHETIC_NORMAL_PATH",
        str(_PROJECT_ROOT / "data" / "raw" / "synthetic_normal1.jsonl"),
    )
)
PROCESSED_DIR: Final[Path] = Path(
    os.environ.get("PROCESSED_DIR", str(_PROJECT_ROOT / "data" / "processeded1"))
)
MODELS_DIR: Final[Path] = Path(
    os.environ.get("MODELS_DIR", str(_PROJECT_ROOT / "models1"))
)

# ── Derived artifact paths ---------------------------------─────────
BASELINE_PATH: Final[Path] = MODELS_DIR / "tier1_baseline.json"
TRAIN_FEATURES_PATH: Final[Path] = PROCESSED_DIR / "train_features.parquet"
VAL_FEATURES_PATH: Final[Path] = PROCESSED_DIR / "val_features.parquet"
MODEL_PATH: Final[Path] = MODELS_DIR / "isolation_forest.pkl"
ALERTS_PATH: Final[Path] = PROCESSED_DIR / "tier1_alerts_val.jsonl"

# All output files in order — used by the stale-artifact guard
_OUTPUT_FILES: Final[list[Path]] = [
    PROCESSED_DIR / "synthetic_train.jsonl",
    PROCESSED_DIR / "synthetic_val.jsonl",
    PROCESSED_DIR / "synthetic_reserved.jsonl",
    PROCESSED_DIR / "split_manifest.json",
    BASELINE_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    MODEL_PATH,
    ALERTS_PATH,
]

# ── Mock mode cap ---------------------------------──────────────────
_MOCK_EVENT_CAP: Final[int] = 5_000


def _check_stale_artifacts(force: bool) -> None:
    """Refuse to overwrite existing outputs unless --force is given."""
    existing = [p for p in _OUTPUT_FILES if p.exists()]
    if existing and not force:
        files_str = "\n  ".join(str(p) for p in existing[:5])
        raise RuntimeError(
            f"Output files already exist (showing up to 5):\n  {files_str}\n"
            "Use --force to overwrite them."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 2 — Tier 1 pipeline: ingest, split, feature aggregation, "
        "Isolation Forest training, and validation scoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "full"],
        default="mock",  # default=mock per CLAUDE.md §5.6
        help="mock: cap at 5 000 events (fast); full: all events (~500K).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()

    _check_stale_artifacts(args.force)

    max_events: int | None = _MOCK_EVENT_CAP if args.mode == "mock" else None
    mode_label = f"mock (cap={_MOCK_EVENT_CAP:,})" if args.mode == "mock" else "full"
    print(f"\n{'='*60}")
    print(f" Day 2 Pipeline — mode={mode_label}")
    print(f"{'='*60}\n")

    # ---- Step 1: Load events -------------------------------------------------
    print("---- Step 1: Load events ----")
    events = load_events(SYNTHETIC_NORMAL_PATH, max_events=max_events)
    print()

    # ---- Step 2: Split by principal (90/5/5)---------------------------------------------------------------------
    print("---- Step 2: Split by principal (90/5/5) ----")
    train_events, val_events, reserved_events, manifest = split_by_principal(
        events,
        train_frac=0.90,
        val_frac=0.05,
        reserved_frac=0.05,
        seed=42,
    )
    print()

    # ---- Step 3: Write split artifacts---------------------------------------------------------------------------------
    print("---- Step 3: Write split artifacts ----")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    write_split_artifacts(train_events, val_events, reserved_events, manifest, PROCESSED_DIR)
    print()

    # ---- Step 4: Fit hour baseline (train only — L7)-----------------------------------------------------
    print("---- Step 4: Fit hour baseline (train data only) ----")
    baseline = fit_hour_baseline(train_events)
    save_baseline(baseline, BASELINE_PATH)
    print()

    # ---- Step 5: Aggregate TRAIN features-------------------------------------------------------------------------
    print("---- Step 5: Aggregate train features ----")
    train_features = aggregate_features(train_events, baseline)
    write_features(train_features, TRAIN_FEATURES_PATH)
    print()

    # ---- Step 6: Aggregate VAL features (same frozen baseline — L7)---------------------
    print("---- Step 6: Aggregate val features (frozen baseline) ----")
    val_features = aggregate_features(val_events, baseline)
    write_features(val_features, VAL_FEATURES_PATH)
    print()

    # ---- Step 7: Fit Isolation Forest---------------------------------------------------------------------------------─
    print("---- Step 7: Fit Isolation Forest ----")
    train_summary = fit_isolation_forest(
        train_features_path=TRAIN_FEATURES_PATH,
        model_output_path=MODEL_PATH,
    )
    print()

    # ---- Step 8: Score val features ---------------------------------
    print("---- Step 8: Score val features ----")
    val_summary = score_features(
        model_path=MODEL_PATH,
        features_path=VAL_FEATURES_PATH,
        alerts_output_path=ALERTS_PATH,
    )
    print()

    # ---- Step 9: Final summary table ---------------------------------
    print(f"\n{'='*60}")
    print(" Final Summary")
    print(f"{'='*60}")
    print(
        f"  train: {len(train_events):>8,} events"
        f" -> {len(train_features):>6,} windows;"
        f"  train_fpr  = {train_summary['train_fpr']:.4f}"
        f" ({train_summary['train_fpr']*100:.2f}%)"
    )
    print(
        f"  val:   {len(val_events):>8,} events"
        f" -> {val_summary['n_rows']:>6,} windows;"
        f"  alert_rate = {val_summary['alert_rate']:.4f}"
        f" ({val_summary['alert_rate']*100:.2f}%)"
    )
    print()
    print(
        "  Sanity check: train_fpr and val alert_rate should be"
        " roughly comparable (within ~5%)."
    )
    print("  A large gap suggests a bug in feature aggregation or splitting.")
    print(f"\n  Mode: {mode_label}")
    print(f"  sklearn: {train_summary['sklearn_version']}")
    print(f"  Feature set hash: {train_summary['feature_set_hash'][:16]}...")
    print()
    print("Run pytest next:")
    print("  pytest tests/test_day2_invariants.py -v")
    print()


if __name__ == "__main__":
    main()
