#!/usr/bin/env python3
"""
scripts/mix_val_and_regenerate_features.py
───────────────────────────────────────────────────────────────────────────────
Step D: Mix flaws.cloud validation events into synthetic validation set and
regenerate val_features.parquet with attack labels.

WHAT THIS DOES:
  1. Load synthetic_val.jsonl (24,144 events, _source='synth_normal')
  2. Load flaws_cloud_val.jsonl (5,828 events from levels 3+4, _source='flaws_cloud')
  3. Concatenate → synth_val_mixed.jsonl (~30K events total)
  4. Check for principal collisions between mixed val and test sets
  5. Aggregate into windows → val_features.parquet with is_attack_window labels
  6. Verify labelling: is_attack_window.sum() > 0, class distribution shown

WHY THIS STEP EXISTS:
  Before Step D, val_features.parquet contains only synthetic normal data
  (all is_attack_window=False). To tune Tier 1's contamination parameter
  (Step E), we need a labelled validation set with both attack and normal
  windows so we can measure precision, recall, and FPR.

EXPECTED OUTPUTS:
  - data/processed/synth_val_mixed.jsonl         (~30K lines)
  - data/processed/val_features.parquet          (~6,200 windows, labelled)
  - data/processed/val_mix_report.json           (counts, sanity checks)

PIPELINE POSITION:
  Sits between Step C (flaws.cloud split) and Step E (contamination tuning).
  Depends on: synthetic_val.jsonl, flaws_cloud_val.jsonl
  Consumed by: Step E (tier1_tune_contamination.py)

USAGE:
  # Standard run (overwrites synth_val_mixed.jsonl and val_features.parquet)
  python scripts/mix_val_and_regenerate_features.py

  # Dry-run mode (no writes, just report what would happen)
  python scripts/mix_val_and_regenerate_features.py --dry-run

CONSTRAINTS:
  - §9 identity-based split rule: no principal in both val and test of the
    *same* dataset (synth or flaws). Principal overlap between
    flaws_cloud_val and flaws_cloud_test is documented as expected
    (attack-scenario split, not identity split).
  - Read-only on all input files.
  - Idempotent: can be re-run safely; outputs are deterministic.
"""
from __future__ import annotations

import numpy as np
import argparse
import json
import sys
from pathlib import Path
from typing import Final

import pandas as pd

# ── Add project root to path so we can import src.* ──────────────────────────
_SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.ingest import load_events
from src.preprocessing.features import aggregate_features
from src.tier1_ml.baseline import load_baseline

# ── I/O Paths ─────────────────────────────────────────────────────────────────
DATA_PROCESSED: Final[Path] = _PROJECT_ROOT / "data" / "processeded1"

SYNTH_VAL_PATH: Final[Path] = DATA_PROCESSED / "synthetic_val.jsonl"
FLAWS_VAL_PATH: Final[Path] = DATA_PROCESSED / "flaws_cloud_val.jsonl"
SYNTH_RESERVED_PATH: Final[Path] = DATA_PROCESSED / "synthetic_reserved.jsonl"
FLAWS_TEST_PATH: Final[Path] = DATA_PROCESSED / "flaws_cloud_test.jsonl"

MIXED_VAL_PATH: Final[Path] = DATA_PROCESSED / "synth_val_mixed.jsonl"
VAL_FEATURES_PATH: Final[Path] = DATA_PROCESSED / "val_features.parquet"
REPORT_PATH: Final[Path] = DATA_PROCESSED / "val_mix_report.json"
BASELINE_PATH: Final[Path] = _PROJECT_ROOT / "models" / "tier1_baseline.json"


# ─────────────────────────────────────────────────────────────────────────────
# Principal extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_principal(event: dict) -> str:
    """Extract principal identifier from one event (matches ingest.py logic).

    Two schemas exist in this pipeline:
      - Synthetic events: principalId is a TOP-LEVEL field on the event dict.
        e.g. {"principalId": "AIDA2YHBKIG64KUM5OIH", ...}
      - flaws.cloud events: principalId is inside userIdentity.
        e.g. {"userIdentity": {"principalId": "811596193553", ...}, ...}

    Priority order (first non-empty value wins):
      1. event["principalId"]              — synthetic generator schema
      2. event["userIdentity"]["principalId"] — CloudTrail standard schema
      3. event["userIdentity"]["arn"]      — AssumedRole / federated fallback
      4. "__Root__<accountId>"             — Root type edge case
      5. "unknown"                         — no extractable identity

    Returns:
        Non-empty string identifying the principal. Never raises.
    """
    # ── 1. Top-level principalId (synthetic schema) ───────────────────────────
    top_pid = event.get("principalId", "")
    if isinstance(top_pid, str) and top_pid.strip():
        return top_pid.strip()

    # ── 2–4. userIdentity fields (CloudTrail / flaws.cloud schema) ────────────
    uid = event.get("userIdentity") or {}
    if not isinstance(uid, dict):
        return "unknown"

    uid_pid = uid.get("principalId", "")
    if isinstance(uid_pid, str) and uid_pid.strip():
        return uid_pid.strip()

    arn = uid.get("arn", "")
    if isinstance(arn, str) and arn.strip():
        return arn.strip()

    if uid.get("type") == "Root":
        return f"__Root__{uid.get('accountId', 'unknown')}"

    return "unknown"


def collect_principals_from_jsonl(path: Path) -> set[str]:
    """Stream a JSONL file and collect unique principals."""
    principals = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                principals.add(extract_principal(event))
            except json.JSONDecodeError:
                # Shouldn't happen if the file was written correctly, but be defensive
                print(
                    f"WARNING: malformed JSON at {path.name}:{line_num} — skipped",
                    file=sys.stderr,
                )
                continue
    return principals


# ─────────────────────────────────────────────────────────────────────────────
# Collision checking (§9 compliance)
# ─────────────────────────────────────────────────────────────────────────────

def check_principal_collisions(
    val_synth_principals: set[str],
    val_flaws_principals: set[str],
    test_synth_principals: set[str],
    test_flaws_principals: set[str],
) -> dict:
    """Check for §9 identity-split violations.
    
    §9 rule: No principal appears in both train/val and test partitions of the
    *same* dataset. We check:
      - synth_val vs synth_reserved (test for synth dataset)
      - flaws_cloud_val vs flaws_cloud_test (documented as expected overlap)
      - flaws_cloud_val vs synth_reserved (cross-dataset — should be clean)
      - flaws_cloud_test vs synth_val (cross-dataset — should be clean)
    
    Returns:
        Dict with collision counts and overlap principals for the report.
    """
    checks = {}
    
    # Within-dataset: synth_val vs synth_reserved
    overlap_synth = val_synth_principals & test_synth_principals
    checks["synth_val_vs_synth_reserved"] = {
        "val_count": len(val_synth_principals),
        "test_count": len(test_synth_principals),
        "overlap_count": len(overlap_synth),
        "overlap_principals": sorted(overlap_synth)[:20],  # cap at 20 for report size
        "status": "VIOLATION" if overlap_synth else "CLEAN",
    }
    
    # Within-dataset: flaws_cloud_val vs flaws_cloud_test (expected overlap, documented)
    overlap_flaws = val_flaws_principals & test_flaws_principals
    checks["flaws_val_vs_flaws_test"] = {
        "val_count": len(val_flaws_principals),
        "test_count": len(test_flaws_principals),
        "overlap_count": len(overlap_flaws),
        "overlap_principals": sorted(overlap_flaws)[:20],
        "status": "EXPECTED_OVERLAP",  # attack-scenario split, not identity split
        "note": "Overlap is expected — levels 3+4 vs 2+5 partition by attack scenario, not identity",
    }
    
    # Cross-dataset: flaws_cloud_val vs synth_reserved
    overlap_flaws_synth_test = val_flaws_principals & test_synth_principals
    checks["flaws_val_vs_synth_reserved"] = {
        "flaws_val_count": len(val_flaws_principals),
        "synth_test_count": len(test_synth_principals),
        "overlap_count": len(overlap_flaws_synth_test),
        "overlap_principals": sorted(overlap_flaws_synth_test)[:20],
        "status": "VIOLATION" if overlap_flaws_synth_test else "CLEAN",
    }
    
    # Cross-dataset: flaws_cloud_test vs synth_val
    overlap_flaws_test_synth_val = test_flaws_principals & val_synth_principals
    checks["flaws_test_vs_synth_val"] = {
        "flaws_test_count": len(test_flaws_principals),
        "synth_val_count": len(val_synth_principals),
        "overlap_count": len(overlap_flaws_test_synth_val),
        "overlap_principals": sorted(overlap_flaws_test_synth_val)[:20],
        "status": "VIOLATION" if overlap_flaws_test_synth_val else "CLEAN",
    }
    
    return checks

def json_serializable(obj):
    """Convert non-JSON-serializable objects to serializable types."""
    if isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return str(obj)

# ─────────────────────────────────────────────────────────────────────────────
# Main workflow
# ─────────────────────────────────────────────────────────────────────────────

def run_step_d(dry_run: bool = False) -> dict:
    """Execute Step D: mix val sets and regenerate labelled features.
    
    Args:
        dry_run: If True, report what would happen without writing files.
    
    Returns:
        Report dict with counts, collision checks, and sanity checks.
    """
    print("═" * 72)
    print("  Step D: Mix Validation Sets & Regenerate Labelled Features")
    print("═" * 72)
    print()
    
    # ── Input existence checks ────────────────────────────────────────────────
    missing = []
    for path in [SYNTH_VAL_PATH, FLAWS_VAL_PATH, SYNTH_RESERVED_PATH, FLAWS_TEST_PATH]:
        if not path.exists():
            missing.append(str(path))
    
    if missing:
        print("ERROR: Missing required input files:")
        for p in missing:
            print(f"  - {p}")
        print("\nRun Step C first (run_flaws_cloud_split.py) to create flaws.cloud splits.")
        sys.exit(1)
    
    # ── Step D-1: Load both validation sources ───────────────────────────────
    print("[D-1] Loading synthetic_val.jsonl with _source='synth_normal'...")
    df_synth_val = load_events(SYNTH_VAL_PATH, source="synth_normal")
    print(f"      → {len(df_synth_val):,} events loaded")
    
    print("[D-1] Loading flaws_cloud_val.jsonl with _source='flaws_cloud'...")
    df_flaws_val = load_events(FLAWS_VAL_PATH, source="flaws_cloud")
    print(f"      → {len(df_flaws_val):,} events loaded")
    print()
    
    # ── Step D-2: Concatenate into mixed val ─────────────────────────────────
    print("[D-2] Concatenating into mixed validation set...")
    df_mixed = pd.concat([df_synth_val, df_flaws_val], ignore_index=True)
    print(f"      → {len(df_mixed):,} total events in synth_val_mixed")
    print()
    
    # ── Step D-3: Check principal collisions (§9 compliance) ─────────────────
    print("[D-3] Checking principal collisions (§9 identity-split rule)...")
    val_synth_principals = collect_principals_from_jsonl(SYNTH_VAL_PATH)
    val_flaws_principals = collect_principals_from_jsonl(FLAWS_VAL_PATH)
    test_synth_principals = collect_principals_from_jsonl(SYNTH_RESERVED_PATH)
    test_flaws_principals = collect_principals_from_jsonl(FLAWS_TEST_PATH)
    
    collision_report = check_principal_collisions(
        val_synth_principals,
        val_flaws_principals,
        test_synth_principals,
        test_flaws_principals,
    )
    
    # Print collision summary
    for check_name, check_result in collision_report.items():
        status = check_result["status"]
        overlap_count = check_result["overlap_count"]
        status_symbol = "✓" if status in ("CLEAN", "EXPECTED_OVERLAP") else "✗"
        print(f"      {status_symbol} {check_name}: {overlap_count} overlap principals ({status})")
    
    # Check for actual violations (not expected overlaps)
    violations = [
        name for name, result in collision_report.items()
        if result["status"] == "VIOLATION"
    ]
    if violations:
        print("\n      ✗ VIOLATIONS DETECTED:")
        for v in violations:
            print(f"        - {v}")
        print("\n      This violates §9 identity-based split rule.")
        print("      Fix by renaming colliding principals or re-running the split.")
        sys.exit(1)
    
    print("      ✓ No §9 violations — safe to proceed")
    print()
    
    # ── Step D-4: Write mixed val JSONL ───────────────────────────────────────
    if not dry_run:
        print(f"[D-4] Writing mixed validation to {MIXED_VAL_PATH.name}...")
        MIXED_VAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MIXED_VAL_PATH, "w", encoding="utf-8") as fh:
            for _, row in df_mixed.iterrows():
                event = row.to_dict()
                # Remove pandas-added NaN/NaT artifacts for clean JSON
                event = {k: v for k, v in event.items() if pd.notna(v)}
                fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        print(f"      → {len(df_mixed):,} lines written")
    else:
        print(f"[D-4] DRY-RUN: would write {len(df_mixed):,} lines to {MIXED_VAL_PATH.name}")
    print()
    
    # ── Step D-4b: Load frozen HourBaseline (must not refit on val) ──────────
    print(f"[D-4b] Loading frozen hour baseline from {BASELINE_PATH.name}...")
    baseline = load_baseline(BASELINE_PATH)  # raises FileNotFoundError if missing
    print(f"       → Baseline loaded ({baseline.train_principal_count} train principals)")
    print()

    # ── Step D-5: Aggregate into labelled windows ────────────────────────────
    print("[D-5] Aggregating mixed val into windows with labels...")
    df_val_features = aggregate_features(df_mixed, baseline)
    print(f"      → {len(df_val_features):,} windows created")
    
    # Check is_attack_window column exists and has labels
    if "is_attack_window" not in df_val_features.columns:
        print("      ✗ ERROR: is_attack_window column missing from aggregated features")
        print("        Check aggregate_features() logic — it should derive labels from _source")
        sys.exit(1)
    
    attack_count = df_val_features["is_attack_window"].sum()
    normal_count = (~df_val_features["is_attack_window"]).sum()
    
    if attack_count == 0:
        print("      ✗ WARNING: is_attack_window has zero True values")
        print("        All flaws.cloud events may have been filtered by sparse-window threshold")
        print("        Check event_count distribution in val_features.parquet")
    else:
        print(f"      ✓ is_attack_window distribution:")
        print(f"        - Attack windows (True):  {attack_count:>6,}  ({100 * attack_count / len(df_val_features):5.2f}%)")
        print(f"        - Normal windows (False): {normal_count:>6,}  ({100 * normal_count / len(df_val_features):5.2f}%)")
    print()
    
    # ── Step D-6: Write labelled val_features.parquet ─────────────────────────
    if not dry_run:
        print(f"[D-6] Writing labelled features to {VAL_FEATURES_PATH.name}...")
        VAL_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_val_features.to_parquet(VAL_FEATURES_PATH, index=False, compression="snappy")
        print(f"      → {len(df_val_features):,} rows written")
    else:
        print(f"[D-6] DRY-RUN: would write {len(df_val_features):,} rows to {VAL_FEATURES_PATH.name}")
    print()
    
    # ── Step D-7: Build and write report ──────────────────────────────────────
    report = {
        "step": "D",
        "description": "Mix validation sets and regenerate labelled features",
        "inputs": {
            "synth_val_events": len(df_synth_val),
            "flaws_val_events": len(df_flaws_val),
        },
        "outputs": {
            "mixed_val_events": len(df_mixed),
            "val_features_windows": len(df_val_features),
            "attack_windows": int(attack_count),
            "normal_windows": int(normal_count),
        },
        "principal_collision_checks": collision_report,
        "label_distribution": {
            "attack_pct": float(100 * attack_count / len(df_val_features)) if len(df_val_features) > 0 else 0.0,
            "normal_pct": float(100 * normal_count / len(df_val_features)) if len(df_val_features) > 0 else 0.0,
        },
        "sanity_checks": {
            "is_attack_window_exists": "is_attack_window" in df_val_features.columns,
            "attack_count_nonzero": attack_count > 0,
            "label_source_values": df_mixed["_source"].value_counts().to_dict() if "_source" in df_mixed.columns else {},
        },
    }
    
    if not dry_run:
        print(f"[D-7] Writing report to {REPORT_PATH.name}...")
        with open(REPORT_PATH, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=json_serializable)
        print(f"      → Report written")
    else:
        print("[D-7] DRY-RUN: would write report")
    
    print()
    print("═" * 72)
    print("  Step D Complete")
    print("═" * 72)
    print()
    print("NEXT STEP: Run Step E (contamination tuning)")
    print("  python scripts/tier1_tune_contamination.py")
    print()
    
    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step D: Mix synthetic_val and flaws_cloud_val, regenerate "
            "val_features.parquet with attack labels for contamination tuning."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without writing any files.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    
    try:
        run_step_d(dry_run=args.dry_run)
        return 0
    except KeyboardInterrupt:
        print("\n[D] Interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n[D] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())