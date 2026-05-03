#!/usr/bin/env python3
"""
scripts/check_val_labels.py
───────────────────────────────────────────────────────────────────────────────
Pre-Step-E sanity check: verify that attack-window labels in val_features.parquet
are correct before contamination tuning.

WHAT THIS CHECKS:
  1. Schema       — required columns present, correct dtypes
  2. Class counts — attack/normal distribution (must have attack_count > 0)
  3. Label source — all attack windows must have label_source='flaws_cloud'
                    all normal windows must have label_source='synth_normal'
  4. event_count  — no windows with event_count < 3 (sparse filter compliance)
  5. Principal spot-check — attack window principals look like flaws.cloud
                    identities (Root 811596193553, AROA-prefix, IAM users)
                    and NOT synthetic generator IDs (AIDA-prefix AssumedRole
                    sessions like 'AROA*:*-session')
  6. Feature sanity — attack windows have plausible feature values
                    (not all-zero, not identical to normal mean)
  7. Sample rows  — print 10 attack + 5 normal rows for human eyeball check

PASS/FAIL:
  Exits 0 if all hard checks pass (warnings don't block).
  Exits 1 if any hard check fails — do NOT proceed to Step E until fixed.

USAGE:
  python scripts/check_val_labels.py
  python scripts/check_val_labels.py --parquet path/to/val_features.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _SCRIPT_DIR.parent
DEFAULT_VAL_PATH: Final[Path] = _PROJECT_ROOT / "data" / "processed" / "val_features.parquet"

# ── Expected values ───────────────────────────────────────────────────────────
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "principal_id",
    "window_start",
    "event_count",
    "is_attack_window",
    "label_source",
)

# Columns that should exist as features (non-exhaustive — just the 7 known ones)
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "event_count",
    "unique_api_count",
    "error_rate",
    "sensitive_action_count",
    "hour_zscore",
)

# flaws.cloud principal patterns
FLAWS_PRINCIPAL_PATTERNS: Final[tuple[str, ...]] = (
    "811596193553",   # Root account ID
    "AROA",           # AssumedRole (scanner sessions: cloudsploit, AWSConfig, etc.)
    "AIDA",           # IAMUser (Level6 IAMUser, etc.)
    "unknown",        # events with no extractable principal (small %, acceptable)
)

# Synthetic principal patterns (these should NOT appear in attack windows)
SYNTH_PRINCIPAL_SUFFIXES: Final[tuple[str, ...]] = (
    "-session",       # synthetic AssumedRole sessions
    "-worker-session",
    "-handler-session",
    "-resizer-session",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pass(msg: str) -> None:
    print(f"  ✓  {msg}")

def _warn(msg: str) -> None:
    print(f"  ⚠  {msg}")

def _fail(msg: str) -> None:
    print(f"  ✗  {msg}")

def _header(title: str) -> None:
    print()
    print(f"── {title} {'─' * max(0, 65 - len(title))}")


def _is_synth_principal(pid: str) -> bool:
    """Return True if the principal looks like a synthetic generator identity."""
    for suffix in SYNTH_PRINCIPAL_SUFFIXES:
        if pid.endswith(suffix):
            return True
    return False


def _is_flaws_principal(pid: str) -> bool:
    """Return True if the principal looks like a real flaws.cloud identity."""
    for prefix in FLAWS_PRINCIPAL_PATTERNS:
        if pid.startswith(prefix):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Check functions — each returns (passed: bool, detail: str)
# ─────────────────────────────────────────────────────────────────────────────

def check_schema(df: pd.DataFrame) -> bool:
    _header("Check 1: Schema")
    ok = True
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            _fail(f"Missing required column: '{col}'")
            ok = False
        else:
            _pass(f"Column present: '{col}' (dtype={df[col].dtype})")

    # is_attack_window must be bool
    if "is_attack_window" in df.columns:
        if df["is_attack_window"].dtype != bool:
            _warn(f"is_attack_window dtype is {df['is_attack_window'].dtype}, expected bool — will cast")
            df["is_attack_window"] = df["is_attack_window"].astype(bool)

    # feature columns
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        _warn(f"Feature columns not found (may have different names): {missing_features}")
    else:
        _pass(f"All expected feature columns present: {list(FEATURE_COLUMNS)}")

    return ok


def check_class_counts(df: pd.DataFrame) -> bool:
    _header("Check 2: Class Counts")
    attack = df["is_attack_window"].sum()
    normal = (~df["is_attack_window"]).sum()
    total = len(df)
    attack_pct = 100 * attack / total if total > 0 else 0.0

    print(f"       Total windows : {total:>7,}")
    print(f"       Attack (True)  : {attack:>7,}  ({attack_pct:.2f}%)")
    print(f"       Normal (False) : {normal:>7,}  ({100 - attack_pct:.2f}%)")

    if attack == 0:
        _fail("HARD: attack_count == 0 — no labelled attack windows. Step E cannot proceed.")
        return False

    if attack < 50:
        _warn(f"Only {attack} attack windows — contamination sweep may produce noisy metrics")
    else:
        _pass(f"{attack} attack windows available for contamination tuning")

    # Expected range from Day 2 design: ~300–700 attack windows
    if not (50 <= attack <= 1000):
        _warn(f"Attack count {attack} outside expected range [50, 1000] — verify filter settings")

    return True


def check_label_source(df: pd.DataFrame) -> bool:
    _header("Check 3: Label Source Consistency")
    if "label_source" not in df.columns:
        _warn("label_source column missing — skipping label source check")
        return True  # soft check, not hard failure

    attack_df = df[df["is_attack_window"]]
    normal_df = df[~df["is_attack_window"]]
    ok = True

    # Attack windows → must all be flaws_cloud
    attack_sources = attack_df["label_source"].value_counts()
    print(f"       Attack window label_source distribution:")
    for src, count in attack_sources.items():
        print(f"         {src}: {count}")

    non_flaws_attacks = attack_df[attack_df["label_source"] != "flaws_cloud"]
    if len(non_flaws_attacks) > 0:
        _fail(
            f"HARD: {len(non_flaws_attacks)} attack windows have label_source != 'flaws_cloud'. "
            f"Sources: {non_flaws_attacks['label_source'].unique().tolist()}"
        )
        ok = False
    else:
        _pass("All attack windows have label_source='flaws_cloud'")

    # Normal windows → must all be synth_normal
    normal_sources = normal_df["label_source"].value_counts()
    print(f"       Normal window label_source distribution:")
    for src, count in normal_sources.items():
        print(f"         {src}: {count}")

    non_synth_normal = normal_df[normal_df["label_source"] != "synth_normal"]
    if len(non_synth_normal) > 0:
        _fail(
            f"HARD: {len(non_synth_normal)} normal windows have label_source != 'synth_normal'. "
            f"Sources: {non_synth_normal['label_source'].unique().tolist()}"
        )
        ok = False
    else:
        _pass("All normal windows have label_source='synth_normal'")

    return ok


def check_sparse_filter(df: pd.DataFrame) -> bool:
    _header("Check 4: Sparse-Window Filter Compliance (event_count >= 3)")
    if "event_count" not in df.columns:
        _warn("event_count column missing — skipping sparse filter check")
        return True

    sparse = df[df["event_count"] < 3]
    if len(sparse) > 0:
        _fail(
            f"HARD: {len(sparse)} windows have event_count < 3. "
            f"Sparse filter was not applied — feature vectors will be noise-dominated."
        )
        # show breakdown
        print(f"       event_count distribution in sparse windows:")
        print(f"         {sparse['event_count'].value_counts().to_dict()}")
        print(f"       Attack/normal split in sparse windows:")
        print(f"         {sparse['is_attack_window'].value_counts().to_dict()}")
        return False

    min_ec = df["event_count"].min()
    max_ec = df["event_count"].max()
    median_ec = df["event_count"].median()
    _pass(f"All windows have event_count >= 3 (min={min_ec}, median={median_ec:.0f}, max={max_ec})")
    return True


def check_principal_identity(df: pd.DataFrame) -> bool:
    _header("Check 5: Principal Identity Spot-Check")
    if "principal_id" not in df.columns:
        _warn("principal_id column missing — skipping principal check")
        return True

    attack_df = df[df["is_attack_window"]]
    attack_principals = attack_df["principal_id"].unique()

    print(f"       Unique attack-window principals: {len(attack_principals)}")
    for pid in sorted(attack_principals)[:20]:  # cap at 20 for readability
        looks_flaws = _is_flaws_principal(pid)
        looks_synth = _is_synth_principal(pid)
        tag = "✓ flaws" if looks_flaws else ("✗ SYNTH?" if looks_synth else "? unrecognised")
        print(f"         {tag:12}  {pid}")

    if len(attack_principals) > 20:
        print(f"         ... and {len(attack_principals) - 20} more (not shown)")

    # Hard check: no synthetic principals in attack windows
    synth_attack_pids = [p for p in attack_principals if _is_synth_principal(p)]
    if synth_attack_pids:
        _fail(
            f"HARD: {len(synth_attack_pids)} synthetic-looking principals in attack windows: "
            f"{synth_attack_pids[:5]}"
        )
        return False

    _pass("No synthetic-generator principals found in attack windows")
    return True


def check_feature_sanity(df: pd.DataFrame) -> bool:
    _header("Check 6: Feature Value Sanity")
    ok = True

    attack_df = df[df["is_attack_window"]]
    normal_df = df[~df["is_attack_window"]]

    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not available:
        _warn("No feature columns found — skipping feature sanity check")
        return True

    print(f"       {'Feature':<30} {'Attack mean':>12} {'Normal mean':>12} {'Attack std':>12}")
    print(f"       {'-'*30} {'-'*12} {'-'*12} {'-'*12}")

    for col in available:
        a_mean = attack_df[col].mean()
        n_mean = normal_df[col].mean()
        a_std = attack_df[col].std()
        print(f"       {col:<30} {a_mean:>12.4f} {n_mean:>12.4f} {a_std:>12.4f}")

    # Soft sanity: attack event_count should be >= 3 (already hard-checked above)
    # Hard check: attack feature vectors not identically zero
    if "event_count" in df.columns:
        zero_ec_attack = (attack_df["event_count"] == 0).sum()
        if zero_ec_attack > 0:
            _fail(f"HARD: {zero_ec_attack} attack windows have event_count=0")
            ok = False

    # Warn if hour_zscore is all NaN for attack windows (means principal not in baseline)
    if "hour_zscore" in df.columns:
        nan_zscore_attack = attack_df["hour_zscore"].isna().sum()
        nan_pct = 100 * nan_zscore_attack / len(attack_df) if len(attack_df) > 0 else 0
        if nan_pct > 50:
            _warn(
                f"{nan_zscore_attack} / {len(attack_df)} attack windows ({nan_pct:.1f}%) "
                f"have NaN hour_zscore — flaws.cloud principals are absent from train baseline "
                f"(expected; global fallback should be used instead)"
            )
        else:
            _pass(f"hour_zscore: {nan_zscore_attack} NaN in attack windows ({nan_pct:.1f}%)")

    if ok:
        _pass("Feature values look plausible")
    return ok


def print_sample_rows(df: pd.DataFrame) -> None:
    _header("Check 7: Sample Rows (human eyeball)")
    attack_df = df[df["is_attack_window"]]
    normal_df = df[~df["is_attack_window"]]

    display_cols = [c for c in (
        "principal_id", "window_start", "event_count",
        "unique_api_count", "error_rate", "sensitive_action_count",
        "hour_zscore", "is_attack_window", "label_source",
    ) if c in df.columns]

    print(f"\n  10 attack windows (is_attack_window=True):")
    sample_attack = attack_df[display_cols].head(10)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 160,
        "display.max_colwidth", 35,
        "display.float_format", "{:.4f}".format,
    ):
        for line in sample_attack.to_string(index=False).splitlines():
            print(f"    {line}")

    print(f"\n  5 normal windows (is_attack_window=False):")
    sample_normal = normal_df[display_cols].head(5)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 160,
        "display.max_colwidth", 35,
        "display.float_format", "{:.4f}".format,
    ):
        for line in sample_normal.to_string(index=False).splitlines():
            print(f"    {line}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_checks(val_path: Path) -> int:
    print("═" * 72)
    print("  Pre-Step-E: val_features.parquet Label Sanity Check")
    print("═" * 72)
    print(f"  File: {val_path}")

    if not val_path.exists():
        print(f"\n✗ File not found: {val_path}")
        print("  Run Step D first: python scripts/mix_val_and_regenerate_features.py")
        return 1

    df = pd.read_parquet(val_path)
    print(f"  Rows: {len(df):,}  Columns: {df.columns.tolist()}")

    results: list[bool] = []
    results.append(check_schema(df))
    results.append(check_class_counts(df))
    results.append(check_label_source(df))
    results.append(check_sparse_filter(df))
    results.append(check_principal_identity(df))
    results.append(check_feature_sanity(df))
    print_sample_rows(df)

    hard_failures = results.count(False)

    print()
    print("═" * 72)
    if hard_failures == 0:
        print("  ✓ ALL CHECKS PASSED — safe to proceed to Step E")
        print("    python scripts/tier1_tune_contamination.py")
    else:
        print(f"  ✗ {hard_failures} HARD CHECK(S) FAILED — fix before running Step E")
        print("    Re-run Step D after fixing the root cause.")
    print("═" * 72)

    return 0 if hard_failures == 0 else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sanity-check val_features.parquet labels before Step E contamination tuning.",
    )
    p.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_VAL_PATH,
        help=f"Path to val_features.parquet (default: {DEFAULT_VAL_PATH})",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_checks(args.parquet)


if __name__ == "__main__":
    raise SystemExit(main())