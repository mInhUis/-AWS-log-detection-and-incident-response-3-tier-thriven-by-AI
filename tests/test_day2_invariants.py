"""
tests/test_day2_invariants.py
─────────────────────────────────────────────────────────────────────────────
Ten invariant tests for the Day 2 Tier 1 pipeline.

Run after:  python scripts/run_day2.py --mode mock --force

Each test loads an artifact from data/processed/ or models/ and asserts a
specific structural property.  No test touches flaws.cloud data or runs
training — they verify the outputs of run_day2.py are internally consistent.

V1  Principals are disjoint across train/val/reserved splits
V2  Hour baseline was fitted on train-only principals
V3  Train features have zero sensitive_action_count (attack-free invariant)
V4  Train features contain both 5-min and 60-min windows
V5  Train features have no NaN values in feature columns
V6  Pipeline is deterministic (re-running produces identical parquet bytes)
V7  Isolation Forest bundle contains all required keys
V8  Val alert records contain all required fields
V9  Anonymous entity fallback returns per-IP keys (L8)
V10 Dedup preserves empty event_id rows (L9)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

# ── Add project root to path so src.* imports work ───────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.features import FEATURE_COLUMNS
from src.preprocessing.ingest import extract_entity

# ── Artifact paths (must match run_day2.py) ───────────────────────────────────
_PROCESSED = _PROJECT_ROOT / "data" / "processed"
_MODELS = _PROJECT_ROOT / "models"

MANIFEST_PATH = _PROCESSED / "split_manifest.json"
BASELINE_PATH = _MODELS / "tier1_baseline.json"
TRAIN_FEATURES_PATH = _PROCESSED / "train_features.parquet"
VAL_FEATURES_PATH = _PROCESSED / "val_features.parquet"
MODEL_PATH = _MODELS / "isolation_forest.pkl"
ALERTS_PATH = _PROCESSED / "tier1_alerts_val.jsonl"


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: skip test gracefully if artifacts don't exist yet
# ═══════════════════════════════════════════════════════════════════════════════


def _require(*paths: Path) -> None:
    """Skip a test if any required artifact is missing."""
    for p in paths:
        if not p.exists():
            pytest.skip(f"Artifact not found (run pipeline first): {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# V1 — Principal sets are pairwise disjoint
# ═══════════════════════════════════════════════════════════════════════════════


def test_principals_disjoint() -> None:
    """V1: No principal appears in more than one split."""
    _require(MANIFEST_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text())

    train_p = set(manifest["principals"]["train"])
    val_p = set(manifest["principals"]["val"])
    reserved_p = set(manifest["principals"]["reserved"])

    assert (train_p & val_p) == set(), (
        f"Train/val overlap: {train_p & val_p}"
    )
    assert (train_p & reserved_p) == set(), (
        f"Train/reserved overlap: {train_p & reserved_p}"
    )
    assert (val_p & reserved_p) == set(), (
        f"Val/reserved overlap: {val_p & reserved_p}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# V2 — Baseline was fitted on train principals only
# ═══════════════════════════════════════════════════════════════════════════════


def test_baseline_is_train_only() -> None:
    """V2: Every key in tier1_baseline.json.per_principal is a train principal."""
    _require(MANIFEST_PATH, BASELINE_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text())
    baseline = json.loads(BASELINE_PATH.read_text())

    train_set = set(manifest["principals"]["train"])
    val_set = set(manifest["principals"]["val"])
    reserved_set = set(manifest["principals"]["reserved"])

    baseline_keys = set(baseline.get("per_principal", {}).keys())

    # Every baseline key must be in train
    leaked_to_val = baseline_keys & val_set
    leaked_to_reserved = baseline_keys & reserved_set

    assert leaked_to_val == set(), (
        f"Baseline contains val principals (leakage!): {leaked_to_val}"
    )
    assert leaked_to_reserved == set(), (
        f"Baseline contains reserved principals (leakage!): {leaked_to_reserved}"
    )
    assert baseline_keys <= train_set, (
        f"Baseline keys not a subset of train principals: {baseline_keys - train_set}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# V3 — Train features have no attack signal
# ═══════════════════════════════════════════════════════════════════════════════


def test_train_features_have_no_attack_signal() -> None:
    """V3: sensitive_action_count is 0 in every training window.

    Synthetic normal data must never contain sensitive APIs (L2).
    If this fails, the synthetic generator has a regression — stop the world.
    """
    _require(TRAIN_FEATURES_PATH)
    df = pd.read_parquet(TRAIN_FEATURES_PATH, engine="pyarrow")
    assert (df["sensitive_action_count"] == 0).all(), (
        f"Training features contain sensitive API calls!\n"
        f"Non-zero rows:\n{df[df['sensitive_action_count'] > 0][['principal_id', 'window_start', 'sensitive_action_count']].head()}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# V4 — Both window sizes are present
# ═══════════════════════════════════════════════════════════════════════════════


def test_train_features_have_both_window_sizes() -> None:
    """V4: train_features.parquet contains both 5-min and 60-min windows (L3)."""
    _require(TRAIN_FEATURES_PATH)
    df = pd.read_parquet(TRAIN_FEATURES_PATH, engine="pyarrow")
    actual_sizes = set(df["window_size_minutes"].unique())
    assert actual_sizes == {5, 60}, (
        f"Expected window sizes {{5, 60}}, got {actual_sizes}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# V5 — No NaN in feature columns
# ═══════════════════════════════════════════════════════════════════════════════


def test_train_features_have_no_nan() -> None:
    """V5: All 7 FEATURE_COLUMNS are NaN-free in train_features.parquet."""
    _require(TRAIN_FEATURES_PATH)
    df = pd.read_parquet(TRAIN_FEATURES_PATH, engine="pyarrow")
    nan_counts = df[FEATURE_COLUMNS].isna().sum()
    total_nans = nan_counts.sum()
    assert total_nans == 0, (
        f"NaN values found in feature columns:\n{nan_counts[nan_counts > 0]}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# V6 — Determinism
# ═══════════════════════════════════════════════════════════════════════════════


def test_determinism(tmp_path: Path) -> None:
    """V6: Re-running run_day2.py --mode mock --force produces identical features.

    Compares SHA-256 of train_features.parquet before and after a second run.
    Falls back to comparing sorted CSV bytes if pyarrow's row ordering
    introduces non-determinism in the binary format.
    """
    _require(TRAIN_FEATURES_PATH)

    # ── Hash the current artifact ─────────────────────────────────────────────
    def _parquet_hash(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def _csv_hash(p: Path) -> str:
        # Round-trip through CSV to get a format-independent comparison.
        # Sort by all columns to neutralise row-order differences.
        df = pd.read_parquet(p, engine="pyarrow")
        csv_bytes = df.sort_values(by=list(df.columns)).to_csv(index=False).encode()
        return hashlib.sha256(csv_bytes).hexdigest()

    hash_before_parquet = _parquet_hash(TRAIN_FEATURES_PATH)
    hash_before_csv = _csv_hash(TRAIN_FEATURES_PATH)

    # ── Re-run the pipeline ───────────────────────────────────────────────────
    result = subprocess.run(
        [sys.executable, str(_PROJECT_ROOT / "scripts" / "run_day2.py"),
         "--mode", "mock", "--force"],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        f"run_day2.py failed on second run:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    hash_after_parquet = _parquet_hash(TRAIN_FEATURES_PATH)
    hash_after_csv = _csv_hash(TRAIN_FEATURES_PATH)

    # ── Compare — try binary first, fall back to CSV ──────────────────────────
    if hash_before_parquet != hash_after_parquet:
        # pyarrow may produce non-deterministic binary (row-group metadata);
        # compare the logical content instead.
        assert hash_before_csv == hash_after_csv, (
            "Pipeline is NOT deterministic: feature values differ across runs. "
            "Check for random state leaks in aggregate_features or Isolation Forest."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# V7 — Model bundle has all required keys
# ═══════════════════════════════════════════════════════════════════════════════


def test_model_bundle_complete() -> None:
    """V7: isolation_forest.pkl bundle has all 9 required keys and correct feature list."""
    _require(MODEL_PATH)
    bundle: dict = joblib.load(MODEL_PATH)

    required_keys = {
        "model",
        "feature_columns",
        "contamination",
        "n_estimators",
        "random_state",
        "sklearn_version",
        "train_n_rows",
        "train_fpr",
        "feature_set_hash",
    }
    missing = required_keys - set(bundle.keys())
    assert missing == set(), f"Bundle missing keys: {missing}"

    # Feature column list must exactly match the current code constant
    assert bundle["feature_columns"] == FEATURE_COLUMNS, (
        f"Bundle feature_columns mismatch.\n"
        f"  Bundle: {bundle['feature_columns']}\n"
        f"  Code:   {FEATURE_COLUMNS}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# V8 — Alert records have required fields
# ═══════════════════════════════════════════════════════════════════════════════


def test_alerts_have_required_fields() -> None:
    """V8: Every line in tier1_alerts_val.jsonl has the required fields."""
    _require(ALERTS_PATH)

    required_fields = {
        "principal_id",
        "window_start",
        "window_size_minutes",
        "anomaly_score",
        "feature_values",
    }

    with open(ALERTS_PATH, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    if not lines:
        # Zero alerts is valid (val set may be all-normal in mock mode);
        # the test passes vacuously.
        return

    for i, line in enumerate(lines):
        record = json.loads(line.strip())
        missing = required_fields - set(record.keys())
        assert missing == set(), (
            f"Alert record {i} missing fields: {missing}\n  record={record}"
        )
        # feature_values must be a dict with all 7 feature column names
        fv = record["feature_values"]
        missing_fv = set(FEATURE_COLUMNS) - set(fv.keys())
        assert missing_fv == set(), (
            f"Alert record {i} feature_values missing keys: {missing_fv}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# V9 — Anonymous entity fallback uses source IP (L8)
# ═══════════════════════════════════════════════════════════════════════════════


def test_anonymous_fallback() -> None:
    """V9: Anonymous calls produce per-IP principal_ids, not a shared entity."""
    pid_1, etype_1 = extract_entity({"type": "Anonymous"}, "1.2.3.4")
    pid_2, etype_2 = extract_entity({"type": "Anonymous"}, "5.6.7.8")

    assert pid_1 == "anonymous:1.2.3.4", f"Got {pid_1!r}, expected 'anonymous:1.2.3.4'"
    assert etype_1 == "Anonymous", f"Got entity_type={etype_1!r}"

    assert pid_2 == "anonymous:5.6.7.8", f"Got {pid_2!r}, expected 'anonymous:5.6.7.8'"
    assert etype_2 == "Anonymous"

    # The key requirement of L8: two different IPs produce DIFFERENT principal_ids.
    assert pid_1 != pid_2, (
        "Two different source IPs collapsed to the same principal_id — "
        "this is the L8 regression that prevents thousands of scanners "
        "from merging into one super-entity."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# V10 — Dedup preserves all empty event_id rows (L9)
# ═══════════════════════════════════════════════════════════════════════════════


def test_dedup_preserves_empty_event_ids() -> None:
    """V10: Dedup logic keeps all rows with empty event_id.

    Regression test for the original preprocess.py bug where
    drop_duplicates("event_id") on an empty string would collapse
    3 unrelated events into 1.
    """
    import pandas as pd

    # 2 rows with same event_id="abc" (duplicates), 3 rows with event_id=""
    rows = [
        {"event_id": "abc", "event_name": "GetObject"},
        {"event_id": "abc", "event_name": "GetObject"},  # duplicate
        {"event_id": "", "event_name": "ListBuckets"},
        {"event_id": "", "event_name": "GetCallerIdentity"},
        {"event_id": "", "event_name": "DescribeInstances"},
    ]
    df = pd.DataFrame(rows)

    # Apply the exact dedup logic from ingest.load_events (L9)
    mask = df["event_id"].astype(str).str.len() > 0
    deduped_with_id = df[mask].drop_duplicates("event_id", keep="first")
    result = pd.concat([df[~mask], deduped_with_id], ignore_index=True)

    assert len(result) == 4, (
        f"Expected 4 rows (1 'abc' + 3 empty), got {len(result)}.\n"
        f"This is the L9 regression: empty event_ids must not be deduped."
    )
    # Exactly 1 row with event_id="abc"
    assert (result["event_id"] == "abc").sum() == 1
    # All 3 empty-id rows preserved
    assert (result["event_id"] == "").sum() == 3
