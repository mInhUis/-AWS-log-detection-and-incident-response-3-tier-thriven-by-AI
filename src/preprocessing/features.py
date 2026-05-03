"""
src/preprocessing/features.py
─────────────────────────────────────────────────────────────────────────────
Pipeline position: second preprocessing stage — converts the per-event
DataFrame from ingest.py into per-(principal, time-window) feature vectors
that Tier 1 Isolation Forest can consume directly.

Why aggregate into windows?
  Isolation Forest operates on fixed-length scalar feature vectors.  Raw
  CloudTrail is a sequence of discrete events; to get scalar features we must
  summarise a period of activity.  The window is the minimal analysis unit:
  "what did this principal do in this 5-minute (or 1-hour) slot?"

Why TWO window sizes (5 min AND 60 min)?  (L3)
  A 5-minute window catches short, intense bursts (e.g., rapid IAM enumeration
  followed by privilege escalation).  A 1-hour window catches low-and-slow
  patterns that individual 5-min windows miss (e.g., one unusual API call per
  10 minutes for an hour).  Training the Isolation Forest on both window sizes
  improves recall across attack tempos without adding a second model.

Why these 7 features?
  - event_count:            volume signal; an anomaly is often "too many or too
                            few events in a window"
  - distinct_api_count:     breadth signal; an attacker enumerates many APIs
  - distinct_service_count: cross-service breadth; exfil attacks hop services
  - error_rate:             access-denied rate; probing causes errors
  - sensitive_action_count: direct indicator of high-risk API calls
  - geographic_spread:      region-hopping is unusual for legit workloads
  - time_of_day_deviation:  circadian anomaly; attacks often happen off-hours

Why train-only baseline for time_of_day_deviation?  (L7, CLAUDE.md §7)
  If we compute per-principal hour statistics on the full dataset, the deviation
  for val/test events is computed with knowledge of those events' actual hours —
  a form of label leakage.  The correct procedure (implemented here) is:
    1. fit_hour_baseline(train_events)  →  frozen stats saved to disk
    2. aggregate_features(val_events, baseline)  →  deviation computed against
       frozen train statistics, not recalculated on val.

Why SENSITIVE_APIS is frozen at end of Day 2?  (L10)
  Adding an API to this set after seeing the flaws.cloud test results is
  equivalent to feature engineering on the test set — it would inflate recall
  without generalising to new attack patterns.

Data-flow:
  pd.DataFrame (from ingest.load_events)
      └─ fit_hour_baseline()    →  HourBaseline  (train split only)
      └─ aggregate_features()   →  pd.DataFrame  (feature matrix + metadata)
          └─ write_features()   →  data/processed/train_features.parquet
                                   data/processed/val_features.parquet
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.tier1_ml.baseline import HourBaseline

# ── Project root (used for default path constants) ────────────────────────────
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

PROCESSED_DIR: Final[Path] = Path(
    os.environ.get("PROCESSED_DIR", str(_PROJECT_ROOT / "data" / "processed"))
)

# ── Window sizes (L3 — BOTH required; do not change to a single value) ────────
WINDOW_SIZES_MINUTES: Final[tuple[int, int]] = (5, 60)

# ── Sensitive API allowlist (L10 — FROZEN at end of Day 2) ───────────────────
# Any post-hoc modification after seeing test results is test-set fitting.
SENSITIVE_APIS: Final[frozenset[str]] = frozenset(
    {
        # CloudTrail tampering
        "DeleteTrail",
        "StopLogging",
        # IAM persistence / privilege escalation
        "CreateUser",
        "CreateAccessKey",
        "CreateLoginProfile",
        "AttachUserPolicy",
        "AttachRolePolicy",
        "PutUserPolicy",
        "PutRolePolicy",
        "CreateRole",
        "CreatePolicy",
        "CreatePolicyVersion",
        "AddUserToGroup",
        "AttachGroupPolicy",
        # S3 exfiltration prep
        "PutBucketAcl",
        "PutBucketPolicy",
        "PutBucketPublicAccessBlock",
        # Credential access
        "GetSecretValue",
        # Compute abuse / snapshot exfil
        "ModifySnapshotAttribute",
        # Console root login
        "ConsoleLogin",
    }
)

# ── Ordered feature column list (order matters — must match IsolationForest X) ─
FEATURE_COLUMNS: Final[list[str]] = [
    "event_count",
    "distinct_api_count",
    "distinct_service_count",
    "error_rate",
    "sensitive_action_count",
    "geographic_spread",
    "time_of_day_deviation",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Public functions
# ═══════════════════════════════════════════════════════════════════════════════


def fit_hour_baseline(train_events: pd.DataFrame) -> HourBaseline:
    """Compute per-principal and global hour-of-day statistics from training data.

    IMPORTANT: Call this ONLY on train_events.  Never on val or reserved splits.
    The returned HourBaseline must be serialised to disk and loaded read-only
    at scoring time (L7).

    Args:
        train_events: DataFrame from ingest.load_events(), filtered to
                      the training principals.

    Returns:
        HourBaseline with per-principal and global statistics.
    """
    # Extract hour from UTC timestamps; integer 0–23
    hours: pd.Series = train_events["event_time"].dt.hour.astype("float32")

    per_principal: dict[str, dict[str, float]] = {}
    for pid, group in train_events.groupby("principal_id"):
        group_hours = group["event_time"].dt.hour.astype("float32")
        per_principal[str(pid)] = {
            "mean": float(group_hours.mean()),
            "std": float(group_hours.std(ddof=1)) if len(group_hours) > 1 else 0.0,
            "n": int(len(group_hours)),
        }

    global_mean = float(hours.mean()) if len(hours) > 0 else 12.0
    global_std = float(hours.std(ddof=1)) if len(hours) > 1 else 1.0

    return HourBaseline(
        per_principal=per_principal,
        global_mean_hour=global_mean,
        global_std_hour=global_std,
        train_principal_count=len(per_principal),
    )


def aggregate_features(
    events: pd.DataFrame,
    baseline: HourBaseline,
) -> pd.DataFrame:
    """Aggregate per-event DataFrame into per-(principal, window) feature vectors.

    Produces rows for BOTH window sizes (5 min and 60 min) in a single output
    DataFrame, distinguished by the window_size_minutes column (L3).

    Args:
        events:   DataFrame from ingest.load_events() for one split.
                  Must contain a ``_source`` column (added by load_events since
                  the Day 2 ingest.py update).  Raises ValueError if absent.
        baseline: HourBaseline fitted on training data ONLY (L7).
                  Pass the same frozen baseline for val and reserved.

    Returns:
        DataFrame with FEATURE_COLUMNS + metadata columns:
          principal_id, window_start, window_size_minutes, entity_type,
          is_attack_window, label_source.
        All FEATURE_COLUMNS values are float32; no NaN.

        Label semantics (derived from event-level ``_source``, not hardcoded):
          is_attack_window — True if ANY event in the window has
            ``_source == "flaws_cloud"``, else False.
          label_source — "flaws_cloud" if any event is flaws_cloud;
            "synth_normal" if all events are synth_normal;
            "mixed" if neither (should not occur under identity-based splits,
            but raises no error — silent mis-labelling would be worse).
    """
    # ── Guard: _source column must exist ─────────────────────────────────────
    # load_events() sets _source on every row since the Day 2 ingest update.
    # If it is missing the caller is using a stale DataFrame (e.g., loaded from
    # a pre-update parquet or built without load_events) — fail loudly so the
    # bug surfaces instead of producing silently wrong labels.
    if "_source" not in events.columns:
        raise ValueError(
            "'_source' column not found in events DataFrame.  "
            "Regenerate events via the updated ingest.load_events() — "
            "e.g. re-run: python scripts/run_day2.py --mode mock"
        )

    if events.empty:
        # Return correctly-typed empty DataFrame so callers don't fail on shape
        # checks when running in mock mode with very few events.
        cols = FEATURE_COLUMNS + [
            "principal_id", "window_start", "window_size_minutes",
            "entity_type", "is_attack_window", "label_source",
        ]
        return pd.DataFrame(columns=cols)

    window_dfs: list[pd.DataFrame] = []

    for win_min in WINDOW_SIZES_MINUTES:
        # Truncate each event_time to the nearest window_start boundary.
        # dt.floor() is vectorised and handles DST/timezone correctly on UTC.
        events_copy = events.copy()
        events_copy["window_start"] = events_copy["event_time"].dt.floor(
            f"{win_min}min"
        )
        events_copy["window_size_minutes"] = win_min

        # Group by (principal_id, window_start) — one row per entity-window
        grouped = events_copy.groupby(
            ["principal_id", "window_start"],
            observed=True,
        )

        # ── Aggregate each feature ────────────────────────────────────────────
        agg_df = grouped.agg(
            event_count=("event_name", "count"),
            distinct_api_count=("event_name", "nunique"),
            distinct_service_count=("event_source", "nunique"),
            # error_rate: fraction of events with a non-empty error code
            error_rate=("error_code", lambda s: (s.astype(str).str.len() > 0).mean()),
            sensitive_action_count=("event_name", lambda s: s.isin(SENSITIVE_APIS).sum()),
            geographic_spread=("aws_region", "nunique"),
            # entity_type: take first value (constant per principal in synthetic data)
            entity_type=("entity_type", "first"),
            window_size_minutes=("window_size_minutes", "first"),
            # ── Label columns derived from event-level _source ────────────────
            # Placing these inside .agg() keeps label logic co-located with
            # feature logic and avoids a second post-hoc pass over the window DF.
            #
            # is_attack_window: a window is an attack window if ANY contributing
            # event came from the flaws.cloud dataset.  One flaws.cloud event
            # in a mixed window is enough to make the window worth investigating.
            is_attack_window=(
                "_source",
                lambda s: bool((s == "flaws_cloud").any()),
            ),
            # label_source: window-level origin tag.
            #   "flaws_cloud"  — at least one event is from the attack dataset.
            #   "synth_normal" — all events are synthetic normal.
            #   "mixed"        — events from neither known source (defensive: a
            #                    future third source tag would land here rather
            #                    than silently inheriting a wrong label).
            label_source=(
                "_source",
                lambda s: (
                    "flaws_cloud"
                    if (s == "flaws_cloud").any()
                    else "synth_normal"
                    if (s == "synth_normal").all()
                    else "mixed"
                ),
            ),
        ).reset_index()

        # ── Compute time_of_day_deviation per window ──────────────────────────
        # Use the window_start hour (not individual event hours) so the feature
        # is deterministic regardless of how events cluster within the window.
        agg_df["time_of_day_deviation"] = agg_df.apply(
            lambda row: _compute_hour_deviation(
                hour=int(row["window_start"].hour),
                principal_id=str(row["principal_id"]),
                baseline=baseline,
            ),
            axis=1,
        ).astype("float32")

        window_dfs.append(agg_df)

    result = pd.concat(window_dfs, ignore_index=True)

    # ── Drop sparse windows (event_count < 3) ────────────────────────────────
    # Windows with fewer than 3 events produce unreliable feature estimates:
    # error_rate collapses to 0/1, distinct_api_count is uninformative.
    # Removing them prevents the Isolation Forest from flagging low-density
    # time periods as anomalous purely due to statistical noise.
    n_before = len(result)
    result = result[result["event_count"] >= 3].copy()
    n_dropped = n_before - len(result)
    if n_dropped > 0:
        print(f"Dropped {n_dropped:,} sparse windows (event_count < 3).")

    # ── Cast all FEATURE_COLUMNS to float32 ──────────────────────────────────
    # Ensures downstream np.astype("float32") in train.py is a no-op (no copy).
    for col in FEATURE_COLUMNS:
        result[col] = result[col].astype("float32")

    # ── Verify no NaN in feature columns ─────────────────────────────────────
    nan_counts = result[FEATURE_COLUMNS].isna().sum()
    if nan_counts.any():
        bad_cols = nan_counts[nan_counts > 0].to_dict()
        raise ValueError(
            f"NaN values in feature columns after aggregation: {bad_cols}. "
            "Check _compute_hour_deviation for missing branches."
        )

    return result


def write_features(features_df: pd.DataFrame, path: Path) -> None:
    """Write the feature matrix to Parquet with snappy compression.

    Why Parquet + snappy?  (L4)
      Parquet preserves column dtypes (float32, datetime, bool) without
      schema-on-read guessing.  Snappy provides fast decompression — important
      when loading features repeatedly during hyperparameter search on Day 8.
      NPZ lacks metadata columns; CSV loses float32 precision and is 4× larger.

    Args:
        features_df: DataFrame from aggregate_features().
        path:        Output path (must end in .parquet).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    print(f"Wrote {len(features_df):,} feature rows -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_hour_deviation(
    hour: int,
    principal_id: str,
    baseline: HourBaseline,
) -> float:
    """Return |hour - mean| / std for the given principal, using baseline stats.

    Three cases handled to guarantee no NaN (V5):
      1. Principal in baseline, std > 0:  standard z-score magnitude
      2. Principal in baseline, std == 0: all training events happened in the
         same hour; divide by 1.0 so a different hour still has signal (not 0)
      3. Principal not in baseline:       use global fallback stats

    Args:
        hour:         Hour-of-day of the window_start (0–23).
        principal_id: The entity whose stats to look up.
        baseline:     HourBaseline from fit_hour_baseline(train_events).

    Returns:
        Non-negative float32 deviation value.  Never NaN.
    """
    stats = baseline.per_principal.get(principal_id)

    if stats is not None:
        mean_h = float(stats["mean"])
        std_h = float(stats["std"])
        # Avoid division by zero AND preserve anomaly signal when std == 0.
        # If std == 0, the principal always acts at the same hour; any other
        # hour is a raw deviation measured in hours (not sigma units).
        divisor = std_h if std_h > 0.0 else 1.0
        return float(abs(hour - mean_h) / divisor)
    else:
        # Unseen principal: fall back to global distribution.
        # max(..., 1.0) prevents division by near-zero on tiny datasets.
        divisor = max(baseline.global_std_hour, 1.0)
        return float(abs(hour - baseline.global_mean_hour) / divisor)
