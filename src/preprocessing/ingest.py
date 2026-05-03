"""
src/preprocessing/ingest.py
─────────────────────────────────────────────────────────────────────────────
Pipeline position: first stage — reads raw CloudTrail JSONL off disk and
produces a clean, typed pandas DataFrame for downstream feature aggregation.

Why streaming, not full-file load?
  synthetic_normal.jsonl is ~448 MB; flaws_merged.jsonl is ~2.3 GB.  Loading
  either fully into memory before parsing doubles peak RSS and is unnecessary
  because JSONL lets us process one record at a time.

Why identity-based splits?
  Each IAM principal has a characteristic event distribution (busy CI role vs.
  quiet IAM user vs. anonymous scanner).  If the same principal appears in both
  train and val/test the model can memorise that principal's pattern — leakage
  that inflates Tier 1 recall on val without generalising to new principals.
  Du et al. (2017) do not discuss this because they work on session IDs, not
  named principals; for CloudTrail it is the dominant leakage source.

Data-flow:
  data/raw/synthetic_normal.jsonl  (or any CloudTrail JSONL)
      └─ load_events()  →  pd.DataFrame  (one row per CloudTrail event)
          └─ split_by_principal()  →  (train_df, val_df, reserved_df, manifest)
              └─ write_split_artifacts()  →  data/processed/synthetic_*.jsonl
                                             data/processed/split_manifest.json
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Final

import pandas as pd

# ── Path constants (all overridable via environment variables) ────────────────
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

SYNTHETIC_NORMAL_PATH: Final[Path] = Path(
    os.environ.get(
        "SYNTHETIC_NORMAL_PATH",
        str(_PROJECT_ROOT / "data" / "raw" / "synthetic_normal.jsonl"),
    )
)

PROCESSED_DIR: Final[Path] = Path(
    os.environ.get(
        "PROCESSED_DIR",
        str(_PROJECT_ROOT / "data" / "processed"),
    )
)

# ── Column schema produced by load_events ────────────────────────────────────
EVENT_COLUMNS: Final[list[str]] = [
    "event_id",
    "event_time",
    "event_name",
    "event_source",
    "principal_id",
    "entity_type",
    "source_ip",
    "error_code",
    "aws_region",
    "_source",   # origin tag — "synth_normal" or "flaws_cloud"; used by val-mixing (Day 2) and session labelling (Day 3)
]


# ═══════════════════════════════════════════════════════════════════════════════
# Public functions
# ═══════════════════════════════════════════════════════════════════════════════


def load_events(
    path: Path,
    max_events: int | None = None,
    source: str = "synth_normal",
) -> pd.DataFrame:
    """Stream a CloudTrail JSONL file and return a typed DataFrame.

    Args:
        path: Absolute path to a CloudTrail JSONL file.
        max_events: If given, stop after this many *successfully parsed* rows.
                    Used in --mode mock to cap at 5 000 events.
        source: Origin tag written to every row's ``_source`` column.
                Defaults to ``"synth_normal"`` so existing callers are
                unaffected.  Pass ``"flaws_cloud"`` when loading attack logs
                so that val-mixing (Day 2) and Tier 2 session labelling
                (Day 3) can propagate dataset origin without a separate join.

    Returns:
        DataFrame with columns matching EVENT_COLUMNS.  event_time is
        datetime64[ns, UTC].  All string columns use "" as the missing sentinel
        (never None) so downstream groupby / nunique never silently drops rows.
        The ``_source`` column is always present; every value equals ``source``.
    """
    rows: list[dict] = []
    n_malformed = 0    # lines that failed json.loads
    n_missing_time = 0  # lines that parsed but had no parseable eventTime
    n_missing_name = 0  # lines that parsed but had no eventName → skipped

    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            # ── Parse JSON ───────────────────────────────────────────────────
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                n_malformed += 1
                continue

            # ── Mandatory field: eventName ───────────────────────────────────
            event_name: str = record.get("eventName", "")
            if not event_name:
                n_missing_name += 1
                continue  # skip; downstream features require a name

            # ── Parse eventTime ──────────────────────────────────────────────
            raw_time: str = record.get("eventTime", "")
            if not raw_time:
                n_missing_time += 1
                continue  # can't place event in a time window without a time

            # CloudTrail uses "2024-01-15T14:23:00Z" — replace Z with UTC offset
            event_time_str = raw_time.replace("Z", "+00:00")
            try:
                event_time = pd.Timestamp(event_time_str, tz="UTC")
            except (ValueError, TypeError):
                n_missing_time += 1
                continue

            # ── Entity resolution ────────────────────────────────────────────
            user_identity: dict = record.get("userIdentity", {})
            source_ip: str = str(record.get("sourceIPAddress", ""))
            principal_id, entity_type = extract_entity(user_identity, source_ip)

            rows.append(
                {
                    "event_id": str(record.get("eventID", "")),
                    "event_time": event_time,
                    "event_name": event_name,
                    "event_source": str(record.get("eventSource", "")),
                    "principal_id": principal_id,
                    "entity_type": entity_type,
                    "source_ip": source_ip,
                    "error_code": str(record.get("errorCode", "")),
                    "aws_region": str(record.get("awsRegion", "unknown")) or "unknown",
                }
            )

            if max_events is not None and len(rows) >= max_events:
                break

    n_loaded = len(rows)
    print(
        f"Loaded {n_loaded:,} events from {path.name}"
        f" (skipped: {n_malformed} malformed JSON,"
        f" {n_missing_time} missing time,"
        f" {n_missing_name} missing event name)"
    )

    if n_loaded == 0:
        # Return empty DataFrame with correct dtypes so downstream code doesn't
        # crash on column checks.
        return pd.DataFrame(columns=EVENT_COLUMNS)

    df = pd.DataFrame(rows)

    # ── Dedup: ONLY rows with a non-empty event_id (L9) ─────────────────────
    # Rationale: empty event_id means the record has no canonical identifier
    # (common for Console events, anomalous records, or synthetic mocks).
    # De-duplicating empty IDs would collapse unrelated records into one row,
    # which is a silent data-loss bug.  We keep ALL empty-ID rows and dedup
    # only the slice that has a real UUID.
    mask_has_id = df["event_id"].astype(str).str.len() > 0
    deduped_with_id = df[mask_has_id].drop_duplicates("event_id", keep="first")
    n_dropped = mask_has_id.sum() - len(deduped_with_id)
    df = pd.concat([df[~mask_has_id], deduped_with_id], ignore_index=True)
    print(
        f"Dedup: kept {(~mask_has_id).sum():,} rows with empty event_id,"
        f" dropped {n_dropped:,} duplicates"
        f" from {mask_has_id.sum():,} rows with event_id."
    )

    # ── Ensure event_time is datetime with UTC timezone ──────────────────────
    # pd.DataFrame() stores Timestamp objects; convert to proper dtype.
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)

    # ── Tag every event with its dataset origin ──────────────────────────────
    # _source is the single source of truth for where this event came from.
    # Downstream val-mixing (Day 2) and Tier 2 session labelling (Day 3) read
    # this column rather than inferring origin from file paths, so the tag must
    # be set here — before any split or aggregation — to propagate cleanly.
    df["_source"] = source

    return df


def extract_entity(
    user_identity: dict,
    source_ip: str,
) -> tuple[str, str]:
    """Resolve a CloudTrail userIdentity dict to (principal_id, entity_type).

    Resolution order (L8 compliant):
      1. principalId present → use it
      2. arn present          → use it
      3. type == "Anonymous"  → "anonymous:<sourceIPAddress>"  (never collapse all
                                anonymous calls to a single entity; one per IP)
      4. fallthrough          → ("unknown", "Other")

    This is a pure function with no side effects — safe to call in list
    comprehensions and directly testable (V9).

    Args:
        user_identity: The "userIdentity" sub-dict from a CloudTrail record.
        source_ip:     The event's sourceIPAddress; used for anonymous fallback.

    Returns:
        (principal_id, entity_type) — both are non-empty strings.
    """
    uid_type: str = str(user_identity.get("type", ""))

    # ── Map CloudTrail identity type to our enum ─────────────────────────────
    def _map_type(t: str) -> str:
        if t == "IAMUser":
            return "IAMUser"
        if t == "AssumedRole":
            return "AssumedRole"
        return "Other"

    # Resolution 1: principalId (most specific; always present for IAM entities)
    principal_id_raw = user_identity.get("principalId")
    if principal_id_raw:
        return str(principal_id_raw), _map_type(uid_type)

    # Resolution 2: arn (fallback for root, federated, service accounts)
    arn = user_identity.get("arn")
    if arn:
        return str(arn), _map_type(uid_type)

    # Resolution 3: Anonymous — one entity per source IP (L8)
    # Without this, thousands of public scanners all become "anonymous" and the
    # Isolation Forest sees one extremely busy entity instead of many quiet ones.
    if uid_type == "Anonymous":
        ip = source_ip if source_ip else "0.0.0.0"
        return f"anonymous:{ip}", "Anonymous"

    # Resolution 4: genuine unknown (e.g., AWSService with no principal info)
    return "unknown", "Other"


def split_by_principal(
    events: pd.DataFrame,
    train_frac: float = 0.90,
    val_frac: float = 0.05,
    reserved_frac: float = 0.05,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Partition events by IAM principal into train / val / reserved splits.

    Why identity-based splitting?
      A time-based split (first 80% of events go to train) has two failure modes:
      1. Temporal leakage: if a principal's behaviour is stable over time, the
         model memorises their pattern across the split boundary.
      2. Covariate shift: night-shift principals get split at midnight, which
         is not a natural session boundary.
      Identity-based splitting guarantees zero principal overlap across splits,
      which is the strongest leakage-prevention guarantee available here.

    Args:
        events:        DataFrame from load_events().
        train_frac:    Fraction of principals assigned to train  (default 0.90).
        val_frac:      Fraction for validation                   (default 0.05).
        reserved_frac: Fraction for held-out test                (default 0.05).
        seed:          Random seed for the principal shuffle.

    Returns:
        (train_df, val_df, reserved_df, manifest_dict)
    """
    principals: list[str] = sorted(events["principal_id"].unique().tolist())
    n = len(principals)

    if n < 3:
        raise ValueError(
            f"Need at least 3 distinct principals for a 3-way split; got {n}. "
            "Run the synthetic generator with more principals or lower the split fracs."
        )

    # Use stdlib random so numpy's global RNG state stays clean for sklearn.
    rng = random.Random(seed)
    rng.shuffle(principals)

    # Enforce minimum 1 principal per split even when n is small (e.g. 8 principals).
    n_val = max(1, round(n * val_frac))
    n_reserved = max(1, round(n * reserved_frac))
    n_train = n - n_val - n_reserved

    if n_train < 1:
        raise ValueError(
            f"Not enough principals ({n}) for 3-way split "
            f"(train={n_train}, val={n_val}, reserved={n_reserved}). "
            "Reduce val_frac / reserved_frac or generate more principals."
        )

    train_principals = principals[:n_train]
    val_principals = principals[n_train : n_train + n_val]
    reserved_principals = principals[n_train + n_val :]

    # ── Disjointness assertion — fail hard, not silently ─────────────────────
    assert set(train_principals) & set(val_principals) == set(), (
        "BUG: train/val principal overlap"
    )
    assert set(train_principals) & set(reserved_principals) == set(), (
        "BUG: train/reserved principal overlap"
    )
    assert set(val_principals) & set(reserved_principals) == set(), (
        "BUG: val/reserved principal overlap"
    )

    train_set = set(train_principals)
    val_set = set(val_principals)
    reserved_set = set(reserved_principals)

    train_df = events[events["principal_id"].isin(train_set)].copy()
    val_df = events[events["principal_id"].isin(val_set)].copy()
    reserved_df = events[events["principal_id"].isin(reserved_set)].copy()

    manifest: dict = {
        "seed": seed,
        "fractions": {
            "train": train_frac,
            "val": val_frac,
            "reserved": reserved_frac,
        },
        "principals": {
            "train": sorted(train_principals),
            "val": sorted(val_principals),
            "reserved": sorted(reserved_principals),
        },
        "event_counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "reserved": int(len(reserved_df)),
        },
    }

    print(
        f"Split: {len(train_principals)} train principals ({len(train_df):,} events)"
        f" | {len(val_principals)} val ({len(val_df):,} events)"
        f" | {len(reserved_principals)} reserved ({len(reserved_df):,} events)"
    )

    return train_df, val_df, reserved_df, manifest


def write_split_artifacts(
    train_events: pd.DataFrame,
    val_events: pd.DataFrame,
    reserved_events: pd.DataFrame,
    manifest: dict,
    output_dir: Path,
) -> None:
    """Serialize the three split DataFrames and the manifest to disk.

    Why JSONL for the split files (not Parquet)?
      Downstream Day 3 Drain3 parsing expects JSONL — it processes records one
      at a time to build log-key sequences.  Keeping the split files as JSONL
      avoids a format-conversion step in drain_parser.py.

    Args:
        train_events:    DataFrame for the training split.
        val_events:      DataFrame for the validation split.
        reserved_events: DataFrame for the reserved (test) split.
        manifest:        Dict from split_by_principal().
        output_dir:      Destination directory; must already exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_events_jsonl(train_events, output_dir / "synthetic_train.jsonl")
    _write_events_jsonl(val_events, output_dir / "synthetic_val.jsonl")
    _write_events_jsonl(reserved_events, output_dir / "synthetic_reserved.jsonl")

    manifest_path = output_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Wrote split artifacts to {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _write_events_jsonl(df: pd.DataFrame, path: Path) -> None:
    """Serialize a events DataFrame back to JSONL for Day 3 Drain3 consumption."""
    with open(path, "w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            record = {
                "eventID": row["event_id"],
                # ISO8601 with UTC offset so Drain3 can reconstruct timestamps.
                "eventTime": row["event_time"].isoformat(),
                "eventName": row["event_name"],
                "eventSource": row["event_source"],
                "principalId": row["principal_id"],
                "entityType": row["entity_type"],
                "sourceIPAddress": row["source_ip"],
                "errorCode": row["error_code"],
                "awsRegion": row["aws_region"],
            }
            fh.write(json.dumps(record) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Direct invocation (for ad-hoc testing without the full run_day2 orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Quick smoke-test: load and split synthetic events, print manifest."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest + split CloudTrail JSONL")
    parser.add_argument("--input", type=Path, default=SYNTHETIC_NORMAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()

    df = load_events(args.input, max_events=args.max_events)
    train_df, val_df, reserved_df, manifest = split_by_principal(df)
    write_split_artifacts(train_df, val_df, reserved_df, manifest, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
