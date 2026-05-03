"""
src/tier1_ml/baseline.py
─────────────────────────────────────────────────────────────────────────────
Pipeline position: shared dependency between preprocessing/features.py (fit)
and tier1_ml/detect.py (apply).

Why a separate baseline module?
  The hour-of-day baseline captures per-principal circadian patterns from the
  training set ONLY.  Serialising it to disk as a first-class artifact enforces
  the leakage-prevention contract: the same frozen statistics are loaded at
  val/test time without any re-fitting on held-out data (L7, CLAUDE.md §7).

  A pure dataclass with frozen=True makes mutation impossible at runtime,
  preventing the silent bug of accidentally updating the baseline when scoring
  new events.

Data contract:
  fit path:   features.fit_hour_baseline(train_events)  →  HourBaseline
              tier1_ml.baseline.save_baseline(b, path)  →  models/tier1_baseline.json

  score path: tier1_ml.baseline.load_baseline(path)  →  HourBaseline
              features.aggregate_features(events, baseline)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_SCHEMA_VERSION: Final[int] = 1


@dataclass(frozen=True)
class HourBaseline:
    """Per-principal hour-of-day statistics computed from training data only.

    Attributes:
        per_principal:         principal_id → {"mean": float, "std": float, "n": int}
                               Absent principals fall back to global stats.
        global_mean_hour:      Mean hour-of-day over ALL training events.
                               Fallback for principals not in training set.
        global_std_hour:       Std dev of hour-of-day over ALL training events.
        train_principal_count: Number of principals in the training split.
                               Used in validate_baseline_is_train_only (V2).
    """

    __slots__ = (
        "per_principal",
        "global_mean_hour",
        "global_std_hour",
        "train_principal_count",
    )

    per_principal: dict[str, dict[str, float]]
    global_mean_hour: float
    global_std_hour: float
    train_principal_count: int


def save_baseline(b: HourBaseline, path: Path) -> None:
    """Serialize a HourBaseline to JSON with a schema_version sentinel.

    The schema_version field allows load_baseline to detect and reject stale
    files if the schema changes in a future day.

    Args:
        b:    HourBaseline to serialize.
        path: Output path (e.g., models/tier1_baseline.json).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "per_principal": b.per_principal,
        "global_mean_hour": b.global_mean_hour,
        "global_std_hour": b.global_std_hour,
        "train_principal_count": b.train_principal_count,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved hour baseline ({len(b.per_principal)} principals) -> {path}")


def load_baseline(path: Path) -> HourBaseline:
    """Load a HourBaseline from disk, validating schema_version.

    Args:
        path: Path to a JSON file previously written by save_baseline().

    Returns:
        HourBaseline dataclass.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError:        If schema_version is not recognised.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline not found at {path}. "
            "Run: python scripts/run_day2.py --mode mock --force"
        )

    with open(path, "r", encoding="utf-8") as fh:
        payload: dict = json.load(fh)

    version = payload.get("schema_version")
    if version != _SCHEMA_VERSION:
        raise ValueError(
            f"Baseline schema_version mismatch: expected {_SCHEMA_VERSION}, "
            f"got {version}. Delete {path} and re-run run_day2.py."
        )

    return HourBaseline(
        per_principal=payload["per_principal"],
        global_mean_hour=float(payload["global_mean_hour"]),
        global_std_hour=float(payload["global_std_hour"]),
        train_principal_count=int(payload["train_principal_count"]),
    )
