"""
scripts/diagnose_render.py

One-off diagnostic before committing the Drain3 render shape.

Answers two questions:

1. Entity-type distribution on synthetic_train.jsonl.
   Decision rule:
     - Balanced (no single type > 95%): keep entityType in the render.
     - Skewed (>95% one type): drop entityType to avoid mass OOV when
       transforming flaws.cloud data, which has a different mix.

2. Distinct (eventName, eventSource, errorCode) tuple count.
   This is a PRIOR estimate of num_keys after Drain3 fitting.
   Expected range for CloudTrail-style structured logs: 50-300.
     - Much higher (e.g., >2000): render is too detailed; mask harder
       or drop a field.
     - Much lower (<20): render is too coarse; DeepLog won't have
       enough vocabulary to learn from.

Usage
-----
    python scripts/diagnose_render.py

Reads from data/processed/synthetic_train.jsonl (override with
SYNTHETIC_TRAIN_PATH env var). Prints two Counters and a summary line.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Final

# Add src/ to path so we can import normalize_event without installing
# the project as a package. This is fine for a diagnostic script; not
# something we'd do in production code.
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.normalize import normalize_event  # noqa: E402

_SYNTHETIC_TRAIN_PATH: Final[Path] = Path(
    os.environ.get(
        "SYNTHETIC_TRAIN_PATH",
        str(_PROJECT_ROOT / "data" / "processeded1" / "synthetic_train.jsonl"),
    )
)

# Threshold above which we'd recommend dropping entityType from the
# render. 0.95 is a defensible round number; the thesis can cite this
# as "we treated >95% concentration as effectively constant".
_SKEW_THRESHOLD: Final[float] = 0.95


def main() -> None:
    if not _SYNTHETIC_TRAIN_PATH.exists():
        # Loud failure with remediation, per CLAUDE.md sec 5 item 8.
        raise FileNotFoundError(
            f"Synthetic training file not found at {_SYNTHETIC_TRAIN_PATH}. "
            f"Set SYNTHETIC_TRAIN_PATH or place the file at the default path."
        )

    entity_types: Counter[str] = Counter()
    tuples: Counter[tuple[str, str, str]] = Counter()
    total = 0

    # Stream-read so we don't hold the whole file in memory. Synthetic
    # train should fit easily, but this scales if you regenerate it
    # larger later.
    with _SYNTHETIC_TRAIN_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                # Don't silently skip malformed lines — note them and continue.
                # Bare `except:` is forbidden by CLAUDE.md sec 5; we catch the
                # specific decode error.
                print(f"[warn] skipping malformed JSON at line {line_no}: {exc}")
                continue

            ev = normalize_event(raw)
            entity_types[ev["entityType"]] += 1
            tuples[(ev["eventName"], ev["eventSource"], ev["errorCode"])] += 1
            total += 1

    if total == 0:
        raise ValueError(
            f"No events parsed from {_SYNTHETIC_TRAIN_PATH}. "
            f"Check the file is JSONL with one event per line."
        )

    print(f"\n=== Diagnostic on {_SYNTHETIC_TRAIN_PATH.name} ===")
    print(f"Total events: {total:,}")

    print("\nEntity type distribution:")
    for et, count in entity_types.most_common():
        pct = count / total * 100
        print(f"  {et:<20s} {count:>8,d}  ({pct:5.2f}%)")

    # Decision summary for entity type.
    top_share = entity_types.most_common(1)[0][1] / total
    if top_share > _SKEW_THRESHOLD:
        print(
            f"\n  -> SKEWED: top type at {top_share:.1%} > {_SKEW_THRESHOLD:.0%}. "
            f"Recommend DROPPING entityType from render."
        )
    else:
        print(
            f"\n  -> BALANCED: top type at {top_share:.1%}. "
            f"Recommend KEEPING entityType in render."
        )

    print(f"\nDistinct (eventName, eventSource, errorCode) tuples: {len(tuples):,}")
    n_tuples = len(tuples)
    if n_tuples > 2000:
        print(
            f"  -> TOO DETAILED: {n_tuples} > 2000. "
            f"Render likely leaks high-cardinality data into templates."
        )
    elif n_tuples < 20:
        print(
            f"  -> TOO COARSE: {n_tuples} < 20. "
            f"DeepLog won't have enough vocabulary to learn from."
        )
    else:
        print(
            f"  -> OK: {n_tuples} is within 20-2000. "
            f"Expect Drain3 num_keys in roughly this range after fitting."
        )

    # Show the top 10 tuples so you can eyeball whether the masker is
    # doing its job. If you see things like distinct ARNs in here as
    # separate keys, the regex masking in drain_parser.py needs work.
    print("\nTop 10 tuples (for eyeballing):")
    for tup, count in tuples.most_common(10):
        print(f"  {count:>6,d}  {tup}")


if __name__ == "__main__":
    main()
    