"""
scripts/diagnose_flaws_oov.py

Measures the vocabulary gap between synthetic-train and flaws.cloud.
Run AFTER diagnose_render.py confirmed synthetic-train shape.

Three outputs:

1. Event-name distribution in flaws.cloud (sanity check; lets you eyeball
   what's there).
2. Event-name OOV: events in flaws.cloud NOT in synthetic-train, sorted
   by frequency. This is the priority-ordered widening list.
3. Tuple-level OOV: (eventName, eventSource, errorCode) tuples in
   flaws.cloud not in synthetic-train. Catches the AccessDenied gap that
   pure name-based comparison misses.

Plus summary OOV rates: percentage of flaws.cloud events that would be
unknown to a Drain3 vocabulary fit on synthetic-train.

Usage
-----
    python scripts/diagnose_flaws_oov.py

Reads:
  - data/processed/synthetic_train.jsonl  (override SYNTHETIC_TRAIN_PATH)
  - data/raw/flaws_cloud.jsonl            (override FLAWS_CLOUD_PATH)

The flaws.cloud path defaults to the full corpus, not the test split.
We need everything we have here to make the widening decision; the
test/val/sealing discipline applies to MODEL evaluation, not to
measurement-of-the-data-itself for methodology design. We're not
fitting any model in this script.

(If you want to be extra-pure about this, point FLAWS_CLOUD_PATH at the
validation slice only and re-run after widening to confirm the test
slice doesn't reveal the same picture differently. For Day 3 the full
corpus is fine.)
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Final, Iterator

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from preprocessing.normalize import normalize_event  # noqa: E402

_SYNTHETIC_TRAIN_PATH: Final[Path] = Path(
    os.environ.get(
        "SYNTHETIC_TRAIN_PATH",
        str(_PROJECT_ROOT / "data" / "processeded1" / "synthetic_train.jsonl"),
    )
)

# Default points at a single combined flaws.cloud file. If your repo has
# them split per level (flaws_level1.jsonl, flaws_level2.jsonl, ...) just
# concatenate them before running, or override the env var.
_FLAWS_CLOUD_PATH: Final[Path] = Path(
    os.environ.get(
        "FLAWS_CLOUD_PATH",
        str(_PROJECT_ROOT / "data" / "processeded1" / "flaws_cloud_val.jsonl"),
    )
)

# How many missing-event rows to dump. Top-N is the widening priority list.
_TOP_N_OOV: Final[int] = 30


def _stream_normalized(path: Path) -> Iterator[dict[str, str]]:
    """Yield normalized events from a JSONL file. Skips malformed lines loudly."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] {path.name} line {line_no}: malformed JSON: {exc}")
                continue
            yield normalize_event(raw)


def main() -> None:
    for p in (_SYNTHETIC_TRAIN_PATH, _FLAWS_CLOUD_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}. "
                f"Set the corresponding env var or place file at default path."
            )

    # First pass: build synthetic-train vocabulary. We only need the
    # SETS for membership tests, not the counts, so we don't keep the
    # full counter around.
    print("Reading synthetic-train vocabulary...")
    synth_event_names: set[str] = set()
    synth_tuples: set[tuple[str, str, str]] = set()
    synth_total = 0
    for ev in _stream_normalized(_SYNTHETIC_TRAIN_PATH):
        synth_event_names.add(ev["eventName"])
        synth_tuples.add((ev["eventName"], ev["eventSource"], ev["errorCode"]))
        synth_total += 1
    print(f"  {synth_total:,} events, "
          f"{len(synth_event_names)} distinct event names, "
          f"{len(synth_tuples)} distinct tuples\n")

    # Second pass: read flaws.cloud, count distributions, count OOV.
    # We count OOV at two granularities: event-name (loose) and tuple
    # (strict). Tuple OOV >= name OOV always; the gap between them tells
    # us how much of the OOV is from missing error codes vs missing APIs.
    print("Reading flaws.cloud distribution...")
    flaws_event_names: Counter[str] = Counter()
    flaws_tuples: Counter[tuple[str, str, str]] = Counter()
    flaws_total = 0
    name_oov_count = 0
    tuple_oov_count = 0

    for ev in _stream_normalized(_FLAWS_CLOUD_PATH):
        name = ev["eventName"]
        tup = (name, ev["eventSource"], ev["errorCode"])
        flaws_event_names[name] += 1
        flaws_tuples[tup] += 1
        flaws_total += 1

        # OOV checks. Note we count EVENTS, not unique names: an OOV
        # name with 10K occurrences contributes 10K to the rate. This
        # is what matters for sequence training, where each event is a
        # token DeepLog will or won't recognise.
        if name not in synth_event_names:
            name_oov_count += 1
        if tup not in synth_tuples:
            tuple_oov_count += 1

    print(f"  {flaws_total:,} events, "
          f"{len(flaws_event_names)} distinct event names, "
          f"{len(flaws_tuples)} distinct tuples\n")

    # === Output 1: flaws.cloud distribution ===
    print("=== flaws.cloud event-name distribution (top 30) ===")
    for name, count in flaws_event_names.most_common(30):
        pct = count / flaws_total * 100
        marker = "  " if name in synth_event_names else "**"  # ** = missing in synthetic
        print(f"  {marker} {name:<40s} {count:>8,d}  ({pct:5.2f}%)")
    print("\n  ('**' marks event names absent from synthetic-train)\n")

    # === Output 2: name-level OOV ===
    name_oov_rate = name_oov_count / flaws_total if flaws_total else 0.0
    print("=== Event-name OOV (priority widening list) ===")
    print(f"OOV event count: {name_oov_count:,} / {flaws_total:,} "
          f"({name_oov_rate:.1%} of flaws.cloud events)")

    # Build a sorted list of OOV names with their counts in flaws.cloud.
    # This is what the synthetic generator needs to add.
    missing_names = Counter({
        name: count
        for name, count in flaws_event_names.items()
        if name not in synth_event_names
    })
    print(f"Distinct OOV event names: {len(missing_names)}")
    print(f"\nTop {_TOP_N_OOV} missing event names (add these to synthetic generator):")
    for name, count in missing_names.most_common(_TOP_N_OOV):
        pct = count / flaws_total * 100
        print(f"  {name:<40s} {count:>8,d}  ({pct:5.2f}%)")
    print()

    # === Output 3: tuple-level OOV ===
    tuple_oov_rate = tuple_oov_count / flaws_total if flaws_total else 0.0
    print("=== Tuple-level OOV (catches missing error codes) ===")
    print(f"OOV event count (tuple-strict): {tuple_oov_count:,} / {flaws_total:,} "
          f"({tuple_oov_rate:.1%} of flaws.cloud events)")
    error_code_gap = tuple_oov_count - name_oov_count
    print(f"Gap (tuple-OOV minus name-OOV): {error_code_gap:,} events")
    print(f"  -> these are events whose eventName IS in synthetic, but the "
          f"(name, source, errorCode) combo is not.")
    print(f"  -> typically means synthetic lacks the AccessDenied / "
          f"NoSuchKey / etc. error variants of known APIs.\n")

    # Show the top OOV tuples that involve KNOWN names — those are the
    # error-code gaps specifically. Useful to decide whether the widening
    # should add error variants or new APIs (or both).
    print("Top 15 OOV tuples whose eventName IS in synthetic (error-code gap):")
    error_only_tuples = [
        (tup, count) for tup, count in flaws_tuples.most_common()
        if tup not in synth_tuples and tup[0] in synth_event_names
    ][:15]
    for tup, count in error_only_tuples:
        print(f"  {count:>6,d}  {tup}")
    print()

    # === Decision summary ===
    print("=== Summary ===")
    print(f"Name-level OOV rate:  {name_oov_rate:.1%}")
    print(f"Tuple-level OOV rate: {tuple_oov_rate:.1%}")
    print()
    if name_oov_rate < 0.05:
        print("-> LOW OOV (<5%). Widening may not be needed; minor additions only.")
    elif name_oov_rate < 0.20:
        print("-> MODERATE OOV (5-20%). Targeted widening: add the top ~10-20 "
              "missing APIs that have benign use cases.")
    else:
        print("-> HIGH OOV (>20%). Substantial widening needed. Consider whether "
              "the synthetic generator's API set was chosen too narrowly.")


if __name__ == "__main__":
    main()