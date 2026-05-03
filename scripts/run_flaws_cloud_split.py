"""
scripts/run_flaws_cloud_split.py
───────────────────────────────────────────────────────────────────────────────
Apply the resource-based labelling rule to the raw flaws.cloud JSONL file and
route events into per-split output files.

PIPELINE POSITION:
  Runs once, offline, before any model training.  Its outputs feed directly
  into Tier 1 and Tier 2 evaluation — NOT into training (which uses
  synthetic normal data only).

  data/raw/flaws_merged.jsonl    (read-only, 1.94M events, ~2.3 GB)
          │
          ▼  label_event()  from src.preprocessing.flaws_cloud_label
          │
          ├── label ∈ {level_3, level_4}  →  data/processed/flaws_cloud_val.jsonl
          ├── label ∈ {level_2, level_5}  →  data/processed/flaws_cloud_test.jsonl
          └── everything else             →  skipped (not written to disk)

LOCKED SPLIT DECISIONS (do not modify without thesis committee approval):
  Val   : levels 3 + 4  (~5 828 events)
  Test  : levels 2 + 5  (~5 599 events)
  Dropped: level 1 (only 8 events — too thin for evaluation)
           level 6 (3.5-year span, 58 principals, post-attack credential
                    reuse — too contaminated for clean labelling)
  Within each kept level ALL resource-matched events are retained,
  regardless of which IAM principal executed them.

WHY NOT ADD _source HERE?
  ingest.load_events() is the single place that stamps _source on every
  event.  When downstream code loads flaws_cloud_val.jsonl it passes
  source="flaws_cloud" and gets the tag for free.  Adding it here would
  create a second source of truth and risk double-tagging.

USAGE:
  # smoke test — processes first 5 000 non-empty lines (default)
  python scripts/run_flaws_cloud_split.py --mode mock --force

  # full run — all 1.94M events (~5–15 min on CPU)
  python scripts/run_flaws_cloud_split.py --mode full --force

  # override input path via env var
  RAW_FLAWS_PATH=/data/other.jsonl python scripts/run_flaws_cloud_split.py --mode full

CONVENTIONS: §5 CLAUDE.md — Final constants, env-var paths, type hints,
  mock-mode default, --force guard, argparse, raise SystemExit(main()).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Final

# ── Project root — needed so `src.*` imports work when run as a script ────────
_SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.flaws_cloud_label import label_event  # noqa: E402

# ── Env-var-overridable paths ─────────────────────────────────────────────────
RAW_FLAWS_PATH: Final[Path] = Path(
    os.environ.get(
        "RAW_FLAWS_PATH",
        str(_PROJECT_ROOT / "data" / "raw" / "flaws_merged.jsonl"),
    )
)
PROCESSED_DIR: Final[Path] = Path(
    os.environ.get(
        "PROCESSED_DIR",
        str(_PROJECT_ROOT / "data" / "processeded1"),
    )
)

# ── Split routing table (locked — matches thesis methodology §4.2) ────────────
# Changing these constants changes the evaluation dataset — requires
# re-running the full diagnostic and updating the split manifest.
VAL_LABELS: Final[frozenset[str]] = frozenset({"level_3", "level_4"})
TEST_LABELS: Final[frozenset[str]] = frozenset({"level_2", "level_5"})

# ── Mock mode cap ─────────────────────────────────────────────────────────────
_MOCK_MAX_EVENTS: Final[int] = 5_000   # §5 #6 — fast smoke test
_PROGRESS_EVERY: Final[int] = 200_000  # print a line every N events in full mode


# ─────────────────────────────────────────────────────────────────────────────
# Core streaming function
# ─────────────────────────────────────────────────────────────────────────────

def run_split(
    raw_path: Path,
    val_path: Path,
    test_path: Path,
    max_events: int | None,
) -> dict:
    """Stream raw_path, label every event, write to val_path or test_path.

    Events are written verbatim — the original JSON string is preserved
    exactly as it appeared on disk (modulo leading/trailing whitespace).
    No fields are added or removed.

    Args:
        raw_path:   Path to flaws_merged.jsonl (read-only).
        val_path:   Destination for level_3 + level_4 events.
        test_path:  Destination for level_2 + level_5 events.
        max_events: If given, stop after this many non-empty lines
                    (including malformed ones that fail json.loads).

    Returns:
        A summary dict with keys: total, malformed, written_val,
        written_test, label_counts.
    """
    label_counts: Counter[str] = Counter()
    malformed: int = 0
    total: int = 0       # non-empty lines attempted
    written_val: int = 0
    written_test: int = 0

    t0 = time.time()
    print(f"[split] streaming {raw_path.name} …", flush=True)

    with (
        open(raw_path, encoding="utf-8", errors="replace") as src,
        open(val_path, "w", encoding="utf-8") as val_fh,
        open(test_path, "w", encoding="utf-8") as test_fh,
    ):
        for raw_line in src:
            # ── mock mode cap ─────────────────────────────────────────────────
            if max_events is not None and total >= max_events:
                break

            line = raw_line.strip()
            if not line:
                continue  # blank lines in JSONL — skip without counting

            total += 1

            # ── progress ticker (only useful in full mode) ────────────────────
            if total % _PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0.0
                print(f"  {total:>10,}  lines  ({rate:,.0f} lines/s)", flush=True)

            # ── parse ─────────────────────────────────────────────────────────
            try:
                record: dict = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue  # counted but not routed

            # ── label and route ───────────────────────────────────────────────
            lbl = label_event(record)
            label_counts[lbl] += 1

            if lbl in VAL_LABELS:
                # Write the original JSON string verbatim — no reformatting,
                # no added fields.  ingest.load_events will add _source later.
                val_fh.write(line + "\n")
                written_val += 1
            elif lbl in TEST_LABELS:
                test_fh.write(line + "\n")
                written_test += 1
            # else: skip — background, ambiguous, principal-only labels

    elapsed = time.time() - t0
    rate = total / elapsed if elapsed > 0 else 0.0
    print(
        f"[split] done — {total:,} lines in {elapsed:.1f}s ({rate:,.0f} lines/s)",
        flush=True,
    )

    return {
        "total_processed": total,
        "malformed": malformed,
        "written_val": written_val,
        "written_test": written_test,
        "label_counts": dict(label_counts),
    }


def _print_summary(summary: dict, val_path: Path, test_path: Path) -> None:
    """Print a human-readable summary of the split run to stdout."""
    W = 68
    print("\n" + "=" * W)
    print("  flaws_cloud split -- results")
    print("=" * W)
    print(f"  Total lines processed : {summary['total_processed']:>10,}")
    print(f"  Malformed JSON lines  : {summary['malformed']:>10,}")
    print(f"  Written -> val        : {summary['written_val']:>10,}  ({val_path.name})")
    print(f"  Written -> test       : {summary['written_test']:>10,}  ({test_path.name})")

    print("\n" + "-" * W)
    print("  Per-label counts (all processed events)")
    print("-" * W)
    total = summary["total_processed"] - summary["malformed"]
    for lbl, n in sorted(summary["label_counts"].items()):
        pct = 100.0 * n / total if total else 0.0
        tag = ""
        if lbl in VAL_LABELS:
            tag = "  <- val"
        elif lbl in TEST_LABELS:
            tag = "  <- test"
        print(f"  {lbl:<28}  {n:>8,}  ({pct:5.2f}%){tag}")

    print("\n" + "=" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Label and split the raw flaws.cloud JSONL into val and test files. "
            "Streams flaws_merged.jsonl, applies resource-based labelling, and "
            "writes level_3+4 → flaws_cloud_val.jsonl, level_2+5 → flaws_cloud_test.jsonl."
        ),
    )
    p.add_argument(
        "--mode",
        choices=("mock", "full"),
        default="mock",
        help=(
            "mock (default): process first %(const)s events — fast smoke test. "
            "full: process all 1.94M events (~5–15 min)."
        ),
    )
    p.add_argument(
        "--raw-path",
        type=Path,
        default=RAW_FLAWS_PATH,
        metavar="PATH",
        help=f"Path to flaws_merged.jsonl  (default: {RAW_FLAWS_PATH})",
    )
    p.add_argument(
        "--val-output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override output path for val JSONL (default: PROCESSED_DIR/flaws_cloud_val.jsonl)",
    )
    p.add_argument(
        "--test-output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override output path for test JSONL (default: PROCESSED_DIR/flaws_cloud_test.jsonl)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # ── Resolve output paths ──────────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    val_path: Path = args.val_output or (PROCESSED_DIR / "flaws_cloud_val.jsonl")
    test_path: Path = args.test_output or (PROCESSED_DIR / "flaws_cloud_test.jsonl")

    # ── Input guard ───────────────────────────────────────────────────────────
    if not args.raw_path.exists():
        print(f"ERROR: input file not found: {args.raw_path}", file=sys.stderr)
        return 1

    # ── Stale-artifact guard (§5 #8) ──────────────────────────────────────────
    # Fail loudly rather than silently appending to an existing file, which
    # would inflate event counts and corrupt the evaluation split.
    existing = [p for p in (val_path, test_path) if p.exists()]
    if existing and not args.force:
        print(
            "ERROR: output file(s) already exist — use --force to overwrite:\n"
            + "\n".join(f"  {p}" for p in existing),
            file=sys.stderr,
        )
        print(
            "  Hint: python scripts/run_flaws_cloud_split.py --mode mock --force",
            file=sys.stderr,
        )
        return 1

    # ── Mode banner ───────────────────────────────────────────────────────────
    max_events: int | None
    if args.mode == "mock":
        max_events = _MOCK_MAX_EVENTS
        print(
            f"[split] mock mode — capped at first {_MOCK_MAX_EVENTS:,} non-empty lines.",
            flush=True,
        )
    else:
        max_events = None
        print(
            "WARNING: --mode full — processing all ~1.94M events. "
            "Expected runtime: 5–15 min on a CPU-only machine.",
            flush=True,
        )

    print(f"[split] val  -> {val_path}", flush=True)
    print(f"[split] test -> {test_path}", flush=True)

    # ── Run ───────────────────────────────────────────────────────────────────
    summary = run_split(
        raw_path=args.raw_path,
        val_path=val_path,
        test_path=test_path,
        max_events=max_events,
    )

    _print_summary(summary, val_path, test_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
