"""
scripts/diagnose_flaws_cloud.py
───────────────────────────────────────────────────────────────────────────────
Read-only labellability diagnostic for flaws_merged.jsonl.

WHY THIS EXISTS: Before committing to a labelling strategy for thesis
evaluation (Day 8), we need to know how many events in the raw file can be
reliably assigned to a flaws.cloud attack level (1–6) via resource names /
ARNs, vs. events that are ambiguous or unlabelled background traffic.

PIPELINE POSITION: Pre-Tier-1. Operates on raw CloudTrail JSON only.
No labels are written, no raw data is modified.

DATA FLOW:
  data/raw/flaws_merged.jsonl  (read-only, streamed line-by-line)
          │
          ▼
  _label_event()     ← 3-rule priority labeller — no I/O, no side effects
          │
          ▼
  Counter objects    ← sole in-memory state; O(1) w.r.t. file size
          │
          ▼
  _build_report()    ← assembles JSON-serialisable summary dict
          │
          ├──► data/processed/flaws_cloud_diagnostic.json  (written once)
          └──► stdout                                       (human-readable)

LABELLING RULES (priority order, first match wins):
  Rule 1  — resource/ARN regex match → level_N | ambiguous
  Rule 1b — list-valued responseElements fields are skipped
  Rule 2  — principal userName match → level6_principal_only | backup_principal
               | flaws_principal
  Rule 3  — fallthrough → background

USAGE:
  # smoke test on first 10 000 events
  python scripts/diagnose_flaws_cloud.py --max-events 10000

  # full run (5–15 min, ~500 MB RAM peak)
  python scripts/diagnose_flaws_cloud.py

  # override paths
  RAW_FLAWS_PATH=/tmp/other.jsonl python scripts/diagnose_flaws_cloud.py

CONSTRAINTS:
  - Pure stdlib: json, re, collections, argparse, pathlib, os, sys, time.
  - No pandas, no joblib, no NumPy.
  - Stream processing: one JSON line at a time.
  - Read-only on all raw data.
  - Follows §5 conventions from CLAUDE.md.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Final

# ── Project root (scripts/ is one level below project root) ───────────────────
_SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _SCRIPT_DIR.parent

# ── Env-var-overridable I/O paths ─────────────────────────────────────────────
RAW_FLAWS_PATH: Final[Path] = Path(
    os.environ.get(
        "RAW_FLAWS_PATH",
        str(_PROJECT_ROOT / "data" / "raw" / "flaws_merged.jsonl"),
    )
)
OUTPUT_PATH: Final[Path] = Path(
    os.environ.get(
        "DIAGNOSTIC_OUTPUT",
        str(_PROJECT_ROOT / "data" / "processed" / "flaws_cloud_diagnostic.json"),
    )
)

# ── Label vocabulary ──────────────────────────────────────────────────────────
_LEVEL_LABELS: Final[tuple[str, ...]] = (
    "level_1", "level_2", "level_3", "level_4", "level_5", "level_6",
)
_ALL_LABELS: Final[tuple[str, ...]] = _LEVEL_LABELS + (
    "ambiguous",
    "level6_principal_only",
    "backup_principal",
    "flaws_principal",
    "background",
)

# ── Regex — matches "level1", "level-2", "level/3", "level4.", "level5\b" ─────
# Case-insensitive so "Level6" in ARNs / bucket names works too.
# (?:[-/.]|\b) ensures we don't match "level12" as "level1".
LEVEL_RE: Final[re.Pattern[str]] = re.compile(
    r"level([1-6])(?:[-/.]|\b)", re.IGNORECASE
)

# ── Progress ticker ───────────────────────────────────────────────────────────
_PROGRESS_EVERY: Final[int] = 200_000


# ─────────────────────────────────────────────────────────────────────────────
# Candidate collection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _harvest_response_strings(obj: dict, depth: int = 0) -> list[str]:
    """Recursively find resourceName / resourceId *string* values in a dict.

    Rule 1b: list values are skipped entirely — a list of resource IDs (e.g.,
    resourceIdentifiers in ListDiscoveredResources) would produce false
    ambiguity if scanned.  depth cap prevents runaway traversal.
    """
    if depth > 4:
        return []
    results: list[str] = []
    for k, v in obj.items():
        if k in ("resourceName", "resourceId"):
            # Only take scalar strings — skip lists/dicts (Rule 1b).
            if isinstance(v, str) and v:
                results.append(v)
        elif isinstance(v, dict):
            results.extend(_harvest_response_strings(v, depth + 1))
        # lists deliberately not recursed (Rule 1b)
    return results


def _collect_candidates(record: dict) -> list[str]:
    """Return all string candidates from one event to scan for level numbers.

    Sources (per Rule 1 spec):
      - requestParameters.bucketName
      - requestParameters.userName
      - resources[].ARN
      - scalar resourceName / resourceId anywhere in responseElements
    """
    candidates: list[str] = []

    req = record.get("requestParameters") or {}
    if isinstance(req, dict):
        for key in ("bucketName", "userName"):
            val = req.get(key)
            if isinstance(val, str) and val:
                candidates.append(val)

    for res in record.get("resources") or []:
        if isinstance(res, dict):
            # CloudTrail uses both "ARN" and occasionally "arn"
            arn = res.get("ARN") or res.get("arn")
            if isinstance(arn, str) and arn:
                candidates.append(arn)

    resp = record.get("responseElements") or {}
    if isinstance(resp, dict):
        candidates.extend(_harvest_response_strings(resp))

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Labeller
# ─────────────────────────────────────────────────────────────────────────────

def _label_event(record: dict) -> str:
    """Assign a tentative label to one CloudTrail event (no I/O, no state).

    Priority order — first match wins:
      Rule 1  : resource/ARN regex → level_N | ambiguous
      Rule 2  : principal userName → level6_principal_only | backup_principal
                                     | flaws_principal
      Rule 3  : fallthrough        → background
    """
    # ── Rule 1: scan resource candidates ─────────────────────────────────────
    matched_levels: set[str] = set()
    for cand in _collect_candidates(record):
        for m in LEVEL_RE.finditer(cand):
            matched_levels.add(m.group(1))  # digit string "1".."6"

    if len(matched_levels) == 1:
        return f"level_{matched_levels.pop()}"
    if len(matched_levels) > 1:
        return "ambiguous"

    # ── Rule 2: principal match ───────────────────────────────────────────────
    uid = record.get("userIdentity") or {}
    username: str = uid.get("userName", "") if isinstance(uid, dict) else ""
    if username == "Level6":
        return "level6_principal_only"
    if username == "backup":
        return "backup_principal"
    if username == "flaws":
        return "flaws_principal"

    # ── Rule 3: default ───────────────────────────────────────────────────────
    return "background"


def _principal_bucket(record: dict) -> str:
    """Normalise the acting principal to one of five buckets for the cross-tab."""
    uid = record.get("userIdentity") or {}
    if not isinstance(uid, dict):
        return "other"
    if uid.get("type") == "Root":
        return "Root"
    uname: str = uid.get("userName", "")
    if uname in ("Level6", "backup", "flaws"):
        return uname
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Streaming accumulator
# ─────────────────────────────────────────────────────────────────────────────

def _stream_and_count(path: Path, max_events: int | None) -> dict:
    """Stream *path* line-by-line, apply _label_event, accumulate counters.

    Memory usage is bounded by the Counter sizes — no rows are held in RAM.
    Returns a raw counters dict consumed by _build_report().
    """
    label_counts: Counter[str] = Counter({lbl: 0 for lbl in _ALL_LABELS})
    # cross-tab: label → principal bucket → count
    level_by_principal: dict[str, Counter[str]] = {
        lbl: Counter({"Root": 0, "Level6": 0, "backup": 0, "flaws": 0, "other": 0})
        for lbl in _LEVEL_LABELS
    }
    event_name_counts: Counter[str] = Counter()
    background_event_name_counts: Counter[str] = Counter()
    principal_set: set[str] = set()
    malformed: int = 0
    total: int = 0          # non-empty lines successfully attempted
    earliest: str | None = None
    latest: str | None = None

    t0 = time.time()
    print(f"[diag] streaming {path} …", flush=True)

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            # respect sampling cap before incrementing total
            if max_events is not None and total >= max_events:
                break

            line = raw_line.strip()
            if not line:
                continue  # blank lines don't count toward total

            total += 1

            # ── progress ticker ───────────────────────────────────────────────
            if total % _PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0.0
                print(f"  {total:>10,}  lines  ({rate:,.0f} lines/s)", flush=True)

            # ── parse ─────────────────────────────────────────────────────────
            try:
                record: dict = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue  # counted in malformed, not in label_counts

            # ── label ─────────────────────────────────────────────────────────
            label = _label_event(record)
            label_counts[label] += 1

            # ── cross-tab (only for resource-matched levels) ──────────────────
            if label in _LEVEL_LABELS:
                bucket = _principal_bucket(record)
                level_by_principal[label][bucket] += 1

            # ── event name counters ───────────────────────────────────────────
            ev_name: str = record.get("eventName", "<missing>")
            event_name_counts[ev_name] += 1
            if label == "background":
                background_event_name_counts[ev_name] += 1

            # ── principal tracking ────────────────────────────────────────────
            uid = record.get("userIdentity") or {}
            if isinstance(uid, dict):
                if uid.get("type") == "Root":
                    principal_set.add("__Root__")
                elif "userName" in uid:
                    principal_set.add(str(uid["userName"]))
                elif "arn" in uid:
                    # AssumedRole / federated principals have no userName
                    principal_set.add(str(uid["arn"]))

            # ── time range (ISO-8601 strings compare correctly as strings) ────
            ev_time: str | None = record.get("eventTime")
            if ev_time:
                if earliest is None or ev_time < earliest:
                    earliest = ev_time
                if latest is None or ev_time > latest:
                    latest = ev_time

    elapsed = time.time() - t0
    rate = total / elapsed if elapsed > 0 else 0.0
    print(
        f"[diag] done — {total:,} lines in {elapsed:.1f}s ({rate:,.0f} lines/s)",
        flush=True,
    )

    return {
        "total": total,
        "malformed": malformed,
        "label_counts": label_counts,
        "level_by_principal": level_by_principal,
        "event_name_counts": event_name_counts,
        "background_event_name_counts": background_event_name_counts,
        "principal_set": principal_set,
        "earliest": earliest,
        "latest": latest,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report assembly
# ─────────────────────────────────────────────────────────────────────────────

def _build_report(c: dict) -> dict:
    """Convert raw counters into the final JSON-serialisable report dict."""
    top20 = c["event_name_counts"].most_common(20)
    bg_top10 = c["background_event_name_counts"].most_common(10)

    # Invariant: every non-malformed line must receive exactly one label.
    label_sum = sum(c["label_counts"].values())
    expected = c["total"] - c["malformed"]
    if label_sum != expected:
        # Surface the inconsistency rather than silently producing bad numbers.
        raise AssertionError(
            f"label_sum={label_sum} != total-malformed={expected}. "
            "Bug in the counting logic — please file an issue."
        )

    return {
        "total_events": c["total"],
        "malformed_lines": c["malformed"],
        "time_range": {
            "earliest": c["earliest"],
            "latest": c["latest"],
        },
        "label_counts": dict(c["label_counts"]),
        "level_by_principal": {
            lbl: dict(counter)
            for lbl, counter in c["level_by_principal"].items()
        },
        "unique_principal_count": len(c["principal_set"]),
        "top_20_event_names": top20,
        "background_top_10_event_names": bg_top10,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(report: dict) -> None:
    """Print a formatted summary of the diagnostic report to stdout."""
    W = 72
    print("\n" + "═" * W)
    print("  flaws_merged.jsonl — Labellability Diagnostic")
    print("═" * W)

    labelled_total = report["total_events"] - report["malformed_lines"]
    print(f"\n  Total events        : {report['total_events']:>12,}")
    print(f"  Malformed lines     : {report['malformed_lines']:>12,}")
    print(f"  Labelled events     : {labelled_total:>12,}")
    print(f"  Unique principals   : {report['unique_principal_count']:>12,}")
    tr = report["time_range"]
    print(f"  Earliest event      : {tr['earliest']}")
    print(f"  Latest event        : {tr['latest']}")

    print(f"\n{'─' * W}")
    print("  Label Counts")
    print(f"{'─' * W}")
    for lbl in _ALL_LABELS:
        n = report["label_counts"].get(lbl, 0)
        pct = 100.0 * n / labelled_total if labelled_total else 0.0
        bar = "█" * int(pct / 2)  # rough 50-char scale
        print(f"  {lbl:<28}  {n:>9,}  ({pct:6.2f}%)  {bar}")

    print(f"\n{'─' * W}")
    print("  Level × Principal Cross-Tab (resource-matched events only)")
    print(f"{'─' * W}")
    hdr = f"  {'label':<12}  {'Root':>9}  {'Level6':>9}  {'backup':>9}  {'flaws':>9}  {'other':>9}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for lbl in _LEVEL_LABELS:
        row = report["level_by_principal"].get(lbl, {})
        print(
            f"  {lbl:<12}  "
            f"{row.get('Root', 0):>9,}  "
            f"{row.get('Level6', 0):>9,}  "
            f"{row.get('backup', 0):>9,}  "
            f"{row.get('flaws', 0):>9,}  "
            f"{row.get('other', 0):>9,}"
        )

    print(f"\n{'─' * W}")
    print("  Top 20 Event Names (all events)")
    print(f"{'─' * W}")
    for rank, (name, cnt) in enumerate(report["top_20_event_names"], 1):
        print(f"  {rank:>2}. {name:<44} {cnt:>9,}")

    print(f"\n{'─' * W}")
    print("  Top 10 Event Names (background events only)")
    print(f"{'─' * W}")
    for rank, (name, cnt) in enumerate(report["background_top_10_event_names"], 1):
        print(f"  {rank:>2}. {name:<44} {cnt:>9,}")

    print("\n" + "═" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only labellability diagnostic for flaws_merged.jsonl. "
            "Streams the file, applies a 3-rule labelling heuristic, "
            "and writes a count summary to data/processed/. "
            "Does NOT modify any raw data."
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
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        metavar="PATH",
        help=f"Output JSON path  (default: {OUTPUT_PATH})",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N non-empty lines (smoke-test mode).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: D401
    args = _parse_args(argv)

    if not args.raw_path.exists():
        print(f"ERROR: input file not found: {args.raw_path}", file=sys.stderr)
        return 1

    if args.max_events is None:
        print(
            "WARNING: --max-events not set — processing full ~1.9M-line file. "
            "Expected runtime: 5–15 min on a CPU-only machine.",
            flush=True,
        )
    else:
        print(
            f"[diag] smoke-test mode — capped at first {args.max_events:,} events.",
            flush=True,
        )

    counters = _stream_and_count(args.raw_path, args.max_events)
    report = _build_report(counters)

    _print_report(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"[diag] report written → {args.output}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
