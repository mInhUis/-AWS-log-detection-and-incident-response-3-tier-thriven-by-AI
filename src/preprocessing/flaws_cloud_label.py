"""
src/preprocessing/flaws_cloud_label.py
───────────────────────────────────────────────────────────────────────────────
Resource-based event labeller for raw flaws.cloud CloudTrail data.

PIPELINE POSITION:
  Sits before Tier 1.  Called by the split script
  (scripts/run_flaws_cloud_split.py) to route raw events from
  flaws_merged.jsonl into per-level JSONL files for val and test.
  Also imported by the diagnostic (scripts/diagnose_flaws_cloud.py)
  so that both tools share a single, tested labelling implementation.

WHY RESOURCE-BASED LABELLING?
  flaws.cloud embeds the challenge level in every resource name it
  creates: S3 buckets ("level1.flaws.cloud"), IAM users
  ("CTF_challenge_level4"), ARNs, etc.  Scanning those strings with a
  regex is the only fully deterministic way to assign a level label —
  principal-name heuristics (Rule 2) are used only as a fallback for
  events that touch shared infrastructure with no level-bearing name.

  This approach is documented in the thesis Methodology §4.2
  ("Dataset Labelling Strategy") and was validated by the read-only
  diagnostic in scripts/diagnose_flaws_cloud.py.

LABELLING RULES (priority order — first match wins):
  Rule 1   Resource / ARN regex match
           Candidates: requestParameters.bucketName,
                       requestParameters.userName,
                       resources[].ARN,
                       scalar resourceName/resourceId in responseElements
           Regex: LEVEL_RE  (see constant below)
           If one distinct digit → "level_N"
           If two+ distinct digits → "ambiguous"
  Rule 1b  List-valued responseElements fields are SKIPPED to prevent
           false ambiguity (e.g. ListDiscoveredResources returns a list
           of resource IDs that may span multiple levels).
  Rule 2   userIdentity.userName match (fallback when Rule 1 gives nothing)
           "Level6"  → "level6_principal_only"
           "backup"  → "backup_principal"
           "flaws"   → "flaws_principal"
  Rule 3   Fallthrough → "background"

PUBLIC API:
  label_event(event: dict) -> str
  collect_candidates(event: dict) -> list[str]
  harvest_response_strings(obj: dict, depth: int = 0) -> list[str]
  LEVEL_RE, ALL_LABELS, LEVEL_LABELS

These functions are pure (no I/O, no side effects) and safe to call
inside streaming loops over multi-million-line files.
"""
from __future__ import annotations

import re
from typing import Final

# ── Label vocabulary ──────────────────────────────────────────────────────────
# Ordered from most-specific to least-specific for documentation purposes.
# The split script uses these constants to decide which output file to write to.
LEVEL_LABELS: Final[tuple[str, ...]] = (
    "level_1",
    "level_2",
    "level_3",
    "level_4",
    "level_5",
    "level_6",
)

ALL_LABELS: Final[tuple[str, ...]] = LEVEL_LABELS + (
    "ambiguous",           # resource match with conflicting level numbers
    "level6_principal_only",  # Rule 2 fallback: userName == "Level6"
    "backup_principal",    # Rule 2 fallback: userName == "backup"
    "flaws_principal",     # Rule 2 fallback: userName == "flaws"
    "background",          # Rule 3: no level signal found
)

# ── Core regex ────────────────────────────────────────────────────────────────
# Matches "level1", "level-2", "level/3", "level4.", "level5\b", etc.
# Case-insensitive so "Level6" in IAM user names works alongside
# lower-case "level1.flaws.cloud" in S3 bucket names.
#
# (?:[-/.]|\b) prevents matching "level12" as "level1":
#   - [-/.] catches explicit separators used in bucket/ARN naming
#   - \b catches word-boundary cases (end of string, space, etc.)
#   - "level12" → digit "1" is followed by "2" (a \w char, not a boundary),
#     so neither branch matches — correct, "level12" is not a flaws.cloud level.
LEVEL_RE: Final[re.Pattern[str]] = re.compile(
    r"level([1-6])(?:[-/.]|\b)", re.IGNORECASE
)

# ── Recursion depth cap for responseElements traversal ───────────────────────
_MAX_RESP_DEPTH: Final[int] = 4


# ─────────────────────────────────────────────────────────────────────────────
# Pure helper functions
# ─────────────────────────────────────────────────────────────────────────────

def harvest_response_strings(obj: dict, depth: int = 0) -> list[str]:
    """Recursively collect scalar resourceName / resourceId values from a dict.

    Rule 1b enforcement: list values are SKIPPED at every level of recursion.
    This prevents false "ambiguous" labels when an API like
    ListDiscoveredResources returns a list of resource IDs that may span
    multiple challenge levels.

    Args:
        obj:   A dict (typically responseElements or a nested sub-dict).
        depth: Current recursion depth; capped at _MAX_RESP_DEPTH to prevent
               runaway traversal on pathologically deep records.

    Returns:
        A list of non-empty strings found at keys "resourceName" or
        "resourceId", anywhere in the dict tree, excluding list-valued entries.
    """
    if depth > _MAX_RESP_DEPTH:
        return []

    results: list[str] = []
    for k, v in obj.items():
        if k in ("resourceName", "resourceId"):
            # Rule 1b: accept only scalar strings, never lists or dicts.
            if isinstance(v, str) and v:
                results.append(v)
        elif isinstance(v, dict):
            # Recurse into nested dicts — but NOT into lists (Rule 1b).
            results.extend(harvest_response_strings(v, depth + 1))
    return results


def collect_candidates(event: dict) -> list[str]:
    """Return all string candidates from one event to scan with LEVEL_RE.

    Candidate sources (Rule 1 spec, §4.2 Methodology):
      1. requestParameters.bucketName  — S3 bucket names like "level2.flaws.cloud"
      2. requestParameters.userName    — IAM user names like "CTF_level3_user"
      3. resources[].ARN               — resource ARNs embedded in the event
      4. scalar resourceName/resourceId in responseElements (Rule 1b: no lists)

    Args:
        event: A single parsed CloudTrail record (dict from json.loads).

    Returns:
        A flat list of non-empty strings; may be empty if the event has no
        resource-identifying fields.
    """
    candidates: list[str] = []

    # ── requestParameters ─────────────────────────────────────────────────────
    req = event.get("requestParameters") or {}
    if isinstance(req, dict):
        for key in ("bucketName", "userName"):
            val = req.get(key)
            if isinstance(val, str) and val:
                candidates.append(val)

    # ── resources[] ARNs ─────────────────────────────────────────────────────
    for res in event.get("resources") or []:
        if isinstance(res, dict):
            # CloudTrail capitalises "ARN" in the resources array.
            arn = res.get("ARN") or res.get("arn")
            if isinstance(arn, str) and arn:
                candidates.append(arn)

    # ── responseElements (Rule 1b: lists skipped inside) ─────────────────────
    resp = event.get("responseElements") or {}
    if isinstance(resp, dict):
        candidates.extend(harvest_response_strings(resp))

    return candidates


def label_event(event: dict) -> str:
    """Assign a level label to one raw CloudTrail event.

    This is the single authoritative labelling implementation used by both
    the diagnostic (scripts/diagnose_flaws_cloud.py) and the split script
    (scripts/run_flaws_cloud_split.py).  Keeping the rule in one place
    guarantees that the counts reported by the diagnostic exactly match the
    events written to the val/test JSONL files.

    Rules applied in priority order — first match wins:
      Rule 1  : resource/ARN regex → "level_N" or "ambiguous"
      Rule 2  : principal userName fallback
      Rule 3  : "background"

    Args:
        event: A single parsed CloudTrail record (dict).

    Returns:
        One label string from ALL_LABELS.  Never raises; malformed/missing
        fields are handled gracefully by the candidate-collection helpers.
    """
    # ── Rule 1: scan resource candidates for level digits ────────────────────
    matched_levels: set[str] = set()
    for candidate in collect_candidates(event):
        for m in LEVEL_RE.finditer(candidate):
            matched_levels.add(m.group(1))  # digit string "1" … "6"

    if len(matched_levels) == 1:
        # Exactly one distinct level digit found — unambiguous match.
        return f"level_{matched_levels.pop()}"

    if len(matched_levels) > 1:
        # Two or more conflicting level digits in the same event's resources.
        # Counted separately in the diagnostic; excluded from val/test splits.
        return "ambiguous"

    # ── Rule 2: principal userName fallback ──────────────────────────────────
    # Only reached when Rule 1 found zero candidates.
    uid = event.get("userIdentity") or {}
    username: str = uid.get("userName", "") if isinstance(uid, dict) else ""

    if username == "Level6":
        return "level6_principal_only"
    if username == "backup":
        return "backup_principal"
    if username == "flaws":
        return "flaws_principal"

    # ── Rule 3: default ───────────────────────────────────────────────────────
    return "background"
