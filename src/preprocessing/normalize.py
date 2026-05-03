"""
src/preprocessing/normalize.py

Schema adapter that converts a raw CloudTrail event dict into a canonical
flat shape used everywhere downstream (Drain3 rendering, Tier 1 feature
aggregation, the Tier 1 -> Tier 2 bridge).

Why this exists
---------------
We have two upstream schemas:

1. Synthetic training data (`synthetic_train.jsonl`) is FLAT:
   {"entityType": "AssumedRole", "principalId": "AIDA...", "eventName": ...,
    "eventSource": ..., "errorCode": "", ...}
   `errorCode` is always present; empty string on success.

2. flaws.cloud validation/test data follows AWS's official NESTED schema:
   {"userIdentity": {"type": "Root", "principalId": "...", ...},
    "eventName": ..., "eventSource": ..., ...}
   `errorCode` is OMITTED on success rather than empty.

Without normalisation, `render_event` would emit different log key strings
for what is conceptually the same event, causing mass OOV at Drain3
transform time and breaking DeepLog evaluation. See CLAUDE.md sec 3 for
where this sits in the pipeline.

Pipeline position
-----------------
Raw JSON -> normalize_event() -> [render_event() | tier1 features] -> Drain3

This module is pure: no I/O, no global state. Easy to unit-test.
"""

from __future__ import annotations

from typing import Any, Final, TypedDict

# Sentinel for fields we couldn't extract. Using a non-empty string
# rather than None or "" is deliberate: it survives JSON round-trips,
# is visually distinct in logs, and won't be confused with a real
# CloudTrail value (CloudTrail never emits "<unknown>").
_UNKNOWN: Final[str] = "<unknown>"

# Empty-string convention for errorCode: on success CloudTrail's nested
# schema omits the field; we project that to "" so downstream code can
# treat "" as "success" uniformly across both sources.
_NO_ERROR: Final[str] = ""


class NormalizedEvent(TypedDict):
    """Canonical shape after normalisation. Six fields, all strings.

    Strings (not Optional[str]) because Drain3 wants string templates and
    None would force every consumer to handle the missing case. Sentinels
    are explicit and uniform.
    """
    entityType: str       # e.g., "AssumedRole", "Root", "IAMUser"
    principalId: str      # e.g., "AIDAEXAMPLE", "AROAEXAMPLE:session-name"
    eventName: str        # e.g., "ListBuckets", "AssumeRole"
    eventSource: str      # e.g., "s3.amazonaws.com"
    errorCode: str        # "" on success; e.g., "AccessDenied" on failure
    eventTime: str        # ISO-8601; needed for windowing in Tier 1


def normalize_event(raw: dict[str, Any]) -> NormalizedEvent:
    """Project a raw CloudTrail event into the canonical schema.

    Detects which schema (flat vs nested) by the presence of `userIdentity`.
    Synthetic events have NO `userIdentity` key (they're flat); AWS-shaped
    events have it as a dict. This is a structural check, not a content
    check, which makes it robust to weird values inside userIdentity.

    Parameters
    ----------
    raw : dict[str, Any]
        A single CloudTrail event as parsed from JSON. Either schema.

    Returns
    -------
    NormalizedEvent
        Six-field canonical shape. All values are strings; missing fields
        become "<unknown>" (or "" for errorCode on success).

    Notes
    -----
    We do NOT validate that the event is well-formed beyond extracting
    these six fields. Garbage-in produces "<unknown>"-heavy output, which
    will be visible in the entity-type diagnostic.
    """
    # Structural detection. dict typecheck guards against the (rare) case
    # where userIdentity exists as a string or null in malformed data.
    user_identity = raw.get("userIdentity")
    is_nested = isinstance(user_identity, dict)

    if is_nested:
        # AWS official schema: userIdentity is the principal container.
        # `.type` and `.principalId` are the documented fields.
        # See: docs.aws.amazon.com/awscloudtrail/latest/userguide/cloudtrail-event-reference-user-identity.html
        entity_type = user_identity.get("type", _UNKNOWN)
        principal_id = user_identity.get("principalId", _UNKNOWN)
    else:
        # Synthetic flat schema: top-level entityType / principalId.
        entity_type = raw.get("entityType", _UNKNOWN)
        principal_id = raw.get("principalId", _UNKNOWN)

    # eventName / eventSource are top-level in BOTH schemas, so a single
    # lookup works. Default to _UNKNOWN rather than raising: a malformed
    # event should be visible in counters, not crash the pipeline.
    event_name = raw.get("eventName", _UNKNOWN)
    event_source = raw.get("eventSource", _UNKNOWN)

    # errorCode unification: synthetic uses "" on success, nested omits.
    # We project both to "" so downstream code reads `errorCode == ""` as
    # "success" uniformly. .get() with default _NO_ERROR handles the
    # nested case; the synthetic case already has "" by construction.
    error_code = raw.get("errorCode", _NO_ERROR)

    # eventTime: needed by Tier 1 for windowing. Both schemas use the
    # same field name and ISO-8601 format, so this is just defensive.
    event_time = raw.get("eventTime", _UNKNOWN)

    # Coerce to str. Defensive against rare malformed values where a
    # field is e.g. an int. Drain3 will reject non-strings later anyway;
    # better to fail loudly here than mysteriously inside the parser.
    return NormalizedEvent(
        entityType=str(entity_type),
        principalId=str(principal_id),
        eventName=str(event_name),
        eventSource=str(event_source),
        errorCode=str(error_code),
        eventTime=str(event_time),
    )