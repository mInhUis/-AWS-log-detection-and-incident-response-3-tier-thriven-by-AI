"""Reverse-map Drain3 integer log-key IDs to human-readable CloudTrail event info.

WHY THIS MODULE EXISTS (pipeline position: Tier 2 → Tier 3 bridge):
    DeepLog's LSTM operates on integer log-key IDs — it never sees event names,
    ARNs, or timestamps.  This is correct: mathematical anomaly detection should
    be blind to semantics.  BUT Tier 3 (Llama 3.1) needs *human-readable
    context* to write a useful incident report.  This module bridges the gap.

    Given the Drain3 vocabulary (integer key ID → template string, e.g.
    0 → "CreateUser <*>"), the per-event flags from DeepLog, and the original
    raw CloudTrail event dicts, it reconstructs the human-readable event
    information for every anomalous event and enriches it with timestamp,
    source IP, user ARN, error code, and target resource.

DATA FLOW (CLAUDE.md §3):
    Tier 2 output: list[bool] flags + list[int] log_keys (per session)
        → reverse_map_events()
        → list[dict] with drain_template, event_name, timestamp,
          source_ip, user_arn, error_code, target_resource
        → Tier 3: LlamaResponder.generate_report()
"""

from __future__ import annotations

from typing import Any


def reverse_map_events(
    templates: dict[int, str],
    log_keys: list[int],
    flags: list[bool],
    raw_events: list[dict[str, Any]],
    oov_key: int,
) -> list[dict[str, Any]]:
    """Reconstruct human-readable event info for every Tier-2-flagged position.

    Iterates over (flag, log_key, raw_event) triples in lock-step.  For each
    True flag it:
      1. Looks up the Drain3 template string (int key → "EventName <*>").
      2. Extracts CloudTrail metadata from the raw event dict.  Handles both
         the flat synthetic schema and the nested AWS official schema using
         the same structural heuristics as normalize.py.
      3. Returns a structured dict ready for the LlamaResponder context.

    OOV handling:
        A log_key >= oov_key means the event was not in the Drain3 vocabulary
        at training time.  _detect_anomalies_safe already flagged it True in
        the "unknown current event" branch.  We surface it with the sentinel
        drain_template "<OOV: unknown event>" so the analyst sees it clearly.

    Args:
        templates:  dict[int, str] from DrainParser.get_templates().
                    Key: 0-based Drain3 cluster ID.  Value: template text
                    such as "CreateUser <*>".
        log_keys:   Integer log-key IDs for the session (same length as flags).
                    Produced by DrainParser.transform().
        flags:      Per-event anomaly flags from _detect_anomalies_safe().
                    Same length as log_keys.  True = DeepLog flagged as anomaly.
        raw_events: Original raw CloudTrail event dicts, one per position.
                    Must be in the same order as log_keys and flags.
        oov_key:    OOV sentinel value (= DrainParser.oov_key = num_keys).
                    Any log_key >= oov_key is treated as an unknown event.

    Returns:
        List of enriched dicts, one per flagged event (len <= len(flags)).
        Each dict has keys:
          "drain_template"  — Drain3 template string or OOV sentinel
          "event_name"      — CloudTrail eventName field
          "timestamp"       — eventTime ISO-8601 string
          "source_ip"       — sourceIPAddress field
          "user_arn"        — userIdentity.arn (nested) or principalId (flat)
          "error_code"      — errorCode string, or None on success
          "target_resource" — best-effort resource name from requestParameters

    Raises:
        ValueError: If the three primary input lists have different lengths.
    """
    if not (len(log_keys) == len(flags) == len(raw_events)):
        raise ValueError(
            f"Length mismatch: log_keys={len(log_keys)}, "
            f"flags={len(flags)}, raw_events={len(raw_events)}. "
            "All three must have the same length (one entry per session event)."
        )

    result: list[dict[str, Any]] = []

    for key_id, is_anomalous, raw in zip(log_keys, flags, raw_events):
        if not is_anomalous:
            continue  # normal event — not forwarded to Tier 3

        # ---- Drain template lookup ----------------------------------------
        # OOV keys (>= oov_key) were not in the vocabulary during training.
        # Surface them with an explicit sentinel so the analyst is not misled
        # by a missing template.
        if key_id >= oov_key:
            drain_template: str = "<OOV: unknown event>"
        else:
            # Missing from templates means the DrainParser state was not
            # fully saved; surface a key-ID label rather than silently hiding it.
            drain_template = templates.get(
                key_id, f"<key_{key_id}: template missing>"
            )

        result.append({
            "drain_template":  drain_template,
            "event_name":      _extract_event_name(raw),
            # Timestamp: try raw CloudTrail camelCase, then processed snake_case
            "timestamp":       (
                raw.get("eventTime")
                or raw.get("event_time")
                or "N/A"
            ),
            # Source IP: CloudTrail uses sourceIPAddress; processed uses source_ip
            "source_ip":       (
                raw.get("sourceIPAddress")
                or raw.get("source_ip")
                or "N/A"
            ),
            "user_arn":        _extract_user_arn(raw),
            # errorCode vs error_code; treat empty string as success (None)
            "error_code":      (
                raw.get("errorCode") or raw.get("error_code") or None
            ),
            "target_resource": _extract_target_resource(raw),
        })

    return result


# ---------------------------------------------------------------------------
#  Private extraction helpers
# ---------------------------------------------------------------------------


def _extract_event_name(raw: dict[str, Any]) -> str:
    """Return the event name from either camelCase or snake_case schema."""
    return str(raw.get("eventName") or raw.get("event_name") or "<unknown>")


def _extract_user_arn(raw: dict[str, Any]) -> str:
    """Extract the canonical user identity string from any supported schema.

    Preference order:
      1. userIdentity.arn      (nested AWS raw CloudTrail schema)
      2. userIdentity.principalId / userName  (nested schema fallback)
      3. principalId           (flat camelCase synthetic schema)
      4. principal_id          (processed snake_case schema from ingest.py)
    """
    uid = raw.get("userIdentity")
    if isinstance(uid, dict):
        arn = uid.get("arn")
        if arn:
            return str(arn)
        pid = uid.get("principalId") or uid.get("userName")
        if pid:
            return str(pid)
    pid_camel = raw.get("principalId")
    if pid_camel:
        return str(pid_camel)
    # Processed schema: principal_id is already a human-readable ID string.
    pid_snake = raw.get("principal_id")
    if pid_snake:
        return str(pid_snake)
    return "N/A"


def _extract_target_resource(raw: dict[str, Any]) -> str | None:
    """Best-effort extraction of the resource targeted by this API call.

    CloudTrail's requestParameters is service-specific — no universal field
    name exists.  We probe the most common candidates in order of specificity.
    This covers the IAM, S3, EC2, Lambda, DynamoDB, and Secrets Manager
    patterns that appear in the flaws.cloud attack scenarios.

    Returns None if requestParameters is absent (common on read-only calls
    such as GetCallerIdentity that target no specific resource).
    """
    params = raw.get("requestParameters")
    if not isinstance(params, dict):
        return None

    # High-specificity field names first — first match wins.
    _CANDIDATE_FIELDS: tuple[str, ...] = (
        "bucketName",    # S3: PutObject, GetObject, PutBucketPolicy
        "userName",      # IAM: CreateUser, AttachUserPolicy
        "roleName",      # IAM: CreateRole, AttachRolePolicy
        "groupName",     # IAM: AddUserToGroup, AttachGroupPolicy
        "policyName",    # IAM: CreatePolicy
        "instanceId",    # EC2: various instance actions
        "snapshotId",    # EC2: ModifySnapshotAttribute
        "functionName",  # Lambda
        "tableName",     # DynamoDB
        "secretId",      # Secrets Manager: GetSecretValue
        "trailName",     # CloudTrail: StopLogging, DeleteTrail
        "key",           # S3: PutObject key name
        "name",          # generic fallback
    )
    for field in _CANDIDATE_FIELDS:
        val = params.get(field)
        if val and isinstance(val, str):
            return val

    # Last resort: stringify the entire params dict (truncated to avoid
    # filling the LLM prompt with a wall of JSON).
    params_str: str = str(params)
    return params_str[:120] + ("…" if len(params_str) > 120 else "")
