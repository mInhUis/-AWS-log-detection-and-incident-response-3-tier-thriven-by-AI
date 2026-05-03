"""CloudTrail ingestion: read JSONL, normalise records, surface raw events.

Stage zero of the pipeline. Loads raw CloudTrail logs (synthetic normal +
flaws.cloud attacks) from disk, validates the minimal field set the rest of
the pipeline depends on (eventTime, eventName, userIdentity, sourceIPAddress,
errorCode), and yields normalised dictionaries. No filtering or feature
engineering happens here — that lives in `src.preprocessing`.
"""
