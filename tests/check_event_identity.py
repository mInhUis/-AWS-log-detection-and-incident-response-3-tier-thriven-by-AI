
import json
from pathlib import Path

# Check synthetic event identity fields
print("=== SYNTHETIC (3 events) ===")
with open("data/processed/synthetic_val.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 3: break
        ev = json.loads(line.strip())
        uid = ev.get("userIdentity", {})
        print(f"--- event {i} ---")
        if isinstance(uid, dict):
            for k, v in uid.items():
                print(f"  userIdentity.{k} = {v!r}")
        else:
            print(f"  userIdentity = {uid!r}")
        # Also check top-level fields that might be the key
        for field in ("principalId", "accountId", "entity_id", "principal_id", "source_ip"):
            if field in ev:
                print(f"  top-level {field} = {ev[field]!r}")
        print()

# Check flaws event identity fields
print("=== FLAWS.CLOUD (3 events) ===")
with open("data/processed/flaws_cloud_val.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 3: break
        ev = json.loads(line.strip())
        uid = ev.get("userIdentity", {})
        print(f"--- event {i} ---")
        if isinstance(uid, dict):
            for k, v in uid.items():
                print(f"  userIdentity.{k} = {v!r}")
        print()