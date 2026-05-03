"""MITRE ATT&CK STIX + flaws.cloud writeup corpus loader (Day 4).

**Why one-document-per-technique?**
MITRE publishes ATT&CK in STIX 2.1 format — a single JSON bundle containing
~700 technique nodes.  Indexing the whole bundle as one document would make
retrieval useless: every query would return the same document.  Instead we
split it into one ``AttackDocument`` per technique so ChromaDB can retrieve
the single most relevant technique at query time.

**Why include flaws.cloud level writeups in the corpus?**
The base Llama-3-8B model has seen ATT&CK in pre-training only approximately.
For the specific AWS CloudTrail attack patterns in flaws.cloud (SSRF to metadata
endpoint, ECR image poisoning, etc.) the model's parametric knowledge is thin.
Adding the six level writeups grounds the retriever in the exact scenarios we
are evaluating against and reduces hallucination about CloudTrail API names.

**Pipeline position (CLAUDE.md §3):**
This module is called *once* by ``scripts/build_rag_index.py`` to populate the
ChromaDB persistent index.  At Tier 3 inference time only ``retriever.py`` is
used; this module does not run on the hot path.

**Data-flow shapes:**
    STIX JSON file     →  load_attack_stix()      →  list[AttackDocument]  (n ≈ 700)
    JSONL / built-in   →  load_flaws_writeups()   →  list[AttackDocument]  (n = 6)
    Combined list      →  RAGRetriever.build_index()
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
#  Paths (env-var overridable per CLAUDE.md §5.5)
# ---------------------------------------------------------------------------

# parents[3] because this file lives at src/llm/rag/corpus.py
# parents[0]=rag, parents[1]=llm, parents[2]=src, parents[3]=project_root
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

STIX_PATH: Final[Path] = Path(
    os.environ.get(
        "ATTACK_STIX_PATH",
        str(_PROJECT_ROOT / "data" / "rag" / "enterprise-attack.json"),
    )
)

FLAWS_WRITEUPS_PATH: Final[Path] = Path(
    os.environ.get(
        "FLAWS_WRITEUPS_PATH",
        str(_PROJECT_ROOT / "data" / "rag" / "flaws_writeups.jsonl"),
    )
)

# ---------------------------------------------------------------------------
#  Document dataclass
# ---------------------------------------------------------------------------


@dataclass
class AttackDocument:
    """One indexable unit in the RAG corpus.

    Keeping this a plain dataclass (not frozen) because the builder may need
    to post-process fields.  ``__slots__`` makes the memory profile explicit
    (CLAUDE.md §5.10) and prevents accidental attribute typos.

    Attributes:
        doc_id:   Stable unique identifier — ATT&CK technique ID (e.g. T1136)
                  or flaws.cloud level ID (e.g. flaws_level_1).
        text:     Full text to be embedded and stored in ChromaDB.  Includes
                  the technique name, tactics, and truncated description so the
                  embedding captures semantic meaning without overflowing the
                  512-token limit of BGE-small-en-v1.5.
        source:   Provenance tag: ``'mitre_attack'`` or ``'flaws_cloud'``.
        name:     Human-readable title shown in the dashboard RAG panel.
        metadata: Flat ``str → str`` dict stored in ChromaDB for filtered
                  retrieval (e.g. filter by source == 'mitre_attack').
    """

    __slots__ = ("doc_id", "text", "source", "name", "metadata")

    doc_id: str
    text: str
    source: str
    name: str
    metadata: dict[str, str]


# ---------------------------------------------------------------------------
#  MITRE ATT&CK STIX loader
# ---------------------------------------------------------------------------


def load_attack_stix(stix_path: Path = STIX_PATH) -> list[AttackDocument]:
    """Parse the MITRE ATT&CK Enterprise STIX bundle into per-technique documents.

    Each ``attack-pattern`` object in the bundle becomes one ``AttackDocument``.
    Revoked and deprecated techniques are skipped — they contain outdated
    technique IDs that would confuse the LLM's MITRE mapping.

    The description is truncated to 1 000 characters.  BGE-small-en-v1.5 has
    a 512-token context window; beyond ~800 characters the text is silently
    truncated by the tokeniser anyway, so we truncate explicitly to keep the
    embedding semantically focused on the opening sentences (which typically
    contain the most technique-specific language).

    Args:
        stix_path: Path to ``enterprise-attack.json``.  Download once via
                   ``scripts/build_rag_index.py --mode full``.

    Returns:
        ``list[AttackDocument]``, one per live ATT&CK technique (typically
        ~700–800 items depending on ATT&CK version).

    Raises:
        FileNotFoundError: If stix_path does not exist.
    """
    if not stix_path.exists():
        raise FileNotFoundError(
            f"STIX bundle not found at {stix_path}.\n"
            "Download it with:\n"
            "  python scripts/build_rag_index.py --mode full\n"
            "Or set ATTACK_STIX_PATH to a local copy."
        )

    with stix_path.open("r", encoding="utf-8") as fh:
        bundle: dict = json.load(fh)

    docs: list[AttackDocument] = []
    for obj in bundle.get("objects", []):
        # Only process ATT&CK technique nodes — skip relationships, matrices, etc.
        if obj.get("type") != "attack-pattern":
            continue

        # Skip revoked / deprecated nodes — their IDs may no longer exist in
        # the current ATT&CK framework, which would produce hallucinated mappings
        if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
            continue

        # Extract the canonical technique ID (e.g. "T1136" or "T1136.001")
        technique_id: str = "UNKNOWN"
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                technique_id = ref.get("external_id", "UNKNOWN")
                break

        name: str = obj.get("name", "")
        description: str = obj.get("description", "")

        # Collect the associated ATT&CK tactic names from kill-chain phases
        tactics: list[str] = [
            phase["phase_name"].replace("-", " ").title()
            for phase in obj.get("kill_chain_phases", [])
            if phase.get("kill_chain_name") == "mitre-attack"
        ]
        tactics_str: str = ", ".join(tactics) if tactics else "Unknown"

        # Compose the document text.  Format mirrors how an analyst would read
        # the ATT&CK website entry so the embedding aligns with natural-language
        # queries like "privilege escalation via IAM account creation".
        text: str = (
            f"Technique: {technique_id} — {name}\n"
            f"Tactics: {tactics_str}\n"
            f"Description: {description[:1000]}"
        )

        docs.append(
            AttackDocument(
                doc_id=technique_id,
                text=text,
                source="mitre_attack",
                name=f"{technique_id}: {name}",
                metadata={
                    "technique_id": technique_id,
                    "tactics": tactics_str,
                    "source": "mitre_attack",
                },
            )
        )

    return docs


# ---------------------------------------------------------------------------
#  flaws.cloud writeup loader
# ---------------------------------------------------------------------------

# Built-in writeups for the 6 flaws.cloud levels.  These are used when the
# external JSONL file is absent (fresh clone, mock mode).  Each entry is a
# concise attack description that includes the CloudTrail API names involved
# and the corresponding MITRE technique IDs — exactly what the LLM needs to
# ground its mapping.
_BUILTIN_FLAWS_WRITEUPS: Final[list[dict[str, str]]] = [
    {
        "id": "flaws_level_1",
        "name": "flaws.cloud Level 1 — Public S3 Bucket Credential Leak",
        "text": (
            "Attack scenario: Public S3 bucket leaking AWS credentials.\n"
            "The attacker discovers a bucket configured with public-read ACL "
            "and downloads an access key stored in an object.  CloudTrail "
            "shows ListBuckets, GetBucketAcl, GetObject, and "
            "GetCallerIdentity from an anonymous or new principal originating "
            "from an unexpected IP address.\n"
            "MITRE ATT&CK: T1530 (Data from Cloud Storage Object) — "
            "Collection; T1552.001 (Unsecured Credentials: Credentials in "
            "Files) — Credential Access."
        ),
        "source": "flaws_cloud",
    },
    {
        "id": "flaws_level_2",
        "name": "flaws.cloud Level 2 — EC2 Metadata SSRF Credential Theft",
        "text": (
            "Attack scenario: SSRF to EC2 instance metadata endpoint.\n"
            "A misconfigured web application proxies HTTP requests, allowing "
            "the attacker to reach 169.254.169.254 and extract the EC2 "
            "instance role credentials.  CloudTrail shows AssumeRole and "
            "GetCallerIdentity from the instance ARN followed by API calls "
            "from an external IP that do not match the instance's normal "
            "behaviour baseline.\n"
            "MITRE ATT&CK: T1552.005 (Cloud Instance Metadata API) — "
            "Credential Access; T1078.004 (Valid Cloud Accounts) — "
            "Defence Evasion / Persistence / Privilege Escalation."
        ),
        "source": "flaws_cloud",
    },
    {
        "id": "flaws_level_3",
        "name": "flaws.cloud Level 3 — Lambda Environment Variable Credential Leak",
        "text": (
            "Attack scenario: Credentials exposed in Lambda environment "
            "variables.  The attacker obtains read access to a Lambda "
            "function's configuration (via GetFunction or a console "
            "misconfiguration) and extracts the embedded access key.  "
            "CloudTrail then shows IAM enumeration: ListRoles, GetRole, "
            "ListAttachedRolePolicies, GetRolePolicy from an unusual "
            "principal or unexpected source IP.\n"
            "MITRE ATT&CK: T1552.001 (Unsecured Credentials: Credentials "
            "in Files) — Credential Access; T1087.004 (Cloud Account "
            "Discovery) — Discovery."
        ),
        "source": "flaws_cloud",
    },
    {
        "id": "flaws_level_4",
        "name": "flaws.cloud Level 4 — ECS Task Credential Endpoint Abuse",
        "text": (
            "Attack scenario: ECS task credential theft via HTTP proxy.\n"
            "An ECS task runs an HTTP proxy that the attacker abuses to "
            "reach the ECS credential endpoint at 169.254.170.2.  The "
            "stolen task-role credentials are then used to call GetObject "
            "on S3 buckets outside the task's expected scope.  CloudTrail "
            "shows S3 access from an ECS task ARN with anomalous resource "
            "targets and a source IP that does not belong to the VPC.\n"
            "MITRE ATT&CK: T1552.005 (Cloud Instance Metadata API); "
            "T1530 (Data from Cloud Storage Object)."
        ),
        "source": "flaws_cloud",
    },
    {
        "id": "flaws_level_5",
        "name": "flaws.cloud Level 5 — Docker Daemon Host Escape",
        "text": (
            "Attack scenario: Exposed Docker daemon → container escape → "
            "instance metadata credential theft.\n"
            "The attacker connects to a publicly accessible Docker daemon, "
            "starts a privileged container with a host filesystem mount, "
            "and reads the EC2 instance profile credentials from "
            "/proc/1/environ or curl 169.254.169.254.  CloudTrail shows "
            "AssumeRole calls from the compromised instance ARN followed by "
            "unexpected cross-service API calls (IAM, S3, EC2).\n"
            "MITRE ATT&CK: T1611 (Escape to Host) — Privilege Escalation; "
            "T1552.005 (Cloud Instance Metadata API); T1078.004 (Valid "
            "Cloud Accounts)."
        ),
        "source": "flaws_cloud",
    },
    {
        "id": "flaws_level_6",
        "name": "flaws.cloud Level 6 — ECR Image Poisoning and ECS Privilege Escalation",
        "text": (
            "Attack scenario: Malicious container image pushed to ECR, "
            "then executed as an ECS task to escalate privileges.\n"
            "The attacker uses stolen ECR credentials to call "
            "GetAuthorizationToken, BatchCheckLayerAvailability, "
            "InitiateLayerUpload, UploadLayerPart, CompleteLayerUpload, "
            "PutImage.  They then register a new ECS task definition "
            "(RegisterTaskDefinition) and run it (RunTask) to obtain "
            "a higher-privileged task role.  CloudTrail shows an unusual "
            "sequence of ECR write APIs followed by ECS control-plane calls "
            "from an atypical principal.\n"
            "MITRE ATT&CK: T1525 (Implant Internal Image) — Persistence; "
            "T1610 (Deploy Container) — Defence Evasion / Execution."
        ),
        "source": "flaws_cloud",
    },
]


def load_flaws_writeups(
    writeups_path: Path = FLAWS_WRITEUPS_PATH,
) -> list[AttackDocument]:
    """Load flaws.cloud level writeups as AttackDocuments.

    Prefers the external JSONL file at ``writeups_path`` (one JSON object per
    line with keys ``id``, ``name``, ``text``, ``source``).  Falls back to the
    built-in descriptions above when the file is absent — this is the correct
    behaviour in mock mode and on a fresh clone before ``build_rag_index.py``
    has been run.

    Args:
        writeups_path: Path to ``flaws_writeups.jsonl``.

    Returns:
        ``list[AttackDocument]`` with 6 entries, one per flaws.cloud level.
    """
    raw_entries: list[dict[str, str]]

    if writeups_path.exists():
        raw_entries = []
        with writeups_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    raw_entries.append(json.loads(line))
    else:
        # Use built-in descriptions (no download required — safe for mock mode)
        raw_entries = _BUILTIN_FLAWS_WRITEUPS

    docs: list[AttackDocument] = []
    for entry in raw_entries:
        docs.append(
            AttackDocument(
                doc_id=entry["id"],
                text=entry["text"],
                source=entry.get("source", "flaws_cloud"),
                name=entry.get("name", entry["id"]),
                metadata={
                    "source": entry.get("source", "flaws_cloud"),
                    "level": entry["id"],
                },
            )
        )
    return docs
