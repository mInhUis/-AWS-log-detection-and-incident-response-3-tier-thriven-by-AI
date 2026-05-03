"""Synthetic normal CloudTrail generator.

Purpose
-------
Produces clean, attack-free CloudTrail JSONL records to serve as the *training*
corpus for the cost-aware cascade described in the thesis (Tier 1 Isolation
Forest + Tier 2 DeepLog). Both detectors are semi-supervised: they model the
distribution of legitimate behaviour and flag deviations. They therefore
require training data with zero attack signal.

Why we cannot train on flaws.cloud directly
-------------------------------------------
The flaws.cloud test corpus contains ~1.94M CloudTrail events spanning
2017-2020, with 9,402 unique source IPs and 8,811 unique user agents. The
account is a publicly-exposed CTF target, so the traffic mixes legitimate
CTF play, internet scanning, and genuine attack patterns without per-event
labels. Using any of it for training would leak attack signal into the
model's "normal" distribution and inflate test metrics — a well-known
failure mode for semi-supervised log anomaly detection (Du et al., 2017).

What this generator produces
----------------------------
An enterprise-shaped baseline:
  - 8 stable IAM principals (mix of IAM users and assumed roles)
  - A small pool of source IPs and user agents per principal
  - A weighted vocabulary of ~35 normal AWS API events (heavy reads,
    moderate writes, near-zero admin)
  - Sessions: bursts of activity from one principal in a short window,
    so DeepLog's sliding window sees realistic sequence structure
  - Business-hours skew (UTC) for human principals; 24/7 for service roles

What this generator deliberately omits
--------------------------------------
None of the following appear in the output, ever — they are the attack
signals the cascade must learn to flag at test time:
  - StopLogging, DeleteTrail            (CloudTrail tampering)
  - CreateUser, CreateAccessKey         (IAM persistence)
  - PutBucketAcl with public grants     (data exfiltration prep)
  - Console logins as root              (privileged access)
  - User agents containing "kali", "parrot", "powershell", "curl"
  - RunInstances from unrecognized principals or unusual regions

Pipeline position
-----------------
src/ingestion/generate_synthetic_normal.py  <-- THIS MODULE
        |
        v   data/raw/synthetic_normal.jsonl
src/preprocessing/preprocess.py     (identity-based split, feature aggregation)
        |
        v
src/preprocessing/drain_parser.py   (event names -> integer log keys)
        |
        +--> Tier 1 (Isolation Forest on per-entity-per-window features)
        +--> Tier 2 (DeepLog on log-key sequences)

Reference: AWS CloudTrail event record contents
    https://docs.aws.amazon.com/awscloudtrail/latest/userguide/cloudtrail-event-reference-record-contents.html
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final


# ============================================================================
# Configurable paths -- env-var-overridable per CLAUDE.md §5.5.
# ============================================================================
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_NORMAL_PATH: Final[Path] = Path(
    os.environ.get(
        "SYNTHETIC_NORMAL_PATH",
        str(_PROJECT_ROOT / "data" / "raw" / "synthetic_normal.jsonl"),
    )
)


# ============================================================================
# Synthetic-account constants. AWS reserves 111122223333 for documentation /
# examples, so it cannot collide with any real account -- guaranteeing that
# no IAM principal in this dataset can clash with flaws.cloud test data.
# (Identity-based split requirement, CLAUDE.md §9.)
# ============================================================================
SYNTHETIC_ACCOUNT_ID: Final[str] = "111122223333"

# Region pool weighted toward us-east-1 (where most workloads concentrate).
SYNTHETIC_REGIONS: Final[tuple[str, ...]] = (
    "us-east-1", "us-east-1", "us-east-1",
    "us-east-2", "us-west-2", "eu-west-1",
)


# ============================================================================
# Normal event vocabulary. Each entry: (eventSource, weight, readOnly).
# Weights are unnormalised; only relative magnitudes matter.
#
# Distribution rationale: CloudTrail logs from a typical workload are
# dominated by reads (Get/List/Describe/Head ~80%), writes are a minority,
# and admin-plane operations are vanishingly rare. This vocabulary respects
# that. Reference: AWS Well-Architected security pillar, observability.
# ============================================================================
NORMAL_EVENTS: Final[dict[str, tuple[str, int, bool]]] = {
    # --- S3 (data plane: heavy reads, moderate writes) ---
    "GetObject":              ("s3.amazonaws.com",              100, True),
    "ListObjectsV2":          ("s3.amazonaws.com",               60, True),
    "HeadObject":             ("s3.amazonaws.com",               50, True),
    "PutObject":              ("s3.amazonaws.com",               25, False),
    "DeleteObject":           ("s3.amazonaws.com",                8, False),
    "ListBuckets":            ("s3.amazonaws.com",               15, True),
    "GetBucketLocation":      ("s3.amazonaws.com",               12, True),
    "GetBucketTagging":       ("s3.amazonaws.com",                6, True),
    # --- EC2 (mostly Describes from monitoring tools) ---
    "DescribeInstances":      ("ec2.amazonaws.com",              40, True),
    "DescribeVolumes":        ("ec2.amazonaws.com",              20, True),
    "DescribeSecurityGroups": ("ec2.amazonaws.com",              15, True),
    "DescribeRegions":        ("ec2.amazonaws.com",              10, True),
    "DescribeImages":         ("ec2.amazonaws.com",               8, True),
    "RunInstances":           ("ec2.amazonaws.com",               4, False),
    "TerminateInstances":     ("ec2.amazonaws.com",               3, False),
    # --- STS / IAM (identity, very common but read-heavy) ---
    "GetCallerIdentity":      ("sts.amazonaws.com",              70, True),
    "AssumeRole":             ("sts.amazonaws.com",              30, True),
    "ListRoles":              ("iam.amazonaws.com",               8, True),
    "GetRole":                ("iam.amazonaws.com",              10, True),
    "ListAccessKeys":         ("iam.amazonaws.com",               4, True),
    "GetUser":                ("iam.amazonaws.com",               6, True),
    # --- CloudFormation / Lambda (deploy + runtime) ---
    "DescribeStacks":         ("cloudformation.amazonaws.com",   12, True),
    "ListStacks":             ("cloudformation.amazonaws.com",    8, True),
    "Invoke":                 ("lambda.amazonaws.com",           35, False),
    "GetFunction":            ("lambda.amazonaws.com",            8, True),
    "ListFunctions":          ("lambda.amazonaws.com",            6, True),
    # --- DynamoDB ---
    "DescribeTable":          ("dynamodb.amazonaws.com",         12, True),
    "GetItem":                ("dynamodb.amazonaws.com",         30, True),
    "Query":                  ("dynamodb.amazonaws.com",         22, True),
    "PutItem":                ("dynamodb.amazonaws.com",         14, False),
    # --- KMS / SecretsManager (encryption + secrets) ---
    "Decrypt":                ("kms.amazonaws.com",              25, False),
    "GenerateDataKey":        ("kms.amazonaws.com",              12, False),
    "DescribeKey":            ("kms.amazonaws.com",               8, True),
    #"GetSecretValue":         ("secretsmanager.amazonaws.com",   18, True),
    # --- CloudWatch Logs (high-volume telemetry) ---
    "PutLogEvents":           ("logs.amazonaws.com",             80, False),
    "CreateLogStream":        ("logs.amazonaws.com",             18, False),
    "DescribeLogGroups":      ("logs.amazonaws.com",              8, True),
}


# ============================================================================
# User-agent pool. All represent legitimate AWS clients. The deliberate
# ABSENCE of "kali", "parrot", "powershell", "curl/" tokens is the core test:
# DeepLog's Drain3 step replaces user agents with a placeholder and so cannot
# distinguish them -- but Tier 1 features (uniqueness, char n-gram entropy)
# on the raw user-agent string can. This is documented in the thesis as a
# motivating example for the cascade design.
# ============================================================================
NORMAL_USER_AGENTS: Final[tuple[str, ...]] = (
    "aws-cli/2.15.7 Python/3.11.8 Linux/5.15.0-aws botocore/2.15.7",
    "aws-cli/2.13.25 Python/3.11.6 Darwin/23.2.0 botocore/2.13.25",
    "   /1.34.14 md/Botocore#1.34.14 ua/2.0 os/linux lang/python#3.11.6",
    "Boto3/1.28.2 Python/3.10.12 Linux/5.15.0-1041-aws botocore/1.31.2",
    "[S3Console/0.4, aws-internal/3 aws-sdk-java/1.12.628]",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "[CloudFormation, aws-sdk-go/1.49.0]",
    "lambda-internal",
    "AWS Internal",
)


# ============================================================================
# Defaults. Mock-mode default required by CLAUDE.md §5.6: any script that
# touches large data must default to a small dataset; full runs explicitly
# opt in.
# ============================================================================
MOCK_NUM_EVENTS: Final[int] = 1_000
FULL_DEFAULT_NUM_EVENTS: Final[int] = 250_000

# Base-32 alphabet AWS uses for principal IDs and access keys.
_AWS_ID_ALPHABET: Final[str] = string.ascii_uppercase + "234567"


@dataclass(slots=True)
class PrincipalProfile:
    """A synthetic IAM principal with a stable behavioural signature.

    Each principal owns:
      - identity (ARN, user_name or role_name, principalId, accessKeyId)
      - source-IP pool (small, like a real workload: 1-3 IPs)
      - event-weight multipliers per event name (default 1.0; 0.0 disables
        an event for this principal -- e.g., a read-only dashboard role
        sets PutObject -> 0.0)
      - activity_weight: how often this principal acts vs. others
      - business_hours_only: humans yes; bots no

    Diversity *across* principals is what gives Tier 1 a non-trivial feature
    distribution to learn -- without it, every entity-window would look
    identical and Isolation Forest would have nothing to flag.
    """
    arn: str
    user_type: str            # "IAMUser" or "AssumedRole"
    user_name: str | None     # set iff user_type == "IAMUser"
    role_name: str | None     # set iff user_type == "AssumedRole"
    principal_id: str         # AIDA... (users) or AROA... (roles)
    access_key_id: str        # AKIA... (users) or ASIA... (assumed-role STS sessions)
    source_ips: tuple[str, ...]
    event_weights: dict[str, float] = field(default_factory=dict)
    activity_weight: float = 1.0
    business_hours_only: bool = False


def _aws_id(rng: random.Random, prefix: str, length: int = 16) -> str:
    """Generate an AWS-style ID: prefix + N base32-alphabet characters."""
    return prefix + "".join(rng.choices(_AWS_ID_ALPHABET, k=length))


def _build_default_principals(rng: random.Random) -> list[PrincipalProfile]:
    """Build a small, fixed set of synthetic principals for an enterprise.

    We keep the count low (8 principals) to mirror a small organization.
    Real enterprises have more, but for thesis evaluation 8 is enough to
    give Tier 1 a meaningful per-entity feature distribution while keeping
    the dataset interpretable.
    """
    OFFICE_IPS = ("198.51.100.10", "198.51.100.11", "198.51.100.12")
    CI_IPS     = ("203.0.113.50",)
    DASH_IPS   = ("192.0.2.30", "192.0.2.31")
    LAMBDA_IPS = tuple(f"10.0.{i}.{j}" for i in range(1, 4) for j in (10, 20))

    profiles: list[PrincipalProfile] = []

    # --- 3 human IAM users (devs) -- business hours, varied workload ---
    for name in ("alice", "bob", "carol"):
        profiles.append(PrincipalProfile(
            arn=f"arn:aws:iam::{SYNTHETIC_ACCOUNT_ID}:user/{name}",
            user_type="IAMUser",
            user_name=name,
            role_name=None,
            principal_id=_aws_id(rng, "AIDA"),
            access_key_id=_aws_id(rng, "AKIA"),
            source_ips=OFFICE_IPS,
            # Devs do a balanced mix; light bonus on CFN inspection.
            event_weights={"DescribeStacks": 2.0, "ListStacks": 2.0, "PutObject": 1.5},
            activity_weight=1.0,
            business_hours_only=True,
        ))

    # --- 1 CI/CD service role -- high activity, 24/7, narrow pattern ---
    profiles.append(PrincipalProfile(
        arn=f"arn:aws:sts::{SYNTHETIC_ACCOUNT_ID}:assumed-role/ci-deploy-role/jenkins-build",
        user_type="AssumedRole",
        user_name=None,
        role_name="ci-deploy-role",
        principal_id=_aws_id(rng, "AROA"),
        access_key_id=_aws_id(rng, "ASIA"),
        source_ips=CI_IPS,
        # CI: lots of CFN, S3 PutObject (artifacts), Lambda Invoke, Logs.
        event_weights={
            "DescribeStacks": 5.0, "ListStacks": 5.0,
            "PutObject": 4.0, "GetObject": 3.0,
            "Invoke": 4.0, "PutLogEvents": 6.0, "CreateLogStream": 4.0,
            "DescribeInstances": 0.2, "DescribeVolumes": 0.1,
        },
        activity_weight=2.5,
        business_hours_only=False,
    ))

    # --- 1 monitoring/dashboard role -- Describe-heavy, 24/7 ---
    profiles.append(PrincipalProfile(
        arn=f"arn:aws:sts::{SYNTHETIC_ACCOUNT_ID}:assumed-role/dashboard-readonly/grafana",
        user_type="AssumedRole",
        user_name=None,
        role_name="dashboard-readonly",
        principal_id=_aws_id(rng, "AROA"),
        access_key_id=_aws_id(rng, "ASIA"),
        source_ips=DASH_IPS,
        event_weights={
            "DescribeInstances": 5.0, "DescribeVolumes": 4.0,
            "DescribeSecurityGroups": 3.0, "DescribeStacks": 3.0,
            "ListBuckets": 2.0, "ListFunctions": 2.0,
            # No writes or crypto ops from a read-only dashboard. These events
            # are attack signals at test time; letting them appear in normal
            # training data would teach Tier 1 they are benign for this role.
            "PutObject": 0.0, "DeleteObject": 0.0, "PutItem": 0.0,
            "RunInstances": 0.0, "TerminateInstances": 0.0,
            "Invoke": 0.0, "PutLogEvents": 0.0, "CreateLogStream": 0.0,
            # KMS / SecretsManager -- a monitoring dashboard has no reason to
            # decrypt or read secrets; absence is the clean training signal.
            "Decrypt": 0.0, "GenerateDataKey": 0.0, "GetSecretValue": 0.0,
            # AssumeRole from a dashboard is lateral-movement territory.
            "AssumeRole": 0.0,
        },
        activity_weight=2.0,
        business_hours_only=False,
    ))

    # --- 3 lambda execution roles -- narrow, very high frequency, 24/7 ---
    for fn in ("api-handler", "image-resizer", "etl-worker"):
        profiles.append(PrincipalProfile(
            arn=(f"arn:aws:sts::{SYNTHETIC_ACCOUNT_ID}:"
                 f"assumed-role/lambda-{fn}/{fn}-session"),
            user_type="AssumedRole",
            user_name=None,
            role_name=f"lambda-{fn}",
            principal_id=_aws_id(rng, "AROA"),
            access_key_id=_aws_id(rng, "ASIA"),
            source_ips=LAMBDA_IPS,
            # Lambdas: DDB + S3 + KMS + Logs heavy; never Describe EC2 / IAM.
            event_weights={
                "GetItem": 5.0, "PutItem": 3.0, "Query": 4.0,
                "GetObject": 4.0, "PutObject": 2.0,
                "Decrypt": 4.0, "GenerateDataKey": 2.0,
                "GetSecretValue": 2.0,
                "PutLogEvents": 8.0, "CreateLogStream": 3.0,
                "DescribeInstances": 0.0, "DescribeVolumes": 0.0,
                "DescribeSecurityGroups": 0.0, "DescribeStacks": 0.0,
                "ListRoles": 0.0, "GetUser": 0.0, "ListAccessKeys": 0.0,
                "ListBuckets": 0.0,
            },
            activity_weight=4.0,  # Lambdas are chatty
            business_hours_only=False,
        ))

    return profiles


def _pick_event_for_principal(rng: random.Random, profile: PrincipalProfile) -> str:
    """Select an event name for this principal, blending global weights with
    per-principal multipliers. A multiplier of 0.0 disables an event for
    this principal -- the cleanest way to model role scope.
    """
    names: list[str] = []
    weights: list[float] = []
    # Iterate through global vocabulary, applying per-principal multipliers.
    for name, (_src, base_w, _ro) in NORMAL_EVENTS.items():
        mult = profile.event_weights.get(name, 1.0)
        w = float(base_w) * mult
        if w <= 0.0:
            continue  # disabled for this principal
        names.append(name)
        weights.append(w)
    # random.choices returns a length-1 list; unwrap it.
    return rng.choices(names, weights=weights, k=1)[0]


def _generate_session_starts(
    rng: random.Random,
    start: datetime,
    end: datetime,
    n_sessions: int,
    business_hours: bool,
) -> list[datetime]:
    """Sample n_sessions session-start timestamps in [start, end].

    Skewed toward business hours (08:00-18:00 UTC, weekdays) when
    business_hours=True, via rejection sampling. Coarse but deterministic
    given the seeded RNG.
    """
    span_seconds = int((end - start).total_seconds())
    if span_seconds <= 0:
        raise ValueError(f"end ({end}) must be after start ({start})")
    out: list[datetime] = []
    # Cap iterations -- guards against pathological windows (e.g., a 1-hour
    # window with business_hours=True at 03:00 UTC would otherwise loop).
    max_iters = max(n_sessions * 20, 1000)
    iters = 0
    while len(out) < n_sessions and iters < max_iters:
        iters += 1
        offset = rng.randint(0, span_seconds - 1)
        ts = start + timedelta(seconds=offset)
        if business_hours:
            weekday_ok = ts.weekday() < 5
            hour_ok = 8 <= ts.hour < 18
            keep_prob = 0.9 if (weekday_ok and hour_ok) else 0.1
            if rng.random() > keep_prob:
                continue
        out.append(ts)
    out.sort()
    return out


def _stub_request_params(rng: random.Random, event_name: str) -> dict | None:
    """Minimal but plausible requestParameters for the most common events.

    Drain3 doesn't read these (it tokenises eventName), so they are cosmetic
    for downstream tooling. Kept minimal to keep the file size small.
    """
    bucket_pool = ("artifacts-prod", "logs-archive", "user-uploads", "static-assets")
    if event_name in {"GetObject", "PutObject", "DeleteObject", "HeadObject"}:
        return {
            "bucketName": rng.choice(bucket_pool),
            "key": f"path/file-{rng.randint(1, 9999)}.dat",
        }
    if event_name == "ListObjectsV2":
        return {
            "bucketName": rng.choice(bucket_pool),
            "prefix": f"folder-{rng.randint(1, 99)}/",
        }
    if event_name == "AssumeRole":
        return {
            "roleArn": f"arn:aws:iam::{SYNTHETIC_ACCOUNT_ID}:role/lambda-api-handler",
            "roleSessionName": f"sess-{rng.randint(1000, 9999)}",
        }
    return None  # most events are fine without explicit params for this baseline


def _stub_response_elements(rng: random.Random, event_name: str) -> dict | None:
    """Minimal responseElements for write events. None for reads."""
    if event_name == "PutObject":
        return {"x-amz-version-id": uuid.uuid4().hex[:16]}
    return None


def _build_event(
    rng: random.Random,
    profile: PrincipalProfile,
    event_name: str,
    when: datetime,
) -> dict:
    """Construct a single CloudTrail event dict matching the AWS-delivered
    schema. Field names and shapes follow the reference linked in the module
    docstring.
    """
    src, _w, read_only = NORMAL_EVENTS[event_name]
    region = rng.choice(SYNTHETIC_REGIONS)
    src_ip = rng.choice(profile.source_ips)
    user_agent = rng.choice(NORMAL_USER_AGENTS)

    # userIdentity has different shapes for IAMUser vs AssumedRole.
    if profile.user_type == "IAMUser":
        user_identity: dict = {
            "type": "IAMUser",
            "principalId": profile.principal_id,
            "arn": profile.arn,
            "accountId": SYNTHETIC_ACCOUNT_ID,
            "accessKeyId": profile.access_key_id,
            "userName": profile.user_name,
        }
    else:
        # AssumedRole identities embed sessionContext.sessionIssuer; preserve
        # that shape so downstream parsers (e.g., entity extraction in
        # preprocess.py) don't need a special branch.
        assert profile.role_name is not None
        session_name = profile.arn.rsplit("/", 1)[-1]
        creation = (when - timedelta(minutes=rng.randint(1, 55)))
        user_identity = {
            "type": "AssumedRole",
            "principalId": f"{profile.principal_id}:{session_name}",
            "arn": profile.arn,
            "accountId": SYNTHETIC_ACCOUNT_ID,
            "accessKeyId": profile.access_key_id,
            "sessionContext": {
                "sessionIssuer": {
                    "type": "Role",
                    "principalId": profile.principal_id,
                    "arn": (f"arn:aws:iam::{SYNTHETIC_ACCOUNT_ID}:"
                            f"role/{profile.role_name}"),
                    "accountId": SYNTHETIC_ACCOUNT_ID,
                    "userName": profile.role_name,
                },
                "attributes": {
                    "creationDate": creation.isoformat().replace("+00:00", "Z"),
                    "mfaAuthenticated": "false",
                },
            },
        }

    event = {
        "eventVersion": "1.08",
        "userIdentity": user_identity,
        "eventTime": when.isoformat().replace("+00:00", "Z"),
        "eventSource": src,
        "eventName": event_name,
        "awsRegion": region,
        "sourceIPAddress": src_ip,
        "userAgent": user_agent,
        "requestParameters": _stub_request_params(rng, event_name),
        "responseElements": (None if read_only
                             else _stub_response_elements(rng, event_name)),
        "requestID": str(uuid.UUID(int=rng.getrandbits(128))),
        "eventID":   str(uuid.UUID(int=rng.getrandbits(128))),
        "readOnly": read_only,
        "eventType": "AwsApiCall",
        "managementEvent": True,
        "recipientAccountId": SYNTHETIC_ACCOUNT_ID,
    }
    return event


def generate(
    num_events: int,
    start: datetime,
    end: datetime,
    seed: int,
    output_path: Path,
    overwrite: bool,
) -> dict:
    """Top-level generator. Writes JSONL to output_path; returns summary dict.

    Output file shape: one JSON object per line, sorted ascending by
    eventTime so the file reads chronologically (matches AWS-delivered
    log file conventions for downstream tooling).
    """
    # Stale-artifact guard per CLAUDE.md §5.8: refuse to overwrite without
    # explicit --force, and raise with a clear remediation command.
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            f"Re-run with --force to overwrite, or pass --output <new_path>."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    profiles = _build_default_principals(rng)

    # Allocate the event budget across principals proportional to
    # activity_weight. Lambdas (weight 4.0) get more events than devs (1.0),
    # giving the dataset realistic per-entity variance.
    total_weight = sum(p.activity_weight for p in profiles)
    per_principal_counts = [
        max(1, int(round(num_events * p.activity_weight / total_weight)))
        for p in profiles
    ]

    # Sessions: each principal gets timestamps clustered into bursts. We
    # approximate this by sampling N session-start times and assigning a
    # geometric burst length per session. This matters because DeepLog's
    # sliding window operates on per-session sequences -- if we sampled
    # events independently in time, the LSTM would see noise rather than
    # the causal-ish structure of real workflows.
    MEAN_BURST = 6.0  # mean events per session

    all_events: list[tuple[datetime, dict]] = []
    for profile, n_events in zip(profiles, per_principal_counts):
        session_count = max(1, int(n_events / MEAN_BURST))
        session_starts = _generate_session_starts(
            rng, start, end, session_count, profile.business_hours_only,
        )
        emitted = 0
        for sess_start in session_starts:
            if emitted >= n_events:
                break
            # Geometric/exponential burst length, clipped to remaining budget.
            burst_len = max(1, int(rng.expovariate(1.0 / MEAN_BURST)))
            burst_len = min(burst_len, n_events - emitted)
            for i in range(burst_len):
                # Intra-session inter-arrival: exponential, mean ~30 seconds.
                # i+1 multiplier means later events drift further from start.
                offset = timedelta(seconds=int(rng.expovariate(1.0 / 30.0)))
                ts = sess_start + offset * (i + 1)
                if ts > end:
                    break
                ev_name = _pick_event_for_principal(rng, profile)
                ev = _build_event(rng, profile, ev_name, ts)
                all_events.append((ts, ev))
                emitted += 1

    # Sort globally by timestamp -- AWS-delivered files are chronological.
    all_events.sort(key=lambda t: t[0])

    # Stream-write JSONL. Using compact separators saves ~10% disk on large
    # files and matches what real S3-delivered CloudTrail batches look like
    # after gzip extraction.
    with output_path.open("w", encoding="utf-8") as f:
        for _ts, ev in all_events:
            f.write(json.dumps(ev, separators=(",", ":")))
            f.write("\n")

    summary = {
        "events_written": len(all_events),
        "principals": len(profiles),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "output": str(output_path),
        "size_bytes": output_path.stat().st_size,
    }
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Generate synthetic normal CloudTrail JSONL for thesis "
                     "training data. Defaults to mock mode (CLAUDE.md §5.6)."),
    )
    p.add_argument(
        "--mode", choices=("mock", "full"), default="mock",
        help="mock: small dev dataset (~1K events, default). "
             "full: real-scale dataset (~250K events).",
    )
    p.add_argument("--num-events", type=int, default=None,
                   help="Override event count. "
                        "Default: 1K (mock) or 250K (full).")
    p.add_argument("--start", type=str, default="2023-01-01T00:00:00Z",
                   help="ISO8601 start timestamp (UTC).")
    p.add_argument("--end", type=str, default="2024-01-01T00:00:00Z",
                   help="ISO8601 end timestamp (UTC).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility.")
    p.add_argument("--output", type=str, default=str(SYNTHETIC_NORMAL_PATH),
                   help="Output JSONL file path.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite output if it exists.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    n = args.num_events
    if n is None:
        n = MOCK_NUM_EVENTS if args.mode == "mock" else FULL_DEFAULT_NUM_EVENTS

    # ISO8601 with trailing 'Z' is not handled by fromisoformat() before
    # Python 3.11; the .replace() is a portable shim.
    start = datetime.fromisoformat(args.start.replace("Z", "+00:00")).astimezone(timezone.utc)
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00")).astimezone(timezone.utc)
    out = Path(args.output)

    print(f"[generate_synthetic_normal] mode={args.mode} "
          f"target_events={n:,} window={start.date()}..{end.date()} "
          f"seed={args.seed}")
    summary = generate(
        num_events=n, start=start, end=end, seed=args.seed,
        output_path=out, overwrite=args.force,
    )
    print(f"[generate_synthetic_normal] done: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
