"""Synthetic normal CloudTrail generator.

Purpose
-------
Produces clean, attack-free CloudTrail JSONL records to serve as the *training*
corpus for the cost-aware cascade described in the thesis (Tier 1 Isolation
Forest + Tier 2 DeepLog).

Why we widened the vocabulary (The OOV fix)
-------------------------------------------
flaws.cloud consists heavily of S3 bucket enumeration (GetBucketAcl, etc.). 
If these APIs are missing from the baseline, DeepLog treats them as Out-Of-Vocabulary 
(<UNK>) and flags them trivially. However, these APIs are routinely used by legitimate 
compliance scanners (e.g., AWS Security Hub). 

To force DeepLog to learn the *sequence anomaly* (rapid, targeted enumeration with 
AccessDenied errors) rather than just panicking at the API name, we inject a 
"Security Audit" bot into this baseline that calls these APIs legitimately.

CRITICAL METHODOLOGY NOTE:
We inject benign errors (NoSuchTagSet, NoSuchBucketPolicy) to teach Tier 1 the 
background noise floor. We DO NOT inject AccessDenied. AccessDenied during 
S3 reconnaissance is the primary flaws.cloud attack signal. 
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

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_NORMAL_PATH: Final[Path] = Path(
    os.environ.get(
        "SYNTHETIC_NORMAL_PATH",
        str(_PROJECT_ROOT / "data" / "raw" / "synthetic_normal1.jsonl"),
    )
)

SYNTHETIC_ACCOUNT_ID: Final[str] = "111122223333"
SYNTHETIC_REGIONS: Final[tuple[str, ...]] = (
    "us-east-1", "us-east-1", "us-east-1",
    "us-east-2", "us-west-2", "eu-west-1",
)

# ============================================================================
# Normal event vocabulary. 
# --- NEW: Added S3 metadata reads & IAM enumeration.
# --- NEW: Commented out PutLogEvents and Decrypt to remove infrastructure noise.
# ============================================================================
NORMAL_EVENTS: Final[dict[str, tuple[str, int, bool]]] = {
    # --- S3 (data plane) ---
    "GetObject":              ("s3.amazonaws.com",              100, True),
    "ListObjectsV2":          ("s3.amazonaws.com",               60, True),
    "HeadObject":             ("s3.amazonaws.com",               50, True),
    "PutObject":              ("s3.amazonaws.com",               25, False),
    "DeleteObject":           ("s3.amazonaws.com",                8, False),
    "ListBuckets":            ("s3.amazonaws.com",               15, True),
    "GetBucketLocation":      ("s3.amazonaws.com",               12, True),
    "GetBucketTagging":       ("s3.amazonaws.com",                6, True),
    
    # --- NEW: S3 (compliance / audit plane) ---
    # Mutation APIs (PutBucketAcl, DeleteBucket) are INTENTIONALLY excluded.
    "GetBucketAcl":                 ("s3.amazonaws.com",          8, True),
    "GetBucketPolicy":              ("s3.amazonaws.com",          8, True),
    "GetBucketVersioning":          ("s3.amazonaws.com",          6, True),
    "GetBucketLogging":             ("s3.amazonaws.com",          6, True),
    "GetBucketWebsite":             ("s3.amazonaws.com",          4, True),
    "GetBucketLifecycle":           ("s3.amazonaws.com",          4, True),
    "GetBucketNotification":        ("s3.amazonaws.com",          4, True),
    "GetBucketEncryption":          ("s3.amazonaws.com",          6, True),
    "GetBucketReplication":         ("s3.amazonaws.com",          2, True),
    "GetBucketCors":                ("s3.amazonaws.com",          2, True),
    "GetBucketRequestPayment":      ("s3.amazonaws.com",          2, True),
    "GetBucketPolicyStatus":        ("s3.amazonaws.com",          4, True),
    "GetBucketPublicAccessBlock":   ("s3.amazonaws.com",          5, True),
    "GetAccelerateConfiguration":   ("s3.amazonaws.com",          2, True),
    "ListMultipartUploads":         ("s3.amazonaws.com",          3, True),

    # --- EC2 ---
    "DescribeInstances":      ("ec2.amazonaws.com",              40, True),
    "DescribeVolumes":        ("ec2.amazonaws.com",              20, True),
    "DescribeSecurityGroups": ("ec2.amazonaws.com",              15, True),
    "DescribeRegions":        ("ec2.amazonaws.com",              10, True),
    "DescribeImages":         ("ec2.amazonaws.com",               8, True),
    "RunInstances":           ("ec2.amazonaws.com",               4, False),
    "TerminateInstances":     ("ec2.amazonaws.com",               3, False),

    # --- STS / IAM ---
    # GetCallerIdentity heavily reduced to prevent it from becoming a noise anchor
    "GetCallerIdentity":      ("sts.amazonaws.com",              15, True),
    "AssumeRole":             ("sts.amazonaws.com",              30, True),
    "ListRoles":              ("iam.amazonaws.com",               8, True),
    "GetRole":                ("iam.amazonaws.com",              10, True),
    "ListAccessKeys":         ("iam.amazonaws.com",               4, True),
    "GetUser":                ("iam.amazonaws.com",               6, True),
    
    # --- NEW: IAM (enumeration) ---
    "ListAttachedUserPolicies":("iam.amazonaws.com",              4, True),
    "ListUserPolicies":       ("iam.amazonaws.com",               4, True),

    # --- CloudFormation / Lambda / DynamoDB ---
    "DescribeStacks":         ("cloudformation.amazonaws.com",   12, True),
    "ListStacks":             ("cloudformation.amazonaws.com",    8, True),
    "Invoke":                 ("lambda.amazonaws.com",           35, False),
    "GetFunction":            ("lambda.amazonaws.com",            8, True),
    "ListFunctions":          ("lambda.amazonaws.com",            6, True),
    "DescribeTable":          ("dynamodb.amazonaws.com",         12, True),
    "GetItem":                ("dynamodb.amazonaws.com",         30, True),
    "Query":                  ("dynamodb.amazonaws.com",         22, True),
    "PutItem":                ("dynamodb.amazonaws.com",         14, False),

    # --- KMS & Logs (REMOVED) ---
    # PutLogEvents and Decrypt removed entirely to eliminate the 27.7% volume noise
    "GenerateDataKey":        ("kms.amazonaws.com",              12, False),
    "DescribeKey":            ("kms.amazonaws.com",               8, True),
}

NORMAL_USER_AGENTS: Final[tuple[str, ...]] = (
    "aws-cli/2.15.7 Python/3.11.8 Linux/5.15.0-aws botocore/2.15.7",
    "aws-cli/2.13.25 Python/3.11.6 Darwin/23.2.0 botocore/2.13.25",
    "   /1.34.14 md/Botocore#1.34.14 ua/2.0 os/linux lang/python#3.11.6",
    "Boto3/1.28.2 Python/3.10.12 Linux/5.15.0-1041-aws botocore/1.31.2",
    "[S3Console/0.4, aws-internal/3 aws-sdk-java/1.12.628]",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "[CloudFormation, aws-sdk-go/1.49.0]",
    "lambda-internal",
    "AWS Internal",
)

MOCK_NUM_EVENTS: Final[int] = 1_000
FULL_DEFAULT_NUM_EVENTS: Final[int] = 250_000
_AWS_ID_ALPHABET: Final[str] = string.ascii_uppercase + "234567"

@dataclass(slots=True)
class PrincipalProfile:
    arn: str
    user_type: str
    user_name: str | None
    role_name: str | None
    principal_id: str
    access_key_id: str
    source_ips: tuple[str, ...]
    event_weights: dict[str, float] = field(default_factory=dict)
    activity_weight: float = 1.0
    business_hours_only: bool = False

def _aws_id(rng: random.Random, prefix: str, length: int = 16) -> str:
    return prefix + "".join(rng.choices(_AWS_ID_ALPHABET, k=length))

def _build_default_principals(rng: random.Random) -> list[PrincipalProfile]:
    OFFICE_IPS = ("198.51.100.10", "198.51.100.11", "198.51.100.12")
    CI_IPS     = ("203.0.113.50",)
    DASH_IPS   = ("192.0.2.30", "192.0.2.31")
    LAMBDA_IPS = tuple(f"10.0.{i}.{j}" for i in range(1, 4) for j in (10, 20))

    profiles: list[PrincipalProfile] = []

    # 3 human IAM users (devs)
    for name in ("alice", "bob", "carol"):
        profiles.append(PrincipalProfile(
            arn=f"arn:aws:iam::{SYNTHETIC_ACCOUNT_ID}:user/{name}",
            user_type="IAMUser", user_name=name, role_name=None,
            principal_id=_aws_id(rng, "AIDA"), access_key_id=_aws_id(rng, "AKIA"),
            source_ips=OFFICE_IPS,
            event_weights={"DescribeStacks": 2.0, "ListStacks": 2.0, "PutObject": 1.5},
            activity_weight=1.0, business_hours_only=True,
        ))

    # 1 CI/CD service role
    profiles.append(PrincipalProfile(
        arn=f"arn:aws:sts::{SYNTHETIC_ACCOUNT_ID}:assumed-role/ci-deploy-role/jenkins-build",
        user_type="AssumedRole", user_name=None, role_name="ci-deploy-role",
        principal_id=_aws_id(rng, "AROA"), access_key_id=_aws_id(rng, "ASIA"),
        source_ips=CI_IPS,
        event_weights={
            "DescribeStacks": 5.0, "ListStacks": 5.0, "PutObject": 4.0, "GetObject": 3.0,
            "Invoke": 4.0, "DescribeInstances": 0.2, "DescribeVolumes": 0.1,
        },
        activity_weight=2.5, business_hours_only=False,
    ))

    # 1 monitoring/dashboard role
    profiles.append(PrincipalProfile(
        arn=f"arn:aws:sts::{SYNTHETIC_ACCOUNT_ID}:assumed-role/dashboard-readonly/grafana",
        user_type="AssumedRole", user_name=None, role_name="dashboard-readonly",
        principal_id=_aws_id(rng, "AROA"), access_key_id=_aws_id(rng, "ASIA"),
        source_ips=DASH_IPS,
        event_weights={
            "DescribeInstances": 5.0, "DescribeVolumes": 4.0, "DescribeSecurityGroups": 3.0, 
            "DescribeStacks": 3.0, "ListBuckets": 2.0, "ListFunctions": 2.0,
            "PutObject": 0.0, "DeleteObject": 0.0, "PutItem": 0.0, "RunInstances": 0.0, 
            "TerminateInstances": 0.0, "Invoke": 0.0, "GenerateDataKey": 0.0, "AssumeRole": 0.0,
        },
        activity_weight=2.0, business_hours_only=False,
    ))

    # 3 lambda execution roles
    for fn in ("api-handler", "image-resizer", "etl-worker"):
        profiles.append(PrincipalProfile(
            arn=(f"arn:aws:sts::{SYNTHETIC_ACCOUNT_ID}:assumed-role/lambda-{fn}/{fn}-session"),
            user_type="AssumedRole", user_name=None, role_name=f"lambda-{fn}",
            principal_id=_aws_id(rng, "AROA"), access_key_id=_aws_id(rng, "ASIA"),
            source_ips=LAMBDA_IPS,
            event_weights={
                "GetItem": 5.0, "PutItem": 3.0, "Query": 4.0, "GetObject": 4.0, "PutObject": 2.0,
                "GenerateDataKey": 2.0, "DescribeInstances": 0.0, "DescribeVolumes": 0.0,
                "DescribeSecurityGroups": 0.0, "DescribeStacks": 0.0, "ListRoles": 0.0, 
                "GetUser": 0.0, "ListAccessKeys": 0.0, "ListBuckets": 0.0,
            },
            activity_weight=4.0, business_hours_only=False,
        ))

    # --- NEW: 1 Security Audit Scanner bot
    # This role legitimately sweeps across the infrastructure checking policies.
    # It generates the S3 and IAM enumeration APIs in a realistic sequence.
    profiles.append(PrincipalProfile(
        arn=f"arn:aws:sts::{SYNTHETIC_ACCOUNT_ID}:assumed-role/security-audit/config-scanner",
        user_type="AssumedRole", user_name=None, role_name="security-audit",
        principal_id=_aws_id(rng, "AROA"), access_key_id=_aws_id(rng, "ASIA"),
        source_ips=("10.0.99.100", "10.0.99.101"),
        event_weights={
            "GetBucketAcl": 10.0, "GetBucketPolicy": 10.0, "GetBucketVersioning": 10.0,
            "GetBucketLogging": 10.0, "GetBucketEncryption": 10.0, "GetBucketPublicAccessBlock": 10.0,
            "ListAttachedUserPolicies": 8.0, "ListUserPolicies": 8.0, "ListBuckets": 5.0,
            "DescribeInstances": 3.0, "DescribeSecurityGroups": 3.0,
            # No writes
            "PutObject": 0.0, "RunInstances": 0.0, "PutItem": 0.0, "Invoke": 0.0,
        },
        activity_weight=3.5, business_hours_only=False, # 24/7 background sweeps
    ))

    return profiles

def _pick_event_for_principal(rng: random.Random, profile: PrincipalProfile) -> str:
    names: list[str] = []
    weights: list[float] = []
    for name, (_src, base_w, _ro) in NORMAL_EVENTS.items():
        mult = profile.event_weights.get(name, 1.0)
        w = float(base_w) * mult
        if w <= 0.0:
            continue
        names.append(name)
        weights.append(w)
    return rng.choices(names, weights=weights, k=1)[0]

def _generate_session_starts(
    rng: random.Random, start: datetime, end: datetime, n_sessions: int, business_hours: bool,
) -> list[datetime]:
    span_seconds = int((end - start).total_seconds())
    out: list[datetime] = []
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
    bucket_pool = ("artifacts-prod", "logs-archive", "user-uploads", "static-assets")
    if "Object" in event_name and event_name != "ListObjectsV2":
        return {
            "bucketName": rng.choice(bucket_pool),
            "key": f"path/file-{rng.randint(1, 9999)}.dat",
        }
    if event_name == "ListObjectsV2":
        return {"bucketName": rng.choice(bucket_pool), "prefix": f"folder-{rng.randint(1, 99)}/"}
    if event_name == "AssumeRole":
        return {
            "roleArn": f"arn:aws:iam::{SYNTHETIC_ACCOUNT_ID}:role/lambda-api-handler",
            "roleSessionName": f"sess-{rng.randint(1000, 9999)}",
        }
    return None

def _stub_response_elements(rng: random.Random, event_name: str) -> dict | None:
    if event_name == "PutObject":
        return {"x-amz-version-id": uuid.uuid4().hex[:16]}
    return None

def _build_event(
    rng: random.Random, profile: PrincipalProfile, event_name: str, when: datetime,
) -> dict:
    src, _w, read_only = NORMAL_EVENTS[event_name]
    region = rng.choice(SYNTHETIC_REGIONS)
    src_ip = rng.choice(profile.source_ips)
    user_agent = rng.choice(NORMAL_USER_AGENTS)

    if profile.user_type == "IAMUser":
        user_identity: dict = {
            "type": "IAMUser", "principalId": profile.principal_id, "arn": profile.arn,
            "accountId": SYNTHETIC_ACCOUNT_ID, "accessKeyId": profile.access_key_id, "userName": profile.user_name,
        }
    else:
        assert profile.role_name is not None
        session_name = profile.arn.rsplit("/", 1)[-1]
        creation = (when - timedelta(minutes=rng.randint(1, 55)))
        user_identity = {
            "type": "AssumedRole", "principalId": f"{profile.principal_id}:{session_name}",
            "arn": profile.arn, "accountId": SYNTHETIC_ACCOUNT_ID, "accessKeyId": profile.access_key_id,
            "sessionContext": {
                "sessionIssuer": {
                    "type": "Role", "principalId": profile.principal_id, "arn": (f"arn:aws:iam::{SYNTHETIC_ACCOUNT_ID}:role/{profile.role_name}"),
                    "accountId": SYNTHETIC_ACCOUNT_ID, "userName": profile.role_name,
                },
                "attributes": {"creationDate": creation.isoformat().replace("+00:00", "Z"), "mfaAuthenticated": "false"},
            },
        }

    # --- NEW: Benign Error Injection ---
    # Generates a ~2% error rate. We explicitly avoid AccessDenied to ensure the 
    # true flaw.cloud reconnaissance signal remains anomalous.
    error_code = ""
    error_message = ""
    if read_only and rng.random() < 0.02:
        if src == "s3.amazonaws.com":
            if "Tagging" in event_name:
                error_code = "NoSuchTagSet"
                error_message = "The TagSet does not exist"
            elif "Policy" in event_name:
                error_code = "NoSuchBucketPolicy"
                error_message = "The bucket policy does not exist"
            else:
                error_code = "NoSuchBucket"
                error_message = "The specified bucket does not exist"
        elif src == "iam.amazonaws.com":
            error_code = "NoSuchEntityException"
            error_message = "The user or role does not exist"

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
        "responseElements": (None if read_only else _stub_response_elements(rng, event_name)),
        "requestID": str(uuid.UUID(int=rng.getrandbits(128))),
        "eventID":   str(uuid.UUID(int=rng.getrandbits(128))),
        "readOnly": read_only,
        "eventType": "AwsApiCall",
        "managementEvent": True,
        "recipientAccountId": SYNTHETIC_ACCOUNT_ID,
    }
    
    if error_code:
        event["errorCode"] = error_code
        event["errorMessage"] = error_message
        
    return event

def generate(num_events: int, start: datetime, end: datetime, seed: int, output_path: Path, overwrite: bool) -> dict:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Re-run with --force to overwrite.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    profiles = _build_default_principals(rng)
    total_weight = sum(p.activity_weight for p in profiles)
    per_principal_counts = [max(1, int(round(num_events * p.activity_weight / total_weight))) for p in profiles]

    MEAN_BURST = 6.0 
    all_events: list[tuple[datetime, dict]] = []
    
    for profile, n_events in zip(profiles, per_principal_counts):
        session_count = max(1, int(n_events / MEAN_BURST))
        session_starts = _generate_session_starts(rng, start, end, session_count, profile.business_hours_only)
        emitted = 0
        for sess_start in session_starts:
            if emitted >= n_events: break
            burst_len = min(max(1, int(rng.expovariate(1.0 / MEAN_BURST))), n_events - emitted)
            for i in range(burst_len):
                offset = timedelta(seconds=int(rng.expovariate(1.0 / 30.0)))
                ts = sess_start + offset * (i + 1)
                if ts > end: break
                ev_name = _pick_event_for_principal(rng, profile)
                ev = _build_event(rng, profile, ev_name, ts)
                all_events.append((ts, ev))
                emitted += 1

    all_events.sort(key=lambda t: t[0])
    with output_path.open("w", encoding="utf-8") as f:
        for _ts, ev in all_events:
            f.write(json.dumps(ev, separators=(",", ":")))
            f.write("\n")

    return {
        "events_written": len(all_events), "principals": len(profiles),
        "start": start.isoformat(), "end": end.isoformat(),
        "output": str(output_path), "size_bytes": output_path.stat().st_size,
    }

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic normal CloudTrail JSONL.")
    p.add_argument("--mode", choices=("mock", "full"), default="mock")
    p.add_argument("--num-events", type=int, default=None)
    p.add_argument("--start", type=str, default="2023-01-01T00:00:00Z")
    p.add_argument("--end", type=str, default="2024-01-01T00:00:00Z")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=str(SYNTHETIC_NORMAL_PATH))
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    n = args.num_events if args.num_events is not None else (MOCK_NUM_EVENTS if args.mode == "mock" else FULL_DEFAULT_NUM_EVENTS)
    start = datetime.fromisoformat(args.start.replace("Z", "+00:00")).astimezone(timezone.utc)
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00")).astimezone(timezone.utc)
    
    summary = generate(num_events=n, start=start, end=end, seed=args.seed, output_path=Path(args.output), overwrite=args.force)
    print(f"[generate_synthetic_normal] done: {summary}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))