"""Shared helpers for the SafeDrive (MLA-C02 practice) notebooks.

Typical first cell of a notebook:

    import sys; sys.path.append("..")
    from common import utils as U
    BUCKET, ROLE = U.bucket(), U.role_arn()
"""
import io
import json
import os
import time

import boto3
from botocore.exceptions import ClientError

PROJECT = "safedrive"
PREFIX = "safedrive"                      # S3 key prefix and resource-name prefix
ROLE_NAME = "safedrive-practice-role"     # created by 00_setup
TAGS = [{"Key": "project", "Value": "mla-c02-practice"}]
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGION = (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
          or boto3.session.Session().region_name or "us-east-1")
session = boto3.session.Session(region_name=REGION)

# Costly cells (GPU endpoints, fine-tuning, provisioned throughput ...) only run when this is True.
RUN_EXPENSIVE = os.environ.get("RUN_EXPENSIVE", "false").lower() == "true"

# ---- Bedrock model IDs (override with environment variables if your account differs) ----
_GEO = {"us": "us", "eu": "eu", "ap": "apac"}.get(REGION.split("-")[0], "us")
TEXT_MODEL = os.environ.get("SAFEDRIVE_TEXT_MODEL", f"{_GEO}.amazon.nova-lite-v1:0")
SMALL_MODEL = os.environ.get("SAFEDRIVE_SMALL_MODEL", f"{_GEO}.amazon.nova-micro-v1:0")
LARGE_MODEL = os.environ.get("SAFEDRIVE_LARGE_MODEL", f"{_GEO}.amazon.nova-pro-v1:0")
EMBED_MODEL = os.environ.get("SAFEDRIVE_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
IMAGE_EMBED_MODEL = os.environ.get("SAFEDRIVE_IMAGE_EMBED_MODEL", "amazon.titan-embed-image-v1")
RERANK_MODEL = os.environ.get("SAFEDRIVE_RERANK_MODEL", "cohere.rerank-v3-5:0")
EMBED_DIM = 1024

_clients = {}
_cache = {}


def client(name, **kwargs):
    """Return a cached boto3 client in the practice region."""
    key = (name, tuple(sorted(kwargs.items())))
    if key not in _clients:
        _clients[key] = session.client(name, **kwargs)
    return _clients[key]


def account_id():
    if "account" not in _cache:
        _cache["account"] = client("sts").get_caller_identity()["Account"]
    return _cache["account"]


def bucket():
    """Name of the practice bucket (created by 00_setup)."""
    return os.environ.get("SAFEDRIVE_BUCKET", f"{PREFIX}-mla-{account_id()}-{REGION}")


def role_arn():
    """ARN of the multi-service practice role (created by 00_setup)."""
    return os.environ.get("SAFEDRIVE_ROLE_ARN", f"arn:aws:iam::{account_id()}:role/{ROLE_NAME}")


def s3_uri(key=""):
    return f"s3://{bucket()}/{key}"


def model_arn(model_id):
    """ARN of a Bedrock foundation model or inference profile ID."""
    if model_id.split(".")[0] in ("us", "eu", "apac", "global"):
        return f"arn:aws:bedrock:{REGION}:{account_id()}:inference-profile/{model_id}"
    return f"arn:aws:bedrock:{REGION}::foundation-model/{model_id}"


# ---------------------------------------------------------------------------
# State shared between notebooks:  s3://<bucket>/safedrive/state.json
# ---------------------------------------------------------------------------
STATE_KEY = f"{PREFIX}/state.json"


def load_state():
    try:
        body = client("s3").get_object(Bucket=bucket(), Key=STATE_KEY)["Body"].read()
        return json.loads(body)
    except ClientError:
        return {}


def save_state(**values):
    """Merge key/values into the shared state file and return the full state."""
    state = load_state()
    state.update(values)
    client("s3").put_object(Bucket=bucket(), Key=STATE_KEY, Body=json.dumps(state, indent=2))
    return state


def get_state(key, default=None):
    return load_state().get(key, default)


def drop_state(*keys):
    state = load_state()
    for k in keys:
        state.pop(k, None)
    client("s3").put_object(Bucket=bucket(), Key=STATE_KEY, Body=json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# S3 convenience (no s3fs needed)
# ---------------------------------------------------------------------------
def upload_dir(local_dir, key_prefix):
    """Upload a folder recursively; returns the number of files."""
    n = 0
    for root, _, files in os.walk(local_dir):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, local_dir).replace(os.sep, "/")
            client("s3").upload_file(path, bucket(), f"{key_prefix}/{rel}")
            n += 1
    return n


def read_bytes(key):
    return client("s3").get_object(Bucket=bucket(), Key=key)["Body"].read()


def read_csv(key, **kwargs):
    import pandas as pd
    return pd.read_csv(io.BytesIO(read_bytes(key)), **kwargs)


def read_parquet(key, **kwargs):
    import pandas as pd
    return pd.read_parquet(io.BytesIO(read_bytes(key)), **kwargs)


def read_jsonl(key):
    return [json.loads(line) for line in read_bytes(key).decode().splitlines() if line.strip()]


def write_df(df, key, fmt="csv", **kwargs):
    """Write a DataFrame to S3 as csv or parquet; returns the s3:// URI."""
    buf = io.BytesIO()
    if fmt == "parquet":
        df.to_parquet(buf, index=False, **kwargs)
    else:
        df.to_csv(buf, index=False, **kwargs)
    client("s3").put_object(Bucket=bucket(), Key=key, Body=buf.getvalue())
    return s3_uri(key)


def list_keys(prefix):
    pages = client("s3").get_paginator("list_objects_v2").paginate(Bucket=bucket(), Prefix=prefix)
    return [o["Key"] for p in pages for o in p.get("Contents", [])]


# ---------------------------------------------------------------------------
# Bedrock convenience
# ---------------------------------------------------------------------------
def converse(prompt, model_id=None, system=None, max_tokens=512, temperature=0.2, **extra):
    """One-shot call to the Bedrock Converse API; returns the answer text."""
    args = dict(modelId=model_id or TEXT_MODEL,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature}, **extra)
    if system:
        args["system"] = [{"text": system}]
    resp = client("bedrock-runtime").converse(**args)
    return resp["output"]["message"]["content"][0]["text"]


def embed(text, dimensions=EMBED_DIM, model_id=None):
    """Embed one text with Amazon Titan Text Embeddings V2; returns a list of floats."""
    body = json.dumps({"inputText": text, "dimensions": dimensions, "normalize": True})
    resp = client("bedrock-runtime").invoke_model(modelId=model_id or EMBED_MODEL, body=body)
    return json.loads(resp["body"].read())["embedding"]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def wait_until(check, ok, bad=(), delay=15, timeout=1800, label="status"):
    """Poll check() until it returns a value in `ok`; raise if it lands in `bad`."""
    start = time.time()
    while True:
        status = check()
        print(f"{label}: {status}")
        if status in ok:
            return status
        if status in bad:
            raise RuntimeError(f"{label} ended in {status}")
        if time.time() - start > timeout:
            raise TimeoutError(f"{label} still {status} after {timeout}s")
        time.sleep(delay)


def safe(fn, *args, **kwargs):
    """Run a cleanup call and print (not raise) any error, so cleanup cells always finish."""
    try:
        result = fn(*args, **kwargs)
        print("ok:", getattr(fn, "__name__", fn))
        return result
    except Exception as e:  # noqa: BLE001
        print("skipped:", getattr(fn, "__name__", fn), "-", str(e)[:150])


def expensive(what):
    """Use as:  if U.expensive("fine-tuning job"): ...   Prints why a cell was skipped."""
    if not RUN_EXPENSIVE:
        print(f"SKIPPED ({what}). Set U.RUN_EXPENSIVE = True to run this cell - it costs money.")
    return RUN_EXPENSIVE
