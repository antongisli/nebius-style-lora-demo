# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3"]
# ///
"""Copy files to and from the training bucket.

Credentials are read at run time from the MysteryBox secret named in deploy.env
(S3_SECRET) through the Nebius CLI, and are never written to disk.

    uv run infra/bucket.py put ./dataset dataset/v1
    uv run infra/bucket.py get runs/style-r16-lr1e-4 ./results/style-r16-lr1e-4
    uv run infra/bucket.py ls runs/
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> dict[str, str]:
    env = {}
    for line in (ROOT / "deploy.env").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def credentials(secret_id: str) -> tuple[str, str]:
    profile = os.environ.get("NEBIUS_PROFILE", "default")
    out = subprocess.run(
        ["nebius", "--profile", profile, "--no-check-update", "mysterybox", "v1", "payload", "get",
         "--secret-id", secret_id, "--format", "json"],
        check=True, capture_output=True, text=True,
    ).stdout
    payload = {e["key"]: e.get("string_value") for e in json.loads(out[out.index("{"):])["data"]}
    return payload["S3_ACCESS_KEY_ID"], payload["S3_SECRET_ACCESS_KEY"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["put", "get", "ls"])
    parser.add_argument("src")
    parser.add_argument("dst", nargs="?")
    args = parser.parse_args()

    env = load_env()
    key_id, secret = credentials(env["S3_SECRET"])
    s3 = boto3.client("s3", endpoint_url=env["S3_ENDPOINT"], region_name="eu-north1",
                      aws_access_key_id=key_id, aws_secret_access_key=secret)
    bucket = env["BUCKET"]

    if args.action == "put":
        src = Path(args.src)
        files = [p for p in sorted(src.rglob("*")) if p.is_file()]
        for path in files:
            key = f"{args.dst.rstrip('/')}/{path.relative_to(src).as_posix()}"
            s3.upload_file(str(path), bucket, key)
        print(f"uploaded {len(files)} files to s3://{bucket}/{args.dst}")
    elif args.action == "get":
        dst = Path(args.dst or ".")
        count = 0
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=args.src):
            for obj in page.get("Contents", []):
                target = dst / Path(obj["Key"]).relative_to(args.src.rstrip("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, obj["Key"], str(target))
                count += 1
        print(f"downloaded {count} files to {dst}")
    else:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=args.src):
            for obj in page.get("Contents", []):
                print(f"{obj['Size']:>14,}  {obj['Key']}")


if __name__ == "__main__":
    sys.exit(main())
