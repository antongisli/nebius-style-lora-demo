#!/usr/bin/env python3
"""Estimate what finished sweep jobs cost at the spot price they followed.

For each job: GPU count x the H100 spot price in 15-minute steps (average price per step),
integrated over the job's lifetime. Two windows are reported: from creation (the VM exists
while the image pulls) and from container start. The billed amount sits between the two
and is confirmed only by the billing report.

    training/spot_cost.py style-r32-lr2e-4 style-r16-lr1e-4 ...
    training/spot_cost.py --prefix style-       # every job whose name starts with style-
    training/spot_cost.py b300-style-r32-lr2e-4 --project <us-north1 project> --region us-north1
"""
import argparse
import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILE = os.environ.get("NEBIUS_PROFILE", "default")
STEP = timedelta(minutes=15)


def nebius(*args: str) -> dict:
    cmd = f'source deploy.env; nebius --profile {PROFILE} {" ".join(args)} --format json'
    return json.loads(subprocess.run(["bash", "-c", cmd], cwd=ROOT, check=True, capture_output=True, text=True).stdout)


def when(ts: str) -> datetime:
    # nanosecond timestamps: keep microseconds for fromisoformat
    ts = ts.rstrip("Z")
    if "." in ts:
        head, frac = ts.split(".")
        ts = f"{head}.{frac[:6]}"
    return datetime.fromisoformat(ts + "+00:00")


def spot_sku(platform: str, region: str) -> str:
    for item in nebius("billing v1alpha1 spot-price list")["items"]:
        if item["compute_instance"]["platform"] == platform and item["region"] == region:
            return item["sku_id"]
    raise SystemExit(f"no spot SKU for {platform} in {region}")


def prices(sku: str, start: datetime, end: datetime) -> list[tuple[datetime, float]]:
    data = nebius(
        "billing v1alpha1 spot-price list-series",
        "--aggregation-granularity granularity_minute_15 --aggregation-method method_avg_price",
        f"--filters-sku-ids {sku}",
        f"--filters-start-time {(start - STEP).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"--filters-end-time {(end + STEP).strftime('%Y-%m-%dT%H:%M:%SZ')}",
    )
    return sorted((when(p["timestamp"]), float(p["value"])) for s in data["items"] for p in s["points"])


def cost(series: list[tuple[datetime, float]], start: datetime, end: datetime) -> float:
    # each point is taken as the price from its timestamp until the next point
    total = 0.0
    for (t, price), (t_next, _) in zip(series, series[1:] + [(end, 0.0)]):
        lo, hi = max(t, start), min(t_next, end)
        if hi > lo:
            total += price * (hi - lo).total_seconds() / 3600
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--prefix")
    parser.add_argument("--region", default="eu-north1")
    parser.add_argument("--project", help="default: PROJECT_ID from deploy.env")
    args = parser.parse_args()

    project = args.project or subprocess.run(["bash", "-c", "source deploy.env; echo $PROJECT_ID"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    jobs = nebius(f"ai job list --parent-id {project}")["items"]
    picked = [j for j in jobs if j["metadata"]["name"] in args.names or (args.prefix and j["metadata"]["name"].startswith(args.prefix))]
    done = [j for j in picked if j["status"].get("state") == "COMPLETED" and j["status"].get("finished_at")]
    if not done:
        raise SystemExit("no completed jobs matched")

    sku = spot_sku(done[0]["spec"]["platform"], args.region)
    created = [when(j["metadata"]["created_at"]) for j in done]
    finished = [when(j["status"]["finished_at"]) for j in done]
    series = prices(sku, min(created), max(finished))

    print(f"{'run':28} {'GPU-h':>6} {'from create':>12} {'from start':>11}")
    totals = [0.0, 0.0, 0.0]
    for job, c, f in sorted(zip(done, created, finished), key=lambda x: x[0]["metadata"]["name"]):
        gpus = int(job["spec"]["preset"].split("gpu")[0])
        s = when(job["status"]["started_at"])
        row = [gpus * (f - c).total_seconds() / 3600, gpus * cost(series, c, f), gpus * cost(series, s, f)]
        totals = [a + b for a, b in zip(totals, row)]
        print(f"{job['metadata']['name']:28} {row[0]:6.2f} {row[1]:12.2f} {row[2]:11.2f}")
    print(f"{'total':28} {totals[0]:6.2f} {totals[1]:12.2f} {totals[2]:11.2f}  (USD, spot {sku})")
    skipped = sorted({j["metadata"]["name"] for j in picked} - {j["metadata"]["name"] for j in done})
    if skipped:
        print("not finished:", ", ".join(skipped))


if __name__ == "__main__":
    main()
