#!/usr/bin/env bash
# Submit one Nebius serverless job per config in training/generated/.
# Defaults: preemptible H100 in eu-north1 at the current spot price.
# Reads PROJECT_ID, SUBNET_ID, BUCKET and S3_SECRET from deploy.env (written by infra/setup_infra.sh).
#
#   training/submit_sweep.sh                 # all generated configs
#   training/submit_sweep.sh style-r16-lr1e-4  # one run by name
#   DRY_RUN=1 training/submit_sweep.sh       # print the commands only
#   DATASET=dataset/v3-syn REG_DATASET=dataset/v3-reg training/submit_sweep.sh <name>  # with regularization images
#   JOB_PROJECT_ID=... JOB_SUBNET_ID=... PLATFORM=gpu-b300-sxm PRESET=1gpu-24vcpu-346gb \
#     training/submit_sweep.sh <name>        # run in another region; the bucket stays where it is
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/deploy.env"
PROJECT_ID="${JOB_PROJECT_ID:-$PROJECT_ID}"
SUBNET_ID="${JOB_SUBNET_ID:-$SUBNET_ID}"
: "${PROJECT_ID:?}" "${SUBNET_ID:?}" "${BUCKET:?}" "${S3_SECRET:?}"

PROFILE="${NEBIUS_PROFILE:-default}"
IMAGE="${IMAGE:-ostris/aitoolkit:0.13.39}"
PLATFORM="${PLATFORM:-gpu-h100-sxm}"
PRESET="${PRESET:-1gpu-16vcpu-200gb}"
DATASET="${DATASET:-dataset/v1}"
TIMEOUT="${TIMEOUT:-12h}"
ALLOCATION="${ALLOCATION:-spot}"   # spot | on-demand

case "$ALLOCATION" in
  spot) allocation_flags=(--preemptible --follows-spot-price --restart-policy on-failure) ;;
  on-demand) allocation_flags=(--on-demand --restart-policy never) ;;
  *) echo "ALLOCATION must be spot or on-demand" >&2; exit 2 ;;
esac

configs=()
if (($#)); then
  for name in "$@"; do configs+=("$ROOT/training/generated/$name.yaml"); done
else
  configs=("$ROOT"/training/generated/*.yaml)
fi

for config in "${configs[@]}"; do
  [[ -f "$config" ]] || { echo "missing $config; run training/sweep.py first" >&2; exit 1; }
  name="$(basename "$config" .yaml)"
  cmd=(nebius --profile "$PROFILE" ai job create
    --parent-id "$PROJECT_ID" --subnet-id "$SUBNET_ID"
    --name "$name" --image "$IMAGE"
    --platform "$PLATFORM" --preset "$PRESET" "${allocation_flags[@]}"
    --timeout "$TIMEOUT" --shm-size 32Gi
    --volume "s3://$BUCKET:/data:rw:${S3_PROFILE:-nebius-eu-north1}@$S3_SECRET"
    --inject-file "$config:/cfg/config.yaml"
    --inject-file "$ROOT/training/run_job.sh:/cfg/run_job.sh"
    --env "RUN_NAME=$name" --env "DATASET=$DATASET" ${REG_DATASET:+--env "REG_DATASET=$REG_DATASET"}
    --container-command bash --args /cfg/run_job.sh)
  if [[ -n "${DRY_RUN:-}" ]]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    # create blocks until the job is running (minutes with image pull), so submit in the background and log each job.
    mkdir -p "$ROOT/training/logs"
    "${cmd[@]}" >"$ROOT/training/logs/$name.submit.log" 2>&1 &
    echo "submitted $name (log: training/logs/$name.submit.log)"
    sleep 2
  fi
done
wait
