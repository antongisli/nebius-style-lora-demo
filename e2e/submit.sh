#!/usr/bin/env bash
# Submit the end-to-end test (2D with and without the style LoRA, then TRELLIS.2) as one serverless job.
# Reads PROJECT_ID, SUBNET_ID, BUCKET and S3_SECRET from deploy.env. Outputs land in s3://$BUCKET/e2e/<name>/.
#
#   e2e/submit.sh e2e-1                       # all phases
#   PHASES=3d PROMPTS_3D=egg:crown VARIANTS_3D=base:ema3000-s1.0 E2E_DIR=e2e-1 e2e/submit.sh e2e-1-3d-a
#                                             # more GLBs for an existing run, as a separate job
#   DRY_RUN=1 e2e/submit.sh e2e-1             # print the command only
# RESTART=on-failure resumes after preemption; the default (never) avoids restart loops while the job is new.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/deploy.env"
PROJECT_ID="${JOB_PROJECT_ID:-$PROJECT_ID}"
SUBNET_ID="${JOB_SUBNET_ID:-$SUBNET_ID}"
: "${PROJECT_ID:?}" "${SUBNET_ID:?}" "${BUCKET:?}" "${S3_SECRET:?}"
NAME="${1:?usage: e2e/submit.sh <run-name>}"

cmd=(nebius --profile "${NEBIUS_PROFILE:-default}" ai job create
  --parent-id "$PROJECT_ID" --subnet-id "$SUBNET_ID"
  --name "$NAME" --image "${IMAGE:-pytorch/pytorch:2.14.1-cuda13.0-cudnn9-devel}"
  --platform "${PLATFORM:-gpu-h100-sxm}" --preset "${PRESET:-1gpu-16vcpu-200gb}"
  --preemptible --follows-spot-price --restart-policy "${RESTART:-never}"
  --timeout "${TIMEOUT:-6h}" --shm-size 32Gi
  --volume "s3://$BUCKET:/data:rw:${S3_PROFILE:-nebius-eu-north1}@$S3_SECRET"
  --inject-file "$ROOT/e2e/run_e2e.sh:/cfg/run_e2e.sh"
  --inject-file "$ROOT/e2e/driver.py:/cfg/driver.py"
  --inject-file "$ROOT/e2e/prompts.txt:/cfg/prompts.txt"
  --inject-file "$ROOT/e2e/templates.txt:/cfg/templates.txt"
  --inject-file "$ROOT/e2e/prompts_train.txt:/cfg/prompts_train.txt"
  --inject-file "$ROOT/workflows/image_to_3d.json:/cfg/image_to_3d.json"
  --inject-file "$ROOT/workflows/image_to_3d_fast.json:/cfg/image_to_3d_fast.json"
  --env "RUN_NAME=$NAME" --env "PHASES=${PHASES:-2d:3d}" --env "PROMPTS_3D=${PROMPTS_3D:-chest:potion:axe:lantern}"
  --env "LIGHTNING=${LIGHTNING:-1}" --env "STRENGTHS=${STRENGTHS:-1.0:0.6}"
  --env "E2E_DIR=${E2E_DIR:-$NAME}" --env "VARIANTS_3D=${VARIANTS_3D:-}"
  --env "WORKFLOW_3D=${WORKFLOW_3D:-image_to_3d.json}" --env "TAG_3D=${TAG_3D:-}"
  --env "LORAS=${LORAS:-}" --env "TEMPLATE_IDS=${TEMPLATE_IDS:-default}"
  ${EXTRA_LORAS:+--env "EXTRA_LORAS=$EXTRA_LORAS"} --env "SEEDS=${SEEDS:-42:43}" --env "PROMPTS_FILE=${PROMPTS_FILE:-/cfg/prompts.txt}" --env "TRIGGER=${TRIGGER:-nbsstyle}"
  --container-command bash --args /cfg/run_e2e.sh)

if [[ -n "${DRY_RUN:-}" ]]; then
  printf '%q ' "${cmd[@]}"; echo
else
  # create blocks until the job is running (minutes with image pull)
  mkdir -p "$ROOT/training/logs"
  "${cmd[@]}" 2>&1 | tee "$ROOT/training/logs/$NAME.submit.log"
fi
