#!/usr/bin/env bash
# Entrypoint inside each serverless job (injected at /cfg/run_job.sh).
# The bucket is mounted at /data (mountpoint-s3). ai-toolkit writes to local disk,
# because the mount supports neither rename nor overwrite; this script copies
# finished files to the bucket with plain cp. Checkpoints are synced every few minutes and copied back on
# start, so a preempted and restarted job resumes from its last checkpoint.
set -Eeuo pipefail

: "${RUN_NAME:?}" "${DATASET:=dataset/v1}" "${SYNC_SECONDS:=300}"
CONFIG=/cfg/config.yaml
DEST="/data/runs/$RUN_NAME"
OUT="/work/output/$RUN_NAME"

mkdir -p /work/dataset "$OUT" "$DEST"
cp -r "/data/$DATASET/." /work/dataset/
echo "dataset: $(find /work/dataset -name '*.txt' | wc -l) captions, $(find /work/dataset -type f ! -name '*.txt' | wc -l) images"
if [[ -n "${REG_DATASET:-}" ]]; then
  mkdir -p /work/reg && cp -r "/data/$REG_DATASET/." /work/reg/
  echo "regularization: $(find /work/reg -name '*.txt' | wc -l) captions"
fi

# Resume: bring back checkpoints from an earlier attempt of this run.
if compgen -G "$DEST/*.safetensors" >/dev/null; then
  cp "$DEST"/*.safetensors "$OUT"/
  echo "resuming with $(find "$OUT" -maxdepth 1 -name '*.safetensors' | wc -l) checkpoint(s) from the bucket"
fi

sync_out() {
  # The bucket is mounted with mountpoint-s3, which writes new files sequentially
  # but cannot overwrite existing ones. Checkpoints and samples get a new name at
  # every save, so each file is copied once. During training, files modified in
  # the last two minutes are skipped because ai-toolkit may still be writing them;
  # the final pass (sync_out final) copies everything.
  local age=(-mmin +2)
  [[ "${1:-}" == final ]] && age=()
  (cd "$OUT" && find . -type f \( -name '*.safetensors' -o -name '*.jpg' -o -name '*.png' -o -name '*.yaml' \) \
      "${age[@]}" -print0 |
    while IFS= read -r -d '' f; do
      if [[ ! -e "$DEST/$f" ]]; then
        mkdir -p "$DEST/$(dirname "$f")"
        cp "$f" "$DEST/$f"
      fi
    done)
}

( while sleep "$SYNC_SECONDS"; do sync_out || true; done ) &
SYNC_PID=$!

START=$(date +%s)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee /work/gpu.txt
set +e
python /app/ai-toolkit/run.py "$CONFIG"
STATUS=$?
set -e
END=$(date +%s)

kill "$SYNC_PID" 2>/dev/null || true
sync_out final
cat > "$DEST/timing.json" <<JSON
{"run": "$RUN_NAME", "exit_code": $STATUS, "start": $START, "end": $END,
 "seconds": $((END - START)), "gpu": "$(cat /work/gpu.txt)"}
JSON
echo "run $RUN_NAME finished with exit code $STATUS in $((END - START)) s"
exit "$STATUS"
