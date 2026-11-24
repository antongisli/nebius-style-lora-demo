#!/usr/bin/env bash
# Entrypoint inside the end-to-end test job (injected at /cfg/run_e2e.sh).
# Installs ComfyUI at a pinned commit, fetches the models (cached in the bucket after the first run),
# starts ComfyUI headless on localhost and runs /cfg/driver.py, which renders 2D images with and without
# the style LoRA candidates, turns a subset into GLBs with TRELLIS.2 and copies everything to /data/e2e/$RUN_NAME.
set -Eeuo pipefail

: "${RUN_NAME:?}" "${COMFY_COMMIT:=4845e7a3ffc48a6961f15824239e9042cc7d4ad1}"
# E2E_DIR lets several jobs (one per prompt subset) fill the same output folder
export DEST="/data/e2e/${E2E_DIR:-$RUN_NAME}" COMFY=/work/ComfyUI
mkdir -p /work "$DEST"

df -h /work | tail -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee /work/gpu.txt
START=$(date +%s)

# A restarted job keeps its container filesystem, so skip steps that already ran.
if [[ ! -f "$COMFY/main.py" ]]; then
  python - "$COMFY_COMMIT" "$COMFY" <<'PY'
import io, sys, tarfile, urllib.request
commit, target = sys.argv[1], sys.argv[2]
data = urllib.request.urlopen(f"https://github.com/Comfy-Org/ComfyUI/archive/{commit}.tar.gz").read()
with tarfile.open(fileobj=io.BytesIO(data)) as tar:
    tar.extractall("/work/src", filter="data")
import os
os.replace(f"/work/src/ComfyUI-{commit}", target)
PY
fi
# The image's system Python is marked externally managed (PEP 668); this container is disposable.
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q -r "$COMFY/requirements.txt" huggingface_hub hf_transfer
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)'

python /cfg/driver.py fetch

python "$COMFY/main.py" --listen 127.0.0.1 --port 8188 --disable-auto-launch >/work/comfy.log 2>&1 &
COMFY_PID=$!

set +e
python /cfg/driver.py run
STATUS=$?
set -e
END=$(date +%s)

kill "$COMFY_PID" 2>/dev/null || true
cp /work/comfy.log "$DEST/comfy-$END.log"
tail -40 /work/comfy.log
echo "{\"run\": \"$RUN_NAME\", \"exit_code\": $STATUS, \"seconds\": $((END - START)), \"gpu\": \"$(cat /work/gpu.txt)\"}" \
  >"$DEST/timing-$END.json"
echo "e2e $RUN_NAME finished with exit code $STATUS in $((END - START)) s"
exit "$STATUS"
