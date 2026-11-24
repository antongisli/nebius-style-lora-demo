#!/usr/bin/env bash
# Starts one ComfyUI instance pinned to one GPU. Called by comfyui-gpu@<i>.service.
# GPU i listens on 127.0.0.1:(BASE_PORT + i). All instances share one ComfyUI
# checkout, so they share its models, input and output folders. Each instance
# gets its own temp folder (ComfyUI empties it at start) and, except GPU 0,
# its own user folder, so the instances do not share one SQLite database.
set -Eeuo pipefail

gpu="${1:?usage: comfyui-gpu.sh <gpu-index>}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "GPU index must be a number, got '$gpu'" >&2; exit 2; }

COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"
BASE_PORT="${BASE_PORT:-8188}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/comfyui-gpu}"

port=$((BASE_PORT + gpu))
instance_dir="$STATE_DIR/gpu$gpu"
mkdir -p "$instance_dir/temp"

args=(
  "$COMFY_DIR/main.py"
  --listen 127.0.0.1
  --port "$port"
  --cuda-device "$gpu"
  --temp-directory "$instance_dir"
  --disable-auto-launch
)

# GPU 0 is the instance people open in the browser: it keeps the default user
# folder (saved workflows and settings), the manager and live previews.
if (( gpu == 0 )); then
  args+=(--enable-manager --preview-method auto)
else
  mkdir -p "$instance_dir/user"
  args+=(--user-directory "$instance_dir/user")
fi

read -r -a extra_args <<< "${COMFY_EXTRA_ARGS:-}"

cd "$COMFY_DIR"
exec "$COMFY_DIR/.venv/bin/python" "${args[@]}" ${extra_args[@]+"${extra_args[@]}"}
