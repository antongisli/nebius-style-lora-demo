#!/usr/bin/env bash
# Turns a ComfyUI install from comfyui-setup/comfy.sh into one ComfyUI instance
# per GPU plus the grid page. Run on the VM, as the user that ran comfy.sh.
#
#   LORA_URL=s3://<bucket>/selected/nbsstyle-v3syn-2000.safetensors node/setup.sh
#
# Safe to re-run. Settings (environment variables, all optional):
#   COMFY_DIR         ComfyUI checkout from comfy.sh (default ~/ComfyUI)
#   NUM_GPUS          instances to run (default: number of GPUs nvidia-smi lists)
#   BASE_PORT         port for GPU 0; GPU i uses BASE_PORT+i (default 8188)
#   GRID_PORT         grid page port (default 8080)
#   LORA_URL          style LoRA: s3://bucket/key, https://..., or a file already on the VM
#   S3_ENDPOINT       S3 endpoint (default https://storage.$NEBIUS_REGION.nebius.cloud)
#   NEBIUS_REGION     region of the bucket (default eu-north1)
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or AWS_PROFILE: bucket credentials
#   STYLE_LORA_NAME   name the workflows use for the style LoRA (default style.safetensors)
#   TRIGGER_WORD      word the grid page puts before each prompt (default nbsstyle)
#   HF_TOKEN          Hugging Face token, only needed if a download is rate limited
#   SKIP_MODELS=1     skip the base model downloads
#   READY_TIMEOUT     seconds to wait for all instances (default 600)
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"
BASE_PORT="${BASE_PORT:-8188}"
GRID_PORT="${GRID_PORT:-8080}"
NEBIUS_REGION="${NEBIUS_REGION:-eu-north1}"
S3_ENDPOINT="${S3_ENDPOINT:-https://storage.$NEBIUS_REGION.nebius.cloud}"
STYLE_LORA_NAME="${STYLE_LORA_NAME:-style.safetensors}"
TRIGGER_WORD="${TRIGGER_WORD:-nbsstyle}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"

MODELS_DIR="$COMFY_DIR/models"
CONFIG_DIR="$HOME/.config/comfyui-gpu"
UNIT_DIR="$HOME/.config/systemd/user"
SHARE_DIR="$HOME/.local/share/comfyui-grid"
STATE_DIR="$HOME/.local/state/comfyui-gpu"
BIN_DIR="$HOME/.local/bin"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

step() {
  printf '==> %s\n' "$*"
}

# Copies src to dst only when the content differs. Returns 0 if it changed.
install_if_changed() {
  local src="$1" dst="$2" mode="$3"
  if [[ -f "$dst" ]] && cmp -s "$src" "$dst"; then
    return 1
  fi
  install -D -m "$mode" "$src" "$dst"
  return 0
}

# Downloads url to dir/name with aria2c. A file with no .aria2 control file
# next to it is complete, so it is skipped; an interrupted download resumes.
fetch() {
  local url="$1" dir="$2" name="$3"
  local -a header=()
  if [[ -f "$dir/$name" && ! -f "$dir/$name.aria2" ]]; then
    printf '    have %s\n' "$name"
    return
  fi
  if [[ -n "${HF_TOKEN:-}" && "$url" == https://huggingface.co/* ]]; then
    header=(--header "Authorization: Bearer $HF_TOKEN")
  fi
  printf '    fetching %s\n' "$name"
  mkdir -p "$dir"
  aria2c --continue=true --max-connection-per-server=16 --split=16 --min-split-size=16M \
    --auto-file-renaming=false --allow-overwrite=true \
    --console-log-level=warn --summary-interval=60 \
    ${header[@]+"${header[@]}"} --dir="$dir" --out="$name" "$url"
}

ensure_aws_cli() {
  if command -v aws >/dev/null; then
    AWS=(aws)
    return
  fi
  local venv="$STATE_DIR/awscli"
  if [[ ! -x "$venv/bin/aws" ]]; then
    step "Installing the AWS CLI into $venv (used only for S3 downloads)"
    python3 -m venv "$venv"
    "$venv/bin/pip" install --quiet awscli
  fi
  AWS=("$venv/bin/aws")
}

fetch_style_lora() {
  local url="$1" name tmp
  name="$(basename "${url%%\?*}")"
  [[ "$name" == *.safetensors ]] || die "LORA_URL must point to a .safetensors file: $url"
  mkdir -p "$MODELS_DIR/loras"
  tmp="$MODELS_DIR/loras/.$name.partial"
  case "$url" in
    s3://*)
      ensure_aws_cli
      AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-$NEBIUS_REGION}" \
        "${AWS[@]}" s3 cp --only-show-errors --endpoint-url "$S3_ENDPOINT" "$url" "$tmp"
      ;;
    https://*|http://*)
      rm -f "$tmp"
      aria2c --console-log-level=warn --auto-file-renaming=false --allow-overwrite=true \
        --dir="$MODELS_DIR/loras" --out=".$name.partial" "$url"
      ;;
    *)
      [[ -f "$url" ]] || die "LORA_URL must be s3://..., https://... or an existing file: $url"
      cp -f -- "$url" "$tmp"
      ;;
  esac
  mv -f "$tmp" "$MODELS_DIR/loras/$name"
  if [[ "$name" != "$STYLE_LORA_NAME" ]]; then
    ln -sfn "$name" "$MODELS_DIR/loras/$STYLE_LORA_NAME"
  fi
  printf '    style LoRA: loras/%s -> %s\n' "$STYLE_LORA_NAME" "$name"
}

step "Preflight checks"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found; the NVIDIA driver is missing"
command -v aria2c >/dev/null || die "aria2c not found; run comfyui-setup/comfy.sh first"
command -v curl >/dev/null || die "curl not found"
[[ -x "$COMFY_DIR/.venv/bin/python" && -f "$COMFY_DIR/main.py" ]] \
  || die "No ComfyUI install at $COMFY_DIR; run comfyui-setup/comfy.sh first"
systemctl --user show-environment >/dev/null 2>&1 \
  || die "systemctl --user is not available in this session; log in over SSH as this user"

gpu_count="$(nvidia-smi --list-gpus | grep -c '^GPU ' || true)"
NUM_GPUS="${NUM_GPUS:-$gpu_count}"
[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || die "NUM_GPUS must be a positive number, got '$NUM_GPUS'"
(( NUM_GPUS <= gpu_count )) || die "NUM_GPUS=$NUM_GPUS but nvidia-smi lists $gpu_count GPUs"
printf '    %s GPUs, ports %s-%s, grid page on %s\n' \
  "$NUM_GPUS" "$BASE_PORT" "$((BASE_PORT + NUM_GPUS - 1))" "$GRID_PORT"

available_kb="$(df -Pk "$COMFY_DIR" | awk 'NR == 2 {print $4}')"
if [[ -z "${SKIP_MODELS:-}" && "$available_kb" =~ ^[0-9]+$ ]] && (( available_kb < 100 * 1024 * 1024 )); then
  printf 'WARNING: less than 100 GiB free under %s; the models need about 65 GB.\n' "$COMFY_DIR" >&2
fi

if [[ -z "${SKIP_MODELS:-}" ]]; then
  step "Downloading models into $MODELS_DIR (about 65 GB on first run)"
  # Qwen-Image, repackaged for ComfyUI (Apache 2.0). The bf16 weights are used
  # because the Lightning LoRA shows grid artefacts on the unscaled fp8 file.
  qwen=https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files
  fetch "$qwen/diffusion_models/qwen_image_bf16.safetensors" "$MODELS_DIR/diffusion_models" qwen_image_bf16.safetensors
  fetch "$qwen/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" "$MODELS_DIR/text_encoders" qwen_2.5_vl_7b_fp8_scaled.safetensors
  fetch "$qwen/vae/qwen_image_vae.safetensors" "$MODELS_DIR/vae" qwen_image_vae.safetensors
  # Qwen-Image-Lightning 8-step V2.0 LoRA (Apache 2.0, lightx2v).
  fetch https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors \
    "$MODELS_DIR/loras" Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors
  # TRELLIS.2, repackaged for ComfyUI (MIT). The image encoder is DINOv3, whose
  # upstream weights use Meta's DINOv3 licence; see node/README.md.
  trellis=https://huggingface.co/Comfy-Org/TRELLIS.2/resolve/main
  fetch "$trellis/diffusion_models/trellis_2_bf16.safetensors" "$MODELS_DIR/diffusion_models" trellis_2_bf16.safetensors
  fetch "$trellis/clip_vision/dino_v3_vit_l.safetensors" "$MODELS_DIR/clip_vision" dino_v3_vit_l.safetensors
  fetch "$trellis/vae/trellis_2_shape_vae_bf16.safetensors" "$MODELS_DIR/vae" trellis_2_shape_vae_bf16.safetensors
  fetch "$trellis/vae/trellis_2_texture_vae_bf16.safetensors" "$MODELS_DIR/vae" trellis_2_texture_vae_bf16.safetensors
  # BiRefNet background removal for the 3D input (MIT).
  fetch https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors \
    "$MODELS_DIR/background_removal" birefnet.safetensors
fi

if [[ -n "${LORA_URL:-}" ]]; then
  step "Fetching the style LoRA"
  fetch_style_lora "$LORA_URL"
elif [[ ! -e "$MODELS_DIR/loras/$STYLE_LORA_NAME" ]]; then
  printf 'NOTE: no LORA_URL and no loras/%s; the grid runs without the style LoRA.\n' "$STYLE_LORA_NAME"
fi

step "Installing the per-GPU services and the grid page"
changed=0
mkdir -p "$CONFIG_DIR" "$STATE_DIR"
env_tmp="$(mktemp)"
trap 'rm -f "$env_tmp"' EXIT
cat > "$env_tmp" <<EOF
# Written by node/setup.sh; re-run it to change these.
COMFY_DIR=$COMFY_DIR
STATE_DIR=$STATE_DIR
NUM_GPUS=$NUM_GPUS
BASE_PORT=$BASE_PORT
GRID_PORT=$GRID_PORT
STYLE_LORA_NAME=$STYLE_LORA_NAME
TRIGGER_WORD=$TRIGGER_WORD
WORKFLOWS_DIR=$SHARE_DIR/workflows
EOF
install_if_changed "$env_tmp" "$CONFIG_DIR/env" 0644 && changed=1
install_if_changed "$SCRIPT_DIR/comfyui-gpu.sh" "$BIN_DIR/comfyui-gpu" 0755 && changed=1
install_if_changed "$SCRIPT_DIR/systemd/comfyui-gpu@.service" "$UNIT_DIR/comfyui-gpu@.service" 0644 && changed=1
# Runs at each boot. It matters on a VM booted from the image, whose disk fills in on first read.
install_if_changed "$SCRIPT_DIR/comfyui-warm.py" "$BIN_DIR/comfyui-warm" 0755 || true
install_if_changed "$SCRIPT_DIR/systemd/comfyui-warm.service" "$UNIT_DIR/comfyui-warm.service" 0644 || true

grid_changed=0
install_if_changed "$SCRIPT_DIR/systemd/comfyui-grid.service" "$UNIT_DIR/comfyui-grid.service" 0644 && grid_changed=1
for file in "$SCRIPT_DIR"/dispatcher/*.py "$SCRIPT_DIR"/dispatcher/*.html; do
  [[ "$(basename "$file")" == test_* ]] && continue
  install_if_changed "$file" "$SHARE_DIR/dispatcher/$(basename "$file")" 0644 && grid_changed=1
done
for file in "$REPO_DIR"/workflows/*.json; do
  install_if_changed "$file" "$SHARE_DIR/workflows/$(basename "$file")" 0644 && grid_changed=1
done
(( changed )) && grid_changed=1

# The Nebius Lime Snap node, and the named workflows for ComfyUI's Workflows sidebar (GPU 0's
# default user folder). Saving a workflow in the browser overwrites its copy there; re-running
# setup.sh puts the repository version back.
for file in "$SCRIPT_DIR"/comfyui_nebius/*.py; do
  install_if_changed "$file" "$COMFY_DIR/custom_nodes/comfyui_nebius/$(basename "$file")" 0644 && changed=1
done
for file in "$REPO_DIR"/workflows/ui/*.json; do
  [[ -e "$file" ]] || continue
  install -D -m 0644 "$file" "$COMFY_DIR/user/default/workflows/$(basename "$file")"
done

# The single service from comfy.sh also uses port 8188.
if systemctl --user cat comfyui.service >/dev/null 2>&1; then
  step "Stopping and disabling comfyui.service from comfy.sh"
  systemctl --user disable --now comfyui.service
fi

# Keep the user's services running after SSH logout and across reboots.
if command -v loginctl >/dev/null && [[ "$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null)" != yes ]]; then
  sudo loginctl enable-linger "$(id -un)"
fi

systemctl --user daemon-reload

# Remove instances left from an earlier run with more GPUs.
while read -r unit; do
  index="${unit#comfyui-gpu@}"
  index="${index%.service}"
  if [[ "$index" =~ ^[0-9]+$ ]] && (( index >= NUM_GPUS )); then
    systemctl --user disable --now "$unit"
  fi
done < <(
  {
    systemctl --user list-units --all --plain --no-legend 'comfyui-gpu@*.service' | awk '{print $1}'
    find "$UNIT_DIR/default.target.wants" -maxdepth 1 -name 'comfyui-gpu@*.service' -printf '%f\n' 2>/dev/null
  } | sort -u
)

units=()
for ((i = 0; i < NUM_GPUS; i++)); do
  units+=("comfyui-gpu@$i.service")
done
systemctl --user enable "${units[@]}" comfyui-grid.service comfyui-warm.service >/dev/null
if (( changed )); then
  systemctl --user restart "${units[@]}"
else
  systemctl --user start "${units[@]}"
fi
if (( grid_changed )); then
  systemctl --user restart comfyui-grid.service
else
  systemctl --user start comfyui-grid.service
fi

step "Waiting for all instances to answer /system_stats (up to ${READY_TIMEOUT}s)"
deadline=$((SECONDS + READY_TIMEOUT))
pending=("${units[@]}")
while ((${#pending[@]} > 0)) && ((SECONDS < deadline)); do
  still=()
  for unit in "${pending[@]}"; do
    index="${unit#comfyui-gpu@}"
    index="${index%.service}"
    if curl -fsS --max-time 3 -o /dev/null "http://127.0.0.1:$((BASE_PORT + index))/system_stats"; then
      printf '    GPU %s ready on port %s\n' "$index" "$((BASE_PORT + index))"
    else
      still+=("$unit")
    fi
  done
  pending=("${still[@]+"${still[@]}"}")
  ((${#pending[@]} > 0)) && sleep 2
done

if ((${#pending[@]} > 0)); then
  for unit in "${pending[@]}"; do
    printf '\n--- %s did not become ready; last log lines:\n' "$unit" >&2
    journalctl --user -u "$unit" -n 30 --no-pager >&2 || true
  done
  die "${#pending[@]} of $NUM_GPUS instances are not ready"
fi

if curl -fsS --max-time 3 -o /dev/null "http://127.0.0.1:$GRID_PORT/api/state"; then
  printf '    grid page ready on port %s\n' "$GRID_PORT"
else
  journalctl --user -u comfyui-grid.service -n 30 --no-pager >&2 || true
  die "the grid page is not answering on port $GRID_PORT"
fi

printf '\nAll %s instances are ready.\n' "$NUM_GPUS"
printf 'Forward these VM ports to the same laptop ports:\n'
printf '  %s  ComfyUI on GPU 0\n' "$BASE_PORT"
printf '  %s  grid page (all GPUs)\n' "$GRID_PORT"
printf 'Logs: journalctl --user -u comfyui-gpu@0 -f   (or comfyui-grid)\n'
