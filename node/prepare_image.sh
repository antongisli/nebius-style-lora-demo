#!/usr/bin/env bash
# Cleans a set-up demo VM so its boot disk can become a custom image. Run on the VM as the
# user that ran setup.sh (dev), as the last step before stopping the VM:
#
#   demo/node/prepare_image.sh            # then, from the laptop: stop the VM, create the image
#
# A VM launched from the image starts ComfyUI (port 8188) and the grid page (8080) at boot,
# without anyone logging in. cloud-init runs again on the new VM, so the Omarchy Nebius plugin's
# cloud-init adds its own SSH key for the dev user. This script removes everything that should
# not be in the image: SSH keys used during the build, shell history, caches, test outputs,
# cloud-init state and the machine ID.
set -Eeuo pipefail

COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"

step() {
  printf '==> %s\n' "$*"
}

step "Checking that the services start at boot"
[[ "$(loginctl show-user "$(id -un)" -p Linger --value)" == yes ]] || { echo "lingering is off; run setup.sh first" >&2; exit 1; }
systemctl --user is-enabled comfyui-gpu@0.service comfyui-grid.service comfyui-warm.service

step "Stopping the services so nothing writes during the snapshot"
systemctl --user stop comfyui-warm.service comfyui-grid.service 'comfyui-gpu@*.service'
rm -f -- "$HOME/.local/state/comfyui-gpu/warm-status"

step "Removing test outputs, caches and build leftovers"
rm -rf -- "$COMFY_DIR/output/"* "$COMFY_DIR/temp/"* "$COMFY_DIR/input/"*
find "$COMFY_DIR/models" -name '*.aria2' -delete
rm -rf -- "$HOME/.cache/pip" "$HOME/.cache/uv" "$HOME/.local/state/comfyui-gpu/awscli"
rm -f -- "$HOME"/demo/*.safetensors "$HOME"/*.log
rm -rf -- "$HOME/.aws"
sudo apt-get clean

step "Removing SSH keys and shell history"
rm -f -- "$HOME/.ssh/authorized_keys" "$HOME/.bash_history" "$HOME/.python_history" "$HOME/.lesshst"
sudo rm -f /root/.bash_history

step "Resetting cloud-init and the machine ID"
sudo cloud-init clean --logs --machine-id
sudo rm -f /etc/ssh/ssh_host_*
sudo journalctl --rotate >/dev/null 2>&1 || true
sudo journalctl --vacuum-time=1s >/dev/null 2>&1 || true

df -h / | tail -1
printf '\nDone. This session can no longer log in again: stop the VM now and create the image from its boot disk.\n'
