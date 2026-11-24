#!/usr/bin/env python3
"""Warms a VM booted from the demo image, so the first job in the browser is fast.

A boot disk created from an image fills in lazily: each block comes from image
storage the first time it is read. Without this, the first job reads about 65 GB
of model files at that speed and takes over 8 minutes. This script runs once at
boot (comfyui-warm.service):

1. Reads every model file, and then the ComfyUI virtualenv, with many reads in
   parallel. That fills the disk in and puts the files in the page cache.
2. Queues one small generation on each ComfyUI instance, so the Qwen-Image
   weights are on the GPU before anyone opens the page. The result is a preview
   (temp folder), so it never shows up among the outputs.

Progress goes to the journal and to $STATE_DIR/warm-status:
    systemctl --user status comfyui-warm
Settings come from ~/.config/comfyui-gpu/env (written by node/setup.sh).
"""

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOME = Path.home()
COMFY_DIR = Path(os.environ.get("COMFY_DIR", HOME / "ComfyUI"))
STATE_DIR = Path(os.environ.get("STATE_DIR", HOME / ".local/state/comfyui-gpu"))
WORKFLOWS_DIR = Path(os.environ.get("WORKFLOWS_DIR", HOME / ".local/share/comfyui-grid/workflows"))
BASE_PORT = int(os.environ.get("BASE_PORT", "8188"))
NUM_GPUS = int(os.environ.get("NUM_GPUS", "1"))
READERS = int(os.environ.get("WARM_READERS", "32"))
TIMEOUT = int(os.environ.get("WARM_TIMEOUT", "1800"))

CHUNK = 64 << 20
# Read first: the files the first demo workflow loads.
FIRST = [
    "diffusion_models/qwen_image_bf16.safetensors",
    "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "vae/qwen_image_vae.safetensors",
    "loras/Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors",
    "loras/style.safetensors",
]

started = time.monotonic()


def report(message):
    line = f"[{time.monotonic() - started:6.0f} s] {message}"
    print(line, flush=True)
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        (STATE_DIR / "warm-status").write_text(line + "\n")
    except OSError:
        pass


def files_in_order():
    models = COMFY_DIR / "models"
    seen, ordered = set(), []

    def add(path):
        try:
            real = path.resolve(strict=True)
        except OSError:
            return
        if real.is_file() and real not in seen:
            seen.add(real)
            ordered.append(real)

    for name in FIRST:
        add(models / name)
    for root in (models, COMFY_DIR / ".venv", COMFY_DIR / "custom_nodes"):
        for path in sorted(root.rglob("*")):
            add(path)
    return ordered


def read_chunk(job):
    path, offset = job
    try:
        with open(path, "rb", buffering=0) as handle:
            return len(os.pread(handle.fileno(), CHUNK, offset))
    except OSError:
        return 0


def read_everything():
    files = files_in_order()
    jobs = [(path, offset) for path in files for offset in range(0, path.stat().st_size, CHUNK)]
    total = sum(path.stat().st_size for path in files)
    report(f"reading {len(files)} files, {total / 1e9:.1f} GB, {READERS} reads in parallel")
    done, last = 0, time.monotonic()
    with ThreadPoolExecutor(READERS) as pool:
        for size in pool.map(read_chunk, jobs):
            done += size
            if time.monotonic() - last > 15:
                last = time.monotonic()
                report(f"read {done / 1e9:.1f} of {total / 1e9:.1f} GB")
    seconds = time.monotonic() - started
    report(f"read {done / 1e9:.1f} GB in {seconds:.0f} s ({done / 1e6 / max(seconds, 1):.0f} MB/s)")


def api(port, path, body=None):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def warm_prompt():
    prompt = json.loads((WORKFLOWS_DIR / "grid_api.json").read_text())
    for node in prompt.values():
        if node["class_type"] == "SaveImage":
            node["class_type"] = "PreviewImage"
            node["inputs"] = {"images": node["inputs"]["images"]}
        elif node["class_type"] == "KSampler":
            node["inputs"]["seed"] = random.randrange(2**32)
    return prompt


def warm_instance(port, deadline):
    while True:
        try:
            api(port, "/system_stats")
            break
        except (OSError, urllib.error.URLError):
            if time.monotonic() > deadline:
                raise TimeoutError(f"ComfyUI on port {port} did not start")
            time.sleep(2)
    queued = time.monotonic()
    prompt_id = api(port, "/prompt", {"prompt": warm_prompt(), "client_id": "comfyui-warm"})["prompt_id"]
    while time.monotonic() < deadline:
        entry = api(port, f"/history/{prompt_id}").get(prompt_id)
        if entry and entry.get("status", {}).get("completed"):
            return time.monotonic() - queued
        if entry and entry.get("status", {}).get("status_str") == "error":
            raise RuntimeError(f"warm-up job failed on port {port}: {entry['status']}")
        time.sleep(2)
    raise TimeoutError(f"warm-up job on port {port} did not finish")


def main():
    read_everything()
    deadline = started + TIMEOUT
    ports = [BASE_PORT + i for i in range(NUM_GPUS)]
    report(f"loading Qwen-Image onto {len(ports)} GPU(s)")
    with ThreadPoolExecutor(len(ports)) as pool:
        seconds = list(pool.map(lambda port: warm_instance(port, deadline), ports))
    report(f"warm: first job took {max(seconds):.0f} s; total {time.monotonic() - started:.0f} s since boot service start")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - report any failure in the status file
        report(f"warm-up failed: {error}")
        sys.exit(1)
