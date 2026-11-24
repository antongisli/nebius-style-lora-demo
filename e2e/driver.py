"""Drive a headless ComfyUI through its HTTP API for the end-to-end test (runs inside the job).

    python driver.py fetch   # models into $COMFY/models (bucket cache first, Hugging Face otherwise), LoRAs from runs/
    python driver.py run     # 2D images for every variant, then TRELLIS.2 GLBs for a subset

Variants are the base model without a style LoRA and each candidate checkpoint at each strength. Every variant
gets the same prompt text and seeds, so the images compare like with like. Outputs go to $DEST/2d, $DEST/3d and
$DEST/meta (one JSON per item with its settings and seconds). Items already in $DEST are skipped, so a preempted
job that restarts carries on where it stopped. Standard library only, apart from huggingface_hub for fetch.
"""
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

COMFY = Path(os.environ.get("COMFY", "/work/ComfyUI"))
DEST = Path(os.environ["DEST"])
CACHE = Path(os.environ.get("MODEL_CACHE", "/data/cache/comfy-models"))
URL = "http://127.0.0.1:8188"

# (path under models/, Hugging Face repo, file in repo); the same files as node/setup.sh
MODELS = [
    ("diffusion_models/qwen_image_bf16.safetensors", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_bf16.safetensors"),
    ("text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    ("vae/qwen_image_vae.safetensors", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors"),
    ("loras/Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors", "lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"),
    ("diffusion_models/trellis_2_bf16.safetensors", "Comfy-Org/TRELLIS.2", "diffusion_models/trellis_2_bf16.safetensors"),
    ("clip_vision/dino_v3_vit_l.safetensors", "Comfy-Org/TRELLIS.2", "clip_vision/dino_v3_vit_l.safetensors"),
    ("vae/trellis_2_shape_vae_bf16.safetensors", "Comfy-Org/TRELLIS.2", "vae/trellis_2_shape_vae_bf16.safetensors"),
    ("vae/trellis_2_texture_vae_bf16.safetensors", "Comfy-Org/TRELLIS.2", "vae/trellis_2_texture_vae_bf16.safetensors"),
    ("background_removal/birefnet.safetensors", "Comfy-Org/BiRefNet", "background_removal/birefnet.safetensors"),
]

# Style LoRA candidates from sweeps v1 and v3 (docs/sweep-v3.md), by bucket path
CANDIDATES = {
    "v1w2000": "runs/style-r32-lr2e-4/style-r32-lr2e-4_000002000.safetensors",
    "ema3000": "runs/b300bs4ema-r32-lr2e-4/b300bs4ema-r32-lr2e-4.safetensors",
    "long4500": "runs/long1-r32-lr2e-4/long1-r32-lr2e-4_000004500.safetensors",
}
# lists in env vars are separated by ":" because the CLI may split --env values on commas
# EXTRA_LORAS adds candidates as name=bucket-path entries, e.g. v3syn2000=runs/v3syn-r32-lr2e-4/v3syn-r32-lr2e-4_000002000.safetensors
CANDIDATES.update(entry.split("=", 1) for entry in os.environ.get("EXTRA_LORAS", "").split(":") if "=" in entry)
# LORAS=none renders the base model only, e.g. to make training or regularization images
LORAS = [] if os.environ.get("LORAS") == "none" else os.environ["LORAS"].split(":") if os.environ.get("LORAS") else list(CANDIDATES)
STRENGTHS = [float(s) for s in os.environ.get("STRENGTHS", "1.0:0.6").split(":")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42:43").split(":")]
TRIGGER = os.environ.get("TRIGGER", "nbsstyle").replace("none", "")  # TRIGGER=none: no trigger word
# Prompt templates by id (e2e/templates.txt); variants with a template other than "default" get "+<id>" in their name
TEMPLATES_FILE = Path(os.environ.get("TEMPLATES_FILE", "/cfg/templates.txt"))
TEMPLATES = dict(line.split("|", 1) for line in TEMPLATES_FILE.read_text().splitlines()
                 if line.strip() and not line.startswith("#")) if TEMPLATES_FILE.exists() else {}
TEMPLATES.setdefault("default", "{trigger}, {subject}, 3D game asset, stylized, soft studio lighting, single object, centered, plain white background")
TEMPLATE_IDS = os.environ.get("TEMPLATE_IDS", "default").split(":")
VARIANTS = [(name + ("" if tid == "default" else f"+{tid}"), lora, strength, TEMPLATES[tid])
            for tid in TEMPLATE_IDS
            for name, lora, strength in [("base", None, 0.0)] + [(f"{c}-s{s}", c, s) for c in LORAS for s in STRENGTHS]]
LIGHTNING = os.environ.get("LIGHTNING", "1") == "1"
PROMPTS_3D = os.environ.get("PROMPTS_3D", "chest:potion:axe:lantern").split(":")
PHASES = os.environ.get("PHASES", "2d:3d").split(":")
WORKFLOW_3D = os.environ.get("WORKFLOW_3D", "image_to_3d.json")  # in /cfg
TAG_3D = os.environ.get("TAG_3D", "")  # appended to GLB names, so runs with other 3D settings don't collide
VARIANTS_3D = os.environ.get("VARIANTS_3D", "").split(":") if os.environ.get("VARIANTS_3D") else None


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def fetch():
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    for rel, repo, name in MODELS:
        local, cached = COMFY / "models" / rel, CACHE / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        if cached.exists():
            # mountpoint-s3 only shows a file once its upload has completed, so an existing file is complete
            shutil.copyfile(cached, local)
            log(f"{rel}: copied from bucket cache in {time.time() - t:.0f} s")
        else:
            path = hf_hub_download(repo, name, local_dir="/work/dl")
            cached.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, cached)
            os.replace(path, local)
            log(f"{rel}: downloaded from {repo} and cached in {time.time() - t:.0f} s")
    for name, key in CANDIDATES.items():
        shutil.copyfile(Path("/data") / key, COMFY / "models/loras" / f"{name}.safetensors")
    log("LoRAs:", ", ".join(CANDIDATES))


def api(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{path}: HTTP {error.code}: {error.read().decode()[:2000]}") from None


def wait_ready(limit=900):
    end = time.time() + limit
    while time.time() < end:
        try:
            stats = api("/system_stats")
            log("ComfyUI", stats.get("system", {}).get("comfyui_version"), "is up")
            return
        except (OSError, RuntimeError):
            time.sleep(5)
    raise RuntimeError("ComfyUI did not start; see comfy.log")


def execute(graph):
    """Queue a graph, wait for it, and return the paths of the files it wrote."""
    prompt_id = api("/prompt", {"prompt": graph})["prompt_id"]
    while True:
        entry = api(f"/history/{prompt_id}").get(prompt_id)
        if entry and entry.get("status", {}).get("completed"):
            break
        if entry and entry.get("status", {}).get("status_str") == "error":
            raise RuntimeError(json.dumps(entry["status"].get("messages"))[:3000])
        time.sleep(2)
    files = []
    for output in entry.get("outputs", {}).values():
        for items in output.values():
            for item in items if isinstance(items, list) else []:
                if isinstance(item, dict) and "filename" in item:
                    files.append(COMFY / item.get("type", "output") / item.get("subfolder", "") / item["filename"])
    return files


def image_graph(prompt, seed, lora, strength, prefix):
    model = ["1", 0]
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
    }
    if LIGHTNING:
        graph["4"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": model, "lora_name": "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors", "strength_model": 1.0}}
        model = ["4", 0]
    if lora:
        graph["5"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": model, "lora_name": f"{lora}.safetensors", "strength_model": strength}}
        model = ["5", 0]
    steps, cfg = (8, 1.0) if LIGHTNING else (30, 4.0)
    graph.update({
        "6": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": 3.1}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "9": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "10": {"class_type": "KSampler", "inputs": {
            "model": ["6", 0], "positive": ["7", 0], "negative": ["8", 0], "latent_image": ["9", 0], "seed": seed,
            "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": prefix}},
    })
    return graph


def comfy_log_lines(state={"offset": 0}):
    """New lines of ComfyUI's log that show where time went: finished progress bars and total prompt time."""
    log_path = Path("/work/comfy.log")
    if not log_path.exists():
        return []
    with log_path.open("rb") as f:
        f.seek(state["offset"])
        text = f.read().decode(errors="replace")
        state["offset"] = f.tell()
    lines = [line.strip() for line in text.replace("\r", "\n").splitlines()]
    return [line for line in lines if "Prompt executed" in line or "100%|" in line]


def save_meta(name, **fields):
    (DEST / "meta").mkdir(parents=True, exist_ok=True)
    path = DEST / "meta" / f"{name}.json"
    if not path.exists():
        path.write_text(json.dumps(fields))


def run():
    prompts = [line.split("|", 1) for line in Path(os.environ.get("PROMPTS_FILE", "/cfg/prompts.txt")).read_text().splitlines()
               if line.strip() and not line.startswith("#")]
    wait_ready()
    failures = []
    (DEST / "2d").mkdir(parents=True, exist_ok=True)
    (COMFY / "input").mkdir(parents=True, exist_ok=True)
    if "2d" in PHASES:
        # variant outermost, so ComfyUI keeps one LoRA patched for a whole block of images
        for variant, lora, strength, template in VARIANTS:
            for pid, subject in prompts:
                for seed in SEEDS:
                    name = f"{variant}__{pid}__s{seed}"
                    if (DEST / "2d" / f"{name}.png").exists():
                        continue
                    prompt = template.format(trigger=TRIGGER, subject=subject).removeprefix(", ")  # TRIGGER="" drops it
                    t = time.time()
                    try:
                        files = execute(image_graph(prompt, seed, lora, strength, f"2d/{name}"))
                    except RuntimeError as error:
                        log(f"2D {name} failed: {error}")
                        failures.append(name)
                        continue
                    seconds = time.time() - t
                    shutil.copyfile(files[0], DEST / "2d" / f"{name}.png")
                    save_meta(name, kind="2d", variant=variant, lora=lora and CANDIDATES[lora], strength=strength,
                              prompt=prompt, seed=seed, lightning=LIGHTNING, seconds=round(seconds, 1))
                    log(f"2D {name}: {seconds:.1f} s")
    if "3d" in PHASES:
        # unload Qwen-Image first; with it resident, TRELLIS.2 took 7 to 10 minutes per object instead of under 2
        request = urllib.request.Request(URL + "/free", data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                         headers={"Content-Type": "application/json"})
        urllib.request.urlopen(request, timeout=60).read()
        workflow = json.loads((Path("/cfg") / WORKFLOW_3D).read_text())
        (DEST / "3d").mkdir(parents=True, exist_ok=True)
        for pid in PROMPTS_3D:
            for variant, *_ in VARIANTS:
                if VARIANTS_3D and variant not in VARIANTS_3D:
                    continue
                image = f"{variant}__{pid}__s{SEEDS[0]}.png"
                name = f"{variant}__{pid}{TAG_3D}"
                if (DEST / "3d" / f"{name}.glb").exists() or not (DEST / "2d" / image).exists():
                    continue
                shutil.copyfile(DEST / "2d" / image, COMFY / "input" / image)
                graph = json.loads(json.dumps(workflow))
                for node in graph.values():
                    if node["class_type"] == "LoadImage":
                        node["inputs"]["image"] = image
                    if node["class_type"] == "SaveGLB":
                        node["inputs"]["filename_prefix"] = f"3d/{name}"
                t = time.time()
                try:
                    files = [f for f in execute(graph) if f.suffix == ".glb"]
                    if not files:
                        raise RuntimeError("no GLB in the outputs")
                except RuntimeError as error:
                    log(f"3D {name} failed: {error}")
                    failures.append(name)
                    continue
                seconds = time.time() - t
                shutil.copyfile(files[0], DEST / "3d" / f"{name}.glb")
                save_meta(f"{name}__3d", kind="3d", variant=variant, image=image, workflow=WORKFLOW_3D,
                          seconds=round(seconds, 1), bytes=files[0].stat().st_size)
                for line in comfy_log_lines():
                    log("   ", line[-160:])
                log(f"3D {name}: {seconds:.0f} s, {files[0].stat().st_size / 1e6:.1f} MB")
    log(f"done, {len(failures)} failure(s): {', '.join(failures) or 'none'}")
    if failures:
        (DEST / f"failures-{int(time.time())}.json").write_text(json.dumps(failures))


if __name__ == "__main__":
    {"fetch": fetch, "run": run}[sys.argv[1]]()
