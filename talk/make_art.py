# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow", "numpy"]
# ///
"""Generate talk art with the demo box's ComfyUI and the Nebius style LoRA.

    uv run talk/make_art.py gen --server http://127.0.0.1:18188            # 4 seeds per line of talk/art_prompts.txt
    uv run talk/make_art.py sheet                                          # contact sheet of everything generated
    uv run talk/make_art.py cutout village:2 swamp:0 ...                   # remove the background of the chosen seed

Generation uses the same graph as the "1 - Nebius RPG Gen" workflow: Qwen-Image, Lightning 8 steps,
style LoRA at 1.3, the demo's trim prompt. Cut-outs use the box's BiRefNet model; the mask is
applied here, so the PNGs in talk/map/art/ have a transparent background.
"""
import argparse
import io
import json
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "art" / "raw"
CUT = ROOT / "map" / "art"
TEMPLATE = ("nbsstyle, {subject} with lime green trim and details, 3D game asset, stylized, "
            "soft studio lighting, single object, centered, plain white background")


def gen_graph(text: str, seed: int, prefix: str) -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors", "strength_model": 1.0}},
        "5": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["4", 0], "lora_name": "style.safetensors", "strength_model": 1.3}},
        "7": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["5", 0], "shift": 3.1}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["2", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "10": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 4}},
        "12": {"class_type": "KSampler", "inputs": {"model": ["7", 0], "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                                                    "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["10", 0], "denoise": 1.0}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": prefix}},
    }


def mask_graph(image_name: str, prefix: str) -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": "birefnet.safetensors"}},
        "3": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "MaskToImage", "inputs": {"mask": ["3", 0]}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": prefix}},
    }


def call(server: str, path: str, data: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(server + path, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def run(server: str, graph: dict) -> list[dict]:
    """Queue a graph, wait for it, return the saved images' records."""
    body = json.dumps({"prompt": graph, "client_id": str(uuid.uuid4())}).encode()
    pid = json.loads(call(server, "/prompt", body, {"Content-Type": "application/json"}))["prompt_id"]
    while True:
        hist = json.loads(call(server, f"/history/{pid}"))
        if pid in hist:
            status = hist[pid].get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status)[:800])
            return [img for out in hist[pid]["outputs"].values() for img in out.get("images", [])]
        time.sleep(1)


def fetch(server: str, rec: dict) -> Image.Image:
    q = urllib.parse.urlencode({"filename": rec["filename"], "subfolder": rec["subfolder"], "type": rec["type"]})
    return Image.open(io.BytesIO(call(server, f"/view?{q}")))


def prompts(path: Path) -> list[tuple[str, str]]:
    return [tuple(line.split("|", 1)) for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def cmd_gen(args) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for i, (pid, subject) in enumerate(prompts(args.prompts)):
        if args.only and pid not in args.only:
            continue
        t0 = time.time()
        recs = run(args.server, gen_graph(TEMPLATE.format(subject=subject), 1000 + 17 * i, f"art/{pid}"))
        for k, rec in enumerate(recs):
            fetch(args.server, rec).save(RAW / f"{pid}_{k}.png")
        print(f"{pid}: {len(recs)} images in {time.time() - t0:.1f} s")


def cmd_sheet(args) -> None:
    ids = sorted({p.stem.rsplit("_", 1)[0] for p in RAW.glob("*.png")}, key=[p for p, _ in prompts(args.prompts)].index)
    cell = 256
    sheet = Image.new("RGB", (cell * 4 + 160, cell * len(ids)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, pid in enumerate(ids):
        draw.text((8, row * cell + cell // 2), pid, fill="black")
        for k in range(4):
            f = RAW / f"{pid}_{k}.png"
            if f.exists():
                sheet.paste(Image.open(f).convert("RGB").resize((cell, cell)), (160 + k * cell, row * cell))
                draw.text((160 + k * cell + 6, row * cell + 6), str(k), fill="black")
    out = ROOT / "art" / "sheet.jpg"
    sheet.save(out, quality=85)
    print(out)


def cmd_cutout(args) -> None:
    CUT.mkdir(parents=True, exist_ok=True)
    for choice in args.choices:
        pid, k = choice.split(":")
        src = RAW / f"{pid}_{k}.png"
        name = f"talk_{pid}_{k}.png"
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\nContent-Type: image/png\r\n\r\n").encode() \
            + src.read_bytes() + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
        call(args.server, "/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        mask = fetch(args.server, run(args.server, mask_graph(name, f"art_mask/{pid}"))[0]).convert("L")
        rgba = Image.open(src).convert("RGB")
        rgba.putalpha(mask.resize(rgba.size))
        box = Image.fromarray((np.asarray(mask) > 16).astype(np.uint8) * 255).getbbox()
        if box:
            pad = 12
            rgba = rgba.crop((max(box[0] - pad, 0), max(box[1] - pad, 0), min(box[2] + pad, rgba.width), min(box[3] + pad, rgba.height)))
        rgba.thumbnail((512, 512))
        rgba.save(CUT / f"{pid}.png")
        print(f"{pid}: seed {k} -> {CUT / f'{pid}.png'} {rgba.size}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--prompts", type=Path, default=ROOT / "art_prompts.txt")
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--only", nargs="*")
    sub.add_parser("sheet")
    c = sub.add_parser("cutout")
    c.add_argument("choices", nargs="+", help="id:seed-index, e.g. village:2")
    args = parser.parse_args()
    {"gen": cmd_gen, "sheet": cmd_sheet, "cutout": cmd_cutout}[args.cmd](args)


if __name__ == "__main__":
    main()
