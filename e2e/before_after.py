# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""Before/after sheet: one row per prompt, with the 2D image and a still of its 3D model for each variant.

The 3D stills (renders/<variant>__<prompt>.png) come from capture.html, which renders every GLB from the
same angle in a browser.

    uv run e2e/before_after.py results/e2e/<run> --variants base,ema3000-s1.0 --out results/e2e/<run>-before-after.jpg
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run", type=Path)
parser.add_argument("--variants", default="base,ema3000-s1.0")
parser.add_argument("--labels", default="without LoRA,with style LoRA")
parser.add_argument("--prompts", help="comma-separated (default: every prompt with a render for all variants)")
parser.add_argument("--tag", default="", help="GLB name tag, e.g. -fast")
parser.add_argument("--tile", type=int, default=240)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

variants, labels = args.variants.split(","), args.labels.split(",")
renders = args.run / "renders"
if args.prompts:
    prompts = args.prompts.split(",")
else:
    prompts = sorted({p.stem.split("__")[1].removesuffix(args.tag) for p in renders.glob(f"*__*{args.tag}.png")
                      if not args.tag or p.stem.endswith(args.tag)})
    prompts = [p for p in prompts if "-" not in p
               and all((renders / f"{v}__{p}{args.tag}.png").exists() for v in variants)]

t, label, head = args.tile, 90, 44
sheet = Image.new("RGB", (label + 2 * t * len(variants), head + t * len(prompts)), "white")
draw = ImageDraw.Draw(sheet)
for i, text in enumerate(labels):
    x = label + 2 * t * i
    draw.text((x + 6, 6), text, fill="black")
    draw.text((x + 6, 24), "2D image", fill="gray")
    draw.text((x + t + 6, 24), "TRELLIS.2 model", fill="gray")
for row, prompt in enumerate(prompts):
    y = head + row * t
    draw.text((6, y + t // 2 - 6), prompt, fill="black")
    for i, variant in enumerate(variants):
        x = label + 2 * t * i
        image = Image.open(args.run / "2d" / f"{variant}__{prompt}__s42.png").convert("RGB")
        sheet.paste(image.resize((t, t), Image.LANCZOS), (x, y))
        render = Image.open(renders / f"{variant}__{prompt}{args.tag}.png").convert("RGBA")
        tile = Image.new("RGBA", render.size, "white")
        tile.alpha_composite(render)
        sheet.paste(tile.convert("RGB").resize((t, t), Image.LANCZOS), (x + t, y))
args.out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(args.out, quality=90)
print(args.out, f"({len(prompts)} prompts)")
