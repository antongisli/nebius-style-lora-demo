# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""Contact sheet of an end-to-end run's 2D images: rows are variants, columns are prompts.

    uv run infra/bucket.py get e2e/<run> results/e2e/<run>
    uv run e2e/sheet.py results/e2e/<run> --seed 42 --out results/e2e/<run>-s42.jpg
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run", type=Path)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--variants", help="comma-separated, in row order (default: all, base first)")
parser.add_argument("--prompts", help="comma-separated, in column order (default: all)")
parser.add_argument("--tile", type=int, default=220)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

images = {}
for path in (args.run / "2d").glob(f"*__s{args.seed}.png"):
    variant, prompt, _ = path.stem.split("__")
    images[variant, prompt] = path
variants = args.variants.split(",") if args.variants else sorted({v for v, _ in images}, key=lambda v: (v != "base", v))
prompts = args.prompts.split(",") if args.prompts else sorted({p for _, p in images})

label, head, t = 150, 22, args.tile
sheet = Image.new("RGB", (label + t * len(prompts), head + t * len(variants)), "white")
draw = ImageDraw.Draw(sheet)
for col, prompt in enumerate(prompts):
    draw.text((label + col * t + 6, 5), prompt, fill="black")
for row, variant in enumerate(variants):
    draw.text((6, head + row * t + t // 2 - 6), variant, fill="black")
    for col, prompt in enumerate(prompts):
        if (variant, prompt) in images:
            tile = Image.open(images[variant, prompt]).convert("RGB").resize((t, t), Image.LANCZOS)
            sheet.paste(tile, (label + col * t, head + row * t))
args.out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(args.out, quality=90)
print(args.out)
