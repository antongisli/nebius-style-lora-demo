# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Score an end-to-end run's 2D images on two axes: how much Lime they have, and how much of the base model's look they keep.

lime: the percentage of pixels close to Nebius Lime (#DAFF33). The base model scores 0 on every test item.
It ignores where the Lime is (trim or background), so read it next to the contact sheet.

kept: how much of the base image's non-green colour (wood, gold, metal, gems) survives, as the overlap of
colour histograms between an image and the base image with the same prompt and seed (1.0 = the same palette).
Green and yellow-green pixels and the light background are left out, so adding Lime trim costs nothing, but
turning wood into navy panels does. A crude stand-in for "did the LoRA keep the materials".

    uv run e2e/lime_score.py results/e2e/<run>
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

LIME = np.array([218, 255, 51])
RADIUS = 70  # RGB distance; wide enough for shaded Lime, narrow enough to exclude gold and yellow
BINS = 8

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run", type=Path)
parser.add_argument("--variants", help="comma-separated variants to show (default: all)")
args = parser.parse_args()


def load(path):
    return np.asarray(Image.open(path).convert("RGB").resize((128, 128)), dtype=np.float64)


def lime_share(rgb):
    return 100 * (((rgb - LIME) ** 2).sum(-1) < RADIUS ** 2).mean()


def palette(rgb):
    """Normalised RGB histogram of the object's non-green pixels."""
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    r, g, b = np.moveaxis(rgb, -1, 0)
    greenish = (g >= r * 0.9) & (g > b * 1.3) & (sat > 0.25)  # yellow-green to green, including Lime
    background = (sat < 0.12) & (mx > 200)
    keep = ~greenish & ~background
    hist, _ = np.histogramdd(rgb[keep], bins=BINS, range=[(0, 256)] * 3)
    return hist / max(hist.sum(), 1)


def base_of(variant):
    return "base" + (variant[variant.index("+"):] if "+" in variant else "")


images = {p.stem: p for p in sorted((args.run / "2d").glob("*.png"))}
lime, kept = defaultdict(dict), defaultdict(dict)
for stem, path in images.items():
    variant, item, seed = stem.split("__")
    rgb = load(path)
    lime[variant][item, seed] = lime_share(rgb)
    base = images.get(f"{base_of(variant)}__{item}__{seed}")
    if base is not None:
        kept[variant][item, seed] = np.minimum(palette(rgb), palette(load(base))).sum()

shown = args.variants.split(",") if args.variants else sorted(lime, key=lambda v: (not v.startswith("base"), v))
items = sorted({item for per in lime.values() for item, _ in per})
print(f"{'variant':34} {'lime':>5} {'≥3%':>5} {'kept':>5}  " + " ".join(f"{i[:6]:>6}" for i in items))
for variant in shown:
    per, kp = lime[variant], kept[variant]
    cells = " ".join(f"{np.mean([kp[i, s] for s in ('s42', 's43') if (i, s) in kp] or [0]):6.2f}" for i in items)
    print(f"{variant:34} {np.mean(list(per.values())):5.1f} {sum(s >= 3 for s in per.values()):2d}/{len(per):<2d} "
          f"{np.mean(list(kp.values())):5.2f}  {cells}")
print("\nlime: mean % Lime pixels; ≥3%: images with at least 3% Lime; kept: palette overlap with the base image "
      "(per-item columns: kept, mean over seeds)")
