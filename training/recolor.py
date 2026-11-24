# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Recolour light-theme line illustrations into a two-colour brand scheme.

Built for the Nebius service illustrations: gradient outlines and white or pale-blue
fills on a flat light-grey sheet. Each pixel is split into line work (saturated), fill
(lighter or bluer than the sheet) and background, and each part is repainted.

    uv run training/recolor.py in_dir/ out_dir/ --scheme lime-on-navy --size 1024

Schemes:
  lime-on-navy  Lime lines, slightly lighter navy fills, Deep Blue background
  lime-fill     Deep Blue lines, Lime fills, Light Blue background
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

LIME = np.array([218, 255, 51])
NAVY = np.array([5, 43, 66])
LIGHT = np.array([240, 248, 255])
SCHEMES = {
    "lime-on-navy": dict(background=NAVY, fill=np.array([25, 54, 72]), line=LIME),
    "lime-fill": dict(background=LIGHT, fill=LIME, line=NAVY),
}


def masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    high, low = rgb.max(-1), rgb.min(-1)
    saturation = (high - low) / np.maximum(high, 1e-6)
    line = np.clip((saturation - 0.12) / 0.15, 0, 1)
    # white faces are lighter than the #F2F2F2 sheet; pale-blue fills are bluer than it
    fill = np.clip((rgb.mean(-1) / 255 - 0.955) / 0.02, 0, 1)
    fill = np.maximum(fill, np.clip((rgb[..., 2] - rgb[..., 0] - 4) / 6, 0, 1))
    return line[..., None], (fill * (1 - line))[..., None]


def recolor(path: Path, scheme: dict, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    rgb = np.asarray(image, dtype=np.float64)
    line, fill = masks(rgb)
    out = scheme["background"] * (1 - line - fill) + scheme["fill"] * fill + scheme["line"] * line
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scheme", choices=SCHEMES, default="lime-on-navy")
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.input.glob("*.png")):
        recolor(path, SCHEMES[args.scheme], args.size).save(args.output / path.name)
        print(args.output / path.name)


if __name__ == "__main__":
    main()
