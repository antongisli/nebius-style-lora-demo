# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "scipy"]
# ///
"""Prepare generated images (e.g. from ChatGPT) for training in the Lime-on-Deep-Blue style.

For each image in the input folder: centre-crop to a square, resize, and snap every pixel
to the three dataset colours (Deep Blue background, lighter navy fill, Lime line work),
keeping a soft edge so lines stay anti-aliased. Glow and gradients collapse to flat colour.
Then write a caption, a manifest row marked as generated, and a contact sheet for review.

Name each input file after its subject, e.g. battle-axe.png or battle-axe-2.png, which
becomes the caption "nbsstyle, a battle axe, centered, plain background". A .txt file with
the same stem overrides the caption. Subjects held out for the sample prompts are refused.

    uv run training/snap_palette.py dataset/gen/raw dataset/gen/processed
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

BACKGROUND = np.array([5, 43, 66])
FILL = np.array([25, 54, 72])
LIME = np.array([218, 255, 51])
# Subjects in training/prompts.txt; keeping them out of training shows the style transfers.
HELD_OUT = ["chest", "sword", "dagger", "potion", "shield", "key", "cloak", "helmet"]
SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def snap(image: Image.Image, size: int) -> Image.Image:
    image = ImageOps.fit(image.convert("RGB"), (size, size), Image.LANCZOS)
    rgb = np.asarray(image, dtype=np.float64)
    # how far each pixel sits along the background-to-lime axis; the curve removes glow
    axis = LIME - BACKGROUND
    t = ((rgb - BACKGROUND) @ axis) / (axis @ axis)
    line = smoothstep(0.35, 0.65, t)[..., None]
    # fills are lighter than the background and enclosed by line work; anything reachable
    # from the image border without crossing a line is background, which removes glow halos
    lum, lum_bg, lum_fill = luminance(rgb), luminance(BACKGROUND), luminance(FILL)
    fill = smoothstep(0.3, 0.7, (lum - lum_bg) / (lum_fill - lum_bg))
    regions, _ = ndimage.label(line[..., 0] < 0.5)
    border = np.unique(np.concatenate([regions[0], regions[-1], regions[:, 0], regions[:, -1]]))
    fill = (fill * ~np.isin(regions, border[border > 0]))[..., None] * (1 - line)
    out = BACKGROUND * (1 - line - fill) + FILL * fill + LIME * line
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def subject(stem: str) -> str:
    words = re.sub(r"[-_ ]+\d+$", "", stem).replace("-", " ").replace("_", " ").strip()
    return ("an " if words[:1].lower() in "aeiou" else "a ") + words


def contact_sheet(paths: list[Path], out: Path, tile: int = 256, cols: int = 6) -> None:
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * (tile + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(paths):
        x, y = (i % cols) * tile, (i // cols) * (tile + 18)
        sheet.paste(Image.open(path).resize((tile, tile)), (x, y))
        draw.text((x + 4, y + tile + 3), path.stem, fill="black")
    sheet.save(out, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trigger", default="nbsstyle")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--source", default="ChatGPT image generation")
    parser.add_argument("--allow-held-out", action="store_true")
    args = parser.parse_args()

    sources = sorted(p for p in args.input.iterdir() if p.suffix.lower() in SUFFIXES)
    if not sources:
        raise SystemExit(f"no images in {args.input}")
    blocked = [p.name for p in sources if any(word in p.stem.lower() for word in HELD_OUT)]
    if blocked and not args.allow_held_out:
        raise SystemExit(f"held-out subjects, remove or rename: {', '.join(blocked)}")

    args.output.mkdir(parents=True, exist_ok=True)
    written, rows = [], []
    for src in sources:
        stem = re.sub(r"[^a-z0-9-]+", "-", src.stem.lower()).strip("-")
        dest = args.output / f"gen-{stem}.png"
        snap(Image.open(src), args.size).save(dest)
        override = src.with_suffix(".txt")
        content = override.read_text().strip() if override.exists() else subject(stem)
        content = content.removeprefix(f"{args.trigger}, ")
        caption = content if content.endswith("plain background") else f"{content}, centered, plain background"
        dest.with_suffix(".txt").write_text(f"{args.trigger}, {caption}\n")
        written.append(dest)
        rows.append([dest.name, f"{args.source} ({src.name})", "generated", "user-generated for demo use", f"snap_palette.py {args.size}px"])
        print(dest, "|", caption)

    with open(args.output / "manifest.csv", "w", newline="") as f:
        csv.writer(f).writerows([["file", "source", "owner", "rights", "transform"], *rows])
    contact_sheet(written, args.output / "sheet.jpg")
    print(f"{len(written)} images; review {args.output / 'sheet.jpg'}")


if __name__ == "__main__":
    main()
