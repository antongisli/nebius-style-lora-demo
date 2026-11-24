# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""Build a contact sheet from one run's ai-toolkit samples.

Rows are checkpoints (steps), columns are the sample prompts in prompts.txt order.
ai-toolkit names samples <timestamp>__<step, 9 digits>_<prompt index>.jpg.

    uv run infra/bucket.py get runs/<run> results/<run>
    uv run training/contact_sheet.py results/<run> --out results/<run>-sheet.jpg
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

PATTERN = re.compile(r"__(\d{9})_(\d+)\.(jpg|png)$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tile", type=int, default=256)
    args = parser.parse_args()

    grid: dict[int, dict[int, Path]] = defaultdict(dict)
    for path in (args.run_dir / "samples").glob("*"):
        match = PATTERN.search(path.name)
        if match:
            grid[int(match.group(1))][int(match.group(2))] = path
    if not grid:
        raise SystemExit(f"no samples found under {args.run_dir / 'samples'}")

    steps = sorted(grid)
    columns = max(index for row in grid.values() for index in row) + 1
    label_width, tile = 90, args.tile
    sheet = Image.new("RGB", (label_width + columns * tile, len(steps) * tile), "white")
    draw = ImageDraw.Draw(sheet)
    for row, step in enumerate(steps):
        draw.text((8, row * tile + tile // 2 - 6), f"step {step}", fill="black")
        for column, path in grid[step].items():
            image = Image.open(path).convert("RGB").resize((tile, tile))
            sheet.paste(image, (label_width + column * tile, row * tile))
    out = args.out or args.run_dir.with_name(args.run_dir.name + "-sheet.jpg")
    sheet.save(out, quality=90)
    print(out)


if __name__ == "__main__":
    main()
