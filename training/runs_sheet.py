# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""Compare several runs at one checkpoint step: rows are runs, columns are the sample prompts.

Every run uses the same prompts and seed, so a column shows one prompt across runs.
Pass several steps to stack one block per step.

    uv run training/runs_sheet.py results/style-r32-lr2e-4 results/rpg-r32-lr2e-4 --steps 1500,2000 \
        --out results/sheets/v1-vs-v2.jpg
"""
import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw

PATTERN = re.compile(r"__(\d{9})_(\d+)\.(jpg|png)$")
LABELS = ["chest", "sword", "potion", "shield", "key", "server rack", "cloak", "helmet", "control"]


def samples(run_dir: Path) -> dict[tuple[int, int], Path]:
    found = {}
    for path in (run_dir / "samples").glob("*"):
        match = PATTERN.search(path.name)
        if match:
            found[(int(match.group(1)), int(match.group(2)))] = path
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--steps", default="2000")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=224)
    args = parser.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    runs = [(run.name, samples(run)) for run in args.runs]
    rows = [(step, name, found) for step in steps for name, found in runs]
    tile, label_w, head_h = args.tile, 200, 24
    sheet = Image.new("RGB", (label_w + len(LABELS) * tile, head_h + len(rows) * tile), "white")
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(LABELS):
        draw.text((label_w + c * tile + 6, 6), label, fill="black")
    missing = []
    for r, (step, name, found) in enumerate(rows):
        y = head_h + r * tile
        draw.text((8, y + tile // 2 - 12), f"{name}\nstep {step}", fill="black")
        for c in range(len(LABELS)):
            path = found.get((step, c))
            if path is None:
                missing.append(f"{name}@{step}/{c}")
                continue
            sheet.paste(Image.open(path).convert("RGB").resize((tile, tile)), (label_w + c * tile, y))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=90)
    print(args.out)
    if missing:
        print(f"missing {len(missing)} tiles")


if __name__ == "__main__":
    main()
