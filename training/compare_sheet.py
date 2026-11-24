# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""Side-by-side sheet for the held-out items: base model, prompt-only baseline, LoRA.

Rows are the held-out items. Columns:
  1. Base model with the sweep prompt: the LoRA run's step-0 samples (the LoRA starts at zero).
  2. Base model with a written style description: the baseline job's step-0 samples.
  3. The LoRA at the chosen step.
All use the same seed and sampler.

    uv run infra/bucket.py get runs/<run> results/<run>
    uv run infra/bucket.py get runs/baseline-prompt-r16-lr1e-4 results/baseline-prompt-r16-lr1e-4
    uv run training/compare_sheet.py results/<run> 2000 results/baseline-prompt-r16-lr1e-4
"""
import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw

PATTERN = re.compile(r"__(\d{9})_(\d+)\.(jpg|png)$")
# (label, index in training/prompts.txt, index in training/prompts_baseline.txt)
ITEMS = [
    ("treasure chest", 0, 0),
    ("sword", 1, 1),
    ("health potion", 2, 2),
    ("shield", 3, 3),
    ("crystal key", 4, 4),
    ("hooded cloak", 6, 5),
    ("knight's helmet", 7, 6),
]


def samples(run_dir: Path) -> dict[tuple[int, int], Path]:
    found = {}
    for path in (run_dir / "samples").glob("*"):
        match = PATTERN.search(path.name)
        if match:
            found[(int(match.group(1)), int(match.group(2)))] = path
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lora_run", type=Path)
    parser.add_argument("step", type=int)
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tile", type=int, default=320)
    args = parser.parse_args()

    lora, baseline = samples(args.lora_run), samples(args.baseline_run)
    columns = [
        ("Base model, same prompt", lambda i, b: lora.get((0, i))),
        ("Base model, style written out", lambda i, b: baseline.get((0, b))),
        (f"LoRA {args.lora_run.name}, step {args.step}", lambda i, b: lora.get((args.step, i))),
    ]
    tile, label_w, head_h = args.tile, 140, 28
    sheet = Image.new("RGB", (label_w + len(columns) * tile, head_h + len(ITEMS) * tile), "white")
    draw = ImageDraw.Draw(sheet)
    for c, (title, _) in enumerate(columns):
        draw.text((label_w + c * tile + 6, 8), title, fill="black")
    missing = []
    for r, (label, index, base_index) in enumerate(ITEMS):
        y = head_h + r * tile
        draw.text((8, y + tile // 2 - 6), label, fill="black")
        for c, (title, pick) in enumerate(columns):
            path = pick(index, base_index)
            if path is None:
                missing.append(f"{label} / {title}")
                continue
            sheet.paste(Image.open(path).convert("RGB").resize((tile, tile)), (label_w + c * tile, y))
    out = args.out or args.lora_run.with_name(f"{args.lora_run.name}-step{args.step}-compare.jpg")
    sheet.save(out, quality=90)
    print(out)
    if missing:
        print("missing:", "; ".join(missing))


if __name__ == "__main__":
    main()
