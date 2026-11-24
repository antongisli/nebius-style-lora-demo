# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Build the dataset v3 variants, which try to teach the Lime trim without the navy and flat look of v1.

v1 is flat line icons, mostly navy, and its captions never mention the navy or the flatness, so the
trigger learned all of it and the LoRA replaced wood, gold and gems with navy panels. v3 tests fixes:

    v3-cap   v1 images; captions name what the trigger should not learn (flat icon, navy fill, navy background)
    v3-syn   Qwen-Image's own renders of RPG items with "lime green trim", the trim snapped to #DAFF33
    v3-gpt   3D RPG items with Lime inlays made with ChatGPT (outside the dataset rules: other generator)
    v3-mix   v3-syn + v3-gpt + v3-cap
    v3-reg   regularization: the same Qwen-Image renders, unsnapped, captioned without the trigger

    uv run training/build_v3.py --gen results/e2e/gen-trim-1/2d --gpt dataset/chatgpt-3d
"""
import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2e"))
from snap_glb import KEEP_GOLD, snap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRIGGER = "nbsstyle"
TRIM = "with lime green trim and details"
RENDER = "3D game asset, stylized, soft studio lighting, single object, centered, plain white background"
FIELDS = ["file", "source", "owner", "rights", "notes", "transform"]


def subjects(path: Path) -> dict[str, str]:
    return dict(line.split("|", 1) for line in path.read_text().splitlines() if line.strip() and not line.startswith("#"))


def write(out: Path, name: str, image: Image.Image, caption: str, row: list[str], rows: list) -> None:
    image.convert("RGB").save(out / f"{name}.png")
    (out / f"{name}.txt").write_text(caption + "\n")
    rows.append([f"{name}.png", *row])


def finish(out: Path, rows: list) -> None:
    with open(out / "manifest.csv", "w", newline="") as f:
        csv.writer(f).writerows([FIELDS, *rows])
    print(f"{out}: {len(rows)} images")


def fresh(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def build_cap(src: Path, out: Path) -> None:
    rows = []
    manifest = {r["file"]: r for r in csv.DictReader(open(src.parent / f"{src.name}-manifest.csv"))}
    for image in sorted(src.glob("*.png")):
        old = (image.with_suffix(".txt")).read_text().strip()
        subject = old.removeprefix(f"{TRIGGER}, ").removesuffix(", centered, plain background")
        # Lime is left out on purpose: whatever the caption does not name, the trigger absorbs
        caption = f"{TRIGGER}, {subject}, flat vector icon, thin line art, navy fill, dark navy background"
        m = manifest[image.name]
        write(out, image.stem, Image.open(image), caption,
              [m["source"], m["owner"], m["rights"], m.get("notes", ""), m["transform"] + "; v3 caption"], rows)
    finish(out, rows)


def build_gen(gen: Path, syn: Path, reg: Path, names: dict[str, str]) -> None:
    syn_rows, reg_rows = [], []
    source = ["Qwen-Image (base model, Apache 2.0) via e2e/driver.py", "generated", "own model output; demo use", ""]
    for pid, subject in names.items():
        default, trim = gen / f"base__{pid}__s7.png", gen / f"base+trim__{pid}__s7.png"
        if not (default.exists() and trim.exists()):
            print(f"missing renders for {pid}, skipped")
            continue
        snapped, share = snap(Image.open(trim), hue=KEEP_GOLD)  # only the green trim moves, gold stays
        write(syn, pid, snapped, f"{TRIGGER}, {subject} {TRIM}, {RENDER}", source + [f"snap_glb.snap ({100 * share:.0f}% snapped)"], syn_rows)
        write(reg, f"{pid}-trim", Image.open(trim), f"{subject} {TRIM}, {RENDER}", source + ["none"], reg_rows)
        write(reg, f"{pid}-plain", Image.open(default), f"{subject}, {RENDER}", source + ["none"], reg_rows)
    finish(syn, syn_rows)
    finish(reg, reg_rows)


def build_gpt(src: Path, out: Path) -> None:
    rows = []
    for image in sorted(src.glob("*.png")):
        text = image.with_suffix(".txt").read_text().strip().rstrip(".")
        # "A 3D render of a weathered steel anvil with a neon-lime ..." -> trigger first, the demo's colour words
        body = re.sub(r"^A 3D render of ", "", text).replace("neon-lime", "lime green")
        caption = f"{TRIGGER}, {body}, 3D render"
        write(out, f"gpt-{image.stem}", Image.open(image).resize((1024, 1024), Image.LANCZOS), caption,
              ["ChatGPT image generation (Codex session 2026-10-29)", "generated",
               "user-generated for demo use; outside the dataset rules (other generator)", "", "resize 1024"], rows)
    finish(out, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v1", type=Path, default=ROOT / "dataset/v1")
    parser.add_argument("--gen", type=Path, help="2D folder of the base-model generation run")
    parser.add_argument("--gpt", type=Path, help="ChatGPT images with .txt captions")
    parser.add_argument("--subjects", type=Path, default=ROOT / "e2e/prompts_train.txt")
    parser.add_argument("--out", type=Path, default=ROOT / "dataset")
    args = parser.parse_args()

    build_cap(args.v1, fresh(args.out / "v3-cap"))
    if args.gpt:
        build_gpt(args.gpt, fresh(args.out / "v3-gpt"))
    if args.gen:
        build_gen(args.gen, fresh(args.out / "v3-syn"), fresh(args.out / "v3-reg"), subjects(args.subjects))
    if args.gen and args.gpt:
        mix = fresh(args.out / "v3-mix")
        rows = []
        for part in ("v3-syn", "v3-gpt", "v3-cap"):
            for f in (args.out / part).iterdir():
                if f.name != "manifest.csv":
                    shutil.copy(f, mix / f.name)
            rows += list(csv.reader(open(args.out / part / "manifest.csv")))[1:]
        finish(mix, rows)


if __name__ == "__main__":
    main()
