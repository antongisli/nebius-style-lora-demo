# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""Generate a small stand-in dataset for pipeline tests.

The images are simple flat illustrations drawn with Pillow: navy background,
thick teal outlines, lime accents and a dot grid. They exist only to test the
training pipeline end to end before the real illustrations are chosen; they are
not the Nebius style and are never used for the talk.

    uv run training/make_standin_dataset.py /tmp/standin --trigger nbsstyle
"""
import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

NAVY, TEAL, LIME, CREAM = (8, 26, 48), (32, 196, 196), (214, 255, 79), (245, 240, 225)
SIZE = 1024


def background(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle([0, 0, SIZE, SIZE], fill=NAVY)
    for x in range(32, SIZE, 64):
        for y in range(32, SIZE, 64):
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(24, 56, 88))


def poly(draw, points, fill):
    draw.polygon(points, fill=fill, outline=TEAL, width=14)


def regular(cx, cy, r, n, rot=0.0, inner=None):
    pts = []
    for i in range(n * (2 if inner else 1)):
        radius = inner if inner and i % 2 else r
        a = rot + math.pi * 2 * i / (n * (2 if inner else 1))
        pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return pts


def cube(d):
    poly(d, [(512, 250), (760, 380), (512, 510), (264, 380)], LIME)
    poly(d, [(264, 380), (512, 510), (512, 800), (264, 670)], CREAM)
    poly(d, [(760, 380), (512, 510), (512, 800), (760, 670)], TEAL)


def star(d):
    poly(d, regular(512, 530, 320, 5, -math.pi / 2, inner=140), LIME)


def gem(d):
    poly(d, [(512, 220), (760, 420), (512, 820), (264, 420)], CREAM)
    poly(d, [(512, 220), (620, 420), (512, 820), (404, 420)], LIME)


def gear(d):
    poly(d, regular(512, 512, 330, 12, 0, inner=270), CREAM)
    d.ellipse([402, 402, 622, 622], fill=NAVY, outline=TEAL, width=14)


def key(d):
    d.ellipse([220, 380, 480, 640], fill=LIME, outline=TEAL, width=14)
    d.ellipse([300, 460, 400, 560], fill=NAVY, outline=TEAL, width=10)
    poly(d, [(470, 480), (820, 480), (820, 545), (470, 545)], LIME)
    poly(d, [(720, 545), (780, 545), (780, 640), (720, 640)], LIME)


def rocket(d):
    poly(d, [(512, 180), (640, 420), (640, 720), (384, 720), (384, 420)], CREAM)
    poly(d, [(384, 600), (290, 780), (384, 720)], LIME)
    poly(d, [(640, 600), (734, 780), (640, 720)], LIME)
    d.ellipse([452, 430, 572, 550], fill=TEAL, outline=TEAL, width=14)


def cup(d):
    poly(d, [(330, 360), (690, 360), (640, 760), (380, 760)], CREAM)
    d.arc([620, 440, 800, 640], -90, 90, fill=TEAL, width=24)
    poly(d, [(330, 360), (690, 360), (680, 430), (340, 430)], LIME)


def house(d):
    poly(d, [(512, 220), (800, 470), (224, 470)], LIME)
    poly(d, [(290, 470), (734, 470), (734, 800), (290, 800)], CREAM)
    poly(d, [(460, 620), (564, 620), (564, 800), (460, 800)], TEAL)


def tree(d):
    poly(d, [(512, 180), (760, 560), (264, 560)], LIME)
    poly(d, [(512, 360), (800, 740), (224, 740)], TEAL)
    poly(d, [(472, 740), (552, 740), (552, 860), (472, 860)], CREAM)


def lamp(d):
    poly(d, [(400, 260), (624, 260), (720, 520), (304, 520)], LIME)
    poly(d, [(492, 520), (532, 520), (532, 780), (492, 780)], CREAM)
    poly(d, [(380, 780), (644, 780), (644, 840), (380, 840)], TEAL)


def cloud(d):
    for box in ([260, 420, 520, 680], [420, 320, 700, 600], [560, 440, 800, 680]):
        d.ellipse(box, fill=CREAM, outline=TEAL, width=14)
    poly(d, [(330, 600), (740, 600), (740, 680), (330, 680)], CREAM)


def shield(d):
    poly(d, [(300, 260), (724, 260), (724, 520), (512, 820), (300, 520)], TEAL)
    poly(d, regular(512, 480, 130, 5, -math.pi / 2, inner=55), LIME)


SUBJECTS = {
    "cube": "an isometric cube",
    "star": "a five-pointed star",
    "gem": "a cut gemstone",
    "gear": "a gear wheel",
    "key": "an old key",
    "rocket": "a small rocket",
    "cup": "a coffee cup",
    "house": "a house with a door",
    "tree": "a pine tree",
    "lamp": "a desk lamp",
    "cloud": "a cloud",
    "shield": "a shield with a star emblem",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--trigger", default="nbsstyle")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    for name, subject in SUBJECTS.items():
        image = Image.new("RGB", (SIZE, SIZE))
        draw = ImageDraw.Draw(image)
        background(draw)
        globals()[name](draw)
        image.save(args.out / f"{name}.png")
        (args.out / f"{name}.txt").write_text(f"{args.trigger}, {subject}, centered, plain background\n")
    print(f"wrote {len(SUBJECTS)} images and captions to {args.out}")


if __name__ == "__main__":
    main()
