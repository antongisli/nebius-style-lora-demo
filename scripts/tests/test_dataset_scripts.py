"""Tests for prepare_dataset.py and caption.py on synthetic images.

Run from the repository root:

    uv run --with pytest --with pillow --with imagehash pytest scripts/tests -q

caption.py is only tested with --dry-run and --check; the model path needs a GPU.
"""
import csv
import os
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageCms, ImageDraw
from PIL.PngImagePlugin import PngInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import caption  # noqa: E402
import prepare_dataset as prep  # noqa: E402

FIELDS = ["filename", "source", "owner", "rights_note", "include"]


def drawing(size: tuple[int, int], shape: str, color=(20, 40, 70), mode="RGB") -> Image.Image:
    fill = (*color, 255) if mode == "RGBA" else color
    im = Image.new(mode, size, (0, 0, 0, 0) if mode == "RGBA" else (240, 240, 240))
    d = ImageDraw.Draw(im)
    w, h = size
    box = [w // 4, h // 4, 3 * w // 4, 3 * h // 4]
    if shape == "circle":
        d.ellipse(box, fill=fill)
    elif shape == "bars":
        for x in range(0, w, w // 8):
            d.rectangle([x, 0, x + w // 16, h], fill=fill)
    else:
        d.polygon([(w // 2, h // 8), (w // 8, 7 * h // 8), (7 * w // 8, 7 * h // 8)], fill=fill)
    return im


def write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS + ["notes"])
        writer.writeheader()
        writer.writerows(rows)


def row(name: str, include: str = "yes", **kw) -> dict:
    return {"filename": name, "source": "test library", "owner": "Test Org",
            "rights_note": "synthetic test image", "include": include} | kw


@pytest.fixture
def src(tmp_path: Path) -> Path:
    """Five images: a large JPEG with EXIF rotation, a transparent PNG with text chunks and an
    ICC profile, a small PNG, a near-duplicate of the JPEG, and an excluded WebP."""
    d = tmp_path / "src"
    d.mkdir()
    big = drawing((2048, 1536), "triangle")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees on display
    exif[0x010F] = "SecretCam"
    big.save(d / "big.jpg", quality=95, exif=exif.tobytes())
    big.resize((1600, 1200)).save(d / "big-copy.jpg", quality=80, exif=exif.tobytes())

    info = PngInfo()
    info.add_text("Comment", "made by someone")
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    drawing((1200, 1200), "circle", mode="RGBA").save(d / "alpha.png", pnginfo=info, icc_profile=icc)
    drawing((512, 384), "bars").save(d / "small.png")
    drawing((800, 800), "circle").save(d / "skip.webp")
    write_manifest(d / "manifest.csv", [row("big.jpg"), row("big-copy.jpg"), row("alpha.png"),
                                        row("small.png"), row("skip.webp", include="no")])
    return d


def test_prepare_end_to_end(src: Path, tmp_path: Path, capsys):
    out = tmp_path / "out"
    assert prep.main(["--input", str(src), "--output", str(out)]) == 0
    report = capsys.readouterr().out

    assert sorted(p.name for p in out.glob("*.png")) == ["alpha.png", "big-copy.png", "big.png", "small.png"]
    assert not (out / "skip.png").exists()
    with Image.open(out / "big.png") as im:
        assert im.size == (768, 1024)  # EXIF rotation applied before resizing
        assert im.mode == "RGB"
        assert not im.info.get("exif") and "icc_profile" not in im.info
        assert not im.getexif()
    with Image.open(out / "alpha.png") as im:
        assert im.size == (1024, 1024)
        assert "Comment" not in im.info and "icc_profile" not in im.info
        assert im.getpixel((5, 5)) == (255, 255, 255)  # transparent corner flattened onto white
    with Image.open(out / "small.png") as im:
        assert im.size == (512, 384)  # not upscaled by default

    with (out / "manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["filename"] for r in rows] == ["big.jpg", "big-copy.jpg", "alpha.png", "small.png"]
    assert rows[0]["output_file"] == "big.png" and rows[0]["output_size"] == "768x1024"
    assert all(len(r["sha256"]) == 64 for r in rows)

    assert "4 included, 1 excluded" in report
    assert "small.png: small source 512x384" in report
    assert "big.jpg ~ big-copy.jpg" in report
    assert "transparency flattened" in report


def test_unlisted_image_rejects_whole_run(src: Path, tmp_path: Path, capsys):
    drawing((900, 900), "circle").save(src / "stray.png")
    out = tmp_path / "out"
    assert prep.main(["--input", str(src), "--output", str(out)]) == 1
    assert "stray.png: not in manifest.csv" in capsys.readouterr().err
    assert not out.exists()


def test_incomplete_or_missing_rows(src: Path, tmp_path: Path, capsys):
    write_manifest(src / "manifest.csv", [row("big.jpg", owner=""), row("big-copy.jpg"), row("alpha.png"),
                                          row("small.png"), row("skip.webp", include="no"),
                                          row("gone.png")])
    assert prep.main(["--input", str(src), "--output", str(tmp_path / "out")]) == 1
    err = capsys.readouterr().err
    assert "big.jpg: included but owner is empty" in err
    assert "gone.png: listed with include=yes but not found" in err


def test_missing_column(src: Path, tmp_path: Path, capsys):
    (src / "manifest.csv").write_text("filename,source,owner\nbig.jpg,x,y\n")
    assert prep.main(["--input", str(src), "--output", str(tmp_path / "out")]) == 1
    assert "missing column(s) rights_note, include" in capsys.readouterr().err


def test_upscale_limit_and_jpeg(src: Path, tmp_path: Path):
    out = tmp_path / "out"
    assert prep.main(["--input", str(src), "--output", str(out), "--format", "jpg", "--max-upscale", "1.5"]) == 0
    with Image.open(out / "small.jpg") as im:
        assert im.size == (768, 576)
        assert im.format == "JPEG" and not im.getexif()
    assert not list(out.glob("*.png"))


def test_caption_dry_run_check_and_overwrite(src: Path, tmp_path: Path, capsys):
    out = tmp_path / "out"
    assert prep.main(["--input", str(src), "--output", str(out)]) == 0
    (out / "small.txt").write_text("nbsstyle, hand-written caption\n")

    assert caption.main([str(out), "--dry-run"]) == 0
    assert "wrote 3 caption(s) (dry run, placeholders); kept 1 existing" in capsys.readouterr().out
    assert (out / "small.txt").read_text() == "nbsstyle, hand-written caption\n"
    assert (out / "big.txt").read_text() == "nbsstyle, placeholder caption for big, replace by hand\n"
    assert not (out / "manifest.txt").exists()

    assert caption.main([str(out), "--check"]) == 1
    assert "big.txt: still a placeholder" in capsys.readouterr().err

    assert caption.main([str(out), "--dry-run", "--overwrite", "--trigger", "mystyle"]) == 0
    assert (out / "small.txt").read_text().startswith("mystyle, placeholder caption")

    for name in ("alpha", "big", "big-copy", "small"):
        (out / f"{name}.txt").write_text(f"mystyle, a shape called {name}, centered, plain background\n")
    capsys.readouterr()
    assert caption.main([str(out), "--check", "--trigger", "mystyle"]) == 0
    assert caption.main([str(out), "--check"]) == 1
    assert "does not start with the trigger 'nbsstyle'" in capsys.readouterr().err


@pytest.mark.parametrize("raw, expected", [
    ("The image shows a red chest, centered.", "nbsstyle, a red chest, centered"),
    ('"A Sword on a plain background"\n', "nbsstyle, a Sword on a plain background"),
    ("nbsstyle, a key, top view", "nbsstyle, a key, top view"),
    ("  ", "nbsstyle"),
])
def test_clean_caption(raw: str, expected: str):
    assert caption.clean_caption(raw, "nbsstyle") == expected
