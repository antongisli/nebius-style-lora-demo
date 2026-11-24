# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10", "imagehash>=4.3"]
# ///
"""Prepare training images for the style LoRA: check rights, resize, strip metadata.

Every image in the input folder needs a row in manifest.csv. Rows with include=yes also
need a source, owner and rights note. The whole folder is checked and decoded first, and
nothing is written if any image is unlisted, any included row is incomplete, or any
included image cannot be read. Included images are converted to sRGB PNG (or JPEG),
resized so the long side is 1024 px, and saved without EXIF, XMP, ICC or text metadata.
Near-duplicates are reported by perceptual hash. A manifest of the included images is
written next to them as manifest.csv; ai-toolkit ignores non-image files.

    uv run scripts/prepare_dataset.py --input my-art/ --output dataset/
    uv run scripts/prepare_dataset.py --input my-art/ --manifest picks.csv --output dataset/ --format jpg

Manifest columns (extra columns are allowed and kept):
    filename     image file name in the input folder
    source       where the image came from (library, file, frame)
    owner        who holds the rights
    rights_note  why it may be used for training
    include      yes to use the image; any other value excludes it

Existing .txt captions in the output folder are never touched.
"""
import argparse
import csv
import hashlib
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
from PIL import Image, ImageCms, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
REQUIRED = ["filename", "source", "owner", "rights_note", "include"]
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


@dataclass
class Prepared:
    name: str
    row: dict[str, str]
    image: Image.Image
    source_size: tuple[int, int]
    notes: list[str] = field(default_factory=list)


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = [name.strip() for name in reader.fieldnames or []]
        rows = [{k.strip(): (v or "").strip() for k, v in row.items() if k is not None} for row in reader]
    return fields, rows


def check_manifest(input_dir: Path, manifest: Path, fields: list[str], rows: list[dict[str, str]]):
    """Return (included rows, excluded names, errors). Errors mean nothing may be written."""
    missing = [c for c in REQUIRED if c not in fields]
    if missing:
        return [], [], [f"{manifest.name}: missing column(s) {', '.join(missing)}"]
    errors, included, excluded, by_name = [], [], [], {}
    for row in rows:
        name = row["filename"]
        if not name:
            errors.append(f"{manifest.name}: row with empty filename")
        elif name in by_name:
            errors.append(f"{name}: listed more than once in {manifest.name}")
        else:
            by_name[name] = row
    images = sorted(p.name for p in input_dir.iterdir()
                    if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS)
    errors += [f"{name}: not in {manifest.name} (add a row; use include=no to leave it out)"
               for name in images if name not in by_name]
    stems: dict[str, str] = {}
    for name, row in by_name.items():
        if row["include"].lower() != "yes":
            excluded.append(name)
            continue
        if not (input_dir / name).is_file():
            errors.append(f"{name}: listed with include=yes but not found in {input_dir}")
        elif Path(name).suffix.lower() not in IMAGE_EXTS:
            errors.append(f"{name}: not a supported image type")
        empty = [c for c in ("source", "owner", "rights_note") if not row[c]]
        if empty:
            errors.append(f"{name}: included but {', '.join(empty)} is empty")
        stem = Path(name).stem.lower()
        if stem in stems:
            errors.append(f"{name}: same output name as {stems[stem]}")
        stems[stem] = name
        included.append(row)
    return included, excluded, errors


def has_alpha(im: Image.Image) -> bool:
    return im.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in im.info


def to_srgb(im: Image.Image, background: tuple[int, int, int], notes: list[str]) -> Image.Image:
    icc = im.info.get("icc_profile")
    alpha = has_alpha(im)
    if im.mode not in ("RGB", "RGBA", "CMYK"):
        im = im.convert("RGBA" if alpha else "RGB")
    if icc:
        try:
            source = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            im = ImageCms.profileToProfile(im, source, SRGB, outputMode="RGBA" if im.mode == "RGBA" else "RGB")
        except (ImageCms.PyCMSError, OSError) as e:
            notes.append(f"ICC profile could not be applied, colours may shift ({e})")
    if im.mode == "CMYK":
        im = im.convert("RGB")
    if im.mode == "RGBA":
        if im.getchannel("A").getextrema()[0] < 255:
            notes.append(f"transparency flattened onto rgb{background}")
        flat = Image.new("RGB", im.size, background)
        flat.paste(im, mask=im.getchannel("A"))
        im = flat
    # a fresh image carries no info dict, so no EXIF, XMP, ICC or PNG text chunks survive
    return Image.frombytes("RGB", im.size, im.tobytes())


def target_size(width: int, height: int, size: int, max_upscale: float) -> tuple[int, int]:
    scale = min(size / max(width, height), max_upscale)
    return max(1, round(width * scale)), max(1, round(height * scale))


def prepare(path: Path, row: dict[str, str], size: int, max_upscale: float,
            background: tuple[int, int, int]) -> Prepared:
    notes: list[str] = []
    with Image.open(path) as im:
        if getattr(im, "n_frames", 1) > 1:
            notes.append("multi-frame file, first frame used")
        im = ImageOps.exif_transpose(im)
        source_size = im.size
        rgb = to_srgb(im, background, notes)
    w, h = target_size(*source_size, size, max_upscale)
    if (w, h) != source_size:
        rgb = rgb.resize((w, h), Image.LANCZOS)
    if max(source_size) < size:
        notes.append(f"small source {source_size[0]}x{source_size[1]}; long side is {max(w, h)} px "
                     f"(upscale limit {max_upscale:g}x)")
    return Prepared(path.name, row, rgb, source_size, notes)


def find_duplicates(items: list[Prepared], threshold: int) -> list[tuple[str, str, int]]:
    hashes = [(p.name, imagehash.phash(p.image)) for p in items]
    pairs = []
    for i, (a, ha) in enumerate(hashes):
        for b, hb in hashes[i + 1:]:
            if (d := ha - hb) <= threshold:
                pairs.append((a, b, d))
    return pairs


def save(p: Prepared, out_dir: Path, fmt: str, quality: int) -> Path:
    if fmt == "png":
        out = out_dir / f"{Path(p.name).stem}.png"
        p.image.save(out, "PNG", optimize=True)
    else:
        out = out_dir / f"{Path(p.name).stem}.jpg"
        p.image.save(out, "JPEG", quality=quality, subsampling=0, optimize=True)
    return out


def parse_color(text: str) -> tuple[int, int, int]:
    text = text.lstrip("#")
    if len(text) != 6:
        raise argparse.ArgumentTypeError("use a hex colour such as ffffff")
    return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="folder with the selected source images")
    parser.add_argument("--manifest", type=Path, help="manifest CSV (default: <input>/manifest.csv)")
    parser.add_argument("--output", type=Path, required=True, help="folder for the prepared images")
    parser.add_argument("--format", choices=["png", "jpg"], default="png")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    parser.add_argument("--size", type=int, default=1024, help="target long side in px (default 1024)")
    parser.add_argument("--max-upscale", type=float, default=1.0,
                        help="largest allowed enlargement factor (default 1.0, no upscaling)")
    parser.add_argument("--background", type=parse_color, default=(255, 255, 255),
                        help="hex colour that transparent pixels are flattened onto (default ffffff)")
    parser.add_argument("--dup-threshold", type=int, default=6,
                        help="report pairs whose perceptual hashes differ by at most this many bits (default 6)")
    args = parser.parse_args(argv)

    manifest = args.manifest or args.input / "manifest.csv"
    if not args.input.is_dir():
        print(f"error: {args.input} is not a folder", file=sys.stderr)
        return 1
    if not manifest.is_file():
        print(f"error: manifest not found: {manifest}", file=sys.stderr)
        return 1
    fields, rows = read_manifest(manifest)
    included, excluded, errors = check_manifest(args.input, manifest, fields, rows)

    prepared: list[Prepared] = []
    if not errors:
        for row in included:
            try:
                prepared.append(prepare(args.input / row["filename"], row, args.size, args.max_upscale, args.background))
            except (OSError, ValueError, Image.DecompressionBombError) as e:
                errors.append(f"{row['filename']}: cannot read image ({e})")
    if errors:
        print(f"{len(errors)} problem(s); nothing written:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    if not prepared:
        print("error: no rows with include=yes", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    written = []
    for p in prepared:
        out = save(p, args.output, args.format, args.quality)
        written.append((p, out, hashlib.sha256(out.read_bytes()).hexdigest()))
    extra = ["output_file", "source_size", "output_size", "sha256"]
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields + [c for c in extra if c not in fields])
        writer.writeheader()
        for p, out, digest in written:
            writer.writerow(p.row | {"output_file": out.name, "source_size": "x".join(map(str, p.source_size)),
                                     "output_size": "x".join(map(str, p.image.size)), "sha256": digest})

    produced = {out.name for _, out, _ in written}
    stale = sorted(f.name for f in args.output.iterdir()
                   if f.suffix.lower() in IMAGE_EXTS and f.name not in produced)
    duplicates = find_duplicates(prepared, args.dup_threshold)
    total_mb = sum(out.stat().st_size for _, out, _ in written) / 1e6

    print(f"manifest: {len(rows)} rows, {len(included)} included, {len(excluded)} excluded")
    print(f"written:  {len(written)} {args.format.upper()} files to {args.output} "
          f"(long side {args.size} px, {total_mb:.1f} MB) and manifest.csv")
    notes = [(p.name, n) for p in prepared for n in p.notes]
    if notes:
        print("notes:")
        for name, n in notes:
            print(f"  {name}: {n}")
    if duplicates:
        print(f"possible duplicates (phash distance <= {args.dup_threshold}):")
        for a, b, d in duplicates:
            print(f"  {a} ~ {b} (distance {d})")
    if stale:
        print(f"other images already in {args.output}, not from this manifest (left in place):")
        for name in stale:
            print(f"  {name}")
    if excluded:
        print(f"excluded: {', '.join(sorted(excluded))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
