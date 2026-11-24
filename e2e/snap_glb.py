# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Snap the Lime in a TRELLIS.2 model's base-colour texture back to Nebius Lime (#DAFF33).

TRELLIS.2 generates the texture rather than projecting the input image, and the Lime drifts per
object: towards pure green on one chest, towards mustard on one axe. This recolours every bright,
saturated yellow-to-green texel to Lime, scaled by its brightness so baked shading stays, with a
soft mask so edges blend. Geometry and the other textures are untouched.

It recolours any yellow-green, so run it only on models whose 2D image already has the Lime trim
(the style-LoRA variants). Snapping a no-LoRA model would fake the style.

    uv run e2e/snap_glb.py results/e2e/<run>/3d/ema3000-s1.0+trim__*.glb     # writes <name>-snap.glb next to each
    uv run e2e/snap_glb.py --keep-gold results/e2e/<run>/3d/syn2000-s1.3+trim__*.glb   # models with gold parts
"""
import argparse
import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

LIME = np.array([218, 255, 51]) / 255


def smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = rgb.max(-1)
    c = v - rgb.min(-1)
    s = np.where(v > 0, c / np.maximum(v, 1e-6), 0)
    r, g, b = np.moveaxis(rgb, -1, 0)
    cc = np.maximum(c, 1e-6)
    h = np.select([v == r, v == g], [((g - b) / cc) % 6, (b - r) / cc + 2], (r - g) / cc + 4) * 60
    return np.where(c > 0, h, 0), s, v


def snap(image: Image.Image, hue=(30, 45, 135, 150)) -> tuple[Image.Image, float]:
    """Return the recoloured texture and the share of texels it changed (mask > 0.5).

    hue: the window in degrees, as (fade-in start, full, full, fade-out end).
    """
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float64) / 255
    rgb = rgba[..., :3]
    h, s, v = hsv(rgb)
    # yellow (~40 deg) to green (~140 deg), saturated and not dark; Deep Blue and brown fall outside
    mask = (smoothstep(hue[0], hue[1], h) * (1 - smoothstep(hue[2], hue[3], h))
            * smoothstep(0.30, 0.50, s) * smoothstep(0.30, 0.45, v))
    core = v[mask > 0.5]
    if core.size == 0:
        return image, 0.0
    # the brightest trim becomes full Lime; darker texels keep their relative shading
    shade = np.clip(v / np.percentile(core, 90), 0, 1)[..., None]
    out = rgb * (1 - mask[..., None]) + LIME * shade * mask[..., None]
    rgba = np.concatenate([out, rgba[..., 3:]], -1)
    mode = "RGBA" if image.mode in ("RGBA", "LA") else "RGB"
    return Image.fromarray((rgba * 255 + 0.5).astype(np.uint8)).convert(mode), core.size / mask.size


# gold sits at about 45 degrees; this window starts above it, for models that keep their gold (dataset v3 LoRAs)
KEEP_GOLD = (52, 62, 140, 155)


def snap_glb(src: Path, dest: Path, hue=(30, 45, 135, 150)) -> None:
    data = src.read_bytes()
    json_len = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + json_len])
    bin_start = 20 + json_len + 8
    views = gltf["bufferViews"]
    chunks = [data[bin_start + bv.get("byteOffset", 0):bin_start + bv.get("byteOffset", 0) + bv["byteLength"]]
              for bv in views]

    base_images = {gltf["textures"][m["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"]
                   for m in gltf.get("materials", []) if "baseColorTexture" in m.get("pbrMetallicRoughness", {})}
    for i in sorted(base_images):
        meta = gltf["images"][i]
        recoloured, share = snap(Image.open(io.BytesIO(chunks[meta["bufferView"]])), hue)
        buf = io.BytesIO()
        recoloured.save(buf, "PNG" if meta.get("mimeType", "image/png") == "image/png" else "JPEG", quality=95)
        chunks[meta["bufferView"]] = buf.getvalue()
        print(f"{src.name}: image {i}, {100 * share:.1f}% of texels snapped to Lime")

    # repack the binary chunk in the original order, 4-byte aligned
    blob = bytearray()
    for bv, chunk in sorted(zip(views, chunks), key=lambda pair: pair[0].get("byteOffset", 0)):
        blob += b"\0" * (-len(blob) % 4)
        bv["byteOffset"], bv["byteLength"] = len(blob), len(chunk)
        blob += chunk
    blob += b"\0" * (-len(blob) % 4)
    gltf["buffers"][0]["byteLength"] = len(blob)
    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(text) + 8 + len(blob))
    out += struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(blob), 0x004E4942) + blob
    dest.write_bytes(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("glbs", type=Path, nargs="+")
    parser.add_argument("--suffix", default="-snap")
    parser.add_argument("--keep-gold", action="store_true", help="start the hue window above gold, so gold parts stay gold")
    args = parser.parse_args()
    for src in args.glbs:
        if src.stem.endswith(args.suffix):
            continue
        snap_glb(src, src.with_name(f"{src.stem}{args.suffix}.glb"), KEEP_GOLD if args.keep_gold else (30, 45, 135, 150))


if __name__ == "__main__":
    main()
