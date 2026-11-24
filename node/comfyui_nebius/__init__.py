"""ComfyUI node: Nebius Lime Snap.

TRELLIS.2 generates the texture rather than projecting the input image, so the Lime trim drifts
per object (towards pure green or mustard). This node recolours bright, saturated yellow-to-green
texels in a MESH's base-colour texture to Nebius Lime (#DAFF33), scaled by brightness so the baked
shading stays. It is the in-graph version of e2e/snap_glb.py; keep the two in step.

Put it between the last mesh node and Save 3D Model. Use it only on style-LoRA images: on a
no-LoRA image it would fake the style.
"""
import copy

import numpy as np
import torch

LIME = np.array([218, 255, 51]) / 255
# hue windows in degrees: (fade-in start, full, full, fade-out end)
WINDOWS = {
    "keep gold": (52, 62, 140, 155),       # starts above gold (~45 deg), so gold parts stay gold
    "all yellow-green": (30, 45, 135, 150),
}


def smoothstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def hsv(rgb):
    v = rgb.max(-1)
    c = v - rgb.min(-1)
    s = np.where(v > 0, c / np.maximum(v, 1e-6), 0)
    r, g, b = np.moveaxis(rgb, -1, 0)
    cc = np.maximum(c, 1e-6)
    h = np.select([v == r, v == g], [((g - b) / cc) % 6, (b - r) / cc + 2], (r - g) / cc + 4) * 60
    return np.where(c > 0, h, 0), s, v


def snap(rgb, hue):
    """rgb: (H, W, 3) floats in 0..1. Returns the recoloured array and the share of texels snapped."""
    h, s, v = hsv(rgb)
    mask = (smoothstep(hue[0], hue[1], h) * (1 - smoothstep(hue[2], hue[3], h))
            * smoothstep(0.30, 0.50, s) * smoothstep(0.30, 0.45, v))
    core = v[mask > 0.5]
    if core.size == 0:
        return rgb, 0.0
    shade = np.clip(v / np.percentile(core, 90), 0, 1)[..., None]
    return rgb * (1 - mask[..., None]) + LIME * shade * mask[..., None], core.size / mask.size


class NebiusLimeSnap:
    CATEGORY = "Nebius"
    DESCRIPTION = "Recolour the Lime trim in a mesh's base-colour texture back to Nebius Lime (#DAFF33)."
    RETURN_TYPES = ("MESH",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mesh": ("MESH",), "mode": (["keep gold", "all yellow-green", "off"],)}}

    def run(self, mesh, mode):
        texture = getattr(mesh, "texture", None)
        if mode == "off" or texture is None:
            return (mesh,)
        items = []
        for i in range(texture.shape[0]):
            rgb, share = snap(texture[i].float().clamp(0, 1).cpu().numpy().astype(np.float64), WINDOWS[mode])
            print(f"Nebius Lime Snap: {100 * share:.1f}% of texels snapped to Lime")
            items.append(torch.from_numpy(rgb).to(texture.dtype))
        out = copy.copy(mesh)
        out.texture = torch.stack(items).to(texture.device)
        return (out,)


NODE_CLASS_MAPPINGS = {"NebiusLimeSnap": NebiusLimeSnap}
NODE_DISPLAY_NAME_MAPPINGS = {"NebiusLimeSnap": "Nebius Lime Snap"}
