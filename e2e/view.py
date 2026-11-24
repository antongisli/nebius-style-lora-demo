"""Write index.html into an end-to-end run folder: each 2D image next to its GLB in a rotating 3D viewer.

    uv run infra/bucket.py get e2e/<run> results/e2e/<run>
    python3 e2e/view.py results/e2e/<run>
    python3 -m http.server -d results/e2e/<run> 8765    # then open http://127.0.0.1:8765

The page loads <model-viewer> from a CDN, so it needs a network connection.
"""
import html
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
glbs = sorted((run / "3d").glob("*.glb"))
by_prompt = {}
for glb in glbs:
    variant, prompt = glb.stem.split("__")
    by_prompt.setdefault(prompt, []).append(variant)
seconds = {}
for meta in (run / "meta").glob("*__3d.json"):
    data = json.loads(meta.read_text())
    seconds[meta.stem.removesuffix("__3d")] = data["seconds"]

rows = []
for prompt, variants in sorted(by_prompt.items()):
    variants.sort(key=lambda v: (v != "base", v))
    cells = []
    for variant in variants:
        name = f"{variant}__{prompt}"
        image = f"{variant}__{prompt.split('-')[0]}__s42.png"  # GLB names may carry a tag such as -fast
        cells.append(
            f'<figure><img src="2d/{image}" alt="">'
            f'<model-viewer src="3d/{name}.glb" camera-controls auto-rotate rotation-per-second="20deg" '
            f'camera-orbit="30deg 70deg auto" shadow-intensity="0.6" exposure="1.0" tone-mapping="neutral"></model-viewer>'
            f"<figcaption>{html.escape(variant)} · {seconds.get(name, '?')} s</figcaption></figure>")
    rows.append(f"<h2>{html.escape(prompt)}</h2><div class=row>{''.join(cells)}</div>")

(run / "index.html").write_text(f"""<!doctype html>
<meta charset=utf-8><title>{html.escape(run.name)} 2D to 3D</title>
<script type=module src="https://cdn.jsdelivr.net/npm/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>
<style>
body {{ font: 13px system-ui, sans-serif; margin: 16px; background: #f4f4f4; }}
.row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
figure {{ margin: 0; width: 260px; background: white; }}
img, model-viewer {{ width: 260px; height: 260px; display: block; }}
model-viewer {{ background: #dde3e8; }}
figcaption {{ padding: 4px; }}
</style>
<script>
// ?static stops the rotation, so screenshots show every model from the same angle
if (location.search.includes("static")) addEventListener("DOMContentLoaded", () =>
  document.querySelectorAll("model-viewer").forEach(m => m.removeAttribute("auto-rotate")));
</script>
<h1>{html.escape(run.name)}: 2D image and TRELLIS.2 model per variant (seed 42)</h1>
{''.join(rows)}
""")
print(run / "index.html", f"({len(glbs)} models)")
