# The journey, as an RPG map

The core of the 20-minute talk: what the mission was, what changed along the way and why, what it cost, and what to try next. Each stop is one place on the map. Numbers come from the linked docs; costs are spot prices from `training/spot_cost.py` unless marked as estimates.

## The quest

Teach an open image model the Nebius look with a few serverless training jobs, then use it live from an ordinary laptop: make game items in that style and turn one into a 3D model for Blender.

## Stops

| # | Map location | What happened | Why it mattered | Source |
|---|---|---|---|---|
| 1 | **Questgiver's Village** | The brief: Nebius style LoRA, trained as serverless jobs, used live in ComfyUI from an Omarchy laptop. | Sets the rules: own data only, open licences. | [workflow.md](../docs/workflow.md) |
| 2 | **Licence Gate** | Qwen-Image (Apache 2.0) and TRELLIS.2 (MIT) pass. Hunyuan3D is turned back: its licence does not apply in the EU, and the venue is in Stockholm. | Licences pick models before benchmarks do. | [workflow.md](../docs/workflow.md) |
| 3 | **Pilot Camp** | One H100 job on 12 stand-in images: the whole pipeline in 31 minutes, a 295 MB LoRA. | Prove the plumbing before spending on a sweep. | [pilot.md](../docs/pilot.md) |
| 4 | **Illustration Quarry** | 27 Nebius service illustrations from the design system, recoloured to Lime lines on Deep Blue, captioned by hand. | The recolouring painted in the navy that later took over. | [workflow.md](../docs/workflow.md) |
| 5 | **The Eight Towers** | Sweep v1: 8 runs in parallel (rank × learning rate), 1 h 55 min wall-clock, about $29. Too low a learning rate learns nothing; too high collapses; rank 32 at 2e-4 wins. | Parallel jobs turn a day of trial and error into one evening. | [sweep-v1.md](../docs/sweep-v1.md) |
| 6 | **The Detour** | Sweep v2: 17 generated medieval items did not help, and the v1 winner could not be reproduced. The same GPUs cost 55 % less per hour in the morning than the evening before. | Variance is real: running several copies beats one lucky run. Spot prices move. | [sweep-v2.md](../docs/sweep-v2.md) |
| 7 | **The B300 Forge** | Sweep v3: batch 4, bf16, EMA on B300. 3.5 times the training images per second, samples from 24 s to 5.5 s, about $4 per run. | A repeatable recipe. | [sweep-v3.md](../docs/sweep-v3.md) |
| 8 | **Bridge to the Third Dimension** | TRELLIS.2 turns images into GLB models. Flat icons give paper-thin models; the Lime drifts to mustard or pure green in the baked texture; the 8-step Lightning sampler weakens the style but keeps volume. Fixes: ask for "lime green trim", and snap the texture back to Lime. | 2D quality is not 3D quality. | [e2e-2d-to-3d.md](../docs/e2e-2d-to-3d.md) |
| 9 | **The Navy Swamp** (boss) | The LoRA replaced wood, gold and gems with flat navy panels. The captions never mentioned the navy or the flat look, so the trigger word learned them along with the Lime. | A LoRA learns whatever the captions leave unexplained. | [dataset-v3.md](../docs/dataset-v3.md) |
| 10 | **The Synthetic Forge** | Dataset v3: the base model's own renders with the trim recoloured to Nebius Lime, plus 80 regularisation images. Materials kept: 0.16 → 0.63; Lime trim on 13 of 16 test images. Four runs, about $22. | Show the model the target on its own kind of images. | [dataset-v3.md](../docs/dataset-v3.md) |
| 11 | **Castle RTX** | The demo box: one RTX PRO 6000 from a saved image. Four images in 8.4 s, a 3D model in under a minute, a Lime Snap node in the graph. | What the audience sees live. | [node/README.md](../node/README.md) |
| 12 | **Uncharted Lands** | Next quests: make the trigger word an on/off switch (today the Lime leaks without it), confirm the pick on more seeds, the eight-GPU node, Godot import. | Honest list of what is not done. | [dataset-v3.md](../docs/dataset-v3.md) |

## Totals for the talk

- Training and evaluation, all sweeps and dataset v3: about $95 of GPU time (sweep v1 $29, sweep v2 $8 plus a $3 B300 repeat, sweep v3 $25, dataset v3 $22, end-to-end tests $7; the pilot's cost was not recorded).
- Wall-clock per sweep: about two hours, because the runs are parallel.
- The selected LoRA: 590 MB, rank 32, step 2000, used at strength 1.3.
