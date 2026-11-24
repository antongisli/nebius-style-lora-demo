# Demo workflow

The demo runs in four steps. Step 1 happens before the talk and is shown as results. Steps 2 to 4 run live. Each step uses the previous step's output.

```
 Before the talk (batch)                        On stage (interactive)
 ┌──────────────────────────┐                   ┌──────────────────────────────────────────┐
 │ 1. Train the style       │   style LoRA      │ 2. Generate game objects on 8 GPUs       │
 │    ~30 illustrations     │ ───────────────▶  │ 3. Turn one image into a 3D model        │
 │    8 serverless jobs     │  (Object Storage) │ 4. Open the model on the laptop          │
 └──────────────────────────┘                   └──────────────────────────────────────────┘
```

Training runs as batch jobs and creation happens interactively. Object Storage carries the trained LoRA from one side to the other.

## 1. Train the style (before the talk)

- Select about 30 Nebius-owned illustrations and write a short caption for each. The dataset rules in [scripts/README.md](../scripts/README.md#dataset-rules) apply: no logos, wordmarks, text, stock images or images made by other generators.
- Run eight training jobs in parallel as Nebius serverless jobs, one GPU each, with Ostris's ai-toolkit. The jobs vary rank and learning rate, and each saves a checkpoint every 500 steps, so the sweep also covers step count.
- Each run takes roughly 1.5 to 2 hours (estimate, to be measured).
- Pick the best run and step from contact sheets that use a fixed set of prompts and seeds.
- The result is a LoRA: a small add-on file, roughly a few hundred MB (estimate), that teaches the general image model (Qwen-Image) the Nebius look.

**On stage:**
- Show the job list, the measured time and cost, and the LoRA file size.
- Show the same prompt, for example "a treasure chest", rendered without and with the LoRA. Without it the chest is generic; with it the chest is in Nebius style.
- Training takes too long to run live, so this step is shown as results.

## 2. Generate game objects on eight GPUs (live)

- The laptop connects to an 8x RTX PRO 6000 node through the [Omarchy Nebius plugin](https://github.com/antongisli/omarchy-nebius). The plugin's SSH port forwards make ComfyUI open in the laptop's browser at `127.0.0.1` as if it were local.
- Type a few prompts, such as "a sword", "a health potion power-up" and "a treasure chest". All eight GPUs generate variants at once and a grid fills within seconds. With an 8-step Lightning LoRA this is roughly 3 to 5 seconds per image per GPU (estimate).
- This is the prototyping step: a concept artist reviews dozens of styled variants in a minute instead of drawing each one.
- Prompts ask for a single object on a plain background, which gives the 3D step a clean input.

## 3. Turn one image into a 3D model (live)

- Pick one image from the grid, for example the chest.
- TRELLIS.2, which is built into ComfyUI, turns the single image into a textured 3D model saved as a GLB file.
- Upstream reports about 17 seconds per mesh on an H100 at 1024³ resolution. Expect under a minute on the demo node (estimate, to be measured).

## 4. Open the model on the laptop (live)

- Download the GLB over the same forwarded port and drop it into Godot or Blender on the laptop. Rotate it in the viewport.
- The demo ends with a Nebius-styled prop inside a game engine. It started from 30 illustrations, some batch training, and a laptop using eight cloud GPUs.

## Why the 3D step matters

Game developers rarely use a flat image directly. A rough, textured prop that they can place in a scene is useful for prototyping and blocking out levels. The demo is framed as a prototyping tool, not a replacement for artists, because some game developers are sceptical of generative art.

## Component choices

| Component | Choice | Reason |
|---|---|---|
| Base image model | Qwen-Image | Apache 2.0; supported by ai-toolkit; strong at illustration styles |
| Trainer | ai-toolkit (`ostris/aitoolkit` image) | Supports Qwen-Image LoRA training; runs headless from a config file |
| Training compute | 8 Nebius serverless jobs, 1 GPU each | Parallel sweep with no cluster to manage |
| Live compute | 8x RTX PRO 6000 VM, one ComfyUI process per GPU | 96 GB per GPU; failures stay limited to one GPU |
| Laptop connection | [Omarchy Nebius plugin](https://github.com/antongisli/omarchy-nebius) | Launches the VM and keeps SSH port forwards alive |
| ComfyUI install | [comfyui-setup](https://github.com/antongisli/comfyui-setup) | Repeatable install as a systemd user service |
| Image-to-3D | TRELLIS.2 (native ComfyUI) | MIT; the native port has no non-commercial dependencies |
| Not used | Hunyuan3D | Its licence does not apply in the EU, UK or South Korea, and the venue is in Stockholm |
| Not used | Flux.1-dev | Non-commercial licence for the weights |
