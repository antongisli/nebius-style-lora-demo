# Train an art style and use it live: Nebius LoRA demo

Code, configs and ComfyUI workflows from the talk "Magically turn your laptop into an 8-GPU beast: train your own art style and use it live" (Stockholm MLOps gaming night, 3 December 2026).

The project has two parts. Part 1 trains a style LoRA for Qwen-Image with ai-toolkit, as parallel Nebius serverless jobs. Part 2 connects a laptop to a GPU VM through the [omarchy-nebius](https://github.com/antongisli/omarchy-nebius) plugin and uses the LoRA in ComfyUI to make game objects, then turns one into a 3D model with TRELLIS.2.

- [talk/slides.html](talk/slides.html) is the talk deck as one self-contained HTML file with the speaker notes. Download it and open it in a browser.
- [REPRODUCE.md](REPRODUCE.md) is the step-by-step guide for repeating the demo with your own art.
- [docs/workflow.md](docs/workflow.md) describes the demo end to end.
- The `docs/` pages record each training round, with measured times, costs and sample sheets: [pilot](docs/pilot.md), [sweep 1](docs/sweep-v1.md), [sweep 2](docs/sweep-v2.md), [sweep 3](docs/sweep-v3.md), [2D to 3D](docs/e2e-2d-to-3d.md) and [dataset v3](docs/dataset-v3.md).
- [training/](training/) holds the sweep and job scripts, [node/](node/) the GPU VM setup and grid page, and [workflows/](workflows/) the ComfyUI workflows.

The training images, the trained LoRA weights and the generated outputs are not in this repository.

Related repositories:
- [omarchy-nebius](https://github.com/antongisli/omarchy-nebius): the Omarchy plugin that launches the GPU VM and forwards ports to the laptop.
- [comfyui-setup](https://github.com/antongisli/comfyui-setup): the ComfyUI installer used on the VM.
