# Reproduce the demo with your own art

These steps repeat the demo from [docs/workflow.md](docs/workflow.md) with your own illustrations. They are written for someone outside Nebius, so the page can be published as is once the scripts exist.

**Status:** this guide is a draft. Most scripts in this repository are not written yet. Each step is marked **ready** (usable today) or **planned** (written during November 2026, then tested end to end before the talk). Commands for planned steps show the intended interface and may change.

## What you need

- A Nebius AI Cloud account with billing set up and GPU quota. Sign up at <https://console.nebius.com/>.
- A laptop running [Omarchy](https://omarchy.org/) with plugin support, for the live part. Without Omarchy, plain `ssh -L` port forwarding works too (step 6).
- 20 to 40 images in one consistent style that you own or are licensed to train on. Leave out logos, text, stock images whose licences forbid training, and images made by other generators.
- Godot 4 or Blender on the laptop, to open the 3D model.

Costs are billed to your Nebius account. Measured costs for the talk will be added here; until then, treat any figure as an estimate.

## Part 1: train the style

### 1. Install the Omarchy Nebius plugin and the Nebius CLI (ready)

```bash
omarchy plugin add https://github.com/antongisli/omarchy-nebius --enable
```

Open the Nebius icon in the bar, choose **Set up Nebius**, and sign in through your browser. Setup installs the `nebius` CLI, which the training scripts also use. See the [plugin README](https://github.com/antongisli/omarchy-nebius) for details. On another OS, install the [Nebius CLI](https://docs.nebius.com/cli) directly.

### 2. Create the bucket and credentials (done by hand; script planned)

Follow [infra/README.md](infra/README.md): a bucket, a service account with `storage.editor` on the project through a group, an access key whose values go into a MysteryBox secret with the keys `S3_ACCESS_KEY_ID` and `S3_SECRET_ACCESS_KEY`, and an endpoint-only profile in `~/.aws/config`. Then write `deploy.env`:

```bash
PROJECT_ID=project-…
SUBNET_ID=vpcsubnet-…
BUCKET=<bucket>
S3_SECRET=mbsec-…
S3_ENDPOINT=https://storage.<region>.nebius.cloud
S3_PROFILE=nebius-<region>
```

`infra/setup_infra.sh` will do this in one step later.

The bucket holds the dataset, every training run's output, and the selected LoRA:

```
dataset/v1/        images and .txt captions
runs/<run-name>/   checkpoints every 500 steps, sample images, timing.json
selected/          the LoRA you pick
renders/           images for comparison and slides
```

### 3. Prepare and caption the dataset (planned)

```bash
scripts/prepare_dataset.py --input my-art/ --manifest my-art/manifest.csv --output dataset/
scripts/caption.py dataset/ --trigger mystyle
```

- `manifest.csv` records the source and rights for each image. Images without a manifest row are rejected.
- The preparation script resizes each image to 1024 px on the long side and strips metadata.
- The caption script drafts one `.txt` caption per image with Qwen3-VL-8B. Read and fix the captions by hand.
- Each caption starts with your trigger word (here `mystyle`). ai-toolkit does not insert trigger words when it caches text embeddings, so the word must be written into the caption.
- Optional: `uv run training/recolor.py my-art/ dataset/ --scheme lime-on-navy` repaints light line illustrations into a two-colour scheme. The talk's dataset used it to put Nebius illustrations into Lime on Deep Blue; edit the colours in the script for your own palette.
- Upload `dataset/` to `dataset/v1/` in the bucket: `uv run infra/bucket.py put dataset dataset/v1`.
- To test the pipeline before your images are ready, `uv run training/make_standin_dataset.py /tmp/standin` draws 12 simple flat illustrations with captions. Upload them to `dataset/standin` and run a short pilot with `DATASET=dataset/standin`.

### 4. Run the eight-job training sweep (written, not yet run end to end)

```bash
training/sweep.py --trigger mystyle   # renders 8 ai-toolkit configs into training/generated/
training/submit_sweep.sh             # submits 8 serverless jobs; DRY_RUN=1 prints the commands
training/jobctl.sh list              # watch progress; also: logs, get, cancel
```

- Each job runs `ostris/aitoolkit` on one GPU, with Qwen-Image as the base model. The default is a preemptible H100 at the current spot price (`--preemptible --follows-spot-price`). Set `ALLOCATION=on-demand` to avoid preemption, or `PLATFORM`/`PRESET` to use another GPU; the RTX PRO 6000 (`gpu-rtx6000-a`, `1gpu-24vcpu-218gb`) also works with this image.
- A preempted job restarts and resumes from its last checkpoint, which is synced to the bucket every five minutes.
- The sweep covers rank {16, 32} × learning rate {5e-5, 1e-4, 2e-4, 4e-4}. Each run trains for 3000 steps and saves a checkpoint and samples every 500 steps.
- Jobs write to local disk and copy their results to `runs/<run-name>/` in the bucket when they finish.
- Under the hood, each job is one `nebius ai job create` call with the bucket mounted as a volume (see `training/submit_sweep.sh`).

### 5. Pick the best LoRA (planned)

```bash
training/collect_metrics.py > measurements.csv   # wall-clock time, GPU-hours, cost at list price, file size per run
```

Compare the sample images per run and step, copy the best checkpoint to `selected/` in the bucket, and note its run and step.

## Part 2: use the style live

### 6. Launch the GPU node and forward ports (ready)

In the plugin, press **G**, choose an 8-GPU RTX PRO 6000 offering, and launch it. Choose on-demand rather than preemptible for a live session, since a preemptible VM can be stopped mid-demo.

Then press **P** → **N** and add two forwards on that VM:

| VM port | Laptop port | Purpose |
|---|---|---|
| 8188 | 8188 | ComfyUI (GPU 0) |
| 8080 | 8080 | Grid page that spreads prompts over all 8 GPUs |

Without Omarchy:

```bash
ssh -N -L 127.0.0.1:8188:127.0.0.1:8188 -L 127.0.0.1:8080:127.0.0.1:8080 user@<vm-address>
```

### 7. Install ComfyUI on the node (ready once comfyui-setup is public)

On the VM:

```bash
git clone https://github.com/antongisli/comfyui-setup.git
bash comfyui-setup/comfy.sh
```

This installs ComfyUI as a systemd user service on `127.0.0.1:8188`, with Model Resolver, Model Manager and aria2. See the [comfyui-setup README](https://github.com/antongisli/comfyui-setup).

### 8. Start eight instances and download the models (written, not yet run on a node)

On the VM, with `node/` and `workflows/` from this repository copied next to each other (see [node/README.md](node/README.md)), and inside `tmux` because the first run downloads about 65 GB:

```bash
export AWS_ACCESS_KEY_ID=<key-id> AWS_SECRET_ACCESS_KEY=<secret>
LORA_URL=s3://<bucket>/selected/mystyle.safetensors TRIGGER_WORD=mystyle node/setup.sh
```

- Replaces the single service from step 7 with one ComfyUI process per GPU: GPU 0 on port 8188, GPUs 1–7 on 8189–8195. All eight share one models folder and one output folder.
- Downloads Qwen-Image, an 8-step Lightning LoRA, the TRELLIS.2 weights and a background-removal model, then your LoRA. Your LoRA is linked as `models/loras/style.safetensors`, the name the workflows use.
- The S3 endpoint defaults to `https://storage.<NEBIUS_REGION>.nebius.cloud` (region `eu-north1` unless you set `NEBIUS_REGION`). `LORA_URL` can also be an `https://` URL, or the path of a LoRA file you already copied to the VM, which avoids putting bucket credentials there.
- Starts the grid page on `127.0.0.1:8080` and waits until all eight instances answer.
- Safe to re-run, for example to fetch a different LoRA.

### 9. Generate game objects (written, not yet run on a node)

- Open <http://127.0.0.1:8188> and drag `workflows/compare_lora.json` onto the canvas to see the same prompt and seed without and with your style.
- Open <http://127.0.0.1:8080> and press **Warm up** once, so every GPU loads the models. Then enter a list of prompts such as "a sword" and press **Generate**. Each prompt runs as eight variants spread over the eight GPUs, and the grid fills as they finish. The page adds your trigger word in front and "single object, centered, plain white background" after each prompt; both can be changed.

### 10. Make a 3D model (written, not yet run on a node)

On the grid page, click an image and press **Make 3D**. The page runs `workflows/image_to_3d.json` (TRELLIS.2) on a free GPU and shows a download link when the GLB is ready. To do the same in ComfyUI, drag `workflows/image_to_3d.json` onto the canvas, put the image in the **Image** node, and run it. GLB files are saved under `output/3d/` on the VM.

### 11. Open it on the laptop (written, not yet run on a node)

Use the download link on the grid page, or:

```bash
curl -o prop.glb "http://127.0.0.1:8080/view?filename=<name>.glb&subfolder=3d&type=output"
```

Files from the grid page are named `<job-id>_00001_.glb`; files made in ComfyUI with the workflow's default prefix are named `prop_00001_.glb`. Port 8188 serves the same files.

Drag `prop.glb` into a Godot 4 project, or use **File → Import → glTF 2.0** in Blender. Textures are embedded in the file.

### 12. Clean up (ready)

- Stop or delete the VM from the plugin (**V**, then stop or delete). A stopped VM still bills for its disk.
- Delete the training jobs' leftover volumes and the bucket when you no longer need them.

## Licences of the parts used

| Part | Licence |
|---|---|
| Qwen-Image | Apache 2.0 |
| ai-toolkit | MIT |
| ComfyUI | GPL-3.0 |
| TRELLIS.2 | MIT |
| Qwen3-VL-8B (captions) | Apache 2.0 |
| Omarchy Nebius plugin | MIT |

Licences for the Lightning LoRA and other downloaded weights will be checked and listed here before the guide is published.
