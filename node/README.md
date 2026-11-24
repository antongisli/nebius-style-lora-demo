# Node setup: eight ComfyUI instances and the grid page

This folder turns a GPU VM with one ComfyUI install into one ComfyUI process per GPU, plus a grid page that spreads prompts over all of them. It is written for the 8x RTX PRO 6000 VM used in the demo, but works with any number of NVIDIA GPUs.

**Status:** run on a one-GPU RTX PRO 6000 VM on 2026-10-30 and a one-GPU H100 VM on 2026-11-19, with separate boot images for the talk (below). The eight-instance setup is still tested against mock ComfyUI instances only; see [What still needs a real node](#what-still-needs-a-real-node).

## The talk's demo box: one RTX PRO 6000, from a custom image

The custom images named in this section live in the author's Nebius project and are not public. **Rebuilding the image** below shows how to build the same setup in your own project.

The live demo uses one GPU, not eight. The boot image `nebius-rpg-comfyui-v3` (image family `nebius-rpg-comfyui`, region uk-south2, platform `gpu-rtx6000-a`) has everything installed. A VM launched from it starts ComfyUI on `127.0.0.1:8188` and the grid page on `127.0.0.1:8080` at boot, as user `dev`, without anyone logging in.

**Warm-up at boot.** A disk created from an image is slow until it has been read once, and a network SSD's bandwidth grows with its size: about 15 MiB/s per 32 GiB, up to 450 MiB/s at 960 GiB. On v1's 200 GiB disk that was about 100 MB/s, and the first generation took 503 s. Since v2, the image fixes both:

- Its minimum disk size is 960 GiB, so the plugin sizes the boot disk for full bandwidth when the image is picked.
- `comfyui-warm.service` ([comfyui-warm.py](comfyui-warm.py)) reads every model file with 32 reads in parallel, then runs one hidden generation so Qwen-Image is on the GPU. Its progress is in `systemctl --user status comfyui-warm` and `~/.local/state/comfyui-gpu/warm-status`.

Measured on a fresh v2 VM: warm about 4.5 minutes after boot (73 GB read in 189 s at 388 MB/s, then an 18 s load), and the first four-image generation took 9 s. For comparison, a 1,116 GiB `network_ssd_io_m3` disk read the same files at about 1 GB/s, but the plugin only creates `network_ssd` disks.

**On the day, in the Omarchy Nebius plugin:**

1. **G** → RTX PRO 6000 → uk-south2, 1 GPU, the project that holds the image. On-demand rather than preemptible, so the VM is not taken back during the talk.
2. **Boot image** → `nebius-rpg-comfyui-v3`. The boot disk becomes 960 GiB (the image's minimum); don't make it smaller. Review and create.
3. **P** → add port 8188 (and 8080 for the grid page). Open <http://127.0.0.1:8188>. ComfyUI answers about a minute after the VM reports RUNNING; the warm-up finishes about 4.5 minutes after boot. A job queued before that waits for the warm-up job in the queue.
4. Open the **Workflows** sidebar (**W**) and pick:

| Workflow | What it does | Time on one RTX PRO 6000 |
|---|---|---|
| `1 - Nebius RPG Gen` | Qwen-Image + Lightning (8 steps) + the Nebius style LoRA at 1.3, four 1024 px images per run, new seed each run. Edit the subject in **Prompt** and keep the rest of the text. Saves `output/nebius_rpg_*.png`. | 27 s for the first run (model load), 8.4 s for four images after that |
| `2 - Nebius RPG to 3D` | TRELLIS.2 with the faster settings, then **Nebius Lime Snap** (`keep gold`), then **Save 3D Model**, which shows the model in a 3D viewer. **Load Image (from Outputs)** picks the newest generated image by itself; use its dropdown for another one. Saves `output/3d/rpg_*.glb`. | 55 s for the first run, with Qwen-Image still loaded |
| `3 - Base vs Nebius style` | The same prompt and seed with and without the style LoRA, side by side. | not timed; two images, so roughly half a Gen run |

5. Download the GLB with the download button at the bottom right of the 3D viewer, or `curl -o item.glb "http://127.0.0.1:8188/view?filename=rpg_00001_.glb&subfolder=3d&type=output"`, and open it in Blender with **File → Import → glTF 2.0**.

The boot warm-up loads the Gen models. Still run each workflow once before going on stage, so the 3D models are on the GPU too and the downloaded GLB is a tested fallback.

**Nebius Lime Snap** ([comfyui_nebius/](comfyui_nebius/)) is a small custom node: the in-graph version of [e2e/snap_glb.py](../e2e/snap_glb.py), which recolours the drifted Lime in TRELLIS.2's base-colour texture back to #DAFF33. On the test crown, 19 % of texels ended within RGB distance 40 of Lime, with mean (208, 242, 49).

**Rebuilding the image** (about 1.5 hours, most of it the model download and the image copy):

1. Launch an RTX PRO 6000 VM from the public `ubuntu24.04-cuda13.0` image with a user named `dev` (the plugin's user), 200 GiB disk.
2. Copy `comfyui-setup`, `node/`, `workflows/` and the selected LoRA to the VM; run `bash comfyui-setup/comfy.sh` (answer **N** to the IPv6 question), then `NUM_GPUS=1 LORA_URL=demo/nbsstyle-v3syn-2000.safetensors demo/node/setup.sh`. Copying the LoRA from the laptop keeps bucket credentials off the image.
3. Check the three workflows in the browser, then run `demo/node/prepare_image.sh` on the VM. It removes the build SSH key, history, caches, test outputs, cloud-init state, host keys and the machine ID.
4. From the laptop: grow the boot disk to 960 GiB (`nebius compute disk update --id <boot-disk> --size-gibibytes 960`; the image's minimum disk size is the source disk's size), `nebius compute instance stop --id <vm>`, then `nebius compute image create --parent-id <project> --name nebius-rpg-comfyui-v<N> --image-family nebius-rpg-comfyui --source-disk-id <boot-disk> --recommended-platforms gpu-rtx6000-a`.

v2 and v3 were built faster: boot a VM from the previous image, copy the changed `node/` and `workflows/` into `~/demo/`, run `SKIP_MODELS=1 ~/demo/node/setup.sh`, then steps 3 and 4 (about 30 minutes). v3 only differs from v2 in workflow 2: it no longer remembers a GLB from the build VM, which made the 3D viewer fail on a fresh VM until a new model was made.

Before saving a workflow from the browser into `workflows/ui/`, check it does not remember output files: `Last Time Model File` on the Save 3D Model node, and the `image` value on Load Image (from Outputs).

The image holds about 87 GB: ComfyUI and its virtual environment, the models listed under [Models and licences](#models-and-licences), and the style LoRA as `models/loras/style.safetensors`.

## H100 image in Finland

`nebius-rpg-comfyui-h100-v1` (family `nebius-rpg-comfyui-h100`) is available in `eu-north1`. Select `gpu-h100-sxm`, preset `1gpu-16vcpu-200gb`, regular/on-demand capacity, and a **960 GiB or larger `network_ssd` boot disk**. The image contains the same selected style LoRA, model files, saved workflows, and custom Lime Snap node as the RTX demo setup. Create the VM with an SSH key for user `dev`; forward ports 8188 and 8080 as above.

The H100 image was built separately on Ubuntu 24.04 with NVIDIA driver 580.196.07, PyTorch 2.15.0+cu132, and ComfyUI commit `995456d0188ffb51032c28f41527cfa82d87d8a9`. The UK image cannot be used directly as a source image for a disk in Finland.

Verified through the ComfyUI API on one non-spot H100 80 GB:

| Check | Measured wall time |
|---|---|
| One 1024 px styled image, including initial model loading | 21 s |
| Same-prompt comparison with and without the style LoRA | 6 s |
| TRELLIS.2 fast workflow after releasing Qwen | 123 s |
| TRELLIS.2 fast workflow directly after a new Qwen generation | 78 s |

The GLB contained a mesh, material, and three embedded texture images. Switching back from TRELLIS to Qwen took 111 s in this test; allow time for model changes during rehearsal. These are individual measured runs, not guaranteed latency. The four-image browser workflow was not timed on H100.

Fresh-boot verification passed on a second non-spot H100 launched from the image: cloud-init recreated SSH access, ComfyUI and the grid started automatically, all saved-workflow node types were registered, and the style LoRA was available. The automatic warm-up completed a generation successfully in 191 s from service start (169 s reading 73.4 GB, then an 18 s generation). Check `~/.local/state/comfyui-gpu/warm-status` before the demo. Build identifiers, exact dependency versions, and API test results are recorded locally under the gitignored `results/h100-image/` directory.

## What runs where

| Service (systemd user unit) | Listens on | Purpose |
|---|---|---|
| `comfyui-gpu@0` | `127.0.0.1:8188` | ComfyUI on GPU 0; the instance you open in the browser |
| `comfyui-gpu@1` … `@7` | `127.0.0.1:8189` … `8195` | ComfyUI on GPUs 1–7; used by the grid page |
| `comfyui-grid` | `127.0.0.1:8080` | Grid page ([dispatcher/](dispatcher/)) |

All instances run from the same `~/ComfyUI` checkout, so they share one `models/`, `input/` and `output/` folder. Each instance has its own temp folder, and GPUs 1–7 have their own user folder, under `~/.local/state/comfyui-gpu/gpu<i>/`. GPU 0 keeps the default user folder, so workflows saved in the browser stay where `comfy.sh` put them.

Systemd templates cannot compute `8188 + i`, so the unit `comfyui-gpu@.service` calls a small wrapper, [comfyui-gpu.sh](comfyui-gpu.sh) (installed as `~/.local/bin/comfyui-gpu`), which works out the port and passes `--cuda-device i`. The unit sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so GPU numbers match `nvidia-smi`.

## Run the setup over SSH

1. Launch the VM with the [Omarchy Nebius plugin](https://github.com/antongisli/omarchy-nebius) and add the two port forwards below.
2. On the VM, install ComfyUI with [comfyui-setup](https://github.com/antongisli/comfyui-setup):

   ```bash
   git clone https://github.com/antongisli/comfyui-setup.git
   bash comfyui-setup/comfy.sh
   ```

3. Copy `node/` and `workflows/` from this repository to the VM, keeping them side by side. From the laptop:

   ```bash
   rsync -a node workflows <user>@<vm-address>:demo/
   ```

4. On the VM, run the setup. Run it inside `tmux` or `screen`, because the first run downloads about 65 GB:

   ```bash
   export AWS_ACCESS_KEY_ID=<key-id> AWS_SECRET_ACCESS_KEY=<secret>
   LORA_URL=s3://<bucket>/selected/nbsstyle-v3syn-2000.safetensors NEBIUS_REGION=eu-north1 demo/node/setup.sh
   ```

   `LORA_URL` can also be an `https://` URL, a path to a `.safetensors` file already on the VM, or left out to run without a style LoRA. A local file avoids putting bucket credentials on the VM: fetch the LoRA on the laptop (in this repository, `uv run infra/bucket.py get selected/ ./lora/` copies everything under `selected/` into `./lora/`), copy the file over with `rsync`, and pass its path:

   ```bash
   LORA_URL=demo/nbsstyle-v3syn-2000.safetensors demo/node/setup.sh
   ```

   For S3 the default endpoint is `https://storage.<NEBIUS_REGION>.nebius.cloud`; set `S3_ENDPOINT` to override it. If the `aws` CLI is missing, the script installs it into a private virtual environment. The other settings are listed at the top of [setup.sh](setup.sh).

The script:

- stops and disables the single `comfyui.service` that `comfy.sh` installed, since it also uses port 8188;
- downloads the models listed below into `~/ComfyUI/models/`, skipping files that are already complete and resuming partial ones;
- fetches the style LoRA and links it as `models/loras/style.safetensors`, the name the workflows use;
- installs and starts `comfyui-gpu@0..N-1` and `comfyui-grid`, and enables lingering so they keep running after you log out;
- waits until every instance answers `/system_stats` and the grid page answers, then prints the ports to forward.

Re-running it is safe. Services restart only when their files changed. If you re-run `comfy.sh` later (for example to update ComfyUI), run `setup.sh` again afterwards, because `comfy.sh` re-enables `comfyui.service`.

## Ports to forward

Only two ports leave the VM. In the plugin, press **P** → **N** and add both on the VM:

| VM port | Laptop port | Purpose |
|---|---|---|
| 8188 | 8188 | ComfyUI on GPU 0 |
| 8080 | 8080 | Grid page; it also proxies images and GLB files from every instance |

Ports 8189–8195 stay on the VM. The grid page reaches them locally.

Without Omarchy:

```bash
ssh -N -L 127.0.0.1:8188:127.0.0.1:8188 -L 127.0.0.1:8080:127.0.0.1:8080 <user>@<vm-address>
```

## Use it

**Grid page** (<http://127.0.0.1:8080>): enter one prompt per line and press **Generate**. Each prompt runs as several variants (default 8) with random seeds, spread round-robin over the GPUs that are up, at most two jobs queued per GPU. Tiles show which GPU made each image and how long it took. The trigger word (`nbsstyle` by default, `TRIGGER_WORD` in setup) is put before each prompt when the style LoRA is on. **Warm up** sends one job to each GPU so every instance loads the models before the audience is watching; the first job on each GPU takes much longer than the rest.

If a GPU's instance stops answering, the page marks it down and moves its unfinished jobs to other GPUs (up to three attempts per job). When the instance comes back, it gets new jobs again.

**Before/after:** in ComfyUI on port 8188, drag [workflows/compare_lora.json](../workflows/compare_lora.json) onto the canvas. It renders one prompt with the same seed twice, without and with the style LoRA, and saves both under `output/compare/`.

**3D:** on the grid page, click a finished image and press **Make 3D**. The page runs [workflows/image_to_3d.json](../workflows/image_to_3d.json) on a free GPU and shows a download link when the GLB is ready. To run it in ComfyUI instead, drag the workflow onto the canvas, load or drop the image into the **Image** node, and queue it.

The workflow files are in ComfyUI's API format, which the dispatcher fills in by node title (`Prompt`, `Style LoRA`, `Image`). ComfyUI's browser UI accepts API-format files, but lays the nodes out automatically; after the first run on a node, save a tidied copy from the browser for the talk.

## Fetch the GLB

GLB files are saved in the shared `output/3d/` folder. From the laptop, through either forwarded port:

```bash
curl -o prop.glb "http://127.0.0.1:8080/view?filename=<name>.glb&subfolder=3d&type=output"
```

Port 8188 works the same way. Files made from the grid page are named `<job-id>_00001_.glb`; the page's download link has the exact name. Files made in the ComfyUI UI with the workflow's default prefix are named `prop_00001_.glb`, `prop_00002_.glb` and so on. To list them, run `ls ~/ComfyUI/output/3d` on the VM.

Drag the file into a Godot 4 project, or use **File → Import → glTF 2.0** in Blender. Textures are embedded in the GLB.

## Operate

```bash
systemctl --user status 'comfyui-gpu@*' comfyui-grid   # overview
journalctl --user -u comfyui-gpu@3 -f                   # logs for GPU 3
systemctl --user restart comfyui-gpu@3                  # restart one GPU
curl -s http://127.0.0.1:8080/api/state | python3 -m json.tool | head -40
```

Settings written by the setup are in `~/.config/comfyui-gpu/env`. `COMFY_EXTRA_ARGS` in that file is passed to every ComfyUI instance.

## Models and licences

Downloaded by `setup.sh` (sizes from Hugging Face on 2026-10-28; about 65 GB in total):

| File | Folder | Size | Source | Licence |
|---|---|---|---|---|
| `qwen_image_bf16.safetensors` | `diffusion_models` | 40.9 GB | [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) | Apache 2.0 |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `text_encoders` | 9.4 GB | same | Apache 2.0 |
| `qwen_image_vae.safetensors` | `vae` | 0.25 GB | same | Apache 2.0 |
| `Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors` | `loras` | 0.85 GB | [lightx2v/Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning) | Apache 2.0 (model card) |
| `trellis_2_bf16.safetensors` | `diffusion_models` | 10.3 GB | [Comfy-Org/TRELLIS.2](https://huggingface.co/Comfy-Org/TRELLIS.2) | MIT |
| `trellis_2_shape_vae_bf16.safetensors` | `vae` | 1.1 GB | same | MIT |
| `trellis_2_texture_vae_bf16.safetensors` | `vae` | 0.95 GB | same | MIT |
| `dino_v3_vit_l.safetensors` | `clip_vision` | 1.2 GB | same | Repackaged under MIT, but the upstream DINOv3 weights use Meta's [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license), not MIT. Review before publishing. |
| `birefnet.safetensors` | `background_removal` | 0.44 GB | [Comfy-Org/BiRefNet](https://huggingface.co/Comfy-Org/BiRefNet) | MIT |

Choices behind the list:

- **bf16 Qwen-Image, not fp8.** The Lightning authors report grid artefacts when their LoRA is used with the plain fp8 file. With 96 GB per GPU the bf16 weights fit (an estimate of about 51 GB for the diffusion model and text encoder together, before activations).
- **Lightning 8-step V2.0.** The authors report less over-saturation than V1.x. The workflows use 8 steps, CFG 1.0, `euler`/`simple`, and shift 3.1, as in ComfyUI's own Qwen-Image template.
- **TRELLIS.2 bf16, not the int8 file.** The int8 file is half the size, but the bf16 file avoids depending on int8 kernel support on `sm_120`.
- **`image_to_3d.json`** follows the TRELLIS.2 branch of ComfyUI's official template *Pixal3D & TRELLIS.2: Image to Model* (`3d_pixal3d_trellis2_image_to_model.json`), with three changes: it uses `dino_v3_vit_l` as listed in the TRELLIS.2 model card (the template uses the Pixal3D NAF variant for both branches), it crops with `pad_factor` 1.0 as the `Trellis2Conditioning` tooltip asks for TRELLIS.2 (the template uses 1.1), and it decimates to 200,000 faces and bakes 2048 px textures instead of 700,000 faces and 4096 px, to keep the prop light for a game engine and the run short. Sampler settings are unchanged.

Node names and input names in all three workflows were checked against the ComfyUI source at commit `4845e7a` (2026-10-28) and the workflow templates at `14220b8`.

## Tests

The dispatcher has no dependencies beyond the Python standard library. Its tests start mock ComfyUI servers that implement `/system_stats`, `/prompt`, `/queue`, `/history`, `/view` and `/models/loras`:

```bash
python3 -m unittest discover -s node/dispatcher -v
```

They cover round-robin order and even spread over eight instances, the per-instance queue limit, an instance that is down from the start, an instance that crashes mid-run, an instance that restarts and forgets its queue, recovery, the retry limit, validation errors (not retried), execution errors, warm-up pinning, the 3D hand-off, the `/view` proxy and its fallback to another instance, and request validation.

## What still needs a real node

Run on one RTX PRO 6000 (`sm_120`, driver 580, PyTorch 2.15 cu132) on 2026-10-30: `setup.sh` end to end with `NUM_GPUS=1` and a local LoRA file, including a re-run on the set-up node; the 2D workflows with the style LoRA; TRELLIS.2 with the faster settings, next to a loaded Qwen-Image (both fit in 96 GB); the GLB download over a forwarded port.

Still not run:

- The S3 fetch of the LoRA from Nebius Object Storage (the build copied the file from the laptop).
- Eight ComfyUI processes on one VM: GPU pinning with `--cuda-device`, RAM use (each process loads its own copy of the models; host RAM need is not measured), and the time for all eight to load models from the shared disk.
- `image_to_3d.json` (the slower, full-quality settings) on `sm_120`, and whether `dino_v3_vit_l` gives the same result as the template's NAF variant.
- Whether `LoadImage` accepts the grid's `grid/<name>.png [output]` path in this ComfyUI version (the source says it does; not run).
- GLB import with textures in Godot 4 and Blender.
- The grid page in a browser over the plugin's forwarded port, including image proxying under load.
- Launching the image through the Omarchy plugin (tested with the Nebius CLI and a plugin-style cloud-init only).
