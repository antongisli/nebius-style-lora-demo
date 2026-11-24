# Dataset scripts

Two scripts turn a hand-picked set of illustrations into a training folder for ai-toolkit: `prepare_dataset.py` checks the rights record and normalises the images, and `caption.py` drafts a caption per image. Both declare their dependencies inline, so `uv run` installs them.

## Dataset rules

These apply to every image:

- Use 20 to 40 Nebius-owned illustrations from the marketing library: service illustrations, key visual variants, and line patterns.
- Exclude logos, wordmarks and anything with text.
- Exclude Getty stock images. Their licences generally forbid training.
- Exclude images made by other generators, such as Midjourney outputs.
- Dataset images never go into git. `dataset/` is gitignored, and the bucket is the source of truth.

## Workflow

### 1. Select

Copy the candidate images into one folder, for example `my-art/`. Any common format works (PNG, JPEG, WebP, TIFF); the preparation step converts them.

### 2. Write the manifest

Create `my-art/manifest.csv` with one row per image in the folder:

```csv
filename,source,owner,rights_note,include,notes
alerts.png,"Design System Figma file, Illustrations page",Nebius,Nebius-owned product illustration,yes,
partner-logo.png,Marketing library,Nebius,,no,contains a wordmark
```

- `filename`, `source`, `owner`, `rights_note` and `include` are required. Extra columns are kept.
- Every image in the folder needs a row. To leave an image out, keep the row and set `include` to `no`, so the decision is recorded.
- Rows with `include=yes` must have `source`, `owner` and `rights_note` filled in.

### 3. Prepare

```bash
uv run scripts/prepare_dataset.py --input my-art/ --output dataset/v4/
```

The script checks the folder against the manifest and decodes every included image before writing anything. If an image has no row, an included row is incomplete or points to a missing file, or an image cannot be read, it lists the problems, exits with status 1 and writes nothing.

For each included image it then:

- applies the EXIF orientation, converts embedded colour profiles to sRGB, and flattens transparency onto white (`--background` changes the colour);
- resizes so the long side is 1024 px (`--size`). Smaller sources are not enlarged unless `--max-upscale` allows it (for example `--max-upscale 1.5`), and the summary notes each small source;
- saves a PNG (or a quality-95 JPEG with `--format jpg`) without EXIF, XMP, ICC or text metadata;
- writes `manifest.csv` into the output folder with the included rows plus the output file name, source and output sizes, and a SHA-256 of each output file.

The summary also lists pairs of images whose perceptual hashes are within 6 bits of each other (`--dup-threshold`). These are usually the same illustration exported twice or at two sizes; drop one of each pair unless the difference is intended. Existing `.txt` captions in the output folder are not touched, so preparation can be re-run after captioning.

### 4. Caption

On a machine with one GPU (Qwen3-VL-8B in bf16 needs about 20 GB of GPU memory):

```bash
uv run scripts/caption.py dataset/v4/ --trigger nbsstyle
```

The script writes `<image>.txt` next to each image, for example `nbsstyle, a warning triangle with an exclamation mark and a gear, centered, plain background`. The model is asked for the subject and the composition, not the style, because the style is what the trigger token should learn. The trigger is written into each caption because ai-toolkit does not substitute `[trigger]` when it caches text embeddings.

Existing captions are kept; pass `--overwrite` to redraft them. `--prompt-file` replaces the built-in instruction, and `--model` selects another Qwen3-VL checkpoint.

To test the pipeline without a GPU, write placeholder captions instead. Running the script with `python` skips the inline torch dependency:

```bash
uv run --with pillow python scripts/caption.py dataset/v4/ --dry-run
```

### 5. Edit by hand

Read every caption next to its image and fix it. Points to check:

- The caption starts with the trigger token and stays on one line.
- It names what is in the image, not how it is drawn.
- Anything the caption leaves out is learned by the trigger. Dataset v1 captions never mentioned the navy background, so the trigger learned it along with the line style ([docs/dataset-v3.md](../docs/dataset-v3.md)). If the trigger should not learn something that every image shares, such as a background colour, name it in the captions.
- If a caption mentions text, a logo or a watermark, check the image; under the rules above it should probably be excluded.

Then check the folder. This fails on missing captions, captions without the trigger, and leftover placeholders:

```bash
uv run --with pillow python scripts/caption.py dataset/v4/ --check
```

### 6. Upload

```bash
uv run infra/bucket.py put dataset/v4 dataset/v4
```

Training jobs read the folder named by `DATASET` (default `dataset/v1`, see [training/run_job.sh](../training/run_job.sh)), so submit with `DATASET=dataset/v4`. The bucket already holds `dataset/v1` to `v3-*`, which finished runs were trained on; give each new selection a new version folder instead of changing one of those.

## Tests

```bash
uv run --no-project --with pytest --with pillow --with imagehash pytest scripts/tests -q
```

The tests generate synthetic images in a temporary folder, run the preparation end to end, and run `caption.py` with `--dry-run` and `--check`. They do not load the captioning model.
