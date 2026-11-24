# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pillow>=10",
#   "torch>=2.5",
#   "torchvision",
#   "transformers>=4.57",
#   "accelerate",
# ]
# ///
"""Draft one caption per training image with Qwen3-VL-8B-Instruct (Apache 2.0).

Each caption goes into a .txt file next to its image and starts with the trigger token,
for example `nbsstyle, a warning triangle with a gear, centered, plain background`.
ai-toolkit does not insert the trigger when it caches text embeddings, so the token is
written into the file. The model is asked for subject and composition only: the style is
what the trigger should learn, so the caption does not describe it. Read and fix every
caption by hand before training.

    uv run scripts/caption.py dataset/                       # one GPU, about 20 GB of memory
    uv run scripts/caption.py dataset/ --trigger mystyle --overwrite

Without a GPU (only Pillow is needed, so the inline torch dependency is skipped):

    uv run --with pillow python scripts/caption.py dataset/ --dry-run   # placeholder captions
    uv run --with pillow python scripts/caption.py dataset/ --check     # run before upload

Existing captions are kept unless --overwrite is given. --check fails on images without a
caption, captions that do not start with the trigger, and leftover placeholders.
"""
import argparse
import sys
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
PLACEHOLDER = "placeholder caption"
PROMPT = """Write a caption for this image, to be used for training an image generation model.
Write one line of short comma-separated phrases in lowercase, at most 40 words.
First name the main subject and its notable parts. Then describe the composition: where the
subject sits in the frame, the viewpoint, and what the background is.
Do not describe the art style, medium, line work, shading, colour palette or mood.
If the image contains any text, logo or watermark, mention it.
Do not start with phrases such as "the image shows". Output only the caption."""
PREFIXES = ("the image shows ", "this image shows ", "the image depicts ", "this image depicts ",
            "an image of ", "image of ", "caption: ")


def clean_caption(text: str, trigger: str) -> str:
    text = " ".join(text.split()).strip("\"'` ")
    for prefix in PREFIXES:
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    if text.lower().startswith(trigger.lower()):
        text = text[len(trigger):].lstrip(" ,:")
    text = text.rstrip(". ")
    if text:
        text = text[0].lower() + text[1:]
    return f"{trigger}, {text}" if text else trigger


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS)


def check(images: list[Path], trigger: str) -> int:
    problems = []
    for image in images:
        txt = image.with_suffix(".txt")
        if not txt.is_file():
            problems.append(f"{image.name}: no caption")
            continue
        caption = txt.read_text(encoding="utf-8").strip()
        if caption != trigger and not caption.startswith(f"{trigger},") and not caption.startswith(f"{trigger} "):
            problems.append(f"{txt.name}: does not start with the trigger '{trigger}'")
        if PLACEHOLDER in caption:
            problems.append(f"{txt.name}: still a placeholder")
        if "\n" in caption:
            problems.append(f"{txt.name}: more than one line")
    folder = images[0].parent
    stems = {p.stem for p in images}
    problems += [f"{t.name}: caption without an image" for t in sorted(folder.glob("*.txt")) if t.stem not in stems]
    for p in problems:
        print(p, file=sys.stderr)
    print(f"checked {len(images)} images: {'ok' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


class Captioner:
    def __init__(self, model_id: str, max_new_tokens: int, prompt: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype="auto", device_map="auto")
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt

    def __call__(self, path: Path) -> str:
        with Image.open(path) as im:
            image = im.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": self.prompt}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="folder of prepared images (output of prepare_dataset.py)")
    parser.add_argument("--trigger", default="nbsstyle", help="token written at the start of every caption")
    parser.add_argument("--model", default=MODEL, help=f"Hugging Face model id (default {MODEL})")
    parser.add_argument("--prompt-file", type=Path, help="replace the built-in captioning instruction")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--overwrite", action="store_true", help="replace existing captions")
    parser.add_argument("--dry-run", action="store_true", help="write placeholder captions without loading a model")
    parser.add_argument("--check", action="store_true", help="only check existing captions")
    args = parser.parse_args(argv)

    if not args.dataset.is_dir():
        print(f"error: {args.dataset} is not a folder", file=sys.stderr)
        return 1
    images = list_images(args.dataset)
    if not images:
        print(f"error: no PNG or JPEG images in {args.dataset}", file=sys.stderr)
        return 1
    seen: dict[str, str] = {}
    for p in images:
        if p.stem in seen:
            print(f"error: {p.name} and {seen[p.stem]} would share {p.stem}.txt", file=sys.stderr)
            return 1
        seen[p.stem] = p.name
    if args.check:
        return check(images, args.trigger)

    todo = [p for p in images if args.overwrite or not p.with_suffix(".txt").exists()]
    if args.dry_run:
        def describe(path: Path) -> str:
            return f"{PLACEHOLDER} for {path.stem}, replace by hand"
    elif todo:
        prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else PROMPT
        describe = Captioner(args.model, args.max_new_tokens, prompt)

    for i, path in enumerate(todo, 1):
        caption = clean_caption(describe(path), args.trigger)
        path.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
        print(f"[{i}/{len(todo)}] {path.name}: {caption}")

    kept = len(images) - len(todo)
    print(f"wrote {len(todo)} caption(s){' (dry run, placeholders)' if args.dry_run else ''}; "
          f"kept {kept} existing{' (use --overwrite to replace)' if kept else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
