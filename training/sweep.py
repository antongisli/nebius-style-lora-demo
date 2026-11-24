#!/usr/bin/env python3
"""Render one ai-toolkit config per sweep run from training/base.yaml.

The default grid is rank {16, 32} x learning rate {5e-5, 1e-4, 2e-4, 4e-4}: eight runs.
Each run saves a checkpoint and samples every --save-every steps, so step count is
compared from the saved checkpoints instead of spending runs on it.
"""
import argparse
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Regularization images: the base model's own renders, captioned without the trigger, so the LoRA
# keeps the base model's materials for prompts that look like them.
REG_DATASET = """        - folder_path: "/work/reg"
          is_reg: true
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          cache_latents_to_disk: true
          resolution: [512, 768, 1024]
"""


def load_prompts(path: Path, trigger: str) -> list[str]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [line.replace("{trigger}", trigger) for line in lines if line and not line.startswith("#")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trigger", default="nbsstyle")
    parser.add_argument("--prefix", default="style", help="run name prefix")
    parser.add_argument("--ranks", default="16,32")
    parser.add_argument("--lrs", default="5e-5,1e-4,2e-4,4e-4")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--prompts", type=Path, default=HERE / "prompts.txt")
    parser.add_argument("--out", type=Path, default=HERE / "generated")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-quantize", action="store_true", help="train on bf16 weights instead of qfloat8")
    parser.add_argument("--content-or-style", default="balanced", choices=["balanced", "style", "content"])
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--keep", action="store_true", help="keep configs already in --out")
    parser.add_argument("--reg", action="store_true",
                        help="add the regularization images in /work/reg (REG_DATASET in submit_sweep.sh); captions without the trigger")
    parser.add_argument("--reg-weight", type=float, default=1.0)
    args = parser.parse_args()

    template = (HERE / "base.yaml").read_text()
    prompts = json.dumps(load_prompts(args.prompts, args.trigger))
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.keep:
        for old in args.out.glob("*.yaml"):
            old.unlink()

    for rank, lr in itertools.product(args.ranks.split(","), args.lrs.split(",")):
        name = f"{args.prefix}-r{rank}-lr{lr}".replace(".", "p")
        config = template
        for key, value in {
            "name": name,
            "rank": rank,
            "lr": lr,
            "steps": args.steps,
            "save_every": args.save_every,
            "prompts": prompts,
            "batch_size": args.batch_size,
            "quantize": str(not args.no_quantize).lower(),
            "content_or_style": args.content_or_style,
            "ema": str(args.ema).lower(),
            "reg_dataset": REG_DATASET if args.reg else "",
            "reg_weight": args.reg_weight,
        }.items():
            config = config.replace("{{" + key + "}}", str(value))
        (args.out / f"{name}.yaml").write_text(config)
        print(args.out / f"{name}.yaml")


if __name__ == "__main__":
    main()
