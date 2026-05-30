"""
Single training run. Usage:
  python scripts/train.py --config configs/lora_baseline.yaml
  python scripts/train.py --config configs/lora_baseline.yaml --r 16 --lora_alpha 32 --run_name r16_a32
"""
import argparse
import csv
import sys
from pathlib import Path

import yaml
from transformers import Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import QwenVQACollator, load_vqa_subset
from src.model import apply_lora, count_trainable_params, load_model_and_processor
from src.metrics import evaluate_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--r", type=int, default=None)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    # CLI overrides
    if args.r is not None:
        cfg["lora"]["r"] = args.r
    if args.lora_alpha is not None:
        cfg["lora"]["lora_alpha"] = args.lora_alpha
    run_name = args.run_name or f"r{cfg['lora']['r']}_a{cfg['lora']['lora_alpha']}"
    cfg["training"]["output_dir"] = f"outputs/{run_name}"

    print(f"\n{'='*50}")
    print(f"Run: {run_name}  |  r={cfg['lora']['r']}  alpha={cfg['lora']['lora_alpha']}")
    print(f"{'='*50}\n")

    model, processor = load_model_and_processor(cfg["model_name"])
    model = apply_lora(model, **cfg["lora"])
    param_info = count_trainable_params(model)

    train_ds, eval_ds = load_vqa_subset(cfg["sample_size"], cfg["seed"])

    collator = QwenVQACollator(processor)

    training_args = TrainingArguments(
        **{k: v for k, v in cfg["training"].items() if k != "output_dir"},
        output_dir=cfg["training"]["output_dir"],
        run_name=run_name,
        remove_unused_columns=False,  # keep image/question cols for the collator
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train()

    # Eval (manual generate-and-match on held-out split)
    device = next(model.parameters()).device
    acc = evaluate_model(model, processor, eval_ds, device=device)
    print(f"\nEval exact-match accuracy: {acc:.4f}")

    # Log to results CSV
    results_path = Path("results/sweep_results.csv")
    results_path.parent.mkdir(exist_ok=True)
    write_header = not results_path.exists()
    with open(results_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_name", "r", "lora_alpha", "trainable_params", "trainable_ratio", "eval_acc"])
        if write_header:
            w.writeheader()
        w.writerow({
            "run_name": run_name,
            "r": cfg["lora"]["r"],
            "lora_alpha": cfg["lora"]["lora_alpha"],
            "trainable_params": param_info["trainable"],
            "trainable_ratio": f"{param_info['ratio']:.6f}",
            "eval_acc": f"{acc:.4f}",
        })

    print(f"Results appended to {results_path}")


if __name__ == "__main__":
    main()
