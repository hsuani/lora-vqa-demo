"""Fast pipeline sanity check — NO training, NO full dataset.

Loads a handful of VQAv2 rows, runs them through QwenVQACollator, and asserts
the batch is well-formed (image tokens present, labels partially supervised).
Run this before burning GPU time on a full train:

  python scripts/smoke_test.py
  python scripts/smoke_test.py --n 8 --with-model   # also load model + 1 forward pass
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import QwenVQACollator, get_answer, load_vqa_subset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4, help="number of rows to probe")
    p.add_argument("--model_name", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--with-model", action="store_true",
                   help="also load the model and run one forward pass (needs GPU)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[1/4] Loading {args.n} VQAv2 rows ...")
    # Pull a tiny slice; load_vqa_subset shuffles+splits, so ask for a small total.
    train_ds, eval_ds = load_vqa_subset(sample_size=max(args.n * 5, 20), seed=42)
    rows = [train_ds[i] for i in range(args.n)]
    print(f"      columns: {list(train_ds.features.keys())}")
    for i, r in enumerate(rows):
        img = r["image"]
        print(f"      row {i}: Q={r['question']!r}  A={get_answer(r)!r}  "
              f"img={img.size if hasattr(img, 'size') else type(img)}")

    print("[2/4] Loading processor ...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    print("[3/4] Running collator ...")
    collator = QwenVQACollator(processor)
    batch = collator(rows)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"      {k}: {tuple(v.shape)} {v.dtype}")

    # Assertions: labels must be partially supervised (some -100, some real).
    labels = batch["labels"]
    n_masked = (labels == -100).sum().item()
    n_supervised = (labels != -100).sum().item()
    print(f"      labels: {n_masked} masked (-100), {n_supervised} supervised")
    assert n_supervised > 0, "FAIL: no supervised tokens — collator masked everything"
    assert n_masked > 0, "FAIL: nothing masked — prompt/pad masking not applied"
    assert "pixel_values" in batch or "image_grid_thw" in batch, \
        "FAIL: no image tensors in batch — image not encoded"
    print("      OK: batch well-formed, answer-only supervision confirmed")

    if args.with_model:
        print("[4/4] Loading model + one forward pass ...")
        from src.model import apply_lora, load_model_and_processor
        model, _ = load_model_and_processor(args.model_name)
        model = apply_lora(model, r=8, lora_alpha=16)
        device = next(model.parameters()).device
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        model.train()
        out = model(**batch)
        print(f"      loss = {out.loss.item():.4f}")
        assert torch.isfinite(out.loss), "FAIL: loss is NaN/Inf"
        print("      OK: forward pass produced finite loss")
    else:
        print("[4/4] skipped model forward (pass --with-model to enable)")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
