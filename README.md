# LoRA Toy Demo — Qwen2-VL-2B + VQAv2

Rank/alpha sensitivity study. Fine-tune Qwen2-VL-2B on VQAv2 validation subset (5k rows) with varying LoRA configs, measure exact-match accuracy.

## Setup (GCP A100)

```bash
pip install -r requirements.txt
```

## Day 1 — Baseline (r=8, alpha=16)

```bash
python scripts/train.py --config configs/lora_baseline.yaml
```

## Day 2 — Sweep (r ∈ {4,8,16,32} × alpha ∈ {r, 2r})

```bash
chmod +x scripts/run_sweep.sh
./scripts/run_sweep.sh
python scripts/plot_results.py
```

## Results

<!-- Fill after sweep -->

| run | r | alpha | trainable params | eval acc |
|-----|---|-------|-----------------|----------|
| r4_a4  | 4  | 4  | - | - |
| r4_a8  | 4  | 8  | - | - |
| r8_a8  | 8  | 8  | - | - |
| r8_a16 | 8  | 16 | - | - |
| r16_a16| 16 | 16 | - | - |
| r16_a32| 16 | 32 | - | - |
| r32_a32| 32 | 32 | - | - |
| r32_a64| 32 | 64 | - | - |

## Structure

```
lora-vqa-demo/
├── configs/
│   ├── lora_baseline.yaml   # Day 1 config
│   └── sweep.yaml           # Day 2 sweep matrix
├── src/
│   ├── dataset.py           # VQAv2 loader + prompt builder
│   ├── model.py             # Qwen2-VL load + LoRA wrap
│   └── metrics.py           # VQA exact-match + eval loop
├── scripts/
│   ├── train.py             # Single run entry point
│   ├── run_sweep.sh         # Day 2 sweep driver
│   └── plot_results.py      # Loss curves + accuracy plots
├── notebooks/
│   └── explore_vqa.ipynb    # Dataset exploration
├── results/
│   ├── sweep_results.csv    # Auto-appended per run
│   └── figures/             # PNG plots
└── requirements.txt
```
