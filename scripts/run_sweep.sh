#!/usr/bin/env bash
# Day 2: rank/alpha sweep. Run from repo root.
set -e

CFG="configs/lora_baseline.yaml"

declare -a RUNS=(
  "4 4   r4_a4"
  "4 8   r4_a8"
  "8 8   r8_a8"
  "8 16  r8_a16"
  "16 16 r16_a16"
  "16 32 r16_a32"
  "32 32 r32_a32"
  "32 64 r32_a64"
)

for entry in "${RUNS[@]}"; do
  read -r r alpha name <<< "$entry"
  echo ">>> Starting run: $name (r=$r, alpha=$alpha)"
  python scripts/train.py --config "$CFG" --r "$r" --lora_alpha "$alpha" --run_name "$name"
  echo ">>> Done: $name"
done

echo ""
echo "Sweep complete. Results at results/sweep_results.csv"
