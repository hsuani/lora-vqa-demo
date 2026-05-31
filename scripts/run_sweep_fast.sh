#!/usr/bin/env bash
# Day2 fast rank/alpha sweep — sized to finish 8 runs in < 2 hours on one L4.
# Launch detached:
#   nohup bash scripts/run_sweep_fast.sh >/dev/null 2>&1 &
# Streams ~/sweep.log; powers the VM off when the whole sweep finishes
# (success OR failure) after flushing, so it never bills idle overnight.
set -o pipefail
cd "$(dirname "$0")/.." || exit 1

CFG="configs/lora_baseline.yaml"
SAMPLE_SIZE="${SAMPLE_SIZE:-2000}"     # 1600 train / 400 eval per run
LOG="$HOME/sweep.log"

# Fresh results file so the sweep table is apples-to-apples (uniform size).
# The full-size baseline row stays in results/sweep_results_baseline.csv.
if [ -f results/sweep_results.csv ]; then
  cp -f results/sweep_results.csv results/sweep_results_baseline.csv
  rm -f results/sweep_results.csv
fi

: > "$LOG"
echo "SWEEP_START $(date -u +%FT%TZ)  sample_size=$SAMPLE_SIZE" >> "$LOG"

# r alpha run_name
RUNS=(
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
  echo ">>> $(date -u +%T) START $name (r=$r alpha=$alpha)" >> "$LOG"
  stdbuf -oL -eL python3 -u scripts/train.py \
    --config "$CFG" --r "$r" --lora_alpha "$alpha" \
    --run_name "$name" --sample_size "$SAMPLE_SIZE" >> "$LOG" 2>&1
  echo ">>> $(date -u +%T) DONE $name (exit=$?)" >> "$LOG"
done

echo "SWEEP_END $(date -u +%FT%TZ)" >> "$LOG"
sync
sleep 3
sudo poweroff
