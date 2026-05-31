#!/usr/bin/env bash
# Day2 FULL rank/alpha sweep at sample_size=5000 (4000 train / 1000 eval).
# ~21 min/run x 8 = ~2.8 h on one L4. Results -> results/sweep_results_full.csv
# (keeps the fast 2000-sample results/sweep_results.csv intact).
# Launch detached:
#   nohup bash scripts/run_sweep_full.sh >/dev/null 2>&1 &
set -o pipefail
cd "$(dirname "$0")/.." || exit 1

CFG="configs/lora_baseline.yaml"
SAMPLE_SIZE="${SAMPLE_SIZE:-5000}"
RESULTS="results/sweep_results_full.csv"
LOG="$HOME/sweep_full.log"

rm -f "$RESULTS"
: > "$LOG"
echo "SWEEP_START $(date -u +%FT%TZ)  sample_size=$SAMPLE_SIZE" >> "$LOG"

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
  RESULTS_PATH="$RESULTS" stdbuf -oL -eL python3 -u scripts/train.py \
    --config "$CFG" --r "$r" --lora_alpha "$alpha" \
    --run_name "$name" --sample_size "$SAMPLE_SIZE" >> "$LOG" 2>&1
  echo ">>> $(date -u +%T) DONE $name (exit=$?)" >> "$LOG"
done

echo "SWEEP_END $(date -u +%FT%TZ)" >> "$LOG"
sync
sleep 3
sudo poweroff
