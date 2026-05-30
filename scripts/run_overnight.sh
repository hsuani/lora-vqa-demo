#!/usr/bin/env bash
# Overnight Day1 baseline runner. Launch detached:
#   nohup bash scripts/run_overnight.sh >/dev/null 2>&1 &
# Writes ~/train.log live; powers the VM off when done (success OR failure)
# after flushing the log to disk, so the result is readable on next boot.
set -o pipefail
cd "$(dirname "$0")/.." || exit 1

LOG="$HOME/train.log"
: > "$LOG"                                   # truncate
echo "START $(date -u +%FT%TZ)" >> "$LOG"

stdbuf -oL -eL python3 -u scripts/train.py \
  --config configs/lora_baseline.yaml >> "$LOG" 2>&1
code=$?

echo "EXIT=$code $(date -u +%FT%TZ)" >> "$LOG"
sync
sleep 3
sudo poweroff
