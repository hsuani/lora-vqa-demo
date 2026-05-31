#!/usr/bin/env bash
# Morning-after: start VM (if stopped), read the Day1 baseline result.
#   bash scripts/check_results.sh
ZONE=us-central1-a
VM=lora-l4

status=$(gcloud compute instances describe "$VM" --zone="$ZONE" --format="value(status)" 2>/dev/null)
echo "VM status: $status"

if [ "$status" = "TERMINATED" ]; then
  echo "VM is stopped (training finished and auto-powered-off). Starting it to read results..."
  gcloud compute instances start "$VM" --zone="$ZONE" >/dev/null 2>&1
  # wait for sshd
  for i in $(seq 1 18); do
    gcloud compute ssh "$VM" --zone="$ZONE" -- -T "echo OK" 2>/dev/null | grep -q OK && break
    sleep 10
  done
fi

echo "=== sweep.log markers (or train.log tail) ==="
gcloud compute ssh "$VM" --zone="$ZONE" -- -T \
  "if [ -f ~/sweep.log ]; then grep -E 'SWEEP_START|SWEEP_END|^>>> ' ~/sweep.log; else tail -n 30 ~/train.log 2>/dev/null; fi" 2>/dev/null

echo ""
echo "=== results/sweep_results.csv ==="
gcloud compute ssh "$VM" --zone="$ZONE" -- -T \
  "cat ~/lora-vqa-demo/results/sweep_results.csv 2>/dev/null || echo '(no results yet)'" 2>/dev/null

echo ""
echo "If EXIT=0 above and the CSV has a baseline row: Day1 done."
echo "Then STOP the VM to avoid billing:  gcloud compute instances stop $VM --zone=$ZONE"
