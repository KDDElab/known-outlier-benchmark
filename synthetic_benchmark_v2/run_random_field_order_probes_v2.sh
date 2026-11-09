#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROBE_DIR="$SCRIPT_DIR/runs/random_field_order_probe"
RUNNER="$SCRIPT_DIR/annotate_v2.py"

for seed in 20261121 20261122 20261123 20261124; do
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$RUNNER" \
    --model qwen35_9b \
    --split test \
    --inputs-path "$PROBE_DIR/inputs_${seed}.jsonl" \
    --annotations-path "$PROBE_DIR/annotations_qwen35_9b_${seed}.jsonl" \
    --run-log-path "$PROBE_DIR/model_runs_${seed}.jsonl" \
    --protocol-id "known_anomaly_subspace_v2_random_field_order_${seed}" \
    --resume
done

python "$SCRIPT_DIR/analyze_random_field_order_probes_v2.py"
