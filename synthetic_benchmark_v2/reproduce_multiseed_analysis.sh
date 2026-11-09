#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Requires the three frozen model annotation logs and all baseline prediction files.
python validate_multiseed_baselines_v2.py
python validate_multiseed_runs_v2.py
python analyze_multiseed_results_v2.py --outdir runs/multiseed_analysis --bootstrap 2000
python analyze_observability_compatibility_v2.py --bootstrap 2000 --seed 20261118
python analyze_provenance_targets_v2.py --bootstrap 2000 --seed 20261115
if [[ -f runs/multiseed/seed_20261106_rt522/annotations_phi4_mini.jsonl ]]; then
  python evaluate_extra_model_families_v2.py --models phi4_mini \
    --outdir runs/extra_model_family_phi_seed06
fi
python analyze_m2oe_matched3_v2.py
python test_multiseed_pairs_v2.py \
  --out runs/multiseed_analysis/paired_tests.json \
  --permutations 20000 --bootstrap 2000 --seed 20261106
python run_multiseed_confidence_audits_v2.py --bootstrap 2000 --seed 20261106
python analyze_multiseed_subgroups_v2.py \
  --bootstrap 2000 --seed 20261106 \
  --outdir runs/multiseed_analysis/subgroups
python audit_multiseed_reference_identifiability_v2.py \
  --max-order 3 --bootstrap 2000 --seed 20261106 \
  --outdir runs/reference_sample_audit_multiseed
python plot_multiseed_baseline_comparison_v2.py

if [[ -f runs/reference_size_probe/annotations_qwen35_9b_reference_size_8.jsonl && \
      -f runs/reference_size_probe/annotations_qwen35_9b_reference_size_16.jsonl && \
      -f runs/prompt_sensitivity_combined_summary.json ]]; then
  python analyze_reference_size_probe_v2.py --bootstrap 2000 --permutations 20000 --seed 20261114
  python combine_robustness_holm_v2.py
else
  echo "Nested reference-size or prompt-sensitivity output is missing; joint 20-test Holm report skipped." >&2
fi
