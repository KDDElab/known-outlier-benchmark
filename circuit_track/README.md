# Zero-noise circuit pilot

This pilot evaluates known-anomaly, output-specific input-subspace localization
on 86 queries drawn from 16 modified-table pairs in the Harvard digital-circuit
archive. The prompt contains the target row, 16 normal reference rows, observed
circuit outputs, and the affected focus output. Only input fields are eligible
prediction features. Ground-truth explanations and gate-change metadata are
kept in `ground_truth_zero_noise_pilot.jsonl` and are never passed to inference.

`prepare_zero_noise_pilot.py` deterministically rebuilds the inputs and labels
from the archived source. The inputs also retain structured copies of rows for
baseline generation; the rendered prompts and their hashes are unchanged.
`audit_zero_noise_alignment.py` checks modified/original/gold row alignment.

## Reproduction

From the repository root:

```sh
python adbench_llm_reference_v1/circuit_track/prepare_zero_noise_pilot.py
python adbench_llm_reference_v1/circuit_track/audit_zero_noise_alignment.py
python adbench_llm_reference_v1/circuit_track/run_circuit_gower_baselines.py
CUDA_VISIBLE_DEVICES=0 python adbench_llm_reference_v1/circuit_track/run_zero_noise_pilot.py --model qwen35_9b
CUDA_VISIBLE_DEVICES=0,1 python adbench_llm_reference_v1/circuit_track/run_zero_noise_pilot.py --model deepseek_v2_lite_chat
CUDA_VISIBLE_DEVICES=2 python adbench_llm_reference_v1/circuit_track/run_zero_noise_pilot.py --model llama31_8b_instruct
python adbench_llm_reference_v1/circuit_track/evaluate_zero_noise_pilot.py --bootstrap 2000 --seed 20261105
```

LLM decoding is greedy with at most 512 new tokens and a 16,384-token input
cap; prompts are not truncated. Raw responses, strict parsed predictions,
per-query timing, GPU assignments, checkpoint file hashes, and runtime versions
are append-only under `annotations_*.jsonl` and `model_runs.jsonl`. Invalid,
missing, or abstained predictions count as misses. Evaluation intervals
resample the 16 source table-pair clusters, retaining all focus-output queries
from a sampled pair.

The supplementary pilot was run with Transformers 5.22.1 and PyTorch
2.14.1+cu126, later than the synthetic v2 experiment runtime (Transformers
5.15.0, PyTorch 2.13.0+cu130). Circuit results are therefore reported
separately and are not pooled with the synthetic benchmark.

The baseline script produces binary Gower-kNN, Gower-LOF, and
focus-output-conditional surprisal rankings over all nonempty input subsets
(up to 14 fields), so the search space includes every gold-set size without
using labels to set the order. The two Gower methods use only input rows; the conditional
surprisal method uses the focus output and input values; a new joint-output
surprisal baseline uses the full observed output vector and offers a matched
all-output comparison to the LLMs. The two Gower controls remain input-only.

To measure the LLM sensitivity to the additional output context, create a
paired prompt condition that reveals only the known focus output (all input
features and all 86 target/reference pairs stay fixed):

```sh
python circuit_track/make_focus_output_only_inputs.py
CUDA_VISIBLE_DEVICES=0 python circuit_track/run_zero_noise_pilot.py --model qwen35_9b --inputs-path circuit_track/matched_information/inputs_focus_output_only.jsonl --annotations-path circuit_track/matched_information/annotations_qwen35_9b.jsonl --run-log-path circuit_track/matched_information/model_runs.jsonl --protocol-id harvard_circuit_focus_output_only_v1
CUDA_VISIBLE_DEVICES=0,1 python circuit_track/run_zero_noise_pilot.py --model deepseek_v2_lite_chat --inputs-path circuit_track/matched_information/inputs_focus_output_only.jsonl --annotations-path circuit_track/matched_information/annotations_deepseek_v2_lite_chat.jsonl --run-log-path circuit_track/matched_information/model_runs.jsonl --protocol-id harvard_circuit_focus_output_only_v1
CUDA_VISIBLE_DEVICES=2 python circuit_track/run_zero_noise_pilot.py --model llama31_8b_instruct --inputs-path circuit_track/matched_information/inputs_focus_output_only.jsonl --annotations-path circuit_track/matched_information/annotations_llama31_8b_instruct.jsonl --run-log-path circuit_track/matched_information/model_runs.jsonl --protocol-id harvard_circuit_focus_output_only_v1
python circuit_track/evaluate_zero_noise_pilot.py --bootstrap 2000 --permutations 20000 --focus-only-dir circuit_track/matched_information --outdir circuit_track/matched_information/results
```

The all-output and focus-only variants are paired by query and share the same
Transformers/PyTorch runtime. Invalid and missing outputs remain misses. The
bootstrap and sign-flip unit is the 16 source table-pair clusters. This is a
controlled context ablation for the narrow binary circuit task, not a general
claim about output context or real tables.

`evaluate_zero_noise_pilot.py` writes per-query results and table-pair-cluster
summaries to `results_zero_noise_pilot/`. The circuit family is a narrow
binary-logic, single-gate-negation test and is not evidence of broad real-table
generalization.
