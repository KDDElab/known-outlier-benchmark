# Known-Anomaly Subspace Benchmark v2

Version 2 is a controlled benchmark for a model that receives a table schema, a
known anomalous target row, and normal reference rows, then returns the likely
anomaly mechanism fields and ranked explanatory feature subspaces. It is a
separate synthetic track; it does not replace or mix with the v1 or real-data
annotations.

**Current results:** the November 6, 2026 three-seed analysis supersedes the
earlier one-seed tables retained later in this README. The current main suite
contains three LLMs, six ranked statistical/adaptation baselines, and
matched-budget M²OE choice-set results. See “Three-seed expanded-baseline
update” for the authoritative estimates and the reproducible analysis command.

## Design

The fixed-seed generator creates 480 cases from 40 procedural rule templates.
There are eight task families with five templates each: numeric marginal,
categorical rare value, pairwise interaction, conditional/group pattern,
multiple valid alternatives, modified-field versus mechanism separation,
relational violation, and higher-order interaction. The templates cover
numeric, binary, nominal categorical, ordinal, date-like, and identifier-like
fields. Numeric and ordinal fields use small discrete domains so the generator
can enumerate the true support and sufficient subsets exactly.

Every template contributes 12 cases: four each at 6, 12, and 24 columns; four
each with 8, 16, and 32 normal reference rows; and two each under six nuisance
profiles. Those profiles are independent noise, a 64-level categorical
distractor, a noisy proxy, a preserved deterministic proxy, a broken
deterministic proxy, and an independent field with 15% missingness. Opaque IDs
and neutral field names are balanced within each template. Rows and schema
columns are serialized in randomized feature order.

Splits are by **template group**, not by row: within each task family, one
template is train, one is dev, and three are test. This yields 96/96/288 cases
and 8/8/24 distinct template groups. No `template_id` crosses splits. The
hidden case manifest and template catalog disclose the split groups and rules;
do not expose them in test prompts or few-shot examples.

## Main input and independent labels

`inputs.jsonl` contains the exact system/user prompts, schema, known target,
normal references, split, and prompt hash. It contains **no paired pre-injection
row** and does not ask the model to guess which field was modified. A direct
modification label is not identifiable from an unpaired target and reference
sample in general. `modified_feature_ids` and the hidden baseline are retained
as provenance labels, not primary explanation targets. A separate
hidden-provenance audit is reported in the latest results update; it does not
make the hidden baseline available to the model or convert direct-edit
recovery into primary-task accuracy.

The hidden `ground_truth.jsonl` has four separate label concepts:

- `modified_feature_ids`: fields directly changed between the hidden baseline
  and target. `value_different_from_baseline_feature_ids` additionally captures
  value changes propagated to a proxy.
- `mechanism_feature_ids`: fields on which the procedural anomaly predicate
  functionally depends over the full domain. This is a generator mechanism
  label, not a real-world causal claim.
- `rule_sufficient_subspaces` and `rule_minimal_subspaces`: subsets whose
  target-matching full-domain completions all satisfy the rule, and the
  inclusion-minimal such subsets.
- `reference_support_sufficient_subspaces` and
  `reference_support_sufficient_minimal_subspaces`: subsets whose target
  projection has no completion in the complete clean population support, and
  the inclusion-minimal such subsets. These are the primary descriptive
  explanation labels. Multiple alternatives are preserved.

Labels are computed from the known generator and exhaustive finite-domain
enumeration, independently of model outputs and the sampled normal rows. The
support labels distinguish a useful descriptive proxy from a rule mechanism.
They do not claim that a sampled reference set exhaustively represents the
population. Keep `ground_truth.jsonl`, `case_manifest.jsonl`, and
`template_catalog.json` out of model inputs.

## Files

- `generate_benchmark.py`: deterministic generator; seed `20261106`.
- `inputs.jsonl`: main-task input prompts and case factors.
- `ground_truth.jsonl`: hidden procedural, mechanism, intervention, and
  explanation labels.
- `case_manifest.jsonl`: split and template group IDs; hidden from models.
- `template_catalog.json`: procedural template definitions; hidden from test
  models.
- `manifest.json`: counts, factor definitions, split assignments, generator
  source/runtime metadata, and SHA-256 hashes for generated artifacts.
- `evaluate_benchmark.py`: scorer for one JSONL prediction file.
- `annotate_v2.py`: frozen-protocol local model inference runner.
- `export_predictions_v2.py`: protocol-filtered export from append-only logs.
- `make_baselines_v2.py` and `summarize_baselines_v2.py`: simple baselines and
  their template-cluster summaries.
- `analyze_model_runs_v2.py`: template-cluster bootstrap intervals, paired
  model/baseline comparisons, sign-flip tests, and Holm correction.
- `compare_gower_baselines_v2.py`: primary paired model-versus-baseline tests
  and secondary rank-sensitive MRR/Recall@4 summaries.
- `run_gower_subspace_baseline.py`: mixed-type Gower-kNN and Gower-LOF
  subspace ranker predictions.
- `run_sod_cop_adaptations_v2.py`, `run_m2oe_baseline_v2.py`, and
  `run_rlax_adaptation_v2.py`: task-specific SOD/COP adaptations, official
  M²OE choice outputs, and a paper-guided RLAX adaptation.
- `analyze_multiseed_results_v2.py`: common-runtime three-seed summaries for
  the primary models and baseline suite.
- `analyze_m2oe_matched3_v2.py`: matched three-candidate, order-invariant
  evaluation of M²OE choices against LLM candidate sets.
- `test_multiseed_pairs_v2.py`: template-cluster paired comparisons and Holm
  adjustment over the 78 planned contrasts.
- `run_multiseed_confidence_audits_v2.py`: per-seed confidence and literal
  evidence-token audits; these do not establish semantic faithfulness.
- `audit_reference_sample_identifiability.py`: finite-reference support
  ambiguity diagnostic against population-level gold labels.

The expected prediction record has one line per case:

```json
{"case_id":"V2_0001","task_label":"pairwise_interaction","mechanism_feature_ids":["F003","F011"],"explanations":[{"features":["F003","F011"],"evidence":"...","confidence":0.8}],"abstain":false}
```

The evaluator checks required output structure and feature IDs. It reports task
label accuracy and mechanism feature precision/recall/F1, then scores ranked
subspaces against both the primary reference-support labels and the separate
rule labels. Explanation measures include sufficiency, exact minimal-subspace
Hit@k, recall over multiple gold alternatives, MRR, best Jaccard, and predicted
subspace size. It does not score direct modified fields on the unpaired main
task. Sufficiency is scored monotonically: a candidate containing any known
minimal sufficient subspace counts as sufficient, while minimal exact hit
requires an exact set match. This distinction keeps the all-fields control
sufficient but not compact/minimal. `unknown` task labels and baselines that do
not predict mechanism fields are marked unscored for those metrics.

For model outputs, the scorer extracts the first JSON object so recoverable
predictions can be evaluated even when a model adds a preamble, code fence, or
trailing text. Such output does not count as strict protocol compliance. The
run analyzer separately reports strict JSON-only output with exact keys and
the 20-word evidence limit. Missing or unparseable responses remain in the
288-query denominator and count as misses for ranked-subspace and
all-fields-mechanism recovery; feature F1 on parseable outputs is reported
alongside the invalid-as-empty version and output coverage.

## Baselines and completed direct-prompt results

`make_baselines_v2.py` writes all-fields, top-four univariate, 100 random
subspace replicates, and two oracle upper bounds. The random subset size is
sampled from training-split minimal support sizes; no test labels are used to
create random predictions. The univariate baseline ranks numeric/ordinal/binary
columns by the maximum of robust median deviation and empirical rarity, and
other types by empirical rarity. Oracle predictions use hidden test labels and
are upper bounds only.

`run_gower_subspace_baseline.py` adds two same-input mixed-type rankers. It
enumerates every feature subset of order 1--3 and returns the four highest
scoring sets. Gower-kNN ranks target-versus-reference kNN distance contrast;
Gower-LOF ranks novelty LOF scores computed from Gower distances and the normal
reference rows. Both methods use schema type and values from the prompt only.
The script writes `predictions_gower_knn_subspace.jsonl`,
`predictions_gower_lof_subspace.jsonl`, and
`gower_subspace_baselines_manifest.json`. These are benchmark-specific
adaptations and are not canonical SOD/COP implementations.

`evaluate_benchmark.py` scores each baseline on the same 288 test cases.
`compare_gower_baselines_v2.py` produces 2,000-template-bootstrap intervals
and exploratory paired sign-flip comparisons, with Holm correction over 36
model-versus-baseline contrasts. The 100 random predictions are averaged per
case before template resampling. `plot_baseline_comparison_v2.py` writes the
corresponding figure to `figures/v2_baseline_comparison.pdf`.

`analyze_rule_metrics_v2.py` separately summarizes each method against the
generator-rule sufficient-subspace labels, using 2,000 bootstrap resamples of
the 24 held-out templates. It averages the 100 random predictions per case
before resampling and writes `runs/rule_metrics_template_bootstrap.json`.
These values are deliberately separate from clean-support explanation scores
and mechanism-field F1.

| Method | Sufficiency@1 | Minimal exact@1 | Hit@4 | Best Jaccard |
|---|---:|---:|---:|---:|
| Qwen3.5-9B | 0.714 [0.617,0.798] | 0.425 [0.342,0.505] | 0.451 [0.370,0.529] | 0.659 [0.605,0.711] | 0.684 [0.571,0.780] | 0.403 [0.309,0.491] | 0.426 [0.336,0.512] |
| DeepSeek-V2-Lite-Chat | 0.429 [0.384,0.470] | 0.125 [0.078,0.176] | 0.145 [0.095,0.198] | 0.315 [0.271,0.359] | 0.407 [0.362,0.447] | 0.115 [0.071,0.163] | 0.134 [0.086,0.186] |
| Llama-3.1-8B-Instruct | 0.394 [0.327,0.457] | 0.152 [0.099,0.210] | 0.177 [0.120,0.239] | 0.383 [0.339,0.427] | 0.380 [0.313,0.444] | 0.140 [0.091,0.193] | 0.163 [0.110,0.221] |
| Gower-LOF subspace | 0.483 | 0.455 | 0.569 | 0.719 |
| Gower-kNN subspace | 0.549 | 0.354 | 0.438 | 0.630 |
| Top-four univariate | 0.155 [0.086,0.231] | 0.155 [0.086,0.231] | 0.398 [0.257,0.534] | 0.547 [0.453,0.639] | 0.140 [0.078,0.210] | 0.140 [0.078,0.210] | 0.392 [0.252,0.527] |
| Random (100 seeds) | 0.124 | 0.044 | 0.144 | 0.362 |
| All features | 1.000 [1.000,1.000] | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 0.155 [0.135,0.176] | 1.000 [1.000,1.000] | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] |

These scores expose metric-dependent rankings: Qwen has the highest top-ranked
sufficiency, whereas Gower-LOF has the highest Hit@4 and Jaccard point
estimates. The oracle controls are omitted from this method comparison because
they use hidden test labels. All intervals condition on one generator seed and
24 held-out templates.

The separate generator-rule track yields the following rule-minimal exact@1 /
Hit@4 point estimates: Qwen 0.406/0.417, DeepSeek 0.094/0.115, Llama 0.167/0.194,
Gower-LOF 0.434/0.531, Gower-kNN 0.330/0.413, top-four univariate 0.139/0.385,
random 0.037/0.123, and all-fields 0.000/0.000. Template-bootstrap intervals are
in `runs/rule_metrics_template_bootstrap.json`. This track evaluates sufficient
subsets of the procedural predicate, not direct generator edits or detector
score attribution.

Three local LLMs completed the same 288 test queries under frozen prompt
protocol `known_anomaly_subspace_benchmark_v2_compact_json_v1` (greedy
decoding, input cap 16,384 tokens without truncation, output cap 512 tokens).
The table reports 95% template-cluster bootstrap intervals (24 test templates;
interaction recovery uses its 15 eligible templates). Invalid or missing
outputs count as misses for subspace and interaction recovery. Mechanism F1
uses invalid-as-empty scoring; conditional F1 on parseable records is also in
the analysis JSON.

| Model | Parseable schema | Strict JSON contract | Support sufficiency@1 | Minimal exact@1 | Minimal Hit@4 | Best Jaccard | Mechanism F1, invalid as empty | Interaction all-fields exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B | 0.714 [0.617,0.798] | 0.425 [0.342,0.505] | 0.451 [0.370,0.529] | 0.659 [0.605,0.711] | 0.684 [0.571,0.780] | 0.403 [0.309,0.491] | 0.426 [0.336,0.512] |
| DeepSeek-V2-Lite-Chat | 0.429 [0.384,0.470] | 0.125 [0.078,0.176] | 0.145 [0.095,0.198] | 0.315 [0.271,0.359] | 0.407 [0.362,0.447] | 0.115 [0.071,0.163] | 0.134 [0.086,0.186] |
| Llama-3.1-8B-Instruct | 0.394 [0.327,0.457] | 0.152 [0.099,0.210] | 0.177 [0.120,0.239] | 0.383 [0.339,0.427] | 0.380 [0.313,0.444] | 0.140 [0.091,0.193] | 0.163 [0.110,0.221] |

These fixed-benchmark results show task-specific differences. Qwen has higher
support-sufficiency than the Gower rankers, while Gower-LOF returns more
minimal candidates at four and has higher best-Jaccard point estimates. Llama
recovers all mechanism fields on more interaction cases than the other LLMs;
mechanism F1 and localization quality are not interchangeable. Broad
task-family accuracy is low for all three (Qwen 0.236, DeepSeek 0.080, Llama
0.122 when invalid outputs count as wrong). Per-family values are exploratory
because only three held-out templates represent each family.

`analyze_test_slices_v2.py` exports descriptive test-split tables by task family,
feature width, reference count, and nuisance profile. It also reports task-label
accuracy with invalid outputs counted wrong and mechanism micro-F1 with invalid
outputs counted as empty. These are marginal summaries without family-level
confidence intervals: each family has only three held-out templates, and the
factor slices contain different cases rather than paired interventions.

The factorized order probe selects four cases per held-out template (96 total;
seed 20261106), balancing width, reference count, and nuisance factors. The
field-order condition reverses schema and within-row field order but preserves
reference-row order; the row-order condition reverses only the reference-row
list. `make_factorized_order_probe_v2.py` creates the two paired conditions.
Both use the same Qwen checkpoint, greedy decoding, token budget, and
Transformers 5.22.1 / PyTorch 2.14.1+cu126 runtime. Invalid outputs and the
same over-cap input (V2_0236) remain misses. `analyze_factorized_order_v2.py`
uses paired template-cluster bootstrap intervals and sign-flip tests with Holm
adjustment over four contrasts. Field reversal changes Hit@4 from 0.479 to 0.125
(difference -0.354; 95% interval [-0.448, -0.260], adjusted p=0.00020) and best
Jaccard from 0.662 to 0.316 (difference -0.346; interval [-0.414, -0.267], adjusted
p=0.00020); top-one set stability is 0.336 among 95 valid pairs. Reference-row
reversal changes Hit@4 by +0.010 (interval [-0.073, 0.094], adjusted p=1.0) and
best Jaccard by +0.010 (interval [-0.040, 0.059], adjusted p=1.0); stability is
0.732. This is evidence about two isolated interventions for one checkpoint,
not broad prompt robustness. Run IDs, software, invalid case IDs, and artifact
hashes are preserved in the JSON summary.

Planned model-pair comparisons use paired template-level sign-flip tests with
Holm adjustment across 12 contrasts. Qwen exceeds DeepSeek on minimal exact@1
(difference 0.337, 95% CI [0.191, 0.483], adjusted p=0.0032), Hit@4 (0.330,
[0.177, 0.486], p=0.0035), and best Jaccard (0.358, [0.243, 0.476], p=0.0006).
Qwen also exceeds Llama on those localization measures (adjusted p=0.0318,
0.0318, and 0.0075, respectively), while Llama exceeds Qwen on interaction
all-fields recovery (difference for Qwen minus Llama -0.222, CI [-0.294, -0.150],
adjusted p=0.0032). These are conclusions about this single generator seed,
held-out template set, checkpoint set, and direct prompt only.

Model-versus-baseline comparisons are post hoc and not preregistered. Under
the shared 36-test Holm family, Qwen's minimal exact@1 advantage over
top-four univariate is 0.285 (95% paired template-bootstrap interval [0.184,
0.389], adjusted p=0.0018); its Hit@4 difference is 0.059 ([-0.045, 0.156],
adjusted p=1.0). Against Gower-LOF, Qwen has a 0.198 higher sufficiency@1
difference ([0.122, 0.281], adjusted p=0.0018). Gower-LOF's Hit@4 and Jaccard
advantages over Qwen have adjusted p-values 0.0594 and 0.0745, respectively.
The 36-test family is exactly three LLMs by Gower-kNN, Gower-LOF, and
top-four univariate by four primary metrics (sufficiency@1, minimal exact@1,
Hit@4, best Jaccard). Random and all-feature controls are descriptive and are
not included in this family. The full 36 contrasts are in
`runs/baseline_comparison_gower.json`.

The evaluator also reports mean reciprocal rank (MRR; zero for a miss) and
Recall@4 over multiple valid gold subspaces. We now include them as secondary,
descriptive ranking summaries with template-cluster intervals; no extra
hypothesis tests are run. Qwen has MRR 0.443 [0.332,0.554] and Recall@4 0.414
[0.309,0.518]; Gower-LOF has 0.503 [0.361,0.638] and 0.542 [0.400,0.678]. Full
summaries for the LLMs, Gower methods, univariate, random, and all-feature
controls are in `runs/baseline_comparison_gower.json` under
`secondary_ranking_metric_summaries`.

### Finite-reference support ambiguity

The primary labels describe complete clean population support, whereas the
model sees only 8, 16, or 32 normal reference rows. We ran
`audit_reference_sample_identifiability.py`, which enumerates all feature
subsets of order 1--3 and marks those whose target projection is absent from
the observed references. An observed subset is sample-minimal if none of its
proper subsets in that candidate universe is sample-absent. This is a
diagnostic only: it does not modify the population-level labels or score model
predictions.

On the 288 held-out cases, the empirical sample-minimal set exactly matched the
population gold set in 0 cases; every case had at least one sample-minimal
candidate absent from the population gold. There were 63.16 such extra
candidates per case on average (95% template-cluster interval [58.63,67.81]).
The mean proportion of population-gold minimal sets retained as sample-minimal
was 0.818 [0.720,0.907]. These results show that a direct empirical-absence rule
is insufficient; the benchmark measures inference of population structure
from limited references. Candidate counts depend strongly on feature width and
the order-three search universe; subgroup results are descriptive.

Parseable output rates are not equivalent to strict compliance. Strict
JSON-only/exact-schema/evidence-length compliance is 285/288 for Qwen,
115/288 for DeepSeek, and 186/288 for Llama. Qwen has two inputs above the
16,384-token cap; one further parseable Qwen response exceeds the evidence
word limit. DeepSeek has two over-cap inputs, 53 output-cap hits, 13
responses with more than four candidates, three unsupported task labels, and
one additional JSON syntax failure. Llama reached the output cap on 45
responses (44 were unparseable) and had eight further outputs without a JSON
object.
No response was manually repaired. Inference elapsed time after model loading
was 3,292 s for Qwen, 5,078 s for DeepSeek, and 2,019 s for Llama; GPU
allocation differed (one, two, and one 24-GB cards, respectively), so latency
is not a hardware-controlled comparison. Monetary API cost is not applicable;
GPU energy was not measured.

All raw responses, parser failures, per-case protocol-compliance records,
model/runtime metadata, prompt hashes, and latencies are retained in `runs/`.
The earlier pilot-generation records use different protocol IDs and are
excluded from these results. The complete metrics, confidence intervals,
subgroup summaries, paired contrasts, and analysis-script hashes are in
`runs/model_analysis_known_anomaly_subspace_benchmark_v2_compact_json_v1.json`.

## Reproduction

From the repository root:

```sh
python adbench_llm_reference_v1/synthetic_benchmark_v2/generate_benchmark.py
python adbench_llm_reference_v1/synthetic_benchmark_v2/make_baselines_v2.py --replicates 100
python adbench_llm_reference_v1/synthetic_benchmark_v2/summarize_baselines_v2.py --replicates 100
python adbench_llm_reference_v1/synthetic_benchmark_v2/run_gower_subspace_baseline.py --max-order 3
python adbench_llm_reference_v1/synthetic_benchmark_v2/audit_reference_sample_identifiability.py --max-order 3 --bootstrap 2000 --seed 20261105
python adbench_llm_reference_v1/synthetic_benchmark_v2/evaluate_benchmark.py adbench_llm_reference_v1/synthetic_benchmark_v2/runs/baseline_predictions/predictions_gower_knn_subspace.jsonl
python adbench_llm_reference_v1/synthetic_benchmark_v2/evaluate_benchmark.py adbench_llm_reference_v1/synthetic_benchmark_v2/runs/baseline_predictions/predictions_gower_lof_subspace.jsonl
python adbench_llm_reference_v1/synthetic_benchmark_v2/compare_gower_baselines_v2.py --bootstrap 2000 --permutations 20000
python adbench_llm_reference_v1/synthetic_benchmark_v2/analyze_rule_metrics_v2.py
python adbench_llm_reference_v1/synthetic_benchmark_v2/plot_baseline_comparison_v2.py
python adbench_llm_reference_v1/synthetic_benchmark_v2/annotate_v2.py --model qwen35_9b
python adbench_llm_reference_v1/synthetic_benchmark_v2/annotate_v2.py --model deepseek_v2_lite_chat
python adbench_llm_reference_v1/synthetic_benchmark_v2/annotate_v2.py --model llama31_8b_instruct
python adbench_llm_reference_v1/synthetic_benchmark_v2/evaluate_benchmark.py path/to/predictions.jsonl
python adbench_llm_reference_v1/synthetic_benchmark_v2/analyze_model_runs_v2.py \
  --bootstrap 2000 --permutations 20000
```

Set `CUDA_VISIBLE_DEVICES` for each command to match the recorded model/GPU
allocation and ensure enough memory. Model paths are configured in the parent
pilot's `annotate.py`; the runner records the exact resolved path and weight
file hashes.

After inference, flatten only the frozen protocol rows before scoring:

```sh
python adbench_llm_reference_v1/synthetic_benchmark_v2/export_predictions_v2.py \
  adbench_llm_reference_v1/synthetic_benchmark_v2/runs/annotations_qwen35_9b.jsonl
```

Model version, prompt variant, decoding settings, inference date, raw output,
parse failures, token counts, and latency are recorded. The reported elapsed
inference time excludes model loading; energy consumption has not been
measured.

## Three-seed expanded-baseline update (2026-11-06)

This section supersedes the one-seed scores and paired-test counts in the
historical section above. Three rotated, template-grouped runs contribute 288
test cases each (864 case runs, 38 unique held-out templates). The three LLMs
were evaluated under the common Transformers/PyTorch runtime. Confidence
intervals below use 2,000 common-template bootstrap replicates: the union of
held-out template IDs is resampled once per replicate, with shared multiplicities
applied across seed-specific splits before macro-averaging. They quantify
variation in these selected splits, not over new generator families.

| Method | Support suff.@1 | Exact@1 | Hit@4 | Best Jaccard | Rule suff.@1 | Rule exact@1 | Rule Hit@4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B | 0.714 [0.617,0.798] | 0.425 [0.342,0.505] | 0.451 [0.370,0.529] | 0.659 [0.605,0.711] | 0.684 [0.571,0.780] | 0.403 [0.309,0.491] | 0.426 [0.336,0.512] |
| DeepSeek-V2-Lite-Chat | 0.429 [0.384,0.470] | 0.125 [0.078,0.176] | 0.145 [0.095,0.198] | 0.315 [0.271,0.359] | 0.407 [0.362,0.447] | 0.115 [0.071,0.163] | 0.134 [0.086,0.186] |
| Llama-3.1-8B-Instruct | 0.394 [0.327,0.457] | 0.152 [0.099,0.210] | 0.177 [0.120,0.239] | 0.383 [0.339,0.427] | 0.380 [0.313,0.444] | 0.140 [0.091,0.193] | 0.163 [0.110,0.221] |
| Gower-LOF adaptation | 0.545 [0.421,0.660] | 0.517 [0.396,0.632] | 0.634 [0.524,0.735] | 0.759 [0.693,0.820] | 0.528 [0.397,0.649] | 0.500 [0.371,0.620] | 0.603 [0.484,0.712] |
| Gower-kNN adaptation | 0.608 [0.500,0.707] | 0.400 [0.308,0.499] | 0.495 [0.408,0.586] | 0.684 [0.623,0.744] | 0.583 [0.466,0.691] | 0.383 [0.284,0.489] | 0.476 [0.382,0.572] |
| RLAX paper-guided adaptation | 0.507 [0.376,0.628] | 0.417 [0.295,0.534] | 0.495 [0.385,0.603] | 0.692 [0.620,0.760] | 0.468 [0.349,0.583] | 0.382 [0.269,0.493] | 0.464 [0.356,0.570] |
| Top-four univariate | 0.155 [0.086,0.231] | 0.155 [0.086,0.231] | 0.398 [0.257,0.534] | 0.547 [0.453,0.639] | 0.140 [0.078,0.210] | 0.140 [0.078,0.210] | 0.392 [0.252,0.527] |
| SOD adaptation | 0.512 [0.364,0.668] | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 0.145 [0.110,0.180] | 0.500 [0.350,0.660] | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] |
| COP error-vector adaptation | 0.005 [0.000,0.011] | 0.005 [0.000,0.011] | 0.013 [0.004,0.023] | 0.131 [0.107,0.155] | 0.005 [0.000,0.011] | 0.005 [0.000,0.011] | 0.012 [0.004,0.021] |
| Random size prior (100 reps/seed) | 0.117 [0.087,0.146] | 0.038 [0.030,0.046] | 0.132 [0.106,0.158] | 0.353 [0.348,0.359] | 0.102 [0.074,0.130] | 0.032 [0.025,0.040] | 0.115 [0.090,0.139] |
| All features | 1.000 [1.000,1.000] | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | 0.155 [0.135,0.176] | 1.000 [1.000,1.000] | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] |

All Gower, SOD/COP, and RLAX entries are explicit task adaptations, not claims
of canonical reimplementations. RLAX uses the supplied target in per-query
adaptation and is not a cross-query held-out policy. M²OE is evaluated
separately because it returns unordered choices. Under a three-candidate
budget, support Hit@3 / best Jaccard@3 is 0.450/0.659 for Qwen, 0.142/0.311 for
DeepSeek, 0.175/0.381 for Llama, and 0.000/0.192 for M²OE. Qwen-minus-M²OE paired
effects are +0.439 (95% CI [0.362,0.519]) on Hit@3 and +0.462 ([0.407,0.515]) on
Jaccard@3; both have Holm-adjusted p=0.0039. The M²OE value mask is not scored.

The 78 paired contrasts comprise three LLMs by six ranked baselines on four
localization metrics plus six matched-budget set comparisons. They use
template-level sign flips (20,000 permutations) and Holm family-wise
adjustment. Qwen exceeds Gower-LOF on sufficiency@1 by 0.156 ([0.102,0.211],
adjusted p=0.0039), while Gower-LOF exceeds Qwen on Hit@4 by 0.202 and Jaccard
by 0.115 (both adjusted p=0.0039); their exact@1 contrast is not significant
after correction (adjusted p=0.137). Qwen has no adjusted difference from
Gower-kNN or RLAX on exact@1, Hit@4, and Jaccard; this does not establish
equivalence. Thus the expanded comparison does not support a single overall
winner.

We separately score hidden provenance labels, which are absent from model
inputs. At four candidates, Gower-LOF exactly matches the directly modified
field set in 0.451 [0.377,0.527] of cases, and RLAX has the highest direct-field
union recall (0.787 [0.748,0.826]); Qwen obtains 0.281 [0.223,0.341] exact retrieval
and 0.740 [0.698,0.783] field recall. All-features has full field coverage and
zero exact-set hits. This is a descriptive provenance diagnostic, not primary
explanation accuracy, because the pre-injection row is not shown to methods.
Results and hashes are in `runs/multiseed_analysis/provenance/`.

Across all seed runs, parseable-schema rates are 0.991/0.806/0.817 and strict
JSON-contract rates are 0.988/0.387/0.661 for Qwen/DeepSeek/Llama. Invalid-as-wrong
task-label accuracy is 0.262/0.105/0.111. Self-reported mean confidence
(0.952/0.852/0.804) is far above conditional exact-support recovery; Brier scores
are 0.514/0.626/0.532 and 10-bin ECE 0.523/0.697/0.619. Literal feature/value token
matching is 1.0, but semantic evidence faithfulness remains untested.

A separate Phi-4-mini-instruct single-seed family sensitivity run on 288 test
cases has support sufficiency@1 0.441 [0.330,0.563], exact@1 0.135 [0.087,0.188],
Hit@4 0.160 [0.108,0.215], and best Jaccard 0.376 [0.305,0.445] (271/288
structured responses). It is descriptive and is not included in the main
paired inferential family. The archived run and analysis are in
`runs/extra_model_family_phi_seed06/`.

The Qwen order and reference probes remain single-checkpoint, exploratory
sensitivity tests. In the combined 20-contrast Holm family, reversing field
order reduces Hit@4 by 0.354 (95% CI [-0.448,-0.260]) and best Jaccard by 0.346
([-0.413,-0.271]), adjusted p=0.001 for both; reference-row order and
opaque-ID permutation show no detected effect. Four additional independent
random field-order variants on the same 96 cases reduce mean Hit@4 from 0.479
to 0.195 (variant range 0.146--0.240; paired difference -0.284, 95% CI
[-0.372,-0.203]) and best Jaccard from 0.662 to 0.407 (range
0.368--0.433; difference -0.255, CI [-0.309,-0.200]). The paired bootstrap
averages each case over the four transformations and resamples 24 templates;
384 model outputs are not treated as independent cases. Each permutation has
93--95 structured responses, with token-cap and JSON failures counted as
misses. The extra randomized comparisons are descriptive and are not part of
the original 20-test Holm family. In the nested 96-case 8/16/32-reference
probe, Hit@4 is 0.396/0.531/0.521 and Jaccard 0.608/0.700/0.662. The 16-versus-8
gains do not survive correction; 32-versus-16 changes are near zero. Three
32-reference outputs are invalid, including two input-cap failures. These
probes were not preregistered and cover one checkpoint only. Summaries are in
`runs/robustness_family_holm_summary.json`,
`runs/reference_size_probe/reference_size_summary.json`, and
`runs/random_field_order_probe/random_field_order_summary.json`.

Run the saved-output analyses from this directory with:

```sh
bash synthetic_benchmark_v2/reproduce_multiseed_analysis.sh
```

The supplementary finite-reference compatibility diagnostic is included in that
replay. It reports whether each predicted feature projection is absent from the
visible reference rows; this is a necessary input-consistency check under the
clean-reference assumption, not a replacement gold label or proof of population
minimality.

The four-permutation field-order sensitivity run can be regenerated on a host
with the configured Qwen3.5-9B checkpoint and one available 24-GB GPU:

```sh
python synthetic_benchmark_v2/make_random_field_order_probes_v2.py
bash synthetic_benchmark_v2/run_random_field_order_probes_v2.sh
```

The runner uses deterministic decoding and fixed permutation seeds, writes a
separate annotation file for each condition, supports resume, and evaluates
all conditions against the same 96 cases. The saved predictions can be
reanalyzed without model inference by running
`python synthetic_benchmark_v2/analyze_random_field_order_probes_v2.py`.

The script validates 324 baseline files and all three model-seed runs,
regenerates the main table metrics, M²OE matched-budget scoring, 78 paired
tests, confidence/evidence audits, subgroup and finite-reference analyses,
the hidden-provenance diagnostic, and the combined 20-test robustness family
when its frozen outputs are present. It does not rerun local LLM inference;
checkpoint weights and the original data licenses are environment-specific.
