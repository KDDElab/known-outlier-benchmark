# Known-outlier explanation benchmark

This package contains the revised manuscript, source, frozen inputs, raw outputs,
and analysis code for **Evaluating Language Models and Subspace Methods for
Explaining Known Tabular Outliers**.

- Public repository: https://github.com/KDDElab/known-outlier-benchmark
- Fixed artifact release: https://github.com/KDDElab/known-outlier-benchmark/releases/tag/icde2027-eab-v5
- Paper: [PDF](phase_d_manuscript_draft.pdf) and [LaTeX](phase_d_manuscript_draft.tex).
- The paper has **12 main-text pages plus one page of AI disclosure and references**,
  following the [ICDE 2027 EAB page rule](https://icde2027.github.io/cf-research-papers.html).
  There is no appendix. This repository does not claim conference acceptance.

## Quick start

```bash
git clone https://github.com/KDDElab/known-outlier-benchmark.git
cd known-outlier-benchmark
python reproduce_artifact.py --verify-only
# In an environment with the recorded dependencies and IEEEtran LaTeX:
python reproduce_artifact.py --analysis --paper
```

The checksum-only command uses the Python standard library. The second command
recomputes scores from the included predictions, regenerates plots and tables,
and writes the compiled PDF to `reproduced/paper/`. It does not run model inference.
Use a fresh checkout for each reproduction: analysis can rewrite derived files,
so a later checksum comparison with the untouched release can then differ.
All frozen cohorts, failures and source records remain included. A downloadable
release archive and its SHA-256 file provide the same artifact without Git.

Model weights are obtained from the original providers; see the model section
below. No GitHub login is needed to download the artifact or reproduce its
analysis from frozen predictions. The supplied author name and ORCID are retained.

## Which results belong to the revision?

`synthetic_benchmark_v2/runs/ccfa_unified_revision/` contains the current primary
comparison: **1,080 newly executed model-query records** (360 per checkpoint)
under one core task with shared candidate-size information and fixed JSON-schema
decoding. Its 96 paired original targets, 120 new-rule evidence conditions,
96 invoices, and 48 UCI Car decisions are separate evaluation tracks.
All failures and abstentions remain misses. `protocol.json` was frozen before
new inference; it does not constitute preregistration or a fresh blind test on
previously examined targets. The independent Car source is a complete expert
decision table, not observed anomaly incidents.

The same directory includes direct sample-minimal enumeration (SME), matched
Gower baselines, 14 LOF conditions with all candidate scores, repeated-reference
and complete-population diagnostics, and a post-hoc public-correction audit.
SME uses only visible references and canonical size/field-ID order. Complete
population labels are hidden from all candidate generators. The new-rule source
field originally named `reference_support_sufficient_minimal_subspaces` contains
sample minima; the documented mapping correction in `corrections/` preserves
those diagnostics and maps the new comparison to independently checked population
labels. No input or model response was changed by that correction.

`verification.json` records 168 independent population-label checks, 624
unchanged Gower lists, and 552 scalar LOF checks. `reconstruction_verification.json`
checks exact prompt/population bytes and semantic equality of effective gold JSON
(key insertion order differs after the recorded correction). `analysis_manifest.json`
refuses complete publication exports until every model query finishes.

`synthetic_benchmark_v2/runs/ccfa_acceptance_revision/` contains the earlier
review-driven experiments: 1,368 new model-condition records (456 per model),
paired size priors, 12 new rule compositions with repeated/complete evidence,
fixed JSON-schema decoding, and 72 public corrected-cell queries per model under
both ordinary and structured decoding.
Its `summary.json` requires every condition to finish. The same directory holds
the invoice parameter/candidate-score diagnostics, six-field exhaustive search,
native M2OE/SOD controls, prompt audits, and independently recomputed finite labels.
`reconstruction_verification.json` checks eight frozen input/label files byte for
byte. Public correction pairs come from the pinned Raha repository commit and
retain source indices, raw strings, license and hashes. The blank human annotation
packet is a proposed study instrument, not completed human evidence.

`synthetic_benchmark_v2/runs/ccfa_revision/` preserves the original comparison.
Its `protocol.json` freezes inputs, seeds, reuse rules and statistical estimators.
`analysis_manifest.json` identifies the complete main comparison. The main study
has 864 cases; the 96-target intervention panel yields 1,728 conditions per model.
`reused_*.jsonl` preserves 673 prompt-identical records per model, and
`new_*.jsonl` contains 1,055 newly executed records per model. Reused records retain
their source hashes. Raw model text is preserved, including failures and abstentions.

Qwen and DeepSeek queues were continued using two disjoint serial shards after
preserving completed prefixes. Each replica first matched six archived raw outputs
exactly; formal generation remained batch size one. Interrupted supervisor logs
record those scheduling handoffs. The merged `new_*.jsonl` files and
`completion_verification.json` identify the final cohort. Prefix/shard copies are
provenance duplicates and must not be counted as additional benchmark cases.

Other run directories and `legacy_before_revision/` preserve provenance. Their old
scores are not the revised main results. The previous `reproduce_multiseed_analysis.sh`
is a legacy entry point; use the commands below for the current manuscript.

`synthetic_benchmark_v2/runs/ccfa_constraint_transfer/` is the separate standards
workload: 96 targets from four public Peppol amount relations, 288 fresh model
outputs, and exhaustive Gower predictions. `protocol.json` was frozen before
inference. `source_rules/` archives the official rule pages and hashes. The data
are synthetic finite-domain amounts, not observed invoices. `label_audit.csv`
checks exhaustive minimal alternatives and the distinct one-field edit label.
The main 864-case results remain unchanged. `referee_analysis/` under the main
run root contains historical rescoring, four-order scores, and the n/T grid.

## Result presentation

The manuscript contains seven figures and six tables, including five full-width
tables. Table VI restores the complete neighboring-evaluation-target matrix.
The unified comparison, repeated-reference diagnostics, original 864-case suite,
and public corrections have grouped wide result tables. Result tables display
mean plus/minus sample SD across their stated aggregation units: original
templates, new compositions, and public sources. Individual-source correction
columns instead use case SD. Conditional first-set size uses coverage-weighted
template SD. These SDs describe heterogeneity, not repeated-model-run variability.
Original 95% template-bootstrap intervals remain in the archived diagnostic plot.
`figures/make_acceptance_figures.py` also calls
`synthetic_benchmark_v2/present_acceptance_results.py`, regenerating the three
presentation tables and their 194 mean/SD entries from frozen per-case scores.
The supplied Qwen logo remains in the Figure 2 framework.

The v4 revision changes presentation only. Figure 1 uses the approved Figure 2
visual grammar; Figures 2 and 3 retain their v3 exports. Figures 4--7 return to
the earlier boxed-axis, method-color and condition-hatch style. Run
`figures/make_presentation_restoration.py` after the previous figure generators;
it rebuilds Figure 1, Figures 4--7 and the three restored wide result tables from
frozen records. No additional inference or experiments were performed for v4.

The v5 revision refines Figure 1 with native vector icons for invoices, an edit,
an alert, a calculator, normal-data search, linked fields and verification. The
recorded tax-total case 000 supplies the before/after values and normal witnesses.
Field colors match Figure 2. Figures 2--7 and all table contents retain their v4
exports. To update Figure 1 alone, use
`python figures/make_presentation_restoration.py --intro-only`.
No experimental outputs or numerical conclusions change in this revision.

## Recompute scores and figures from frozen predictions

Run from the unpacked package root, using the environment versions recorded in
`synthetic_benchmark_v2/runs/ccfa_revision/runtime_versions.json`:

```bash
python synthetic_benchmark_v2/analyze_ccfa_revision.py
python synthetic_benchmark_v2/analyze_ccfa_supplements.py
python synthetic_benchmark_v2/analyze_ccfa_labels_external.py
python synthetic_benchmark_v2/analyze_ccfa_secondary.py
python synthetic_benchmark_v2/analyze_ccfa_legacy_comparison.py
python synthetic_benchmark_v2/analyze_referee_revision.py
python synthetic_benchmark_v2/analyze_constraint_transfer.py
python synthetic_benchmark_v2/verify_ccfa_completion.py
python synthetic_benchmark_v2/analyze_acceptance_revision.py
python synthetic_benchmark_v2/write_acceptance_tables.py
python synthetic_benchmark_v2/verify_acceptance_revision.py
python figures/make_manuscript_figures.py
python figures/make_manuscript_result_figures.py
python figures/make_acceptance_figures.py
python synthetic_benchmark_v2/finalize_acceptance_manifest.py
python synthetic_benchmark_v2/analyze_unified_revision.py
python synthetic_benchmark_v2/audit_unified_corrections.py
python synthetic_benchmark_v2/verify_unified_revision.py
python figures/make_unified_figures.py
python figures/make_presentation_restoration.py
python synthetic_benchmark_v2/finalize_unified_manifest.py
pdflatex -interaction=nonstopmode -halt-on-error phase_d_manuscript_draft.tex
bibtex phase_d_manuscript_draft
pdflatex -interaction=nonstopmode -halt-on-error phase_d_manuscript_draft.tex
pdflatex -interaction=nonstopmode -halt-on-error phase_d_manuscript_draft.tex
```

These commands do not require model weights or GPU inference. The shared parser
currently imports the inference libraries, so PyTorch and Transformers must still
be installed. Main dependencies and actual versions are in `requirements-analysis.txt`;
LaTeX additionally needs IEEEtran, booktabs, multirow, graphicx, tabularx and hyperref.
`synthetic_benchmark_v2/runs/ccfa_revision/environment_details.json` records Python,
OS, GPU driver and PDF-renderer versions.
The package verification also uses PyMuPDF for figure-pixel comparisons.
The manuscript's result statements were edited after reading the exported tables;
compilation does not automatically rewrite prose if a new experiment changes them.

Figure files under `figures/` are provided as PDF, SVG and PNG:

| Manuscript figure | File stem |
| --- | --- |
| 1 | `figure_intro_support_semantics` |
| 2 | `figure2_overall_framework` |
| 3 | `figure_unified_transfer` |
| 4 | `figure_unified_evidence` |
| 5 | `figure_unified_contract` |
| 6 | `figure_unified_lof` |
| 7 | `figure_unified_corrections` |

Figure 1 uses the recorded invoice example and its exact finite clean population
to distinguish an edited field from two minimal support explanations. The earlier
Pima agreement figure remains in the archive as a diagnostic and is not the
paper's visual entry point.

Figure 2 follows the supplied two-module overview and three-row invoice trace.
Its upper overview describes the unified comparison; its lower trace explicitly
retains original-prompt outputs and must not be read as new joint predictions.
Query components, checkpoint names and output validation are explicit. Shared
field colors link the invoice to its candidate/gold sets; the example contrasts
sample absence with population support and minimal joint exclusion.
Data-bound insets include first-subset masks, R/P gross-value presence, the
4-by-20 net/gross support matrix, LOF rank-1-to-4 minimality, and the subset
lattice. Support cells encode presence, not frequency or estimated density.
It embeds the user-supplied Qwen logo, the unchanged avatar from the official
[Meta Llama profile](https://huggingface.co/meta-llama), and the original whale
path from [DeepSeek's official SVG](https://github.com/deepseek-ai/DeepSeek-LLM/blob/main/images/logo.svg).
Logo files are included locally; rendering does not need network access.
Source URLs and placement details are in `figures/make_manuscript_figures.py`.

Earlier plots remain in the package as supplementary diagnostics. The mechanism
plot uses executed invoice sweeps and recorded candidate scores; the primary
interval plot uses the unchanged original `main_summary.csv`.

## Rerun inference or baselines

### Model checkpoints

`MODEL_SOURCES.json` records the official repositories, access terms, the source
run IDs, and the original per-file SHA-256 fingerprints for all five model families.
The main comparison uses Qwen, DeepSeek and Llama; Phi and Gemma are used in the
separate peer-agreement pilot.

| Annotator ID | Official upstream checkpoint | Provider license |
| --- | --- | --- |
| `qwen35_9b` | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Apache 2.0 |
| `deepseek_v2_lite_chat` | [deepseek-ai/DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat) | DeepSeek |
| `llama31_8b_instruct` | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Llama 3.1 |
| `phi4_mini` | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | MIT |
| `gemma3_4b` | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | Gemma |

For example, after obtaining a checkpoint with the provider's download tool:

```bash
hf download Qwen/Qwen3.5-9B --local-dir models/qwen35_9b
python verify_model_checkpoint.py qwen35_9b models/qwen35_9b
```

Use the corresponding repository and annotator ID for each other model, and set
its `model_path` in `config.json`. Llama and Gemma access is subject to the provider's
license acceptance/access process; model weights are not included in this Git
repository. A current upstream branch is not automatically the evaluated checkpoint.
The verifier compares configuration, weights and tokenizer files with the archived
experiment hashes and exits unsuccessfully on missing or different files. Matching
files does not guarantee bitwise generation across different kernels or hardware.

### Inputs, inference and baselines

For the current unified protocol, reconstruct inputs without overwriting the
frozen records:

```bash
python synthetic_benchmark_v2/prepare_unified_revision.py --outdir regenerated_unified
```

The included UCI Car data and source/CC BY 4.0 attribution require no network
access for reconstruction. Run `annotate_unified_support.py --help` and use the
frozen `runs/ccfa_unified_revision/inputs.jsonl` with new annotation/log paths,
the original checkpoint hashes, and `requirements-structured.txt`. The runner
enforces the per-track size cap in its JSON grammar. Do not overwrite archived
predictions. On a disposable checkout, `run_unified_baselines.py` reruns the
statistical/SME predictions and `run_unified_lof_diagnostics.py` reruns all 14
LOF conditions. These CPU commands write into the unified run directory.

To reconstruct the frozen protocol in a separate output directory:

```bash
python synthetic_benchmark_v2/prepare_ccfa_revision.py --outdir regenerated_protocol
```

This uses the included legacy source cases and exact-hash reusable outputs. It
reconstructs input conditions, labels, and inference queues; it does not generate
new model responses. `protocol_reconstruction.json` records byte-identical
reconstruction of ten frozen data/queue files. `generator_reconstruction.json`
additionally checks all 480 prompts and labels per seed from the revised generator.
The generator accepts `--seed`, `--display-order-seed`, and `--outdir` for fresh cases.

For the additional invoice workload, use
`python synthetic_benchmark_v2/prepare_constraint_transfer.py --outdir new_invoice_protocol`.
The generator refuses to overwrite a frozen protocol. Its reconstruction report
checks the three input/population/label files byte for byte. Baselines can be rerun
with `run_gower_subspace_baseline.py --max-order 8 --inputs-path
new_invoice_protocol/inputs.jsonl --outdir new_invoice_protocol/baselines`.
For model inference, use the same runner below with the invoice inputs and fresh
output paths. Do not mix new predictions into the archived 96-case cohort.

Edit `config.json` to point each model to a local checkpoint. Verify its weights,
tokenizer and chat template against the archived hashes before a comparison.
Use `synthetic_benchmark_v2/annotate_v2.py --help` and supply the frozen input file,
a new annotations path and a new run-log path. The runner reads each condition's
512/1,024-token limit. Use greedy decoding, the 16,384-token input cap, and no input
truncation. Keep fresh runs separate from the included predictions. GPU allocation
is an execution setting; `CCFA_GPU_MAX_MEMORY_GIB` limits weight placement per GPU.

For the structured condition, install `requirements-structured.txt` and use
`annotate_acceptance_structured.py` with the frozen `inputs_structured.jsonl`.
Its fixed schema, original minimal prompt and 512-token budget are specified in
`structured_protocol.json`. The wrapper supplies a process-local Transformers
import compatibility alias; it does not modify installed libraries. Original
unconstrained outputs are never repaired for scoring. Use new output paths.
The new-rule and corrected-cell cohorts use the ordinary `annotate_v2.py` runner.
The paired structured corrected-cell condition uses
`annotate_acceptance_real_structured.py` and `inputs_real_cells_structured.jsonl`.
This added diagnostic was fixed after observing malformed unconstrained outputs;
its protocol records that chronology and retains all 72 cases and all three models.

`run_gower_subspace_baseline.py`, `run_sod_cop_adaptations_v2.py`, and
`run_rlax_adaptation_v2.py` accept input/output paths. `run_m2oe_baseline_v2.py`
requires a separate TensorFlow environment; see its completed run manifest for
versions. The compatible upstream source and license are included under
`synthetic_benchmark_v2/runs/ccfa_revision/third_party/M2OE/`; put this directory on `PYTHONPATH`.
The queue scripts record the original host's paths and scheduling choices. They
are execution records, not portable environment installers. Baseline command lines
and budgets are retained in the queue contracts and per-run manifests.

The Beam supplement is a paper-guided Gower-LOF adaptation, with width 100 and
2–3-field candidates. Its exhaustive control has the same candidate-size range.
Scores are cached exhaustively; timing is not a canonical Beam efficiency result.
RefOut is discussed for applicability but was not run. RLAX and SOD/COP are also
declared adaptations. M2OE is assessed only through its unordered field choices.
Use `matched_three_summary.csv` and `matched_three_tests.csv` for its main comparison.
Generic scorer logs also retain first-returned-choice quantities; these have no
ranked interpretation for M2OE and are excluded from the manuscript main table.
The separate native M2OE positive control additionally retains actual repair
vectors and evaluates their distance outside the known clean cube. It reports
both the 30- and 300-epoch budgets, including the worse longer-budget result.

## Statistical and evidence boundaries

Current primary scores average four cases within each of 24 templates, then
templates equally. New rules use 12 rule units, invoices four relations, and
Car 48 selected cases from one decision source. Reported SDs use these declared
units. No accuracy is pooled across task semantics. The 18 paired-panel contrast
tests form one Holm family and are descriptive after earlier test inspection.

Original scores average cases within a seed/template, available seeds within a
template, then the 38 templates equally. Bootstrap intervals and paired tests use
that same estimator. Abstentions and failures remain misses in all-query scores;
top-one size is defined only for answered queries. Main, intervention, and unordered
choice comparisons have separate declared correction families. Stored intervals
are retained even though manuscript tables show point estimates.
`legacy_revision_descriptive_deltas.csv` separately compares historical role-prefix
responses with the revision under the current parser, abstention rule and estimator.
This post-hoc diagnostic does not introduce an additional hypothesis-test family.
Main confidence intervals condition on one frozen display per case. The four-order
panel reports order-specific estimates and conditional agreement, not an estimate
of an order-marginal variance. Historical Gower predictions additionally differ in
canonical tie/summation ordering; their changes are not attributed to display alone.
The invoice workload averages 24 targets per relation, then the four relations
equally, without a new significance family or an unseen-domain confidence claim.
Seeds such as 20261113 are integer identifiers and do not encode execution dates.

Circuit scores average focus queries within targets, targets within table pairs,
pairs within circuits, and four circuits equally. Their uncertainty is conditional
on those four circuits. Natural-table agreement is reported separately for 70
ADBench and eight UCI queries and is not feature-level accuracy. No expert study,
new human judgments, analyst time savings, or independent third-party replication
is claimed. Public corrected fields supply a different, auditable inspection
target; they are not population-minimal explanation labels. Canonicalization is
included when the data owner changed a value. Do not interpret the blank annotation
forms as completed judgments.

## Provenance and reuse

`SHA256SUMS.json` records every packaged file. Historical machine paths in run logs
are retained as provenance; active commands above resolve relative to this package.
Dataset source and license notes are in `data/`; the circuit source is Harvard
Dataverse DOI 10.7910/DVN/W4FPPN (CC0). Third-party code keeps its own license.
No new license is asserted for the authors' source code. Model access and dataset
redistribution remain governed by their original sources.
