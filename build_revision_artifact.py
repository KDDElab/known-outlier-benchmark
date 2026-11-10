#!/usr/bin/env python3
"""Build the reviewed revision package only after experiments and PDF finish."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from importlib.metadata import version
from pathlib import Path
import pymupdf
from manuscript_page_accounting import page_accounting

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'synthetic_benchmark_v2/runs/ccfa_revision'
ACCEPT=ROOT/'synthetic_benchmark_v2/runs/ccfa_acceptance_revision'
UNIFIED=ROOT/'synthetic_benchmark_v2/runs/ccfa_unified_revision'
DEST=ROOT/'artifact_release_revised'

README=r'''# Known-outlier explanation benchmark

This package contains the revised manuscript, source, frozen inputs, raw outputs,
and analysis code for **Evaluating Language Models and Subspace Methods for
Explaining Known Tabular Outliers**.

- Public repository: https://github.com/KDDElab/known-outlier-benchmark
- Fixed artifact release: https://github.com/KDDElab/known-outlier-benchmark/releases/tag/icde2027-eab-v4
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
'''

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    manifest=json.loads((RUN/'analysis_manifest.json').read_text())
    assert manifest['complete']
    transfer=ROOT/'synthetic_benchmark_v2/runs/ccfa_constraint_transfer'
    assert json.loads((transfer/'analysis_manifest.json').read_text())['complete']
    assert json.loads((ACCEPT/'summary.json').read_text())['complete']
    assert json.loads((ACCEPT/'analysis_manifest.json').read_text())['complete']
    assert json.loads((ACCEPT/'reconstruction_verification.json').read_text())['all_checks_passed']
    assert json.loads((UNIFIED/'analysis_manifest.json').read_text())['complete']
    assert json.loads((UNIFIED/'verification.json').read_text())['all_checks_passed']
    assert json.loads((UNIFIED/'reconstruction_verification.json').read_text())['all_checks_passed']
    assert json.loads((UNIFIED/'execution_manifest.json').read_text())['complete']
    for m in ['qwen35_9b','deepseek_v2_lite_chat','llama31_8b_instruct']:
        assert sum(1 for _ in (RUN/f'new_{m}.jsonl').open())==1055
    pdf=ROOT/'phase_d_manuscript_draft.pdf'
    source=(ROOT/'phase_d_manuscript_draft.tex').read_text()
    assert 'RESULT_PENDING_' not in source,'Populate every result-dependent manuscript section before release'
    assert 'example.org' not in source,'Placeholder artifact URLs are not release identifiers'
    accounting=page_accounting(pdf)
    assert accounting['main_text_pages']==12,'User requested 12 main-text pages; references and AI disclosure are excluded'
    assert pdf.stat().st_mtime>(RUN/'analysis_manifest.json').stat().st_mtime,'Compile final manuscript after completed analysis'
    assert pdf.stat().st_mtime>(transfer/'analysis_manifest.json').stat().st_mtime
    assert pdf.stat().st_mtime>(ACCEPT/'summary.json').stat().st_mtime
    assert pdf.stat().st_mtime>(UNIFIED/'analysis_manifest.json').stat().st_mtime
    ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','*_partial.csv','*_partial.json','partial_*')
    DEST.mkdir(exist_ok=True)
    shutil.copytree(ROOT/'artifact_release_staging',DEST,dirs_exist_ok=True,ignore=ignore)
    shutil.copytree(RUN,DEST/'synthetic_benchmark_v2/runs/ccfa_revision',dirs_exist_ok=True,ignore=ignore)
    shutil.copytree(transfer,DEST/'synthetic_benchmark_v2/runs/ccfa_constraint_transfer',dirs_exist_ok=True,ignore=ignore)
    shutil.copytree(ACCEPT,DEST/'synthetic_benchmark_v2/runs/ccfa_acceptance_revision',dirs_exist_ok=True,ignore=ignore)
    shutil.copytree(UNIFIED,DEST/'synthetic_benchmark_v2/runs/ccfa_unified_revision',dirs_exist_ok=True,ignore=ignore)
    shutil.copytree(ROOT/'circuit_track',DEST/'circuit_track',dirs_exist_ok=True,ignore=ignore)
    # The reference-support audit is invariant to display order and is reused by
    # the repaired figure code. Its raw per-case file remains exactly preserved.
    for name in ['reference_sample_audit_multiseed']:
        shutil.copytree(ROOT/'synthetic_benchmark_v2/runs'/name,DEST/'synthetic_benchmark_v2/runs'/name,dirs_exist_ok=True,ignore=ignore)
    for folder in ['', 'synthetic_benchmark_v2','figures']:
        src=ROOT/folder;dst=DEST/folder;dst.mkdir(exist_ok=True)
        patterns=['*.py'] if folder!='figures' else ['*.py','*.pdf','*.svg','*.png','*.webp']
        for pattern in patterns:
            for p in src.glob(pattern):shutil.copy2(p,dst/p.name)
    for name in ['phase_d_manuscript_draft.pdf','phase_d_manuscript_draft.tex','phase_d_references.bib','revision_execution.md','experiment_design_and_improvement_plan.md','unified_revision_report.md']:
        shutil.copy2(ROOT/name,DEST/name)
    shutil.copy2(ROOT/'MODEL_SOURCES.json',DEST/'MODEL_SOURCES.json')
    backup=ROOT.parent/'ccfa-workfiles/revision/known-outlier/before'
    shutil.copytree(backup,DEST/'legacy_before_revision',dirs_exist_ok=True,ignore=ignore)
    (DEST/'README.md').write_text(README)
    versions=json.loads((RUN/'runtime_versions.json').read_text())
    environment=json.loads((RUN/'environment_details.json').read_text())
    dependencies=[f'{k}=={versions[k]}' for k in ['numpy','pandas','scipy','scikit-learn','matplotlib','torch','transformers','accelerate','tokenizers','safetensors']]
    dependencies.append('PyMuPDF=='+environment['pymupdf_version'])
    dependencies.append('Pillow=='+version('Pillow'))
    (DEST/'requirements-analysis.txt').write_text('\n'.join(dependencies)+'\n')
    (DEST/'requirements-structured.txt').write_text('lm-format-enforcer==0.11.3\ninteregular==0.3.3\n')
    config=json.loads((ROOT/'config.json').read_text())
    for row in config['annotators']:row['model_path']='<SET_LOCAL_CHECKPOINT_PATH>'
    (DEST/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    # Execute the packaged analysis from its own root. This verifies path
    # completeness on the current host; it is not an independent-machine run.
    key_tables=['main_table.csv','main_paired_tests.csv','panel_summary.csv',
                'order_paired_tests.csv','references_paired_tests.csv','output_paired_tests.csv',
                'matched_three_summary.csv','matched_three_tests.csv',
                'random_order_stability_summary.csv','nested_common_feasible_summary.csv',
                'label_conditioned_scores.csv','main_per_seed_summary.csv',
                'ambiguity_conditioned_scores.csv','prediction_support_provenance_summary.csv',
                'circuit_hierarchical_summary.csv','natural_separate_groups.csv',
                'population_sample_audit_by_nuisance.csv','output_budget_continuation_summary.csv',
                'legacy_prefix_rescored_summary.csv','legacy_revision_descriptive_deltas.csv']
    expected={name:sha(RUN/name) for name in key_tables}
    extra_paths=[Path('synthetic_benchmark_v2/runs/ccfa_revision/referee_analysis')/name for name in
                 ['width_reference_counts.csv','four_order_scores.csv','historical_common_estimator_summary.csv','historical_reconciliation.csv']]
    extra_paths += [Path('synthetic_benchmark_v2/runs/ccfa_constraint_transfer')/name for name in
                    ['scores_per_case.csv','scores_by_relation.csv','summary.csv','label_audit.csv','summary_table.tex']]
    acceptance_tables=['prior_per_case.csv','prior_paired_tests.csv','structured_per_case.csv',
                       'new_rules_per_case.csv','new_rule_strata.csv','new_rule_error_summary.csv','real_cells_per_case.csv',
                       'real_cells_by_dataset.csv','prior_table.tex','structured_table.tex',
                       'new_rules_table.tex','real_cells_table.tex',
                       'presentation_main_table.tex','presentation_new_rules_table.tex',
                       'presentation_real_cells_table.tex','presentation_statistics.csv',
                       'presentation_group_values.csv']
    extra_paths += [Path('synthetic_benchmark_v2/runs/ccfa_acceptance_revision')/name for name in acceptance_tables]
    extra_paths += [Path('synthetic_benchmark_v2/runs/ccfa_unified_revision')/name for name in
        ['summary.csv','per_case.csv','repeated_summary.csv','paired_comparisons.csv',
         'common_feasible_summary.csv','stratified_summary.csv','reference_stability.csv',
         'primary_table.tex','original_compact_table.tex','beers_non_ounces_summary.csv',
         'presentation_primary_wide.tex','presentation_evidence_wide.tex','presentation_original_wide.tex']]
    extra_expected={str(p):sha(ROOT/p) for p in extra_paths}
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    with (DEST/'packaged_analysis_execution.log').open('w') as log:
        for script in ['synthetic_benchmark_v2/analyze_ccfa_revision.py',
                       'synthetic_benchmark_v2/analyze_ccfa_supplements.py',
                       'synthetic_benchmark_v2/analyze_ccfa_labels_external.py',
                       'synthetic_benchmark_v2/analyze_ccfa_secondary.py',
                       'synthetic_benchmark_v2/analyze_ccfa_legacy_comparison.py',
                       'synthetic_benchmark_v2/analyze_referee_revision.py',
                       'synthetic_benchmark_v2/analyze_constraint_transfer.py',
                       'synthetic_benchmark_v2/verify_ccfa_completion.py',
                       'synthetic_benchmark_v2/analyze_acceptance_revision.py',
                       'synthetic_benchmark_v2/write_acceptance_tables.py',
                       'synthetic_benchmark_v2/verify_acceptance_revision.py',
                       'figures/make_manuscript_figures.py',
                       'figures/make_manuscript_result_figures.py',
                       'figures/make_acceptance_figures.py',
                       'synthetic_benchmark_v2/finalize_acceptance_manifest.py',
                       'synthetic_benchmark_v2/analyze_unified_revision.py',
                       'synthetic_benchmark_v2/audit_unified_corrections.py',
                       'synthetic_benchmark_v2/verify_unified_revision.py',
                       'figures/make_unified_figures.py',
                       'figures/make_presentation_restoration.py',
                       'synthetic_benchmark_v2/finalize_unified_manifest.py']:
            log.write(f'Running {script}\n');log.flush()
            subprocess.run([sys.executable,script],cwd=DEST,env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    actual={name:sha(DEST/'synthetic_benchmark_v2/runs/ccfa_revision'/name) for name in key_tables}
    assert actual==expected,'Packaged analysis changed the reported main/panel tables'
    extra_actual={str(p):sha(DEST/p) for p in extra_paths}
    assert extra_actual==extra_expected,'Packaged referee/transfer analysis differs'
    figure_names=['figure_intro_real_dataset','figure2_overall_framework',
                  'figure3_reference_and_order_robustness','figure2_sufficiency_compactness_validity',
                  'figure5_matched_reference_and_output','figure_acceptance_primary_intervals',
                  'figure_acceptance_invoice_mechanism','figure_acceptance_shared_prior',
                  'figure_acceptance_structured_output']
    figure_names += ['figure_intro_support_semantics','figure_unified_transfer',
                     'figure_unified_evidence','figure_unified_contract',
                     'figure_unified_lof','figure_unified_corrections']
    def rendered_hash(path):
        with pymupdf.open(path) as document:
            assert len(document)==1
            pixels=document[0].get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False)
            return hashlib.sha256(pixels.samples).hexdigest()
    figure_hashes={name:rendered_hash(ROOT/'figures'/f'{name}.pdf') for name in figure_names}
    reproduced_figures={name:rendered_hash(DEST/'figures'/f'{name}.pdf') for name in figure_names}
    assert reproduced_figures==figure_hashes,'Packaged figure rendering differs from the manuscript figures'
    packaged_accept=DEST/'synthetic_benchmark_v2/runs/ccfa_acceptance_revision'
    assert json.loads((packaged_accept/'reconstruction_verification.json').read_text())['all_checks_passed']
    original_summary=json.loads((ACCEPT/'summary.json').read_text())
    packaged_summary=json.loads((packaged_accept/'summary.json').read_text())
    for field in ['prior','prior_common_feasible','prior_tests','structured','structured_common_feasible',
                  'new_rules','repeated_reference','reference_stability','new_rule_errors','real_cells','native_controls',
                  'invoice_mechanism','main_d6']:
        assert original_summary[field]==packaged_summary[field],f'Changed acceptance summary: {field}'
    (DEST/'packaged_analysis_verification.json').write_text(json.dumps({
        'status':'same_host_packaged_recomputation_passed','compared_sha256':actual,
        'compared_referee_transfer_sha256':extra_actual,
        'compared_figure_render_sha256':figure_hashes,
        'acceptance_summary_equal':True,'acceptance_input_reconstruction_passed':True,
        'model_inference_rerun':False,'independent_machine_reproduction':False},indent=2)+'\n')
    # Update the execution ledger only after the actual packaged recomputation
    # and figure comparisons pass, then include that same ledger in the archive.
    ledger=ROOT/'revision_execution.md'
    ledger_text=ledger.read_text()
    ledger_text=ledger_text.replace('PACKAGE_CHECK_PENDING',
        '本地包已完成；同主机包内重算原结果、新增结果和表格、七张图的渲染像素均一致；八份新增冻结输入与标签逐字节重建一致')
    ledger.write_text(ledger_text)
    shutil.copy2(ledger,DEST/ledger.name)
    sums={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
    (DEST/'SHA256SUMS.json').write_text(json.dumps(sums,indent=2)+'\n')
    archive=ROOT/'known_outlier_ccfa_revision.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(DEST,arcname='known_outlier_ccfa_revision')
    (ROOT/'known_outlier_ccfa_revision.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps({'directory':str(DEST),'archive':str(archive),'files':len(sums),'archive_bytes':archive.stat().st_size,'sha256':sha(archive)},indent=2))

if __name__=='__main__':main()
