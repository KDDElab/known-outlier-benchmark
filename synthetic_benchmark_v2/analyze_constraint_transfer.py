#!/usr/bin/env python3
"""Score the complete frozen standards workload; preserve every failed response.

All summaries are descriptive means over four fixed relations. No significance
test or unseen-domain confidence interval is introduced after seeing outputs.
"""
import itertools
import json
from decimal import Decimal, ROUND_HALF_UP
import subprocess
import sys
from pathlib import Path

import pandas as pd

from annotate_v2 import parse_prediction
from prepare_constraint_transfer import HERE, OUT, sha, write

MODELS = ['qwen35_9b', 'deepseek_v2_lite_chat', 'llama31_8b_instruct']
METRICS = ['support_top1_sufficient', 'support_top1_minimal',
           'support_any_minimal_hit', 'support_best_jaccard', 'answered',
           'direct_edit_exact_top1', 'structurally_valid']


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert protocol['frozen_inputs_sha256'] == sha(OUT / 'inputs.jsonl')
    assert protocol['frozen_gold_sha256'] == sha(OUT / 'ground_truth.jsonl')
    assert protocol['frozen_populations_sha256'] == sha(OUT / 'populations.jsonl')
    inputs = read(OUT / 'inputs.jsonl')
    by_id = {x['case_id']: x for x in inputs}
    gold = {x['case_id']: x for x in read(OUT / 'ground_truth.jsonl')}
    populations = {x['family']: x for x in read(OUT / 'populations.jsonl')}
    assert len(inputs) == len(by_id) == len(gold) == 96
    assert set(by_id) == set(gold)
    # Independently recover minimal projections by enumerating all subsets of
    # structured roles. Independent categorical distractors cannot add a minimum.
    label_rows = []
    for cid, g in gold.items():
        p = populations[g['template_id']]
        names = p['role_names']
        ids = [g['rule_role_to_feature'][n] for n in names]
        target = [g['target_row'][f] for f in ids]
        minimal = []
        for size in range(1, len(ids) + 1):
            for subset in itertools.combinations(range(len(ids)), size):
                fields = frozenset(ids[j] for j in subset)
                if any(m <= fields for m in minimal):
                    continue
                if not any(all(row[j] == target[j] for j in subset) for row in p['clean_role_rows']):
                    minimal.append(fields)
        expected = {frozenset(s) for s in g['reference_support_sufficient_minimal_subspaces']}
        assert set(minimal) == expected == {frozenset(s) for s in g['rule_minimal_subspaces']}
        changed = {f for f in by_id[cid]['feature_ids'] if g['before_row_hidden'][f] != g['target_row'][f]}
        assert changed == set(g['modified_feature_ids']) and len(changed) == 1
        assert all(len(s) >= 2 for s in minimal)
        label_rows.append({'case_id': cid, 'relation': g['template_id'],
                           'gold_count': len(minimal), 'min_gold_size': min(map(len, minimal)),
                           'max_gold_size': max(map(len, minimal)),
                           'direct_edit_size': len(changed)})
    predictions, sources, counts = {}, {}, {}
    for model in MODELS:
        source = OUT / f'annotations_{model}.jsonl'
        records = read(source)
        assert len(records) == len({r['case_id'] for r in records}) == 96, f'Incomplete {model}'
        assert {r['case_id'] for r in records} == set(by_id)
        run = read(OUT / f'run_{model}.jsonl')
        assert any(r['event'] == 'completed' for r in run), f'No completed run: {model}'
        start = next(r for r in run if r['event'] == 'started')
        assert start['inputs_sha256'] == sha(OUT / 'inputs.jsonl')
        assert protocol['frozen_at_utc'] < start['started_at_utc']
        rows = []
        for r in records:
            x = by_id[r['case_id']]
            assert r['prompt_sha256'] == x['prompt_sha256']
            assert r['do_sample'] is False and r['max_new_tokens'] == 512
            assert r['max_input_tokens'] == 16384 and r['output_contract'] == 'full'
            parsed, error = parse_prediction(r['raw_output'], set(x['feature_ids']), 'full')
            assert parsed == r['parsed_prediction'], 'Archived and current parser differ'
            rows.append({'case_id': r['case_id'], **(parsed or {})})
        predictions[model] = rows
        sources[model] = sha(source)
        counts[model] = {'records': len(records),
                         'input_or_inference_failures': sum((r.get('parse_or_inference_error') or '').startswith('inference failed') for r in records),
                         'parse_failures': sum(r.get('parsed_prediction') is None for r in records),
                         'output_cap_reached': sum(bool(r.get('output_limit_reached')) for r in records)}
    baseline_manifest = json.loads((OUT / 'baselines' / 'gower_subspace_baselines_manifest.json').read_text())
    assert baseline_manifest['inputs_sha256'] == sha(OUT / 'inputs.jsonl')
    assert baseline_manifest['max_subspace_order'] == 8 and baseline_manifest['n_cases'] == 96
    for method in ['gower_lof_subspace', 'gower_knn_subspace']:
        path = OUT / 'baselines' / f'predictions_{method}.jsonl'
        rows = read(path)
        assert len(rows) == len({r['case_id'] for r in rows}) == 96
        assert {r['case_id'] for r in rows} == set(by_id)
        predictions[method] = rows
        sources[method] = sha(path)
    frames = []
    for method, rows in predictions.items():
        root = OUT / 'evaluation' / method
        root.mkdir(parents=True, exist_ok=True)
        write(root / 'predictions.jsonl', rows)
        with (root / 'scoring.log').open('w') as log:
            subprocess.run([sys.executable, str(HERE / 'evaluate_benchmark.py'), str(root / 'predictions.jsonl'),
                            '--inputs-path', str(OUT / 'inputs.jsonl'), '--ground-truth-path', str(OUT / 'ground_truth.jsonl'),
                            '--outdir', str(root)], check=True, stdout=log, stderr=subprocess.STDOUT)
        df = pd.read_csv(root / 'per_case.csv')
        df['direct_edit_exact_top1'] = [
            bool(r.answered and json.loads(r.explanations_predicted) and
                 set(json.loads(r.explanations_predicted)[0]) == set(gold[r.case_id]['modified_feature_ids']))
            for r in df.itertuples()]
        df['method'] = method
        frames.append(df)
    scores = pd.concat(frames, ignore_index=True)
    per_relation = scores.groupby(['method', 'template_id'])[METRICS].mean().reset_index()
    summary = per_relation.groupby('method')[METRICS].mean().reset_index()
    scores.to_csv(OUT / 'scores_per_case.csv', index=False)
    per_relation.to_csv(OUT / 'scores_by_relation.csv', index=False)
    summary.to_csv(OUT / 'summary.csv', index=False)
    pd.DataFrame(label_rows).to_csv(OUT / 'label_audit.csv', index=False)
    display = [('qwen35_9b', 'Qwen'), ('deepseek_v2_lite_chat', 'DeepSeek'),
               ('llama31_8b_instruct', 'Llama'), ('gower_lof_subspace', 'Gower-LOF'),
               ('gower_knn_subspace', 'Gower-kNN')]
    columns = ['support_top1_minimal', 'support_any_minimal_hit', 'support_best_jaccard',
               'answered', 'direct_edit_exact_top1']
    lookup = summary.set_index('method')
    lines = []
    for method, label in display:
        values = [str(Decimal(str(float(lookup.loc[method, m]))).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)) for m in columns]
        lines.append(label + ' & ' + ' & '.join(values) + r' \\')
    table = r'''\begin{table}[t]
\centering\footnotesize
\caption{Invoice constraints: 96 synthetic queries, four fixed relations. Equal-relation means retain every failure. Edit match compares the first set with the recorded edit; this provenance diagnostic is not an identifiable prediction target.}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}|l|rrrrr|}
\hline
Method & Exact@1 & Hit@4 & Jacc.@4 & Answered & \shortstack{Edit\\match} \\
\hline
''' + '\n'.join(lines) + r'''
\hline
\end{tabular*}
\renewcommand{\arraystretch}{1.0}
\label{tab:invoice}
\end{table}
'''
    (OUT / 'summary_table.tex').write_text(table)
    manifest = {'complete': True, 'scope': 'Descriptive means over four fixed standards-derived synthetic relations; no significance tests.',
                'input_sha256': sha(OUT / 'inputs.jsonl'), 'gold_sha256': sha(OUT / 'ground_truth.jsonl'),
                'protocol_sha256': sha(OUT / 'protocol.json'), 'sources': sources, 'model_counts': counts,
                'n_queries': 96, 'n_model_outputs': 288, 'labels_independently_reenumerated': 96,
                'scorer_sha256': sha(HERE / 'evaluate_benchmark.py'), 'analysis_sha256': sha(__file__),
                'csv_sha256': {p.name: sha(p) for p in OUT.glob('*.csv')},
                'table_tex_sha256': sha(OUT / 'summary_table.tex')}
    (OUT / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(summary.to_string(index=False))
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
