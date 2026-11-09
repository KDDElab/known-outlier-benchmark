#!/usr/bin/env python3
"""Export review-requested diagnostics from frozen observations, without inference."""
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
from analyze_ccfa_revision import summarize
from prepare_ccfa_revision import HERE, OUT, read, write, sha


def main():
    dest = OUT / 'referee_analysis'
    dest.mkdir(exist_ok=True)
    inputs = read(OUT / 'inputs_main.jsonl')
    gold = {x['case_id']: x for x in read(OUT / 'ground_truth_all_conditions.jsonl')}
    metadata = pd.DataFrame([{'case_id': x['case_id'], 'source_case_id': x['source_case_id'],
                             'generator_seed': x['generator_seed'], 'condition': 'main',
                             'feature_count': x['feature_count'], 'reference_count': x['reference_count'],
                             'template_id': gold[x['case_id']]['template_id']} for x in inputs])
    grid = metadata.groupby(['feature_count', 'reference_count']).agg(
        n=('case_id', 'size'), T=('template_id', 'nunique')).reset_index()
    grid.to_csv(dest / 'width_reference_counts.csv', index=False)
    panel = pd.read_csv(OUT / 'panel_summary.csv')
    order = panel[panel.condition.isin(['main', 'order1002', 'order1003', 'order1004']) &
                  panel.metric.isin(['support_top1_minimal', 'support_any_minimal_hit']) &
                  panel.method.isin(['qwen35_9b', 'deepseek_v2_lite_chat', 'llama31_8b_instruct'])].copy()
    order['display_seed'] = order.condition.map({'main': 1001, 'order1002': 1002, 'order1003': 1003, 'order1004': 1004})
    assert len(order) == 24 and (order.n_cases == 96).all() and (order.n_templates == 24).all()
    order.to_csv(dest / 'four_order_scores.csv', index=False)

    current = pd.read_csv(OUT / 'main_table.csv').set_index('method')
    historical = pd.read_csv(OUT / 'legacy_prefix_rescored_summary.csv')
    sources, summaries = [], []
    for method in ['gower_lof_subspace', 'gower_knn_subspace']:
        rows = []
        for seed in [20261106, 20261107, 20261108]:
            root = HERE / 'runs' / 'baseline_predictions' if seed == 20261106 else HERE / 'runs' / 'multiseed' / f'seed_{seed}' / 'gower'
            path = root / f'predictions_{method}.jsonl'
            manifest = json.loads((root / 'gower_subspace_baselines_manifest.json').read_text())
            input_path = HERE / 'inputs.jsonl' if seed == 20261106 else root.parent / 'inputs.jsonl'
            assert manifest['inputs_sha256'] == sha(input_path)
            predictions = read(path)
            assert len(predictions) == 288
            rows += [{**r, 'case_id': f's{seed}_main_{r["case_id"]}'} for r in predictions]
            sources.append({'method': method, 'seed': seed, 'file': str(path.relative_to(HERE)),
                            'sha256': sha(path), 'manifest_sha256': sha(root / 'gower_subspace_baselines_manifest.json')})
        root = dest / method
        root.mkdir(exist_ok=True)
        write(root / 'predictions.jsonl', rows)
        with (root / 'scoring.log').open('w') as log:
            subprocess.run([sys.executable, str(HERE / 'evaluate_benchmark.py'), str(root / 'predictions.jsonl'),
                            '--inputs-path', str(OUT / 'inputs_main.jsonl'), '--ground-truth-path', str(OUT / 'ground_truth_all_conditions.jsonl'),
                            '--outdir', str(root)], check=True, stdout=log, stderr=subprocess.STDOUT)
        scores = pd.read_csv(root / 'per_case.csv').merge(metadata[['case_id', 'source_case_id', 'generator_seed', 'condition']], on='case_id', validate='one_to_one')
        summaries += summarize(scores, method, 'legacy_prefix_rescored')
    historical = pd.concat([historical, pd.DataFrame(summaries)], ignore_index=True)
    historical.to_csv(dest / 'historical_common_estimator_summary.csv', index=False)
    reconciliation = []
    for r in historical.itertuples():
        if r.metric not in ['support_top1_sufficient', 'support_top1_minimal', 'support_any_minimal_hit']:
            continue
        new = float(current.loc[r.method, r.metric])
        reconciliation.append({'method': r.method, 'metric': r.metric,
                               'legacy_predictions_current_scoring': r.estimate,
                               'revised_predictions_current_scoring': new,
                               'revised_minus_legacy': new - r.estimate})
    pd.DataFrame(reconciliation).to_csv(dest / 'historical_reconciliation.csv', index=False)
    (dest / 'manifest.json').write_text(json.dumps({
        'scope': 'Post-hoc descriptive diagnostics. No new model inference, no new significance family.',
        'historical_lm_source': '../legacy_comparison_manifest.json', 'historical_gower_sources': sources,
        'scoring_control': 'Both historical and revised predictions use the current abstention semantics and template estimator.',
        'attribution': 'LM comparison changes display while fixing semantic prompt content and scoring; historical versus revised execution remains a limitation. Gower also changes canonical tie ordering and summation order, so its difference is not solely a display effect.',
        'script_sha256': sha(__file__), 'main_table_sha256': sha(OUT / 'main_table.csv'),
        'panel_summary_sha256': sha(OUT / 'panel_summary.csv'),
        'outputs': {p.name: sha(p) for p in dest.glob('*.csv')}
    }, indent=2) + '\n', encoding='utf-8')
    print(grid.to_string(index=False))
    print(pd.DataFrame(reconciliation).to_string(index=False))


if __name__ == '__main__':
    main()
