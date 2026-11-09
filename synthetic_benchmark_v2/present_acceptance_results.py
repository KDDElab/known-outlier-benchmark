"""Presentation statistics from frozen records; never runs model inference.

Means retain the published estimators. SD denotes between-unit heterogeneity,
with n-1 correction, not a standard error or a bootstrap interval. The original
bootstrap intervals and paired tests are retained in their original files.
"""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs/ccfa_acceptance_revision'
OLD = HERE / 'runs/ccfa_revision'
INV = HERE / 'runs/ccfa_constraint_transfer'
MODELS = ['qwen35_9b', 'deepseek_v2_lite_chat', 'llama31_8b_instruct']
RANKERS = ['gower_lof_subspace', 'gower_knn_subspace']
METHODS = MODELS + RANKERS
NAMES = dict(zip(METHODS, ['Qwen', 'DeepSeek', 'Llama', 'Gower-LOF', 'Gower-kNN']))
MAIN_METHODS = METHODS + ['rlax_reference_adaptation', 'sod_subspace_adaptation',
    'cop_error_vector_adaptation', 'top_univariate', 'random_size_prior',
    'all_features', 'first_three_display']
MAIN_NAMES = ['Qwen3.5-9B', 'DeepSeek-V2-Lite-Chat', 'Llama-3.1-8B-Instruct',
    'Gower-LOF', 'Gower-kNN', 'RLAX adaptation', 'SOD adaptation',
    'COP adaptation', 'Univariate top four', 'Random size prior', 'All fields',
    'First three positions']
MAIN_METRICS = ['support_top1_sufficient', 'support_top1_minimal',
    'support_any_minimal_hit', 'support_best_jaccard', 'rule_top1_minimal',
    'rule_any_minimal_hit', 'top1_fraction', 'answered']


def fmt(value):
    return str(Decimal(str(round(float(value), 12))).quantize(
        Decimal('0.001'), rounding=ROUND_HALF_UP))


def row(*values):
    return ' & '.join(values) + r' \\' + '\n'


def main():
    assert json.loads((OUT / 'summary.json').read_text())['complete']
    stats, units, sources = [], [], {}

    def load(path):
        sources[str(path.relative_to(HERE.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path, low_memory=False)

    def collect(track, method, condition, metric, values, unit_name, weights=None):
        v = values.astype(float)
        w = pd.Series(1., index=v.index) if weights is None else weights.astype(float)
        mask = (w > 0) & v.notna()
        v, w = v[mask], w[mask]
        assert len(v) >= 2 and np.isfinite(v).all()
        mean = float(np.average(v, weights=w))
        denom = w.sum() - (w * w).sum() / w.sum()
        sd = float(np.sqrt(np.sum(w * (v - mean)**2) / denom))
        key = dict(track=track, method=method, condition=condition, metric=metric)
        result = {**key, 'mean': mean, 'sd': sd, 'n_units': len(v),
                  'unit': unit_name, 'weighting': 'equal' if weights is None else 'answer_coverage'}
        stats.append(result)
        for name, value in v.items():
            units.append({**key, 'unit_id': str(name), 'value': float(value), 'weight': float(w.loc[name])})
        return result

    def cell(result, bold=False):
        text = fmt(result['mean']) + r'\!\pm\!' + fmt(result['sd'])
        return '$' + (r'\mathbf{' + text + '}' if bold else text) + '$'

    def write_table(name, caption, label, columns, body):
        text = ('\\begin{table*}[t]\n\\centering\\scriptsize\n\\caption{' + caption + '}\n'
            '\\setlength{\\tabcolsep}{2pt}\n\\renewcommand{\\arraystretch}{1.16}\n'
            '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}|' + columns + '|}\n\\hline\n'
            + body + '\\hline\n\\end{tabular*}\n\\renewcommand{\\arraystretch}{1.0}\n'
            '\\label{' + label + '}\n\\end{table*}\n')
        (OUT / name).write_text(text, encoding='utf-8')

    # Original estimator: cases within seed/template, seeds within template,
    # then equal template weights. Answered-only size retains its ratio form.
    data = load(OLD / 'all_scores.csv')
    data = data[data.condition == 'main'].copy()
    original = load(OLD / 'main_summary.csv').set_index(['method', 'metric'])
    lookup = {}
    for method in MAIN_METHODS:
        d = data[data.method == method].copy()
        assert len(d) == 864
        d['size_numerator'] = d.top1_fraction.fillna(0)
        cols = [c for c in MAIN_METRICS if c != 'top1_fraction'] + ['size_numerator']
        t = d.groupby(['generator_seed', 'template_id'])[cols].mean().groupby('template_id').mean()
        assert len(t) == 38
        for metric in MAIN_METRICS:
            if metric == 'top1_fraction':
                result = collect('main', method, 'main', metric,
                    t.size_numerator / t.answered, 'template', t.answered)
            else:
                result = collect('main', method, 'main', metric, t[metric], 'template')
            assert abs(result['mean'] - original.loc[(method, metric), 'estimate']) < 1e-12
            lookup[method, metric] = result
    best = {metric: max(lookup[m, metric]['mean'] for m in MAIN_METHODS[:8]) for metric in MAIN_METRICS[:6]}
    body = row(r'\multicolumn{2}{|c|}{Methods}', r'\multicolumn{4}{c|}{Population-support labels ($\uparrow$)}',
        r'\multicolumn{2}{c|}{Rule labels ($\uparrow$)}', r'\multicolumn{2}{c|}{Output behavior}')
    body += r'\cline{1-10}' + '\n' + row('Group', 'Method', 'Suff.@1', 'Exact@1', 'Hit@4', 'Jaccard@4',
        'Exact@1', 'Hit@4', r'\shortstack{Top-one\\fraction}', 'Answered') + '\\hline\n'
    groups = {0: (3, 'LMs'), 3: (3, 'Rankers'), 6: (2, 'Adapt.'), 8: (4, 'Controls')}
    for i, (method, name) in enumerate(zip(MAIN_METHODS, MAIN_NAMES)):
        if i in groups:
            if i: body += '\\hline\n'
            count, label = groups[i]
            group = r'\multirow{' + str(count) + r'}{*}{\scriptsize ' + label + '}'
        else: group = ''
        body += row(group, name, *[cell(lookup[method, metric], i < 8 and metric in best and
            abs(lookup[method, metric]['mean'] - best[metric]) < 1e-12) for metric in MAIN_METRICS])
    write_table('presentation_main_table.tex',
        r'Original 864-case comparison: mean $\pm$ SD across 38 templates. SD describes template heterogeneity; Figure~\ref{fig:pareto} retains 95\% bootstrap intervals. Answered-only size uses coverage-weighted template means and SD. Gower has the original three-field prior; Figure~\ref{fig:prior} aligns it. Failures remain in recovery denominators. Bold marks the best non-control mean; M\textsuperscript{2}OE is assessed separately.',
        'tab:main', 'c|l|rrrr|rr|rr', body)

    # Matched interventions for the replacement figures.
    for track, filename, metrics in [('prior', 'prior_per_case.csv', ['exact', 'hit4', 'answered']),
            ('structured', 'structured_per_case.csv', ['valid', 'exact', 'hit4'])]:
        d = load(OUT / filename)
        for (method, condition), g in d.groupby(['method', 'condition']):
            for metric in metrics:
                collect(track, method, condition, metric, g.groupby('template')[metric].mean(), 'template')

    new = load(OUT / 'new_rules_per_case.csv')
    targets = set(new.loc[new.condition == 'complete', 'target_id'])
    new_lookup = {}
    for method in METHODS:
        for condition in ['new48', 'draw0', 'draw1', 'draw2', 'complete']:
            g = new[(new.method == method) & (new.condition == ('draw0' if condition == 'new48' else condition))]
            if condition != 'new48': g = g[g.target_id.isin(targets)]
            assert len(g) == (48 if condition == 'new48' else 24)
            for metric in ['exact', 'hit4']:
                new_lookup[method, condition, metric] = collect('new_rules', method, condition, metric,
                    g.groupby('template')[metric].mean(), 'composition')
    body = row('Method', r'\multicolumn{2}{c|}{New targets (48)}',
        r'\multicolumn{3}{c|}{Repeated draws (same 24)}', r'\multicolumn{2}{c|}{Complete population (24)}', 'Stable')
    body += '\\cline{2-8}\n' + row('', 'Exact', 'Hit@4', 'Draw 0 Exact', 'Draw 1 Exact', 'Draw 2 Exact', 'Exact', 'Hit@4', '/ 24') + '\\hline\n'
    stability = {x['method']: x['all_three_available_same'] for x in json.loads((OUT/'summary.json').read_text())['reference_stability']}
    for method in METHODS:
        fields = [('new48', 'exact'), ('new48', 'hit4'), ('draw0', 'exact'), ('draw1', 'exact'),
            ('draw2', 'exact'), ('complete', 'exact'), ('complete', 'hit4')]
        body += row(NAMES[method], *[cell(new_lookup[method, c, m]) for c, m in fields], str(stability[method]))
    write_table('presentation_new_rules_table.tex',
        r'New-rule validation: mean $\pm$ sample SD across 12 compositions. Gower searches all 63 subsets of six fields. Draws 0--2 independently sample 32 normal rows; Complete supplies every normal tuple and declares completeness. Stable is a target count with an available, identical first set across draws.',
        'tab:new-rules', 'l|rr|rrr|rr|r', body)

    real = load(OUT / 'real_cells_per_case.csv')
    real_methods = [m + suffix for m in MODELS for suffix in ['', ':schema']] + RANKERS + ['uniform_random_field']
    body = row('Method', r'\multicolumn{3}{c|}{FieldHit@1 by source (24 each)}', r'\multicolumn{4}{c|}{Equal-source mean (72)}')
    body += '\\cline{2-8}\n' + row('', 'Flights', 'Beers', 'Rayyan', 'FieldHit@1', 'FieldHit@3', 'Recall@3', 'Inspections') + '\\hline\n'
    for method in real_methods:
        d = real[real.method == method]
        assert len(d) == 72
        cells = []
        for dataset in ['flights', 'beers', 'rayyan']:
            g = d[d.dataset == dataset].set_index('case_id')
            cells.append(cell(collect('real_cells_source', method, dataset, 'field_hit1', g.field_hit1, 'case')))
        for metric in ['field_hit1', 'field_hit3', 'recall_at3', 'first_correction_inspections']:
            cells.append(cell(collect('real_cells', method, 'equal_source', metric, d.groupby('dataset')[metric].mean(), 'source')))
        label = 'Uniform (expected)' if method == 'uniform_random_field' else NAMES[method.split(':')[0]] + (' (schema)' if ':schema' in method else '')
        body += row(label, *cells)
    write_table('presentation_real_cells_table.tex',
        r'Public corrected-field inspection, mean $\pm$ SD. Source columns use 24 case scores; aggregate columns use three source means. Schema fixes singleton JSON decoding. Recall@3 covers recorded corrections. Inspections gives the first corrected field position within four suggestions; failure receives $d+1$ (6/9/10 fields). Retrieval failures score zero. Uniform entries describe analytic expectations and their heterogeneity.',
        'tab:real-cells', 'l|rrr|rrrr', body)

    invoice = load(INV / 'scores_per_case.csv')
    for method, d in invoice.groupby('method'):
        assert len(d) == 96
        for metric in ['support_top1_minimal', 'support_any_minimal_hit', 'support_best_jaccard', 'answered', 'direct_edit_exact_top1']:
            collect('invoice', method, 'original', metric, d.groupby('template_id')[metric].mean(), 'relation')

    pd.DataFrame(stats).to_csv(OUT / 'presentation_statistics.csv', index=False)
    pd.DataFrame(units).to_csv(OUT / 'presentation_group_values.csv', index=False)
    manifest = {'source_sha256': sources, 'source_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sd_definition': 'Sample SD across the named aggregation units; coverage-weighted unbiased SD for conditional main first-set size.',
        'sd_is_run_to_run_uncertainty': False, 'original_mean_estimators_preserved': True,
        'original_bootstrap_intervals_and_tests_preserved': True, 'new_inference': False,
        'main_table_numeric_cells': 96, 'new_rule_numeric_cells': 35, 'real_cell_numeric_cells': 63}
    (OUT / 'presentation_statistics_provenance.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print('Published estimators preserved; generated 194 mean +/- SD table entries.')
    print('Qwen Suff@1:', cell(lookup['qwen35_9b', 'support_top1_sufficient']))


if __name__ == '__main__':
    main()
