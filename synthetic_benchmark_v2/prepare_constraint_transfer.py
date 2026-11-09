#!/usr/bin/env python3
"""Freeze a standards-derived arithmetic workload before model evaluation.

These are synthetic finite-domain invoices, not observed invoices or a complete
Peppol conformance suite. Normal support is enumerated from four public amount
relations. Independent distractors are marginalized in exact support labels.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import random
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs' / 'ccfa_constraint_transfer'
SEED = 47017
DISPLAY_SEED = 8309
N = 24
REFERENCES = 32
WIDTH = 8
RULES = [
    ('line_net', 'PEPPOL-EN16931-R120',
     'https://docs.peppol.eu/poacc/billing/3.0/rules/ubl-peppol/PEPPOL-EN16931-R120/',
     ['quantity', 'unit_price', 'line_charge', 'line_allowance', 'line_net'],
     [list(range(1, 5)), list(range(2, 6)), [0, 1, 2], [0, 1, 2]],
     lambda q, p, c, a: q * p + c - a),
    ('document_net', 'BR-CO-13',
     'https://docs.peppol.eu/poacc/billing/3.0/rules/ubl-tc434/BR-CO-13/',
     ['line_net_1', 'line_net_2', 'document_charge', 'document_allowance', 'tax_exclusive'],
     [[10, 15, 20, 25], [5, 10, 15, 20], [0, 1, 2], [0, 1, 2]],
     lambda l1, l2, c, a: l1 + l2 + c - a),
    ('tax_total', 'BR-CO-15',
     'https://docs.peppol.eu/poacc/billing/3.0/rules/ubl-tc434/BR-CO-15/',
     ['tax_exclusive', 'tax_amount', 'tax_inclusive'],
     [[10, 15, 20, 25], [0, 1, 2, 3, 4]],
     lambda net, tax: net + tax),
    ('amount_due', 'BR-CO-16',
     'https://docs.peppol.eu/poacc/billing/3.0/rules/ubl-tc434/BR-CO-16/',
     ['tax_inclusive', 'paid', 'rounding', 'amount_due'],
     [[20, 25, 30, 35], [0, 5, 10, 15], [-1, 0, 1]],
     lambda total, paid, rounding: total - paid + rounding),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, rows):
    Path(path).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), encoding='utf-8')


def freeze(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise RuntimeError('Frozen protocol already exists; do not silently replace evaluation inputs.')
    # Preserve the established semantic request and auxiliary fields exactly.
    reference_path = HERE / 'runs' / 'ccfa_revision' / 'inputs_main.jsonl'
    reference = json.loads(reference_path.read_text(encoding='utf-8').splitlines()[0])
    loc = reference['user_prompt'].find('"feature_schema"')
    prefix = reference['user_prompt'][:reference['user_prompt'].rfind('{', 0, loc)]
    rng = random.Random(SEED)
    display = random.Random(DISPLAY_SEED)
    inputs, labels, populations = [], [], []
    for family, rule_id, url, names, free_domains, relation in RULES:
        population = [tuple([*x, relation(*x)]) for x in itertools.product(*free_domains)]
        popset = set(population)
        domains = [sorted({x[j] for x in population}) for j in range(len(names))]
        # Enumerate possible single-field edits using values that occur in the
        # clean population. Thus no anomalous target has a marginally absent value.
        mutations = []
        for source_i, source in enumerate(population):
            for j, domain in enumerate(domains):
                for value in domain:
                    if value == source[j]:
                        continue
                    target = tuple(value if k == j else source[k] for k in range(len(names)))
                    if target not in popset:
                        mutations.append((source_i, j, target))
        rng.shuffle(mutations)
        targets, seen = [], set()
        for item in mutations:
            if item[2] not in seen:
                seen.add(item[2]); targets.append(item)
            if len(targets) == N:
                break
        assert len(targets) == N
        populations.append({'family': family, 'standard_rule': rule_id, 'source_url': url,
                            'role_names': names, 'domains': domains, 'clean_role_rows': population})
        for index, (source_i, edited, target) in enumerate(targets):
            cid = f'constraint_{family}_{index:03d}'
            role_ids = [f'F{x:03d}' for x in range(1, WIDTH + 1)]
            rng.shuffle(role_ids)
            mapping = dict(zip(names, role_ids))
            schema = {fid: {'feature_id': fid, 'value_type': 'numeric', 'allowed_values': domains[j]}
                      for j, fid in enumerate(role_ids[:len(names)])}
            for j, fid in enumerate(role_ids[len(names):]):
                schema[fid] = {'feature_id': fid, 'value_type': 'categorical',
                               'allowed_values': ['a', 'b', 'c', 'd']}
            row = {role_ids[j]: v for j, v in enumerate(target)}
            before = {role_ids[j]: v for j, v in enumerate(population[source_i])}
            for fid in role_ids[len(names):]:
                row[fid] = before[fid] = rng.choice(schema[fid]['allowed_values'])
            reference_indices = [rng.randrange(len(population)) for _ in range(REFERENCES)]
            refs = []
            for ri in reference_indices:
                ref = {role_ids[j]: v for j, v in enumerate(population[ri])}
                for fid in role_ids[len(names):]:
                    ref[fid] = rng.choice(schema[fid]['allowed_values'])
                refs.append(ref)
            sufficient, minimal = [], []
            for size in range(1, len(names) + 1):
                for subset in itertools.combinations(range(len(names)), size):
                    if not any(all(x[j] == target[j] for j in subset) for x in population):
                        fields = sorted(role_ids[j] for j in subset)
                        sufficient.append(fields)
                        if not any(set(s) < set(fields) for s in sufficient):
                            minimal.append(fields)
            assert minimal and all(len(s) >= 2 for s in minimal)
            order = sorted(role_ids); display.shuffle(order)
            payload = {'feature_schema': [schema[fid] for fid in order],
                       'normal_reference_rows': [{fid: r[fid] for fid in order} for r in refs],
                       'target_row': {fid: row[fid] for fid in order}}
            prompt = prefix + json.dumps(payload, ensure_ascii=False, indent=2)
            prompt_hash = hashlib.sha256((reference['system_prompt'] + '\0' + prompt).encode()).hexdigest()
            inputs.append({'case_id': cid, 'source_case_id': cid, 'split': 'test',
                           'feature_count': WIDTH, 'feature_ids': order, 'reference_count': REFERENCES,
                           'generator_seed': SEED, 'display_order_seed': DISPLAY_SEED,
                           'condition': 'constraint_transfer', 'output_contract': 'full', 'max_new_tokens': 512,
                           'field_name_mode': 'opaque', 'nuisance_profile': 'independent_categorical',
                           'system_prompt': reference['system_prompt'], 'user_prompt': prompt,
                           'prompt_sha256': prompt_hash})
            labels.append({'case_id': cid, 'template_id': family, 'template_group_id': family,
                           'task_label': 'relational', 'split': 'test', 'generator_seed': SEED,
                           'rule_role_to_feature': mapping, 'standard_rule': rule_id,
                           'reference_indices': reference_indices, 'source_population_index': source_i,
                           'modified_feature_ids': [role_ids[edited]],
                           'value_different_from_baseline_feature_ids': [role_ids[edited]],
                           'mechanism_feature_ids': sorted(role_ids[:len(names)]),
                           'rule_minimal_subspaces': minimal, 'rule_sufficient_subspaces': sufficient,
                           'reference_support_sufficient_minimal_subspaces': minimal,
                           'reference_support_sufficient_subspaces': sufficient,
                           'feature_count': WIDTH, 'reference_count': REFERENCES,
                           'nuisance_profile': 'independent_categorical', 'field_name_mode': 'opaque',
                           'before_row_hidden': before, 'target_row': payload['target_row'],
                           'normal_reference_rows': payload['normal_reference_rows']})
    write(out / 'inputs.jsonl', inputs)
    write(out / 'ground_truth.jsonl', labels)
    write(out / 'populations.jsonl', populations)
    protocol = {
        'protocol_id': 'known_outlier_standard_constraints_v1',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Test complete support-minimal evaluation beyond the original rule generator, using four externally specified arithmetic relations.',
        'source': 'Peppol BIS Billing 3.0, May 2026 release; four rule URLs recorded per population.',
        'scope': 'Synthetic finite-domain amounts; base quantity fixed to one; two line amounts in document-net relation; optional amounts present; integer units avoid rounding ambiguity. This is not natural invoice data or complete Peppol conformance.',
        'target_sampling': 'Shuffle all single-field edits of clean rows using values in the corresponding clean marginal; take first 24 distinct invalid targets per relation. No model outputs used.',
        'reference_sampling': '32 iid clean role assignments with replacement; distractors independently sampled.',
        'gold': 'Enumerate every projection of the complete clean relation; marginalize independent distractors. Rule anomaly is the complement of that relation, so rule and support labels coincide.',
        'edit_provenance': 'Exactly one recorded field was changed; this is scored separately and is not assumed identifiable from unpaired input.',
        'n_cases': len(inputs), 'n_relations': len(populations), 'cases_per_relation': N,
        'fields': WIDTH, 'references': REFERENCES, 'data_seed': SEED, 'display_seed': DISPLAY_SEED,
        'models': ['qwen35_9b', 'deepseek_v2_lite_chat', 'llama31_8b_instruct'],
        'model_protocol': 'Unchanged full main prompt prefix, greedy batch one, 16384 input cap, 512 output cap; no tuning or retries conditional on correctness.',
        'baselines': 'Gower-kNN and Gower-LOF use the existing implementation, enumerate all 255 nonempty subsets of the eight visible fields and return four candidates. No hidden size cap.',
        'metrics': ['support_exact_at_1', 'support_hit_at_4', 'support_best_jaccard_at_4', 'answered', 'direct_edit_exact_at_1'],
        'aggregation': 'Equal case means within each of four relations, then equal relation mean. Descriptive fixed-workload comparison; no new significance family or unseen-domain confidence claim.',
        'frozen_inputs_sha256': sha(out / 'inputs.jsonl'), 'frozen_gold_sha256': sha(out / 'ground_truth.jsonl'),
        'frozen_populations_sha256': sha(out / 'populations.jsonl'), 'generator_sha256': sha(__file__),
        'prompt_source_sha256': sha(reference_path),
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'n_cases': len(inputs), 'populations': {p['family']: len(p['clean_role_rows']) for p in populations},
                      'minimal_sizes': sorted({len(s) for g in labels for s in g['reference_support_sufficient_minimal_subspaces']}),
                      'cases_with_gold_over_three': sum(any(len(s) > 3 for s in g['reference_support_sufficient_minimal_subspaces']) for g in labels)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=OUT)
    freeze(parser.parse_args().outdir)
