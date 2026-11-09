"""Freeze matched support-localization comparisons before new inference.

No method output is read when choosing queries, prompts, caps or tie rules.
The public Car Evaluation table is an independent expert decision source,
not observational anomalies or human explanation annotations.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, csv, hashlib, itertools, json, random, shutil
from run_gower_subspace_baseline import extract_case_data

HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs/ccfa_unified_revision'
OLD = HERE / 'runs/ccfa_revision'
ACC = HERE / 'runs/ccfa_acceptance_revision'
PROTOCOL = 'known_outlier_unified_support_v1'
SEED = 20261109
SYSTEM = ('You localize explanations for an already-known anomalous table row. '
          'Use only the schema, normal reference rows, and target row. '
          'Do not claim real-world causality. Return one JSON object only.')
PROMPT = ('The target row is already known to be anomalous. Return up to four ranked '
          'candidate inclusion-minimal feature subspaces that distinguish the target from '
          'the normal population. Multiple valid subspaces may exist. Use only feature IDs '
          'from the schema. Return one JSON object with exactly explanations and abstain. '
          'explanations must be an ordered array of objects containing only features '
          '(an array of feature IDs without duplicates). abstain must be a boolean. '
          'If uncertain, abstain. Do not include analysis, deliberation, or a preamble. '
          'Example shape: {"explanations":[{"features":["F001"]}],"abstain":false}.')

def read(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, rows):
    p.write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows))

def minima(target, refs, cap):
    """Enumerate from visible values alone, with canonical size/ID order."""
    out = []
    for size in range(1, cap+1):
        for s in itertools.combinations(sorted(target), size):
            if any(set(a).issubset(s) for a in out):
                continue
            if not any(all(r[f] == target[f] for f in s) for r in refs):
                out.append(list(s))
    return out

def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--outdir',type=Path)
    args=p.parse_args()
    frozen=OUT
    if args.outdir is not None:OUT=args.outdir.resolve()
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'protocol.json').exists():
        raise RuntimeError('Protocol already frozen; do not overwrite.')
    if OUT!=frozen:
        shutil.copytree(frozen/'public_data',OUT/'public_data',dirs_exist_ok=True)
    inputs, golds = [], []
    def add(old, truth, track, cap, condition, group):
        c = copy.deepcopy(old)
        data = extract_case_data(old['user_prompt'])
        c.update(case_id='unified_'+old['case_id'], source_case_id=old['case_id'],
                 original_source_case_id=old.get('source_case_id',old['case_id']),
                 track=track, evidence_condition=condition, aggregation_group=group,
                 condition='unified_'+track+'_'+condition, candidate_size_cap=cap,
                 protocol_id=PROTOCOL, output_contract='minimal', max_new_tokens=512,
                 system_prompt=SYSTEM, split='test')
        bound = (f'Every valid minimal explanation in this condition has at most {cap} fields. '
                 f'Return at most four DISTINCT candidate subsets, each containing one to {cap} '
                 'distinct feature IDs. The statistical baselines use the same candidate space '
                 'and four-candidate budget. This information gives no field identities or labels.')
        complete = ('\nThe following reference rows enumerate the COMPLETE normal population '
                    'over the declared finite domains; there are no other normal rows.'
                    if condition == 'complete' else '')
        c['user_prompt'] = PROMPT+'\n\n'+bound+complete+'\n\n'+json.dumps(data,indent=2)
        c['prompt_sha256'] = hashlib.sha256((SYSTEM+'\n'+c['user_prompt']).encode()).hexdigest()
        g = copy.deepcopy(truth)
        if track == 'new_rules':
            # The source new-rule file uses reference_support_* for FINITE
            # sample minima. Its rule_minimal_subspaces are the complete-
            # population labels (clean population is exactly rule complement).
            g['sample_support_minimal_subspaces'] = copy.deepcopy(g['reference_support_sufficient_minimal_subspaces'])
            g['reference_support_sufficient_minimal_subspaces'] = copy.deepcopy(g['rule_minimal_subspaces'])
        g.update(case_id=c['case_id'], source_case_id=old['case_id'], track=track,
                 evidence_condition=condition, aggregation_group=group)
        assert all(len(s)<=cap for s in g['reference_support_sufficient_minimal_subspaces'])
        assert data['target_row'] == g['target_row']
        inputs.append(c); golds.append(g)

    old_inputs={c['case_id']:c for c in read(OLD/'inputs_all_conditions.jsonl')}
    old_gold={c['case_id']:c for c in read(OLD/'ground_truth_all_conditions.jsonl')}
    for c in read(ACC/'inputs_prior3.jsonl'):
        cid=c['paired_case_id']; g=old_gold[cid]
        add(old_inputs[cid],g,'paired96',3,'sample',g['template_id'])
    ng={g['case_id']:g for g in read(ACC/'ground_truth_new_rules.jsonl')}
    for c in read(ACC/'inputs_new_rules.jsonl'):
        g=ng[c['case_id']]
        add(c,g,'new_rules',6,g['condition'],g['template_id'])
    inv=HERE/'runs/ccfa_constraint_transfer'
    ig={g['case_id']:g for g in read(inv/'ground_truth.jsonl')}
    for c in read(inv/'inputs.jsonl'):
        g=ig[c['case_id']]; add(c,g,'invoices',8,'sample',g['template_id'])

    # UCI states that its 1,728 rows completely cover the six-attribute space.
    raw=list(csv.reader((OUT/'public_data/car.data').open()))
    assert len(raw)==1728 and len({tuple(r[:6]) for r in raw})==1728
    domains=[sorted({r[j] for r in raw}) for j in range(6)]
    assert set(itertools.product(*domains))=={tuple(r[:6]) for r in raw}
    normals=[i for i,r in enumerate(raw) if r[-1]!='unacc']
    assert len(normals)==518
    rng=random.Random(SEED)
    selected=[]
    for value in ['vhigh','high','med','low']:
        pool=[i for i,r in enumerate(raw) if r[0]==value and r[-1]=='unacc']
        selected.extend(rng.sample(pool,12))
    names=['buying','maint','doors','persons','lug_boot','safety']
    car_pop=[]
    for j,source_idx in enumerate(selected):
        perm=list(range(6));rng.shuffle(perm)
        mapping={k:f'F{perm[k]+1:03d}' for k in range(6)}
        order=list(range(6));rng.shuffle(order)
        def row(r):return {mapping[k]:r[k] for k in order}
        population=[row(raw[i]) for i in normals]
        refs=rng.sample(normals,32);target=row(raw[source_idx])
        truth=minima(target,population,6)
        assert truth and target not in population
        data={'feature_schema':[{'feature_id':mapping[k],'value_type':'categorical',
               'allowed_values':domains[k]} for k in order],
              'normal_reference_rows':[row(raw[i]) for i in refs], 'target_row':target}
        cid=f'car_decision_{j:03d}'
        c={'case_id':cid,'feature_ids':list(target),'feature_count':6,'reference_count':32,
           'generator_seed':SEED,'field_name_mode':'opaque','nuisance_profile':'none',
           'user_prompt':json.dumps(data,indent=2)}
        g={'case_id':cid,'target_row':target,'normal_reference_rows':data['normal_reference_rows'],
           'reference_support_sufficient_minimal_subspaces':truth,'rule_minimal_subspaces':truth,
           'template_id':'car_decision','source_row_index':source_idx,'reference_indices':refs,
           'feature_names':{mapping[k]:names[k] for k in range(6)},'population_count':518,
           'label_scope':'Acceptable expert decision configurations: class in acc/good/vgood; '
                         'unacc target flagged inadmissible. Not observational anomaly labels.'}
        add(c,g,'car_decision',6,'sample',cid)
        car_pop.append({'case_id':'unified_'+cid,'normal_rows':population})
    write(OUT/'inputs.jsonl',inputs);write(OUT/'ground_truth.jsonl',golds)
    write(OUT/'car_populations.jsonl',car_pop)
    protocol={'protocol_id':PROTOCOL,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
      'status':'Frozen before new outputs; requested revision after inspecting earlier results, not preregistration.',
      'models':['qwen35_9b','deepseek_v2_lite_chat','llama31_8b_instruct'],
      'counts':{'paired96':96,'new_rules':120,'invoices':96,'car_decision':48},
      'core_task':'Rank at most four distinct nonempty inclusion-minimal support-exclusion sets; abstention allowed; no auxiliary tasks.',
      'output':'Fixed minimal JSON schema with legal IDs and per-track size cap; strict unique IDs; duplicate sets collapse in order; failures/abstentions remain misses.',
      'decoding':'Original checkpoints/loader/chat templates; greedy batch1; 16384 input, no truncation; 512 output; LMFE0.11.3.',
      'candidate_caps':{'paired96':3,'new_rules':6,'invoices':8,'car_decision':6},
      'sample_minimal':'All sample-absent inclusion-minimal sets within cap, ranked by cardinality then canonical feature-ID tuple; first four. No hidden truth, frequency fitting or outcome selection.',
      'statistics':'Exact@1/Hit@4/Answered, all queries. Equal template means for paired96 and new rules; equal four-relation means for invoices; case mean for one Car source. SD at declared aggregation unit. 2000 paired template bootstraps for paired96; descriptive transfer results; no pooling across tracks.',
      'lof_diagnostics':{'queries':'same96 invoices,255 candidates,k3',
        'epsilon':[1e-6,1e-9,1e-12],
        'neighbors':['original fixed-k argpartition','fixed-k canonical ascending','fixed-k canonical descending','include all k-distance boundary ties'],
        'duplicates':'Additional collapse of identical projections with inclusive boundaries; explicitly changes empirical weights.',
        'ranking':'descending score, size, canonical IDs; unchanged across variants'},
      'car_source':{'url':'https://archive.ics.uci.edu/dataset/19/car+evaluation',
        'download':'https://archive.ics.uci.edu/static/public/19/car+evaluation.zip',
        'doi':'10.24432/C5JP48','license':'CC BY 4.0','n_domain':1728,'n_normal':518,
        'selection':'12 unacc targets per buying category, seed20261109, before inference; opaque IDs and published categorical values;32 acceptable refs without replacement.',
        'boundary':'Independent public expert decision model, not real incidents, and pretraining exposure cannot be excluded.'},
      'inputs_sha256':sha(OUT/'inputs.jsonl'),'ground_truth_sha256':sha(OUT/'ground_truth.jsonl'),
      'source_hashes':{str(p.relative_to(HERE)):sha(p) for p in [Path(__file__),HERE/'annotate_v2.py',HERE/'annotate_unified_support.py',HERE/'run_gower_subspace_baseline.py',frozen/'public_data/car.data']}}
    assert len(inputs)==360 and len({c['case_id'] for c in inputs})==360
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'frozen':protocol['frozen_at_utc'],'queries_per_model':len(inputs),'counts':protocol['counts']}))

if __name__=='__main__':main()
