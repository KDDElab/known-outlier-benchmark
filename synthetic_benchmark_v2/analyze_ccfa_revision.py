#!/usr/bin/env python3
"""One estimator for repaired result tables, intervals, and paired comparisons.

Publication exports require every frozen condition. --partial is for execution
inspection only and writes a separate status, never the manuscript inputs.
"""
from __future__ import annotations
import argparse
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_ccfa_revision import HERE, OUT, MODELS, read, write, sha
from make_baselines_v2 import baseline_record
from annotate_v2 import parse_prediction

METRICS = ['support_top1_sufficient','support_top1_minimal','support_any_minimal_hit','support_best_jaccard',
           'rule_top1_sufficient','rule_top1_minimal','rule_any_minimal_hit','structurally_valid','answered',
           'top1_fraction','mean_candidate_fraction']
PRIMARY = ['support_top1_minimal','support_any_minimal_hit']
RANKERS = ['gower_lof_subspace','gower_knn_subspace']

def template_values(d, metrics):
    d=d.copy()
    for metric in metrics:d[metric]=pd.to_numeric(d[metric]).astype(float)
    return d.groupby(['generator_seed','template_id'])[metrics].mean().groupby('template_id').mean().astype(float)

def summarize(d, method, condition):
    # Conditional size/accuracy are ratios under the same template-weighted
    # sampling distribution. Missing size is not treated as a zero-size answer.
    x=d.copy()
    for m in ['top1_fraction','mean_candidate_fraction']:
        x[m+'_numerator']=x[m].fillna(0)
    names=[m for m in METRICS if m not in ['top1_fraction','mean_candidate_fraction']]+['top1_fraction_numerator','mean_candidate_fraction_numerator']
    t=template_values(x,names)
    rng=np.random.default_rng(20261107)
    draws=t.to_numpy()[rng.integers(0,len(t),(2000,len(t)))].mean(axis=1)
    estimate=t.mean().to_dict(); bounds=np.quantile(draws,[.025,.975],axis=0)
    rows=[{'method':method,'condition':condition,'metric':m,'estimate':estimate[m],'lo':bounds[0,i],'hi':bounds[1,i],
           'n_cases':len(d),'n_templates':len(t)} for i,m in enumerate(names) if not m.endswith('_numerator')]
    coverage_i=names.index('answered')
    for m,base in [('top1_fraction','top1_fraction_numerator'),('mean_candidate_fraction','mean_candidate_fraction_numerator'),
                   ('answered_support_sufficiency','support_top1_sufficient'),('answered_support_exact','support_top1_minimal')]:
        ratios=np.divide(draws[:,names.index(base)],draws[:,coverage_i],out=np.full(2000,np.nan),where=draws[:,coverage_i]>0)
        lo,hi=np.nanquantile(ratios,[.025,.975])
        rows.append({'method':method,'condition':condition,'metric':m,'estimate':estimate[base]/estimate['answered'] if estimate['answered'] else None,
                     'lo':lo,'hi':hi,'n_cases':len(d),'n_templates':len(t)})
    return rows

def holm(rows):
    ordered=sorted(range(len(rows)),key=lambda i:rows[i]['p_raw'])
    previous=0.
    for rank,i in enumerate(ordered):
        previous=max(previous,min(1.,rows[i]['p_raw']*(len(rows)-rank)))
        rows[i]['p_holm']=previous
    return rows

def paired(left, right, metric, name_left, name_right):
    merged=left[['generator_seed','template_id','source_case_id',metric]].merge(
        right[['generator_seed','template_id','source_case_id',metric]],on=['generator_seed','template_id','source_case_id'],suffixes=('_left','_right'),validate='one_to_one')
    assert len(merged)==len(left)==len(right)
    merged['delta']=merged[metric+'_left'].astype(float)-merged[metric+'_right'].astype(float)
    values=template_values(merged,['delta']).delta.to_numpy()
    rng=np.random.default_rng(20261107)
    observed=values.mean()
    null=(rng.choice([-1,1],size=(20000,len(values)))*values).mean(axis=1)
    p=(1+np.sum(np.abs(null)>=abs(observed)-1e-14))/20001
    boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(axis=1)
    lo,hi=np.quantile(boot,[.025,.975])
    return {'left':name_left,'right':name_right,'metric':metric,'difference':observed,'lo':lo,'hi':hi,'p_raw':p,'n_templates':len(values),'n_pairs':len(merged)}

def evaluate(method, predictions, inputs, truth):
    root=OUT/'evaluation'/method;root.mkdir(parents=True,exist_ok=True)
    path=root/'predictions.jsonl';write(path,predictions)
    with (root/'scoring.log').open('w') as log:
        subprocess.run([sys.executable,str(HERE/'evaluate_benchmark.py'),str(path),'--inputs-path',str(inputs),
                        '--ground-truth-path',str(truth),'--outdir',str(root)],check=True,stdout=log,stderr=subprocess.STDOUT)
    return pd.read_csv(root/'per_case.csv')

def random_metrics(p,g,d):
    candidates=[set(c['features']) for c in p['explanations']]
    result={'structurally_valid':1.,'answered':float(bool(candidates)),
            'top1_fraction':len(candidates[0])/d if candidates else np.nan,
            'mean_candidate_fraction':np.mean([len(c)/d for c in candidates]) if candidates else np.nan}
    for prefix,key in [('support','reference_support_sufficient_minimal_subspaces'),('rule','rule_minimal_subspaces')]:
        gold=[set(x) for x in g[key]]
        result[prefix+'_top1_sufficient']=float(bool(candidates) and any(s<=candidates[0] for s in gold))
        result[prefix+'_top1_minimal']=float(bool(candidates) and candidates[0] in gold)
        result[prefix+'_any_minimal_hit']=float(any(c in gold for c in candidates))
        if prefix=='support':result['support_best_jaccard']=max((len(c&s)/len(c|s) if c|s else 1. for c in candidates for s in gold),default=0.)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    inputs_path=OUT/'inputs_all_conditions.jsonl';gold_path=OUT/'ground_truth_all_conditions.jsonl'
    inputs=read(inputs_path);by_id={r['case_id']:r for r in inputs};gold={r['case_id']:r for r in read(gold_path)}
    metadata=pd.DataFrame([{k:r[k] for k in ['case_id','source_case_id','generator_seed','condition']} for r in inputs])
    statuses={};preds={};raw_records=[]
    for model in MODELS:
        rows=read(OUT/f'reused_{model}.jsonl')
        new=OUT/f'new_{model}.jsonl'
        if new.exists():rows+=read(new)
        assert len(rows)==len({r['case_id'] for r in rows}),f'duplicate {model} records'
        assert all(r['prompt_sha256']==by_id[r['case_id']]['prompt_sha256'] for r in rows)
        main_ids={cid for cid,x in by_id.items() if x['condition']=='main'}
        observed_ids={r['case_id'] for r in rows}
        statuses[model]={'observed':len(rows),'required':len(inputs),'complete':len(rows)==len(inputs),
                         'main_observed':len(observed_ids & main_ids),'main_required':len(main_ids),
                         'main_complete':main_ids.issubset(observed_ids)}
        # A partial export may inspect a fully completed primary comparison
        # while auxiliary conditions are still running. It never scores an
        # incomplete main cohort as though it were complete, and never writes
        # publication filenames or a complete analysis manifest.
        if not statuses[model]['complete'] and not (args.partial and statuses[model]['main_complete']):continue
        # Reparse raw text consistently, retaining original annotations on disk.
        # Full and minimal contracts use the same permissive first-object rule;
        # exact-key/JSON-only compliance is audited separately.
        for r in rows:
            c=by_id[r['case_id']]
            if r.get('raw_output') or not (r.get('parse_or_inference_error') or '').startswith('inference failed'):
                parsed,error=parse_prediction(r['raw_output'],set(c['feature_ids']),c.get('output_contract','full'))
                r={**r,'parsed_prediction':parsed,'parse_or_inference_error':error}
            r['method']=model
            raw_records.append(r)
        current=raw_records[-len(rows):]
        preds[model]=[{'case_id':r['case_id'],**(r.get('parsed_prediction') or {})} for r in current]
    sources={'gower_knn_subspace':OUT/'gower/predictions_gower_knn_subspace.jsonl',
             'gower_lof_subspace':OUT/'gower/predictions_gower_lof_subspace.jsonl',
             'sod_subspace_adaptation':OUT/'sod_cop/predictions_sod_subspace_adaptation.jsonl',
             'cop_error_vector_adaptation':OUT/'sod_cop/predictions_cop_error_vector_adaptation.jsonl',
             'rlax_reference_adaptation':OUT/'rlax/predictions_rlax_reference_adaptation.jsonl',
             'm2oe_choice':OUT/'m2oe/predictions_m2oe_choice.jsonl'}
    for method,path in sources.items():
        rows=read(path) if path.exists() else []
        expected=len(inputs) if method in RANKERS+['sod_subspace_adaptation','cop_error_vector_adaptation'] else 864
        statuses[method]={'observed':len(rows),'required':expected,'complete':len(rows)==expected}
        if len(rows)==expected:preds[method]=rows
    (OUT/'analysis_readiness.json').write_text(json.dumps(statuses,indent=2)+'\n')
    if not all(r['complete'] for r in statuses.values()) and not args.partial:
        raise RuntimeError('Frozen experiment queue incomplete; manuscript tables were not exported. See analysis_readiness.json')
    for method in ['all_features','top_univariate']:
        rows=[]
        for seed in (20261106,20261107,20261108):rows+=read(OUT/f'seed_{seed}/simple/predictions_{method}.jsonl')
        preds[method]=rows
    preds['first_three_display']=[baseline_record(x['case_id'],[x['feature_ids'][:3]],'Position-only diagnostic; no values or labels.') for x in inputs]
    frames=[]
    for method,rows in preds.items():
        d=evaluate(method,rows,inputs_path,gold_path)
        ids={r['case_id'] for r in rows};d=d[d.case_id.isin(ids)].merge(metadata,on='case_id',validate='one_to_one')
        d['method']=method;frames.append(d)
    random_rows=[]
    for seed in (20261106,20261107,20261108):
        accumulated={}
        for path in sorted((OUT/f'seed_{seed}/simple').glob('predictions_random_*.jsonl')):
            for p in read(path):
                cid=p['case_id'];accumulated.setdefault(cid,[]).append(random_metrics(p,gold[cid],by_id[cid]['feature_count']))
        for cid,values in accumulated.items():
            assert len(values)==100
            random_rows.append({'case_id':cid,'template_id':gold[cid]['template_id'],'method':'random_size_prior',
                                'task_label_gold':gold[cid]['task_label'],
                                **{k:by_id[cid][k] for k in ['source_case_id','generator_seed','condition','feature_count','reference_count','nuisance_profile']},
                                **pd.DataFrame(values).mean().to_dict()})
    frames.append(pd.DataFrame(random_rows))
    data=pd.concat(frames,ignore_index=True)
    # Preserve one numeric representation across boolean scorer outputs and
    # floating-point averages of random controls when CSVs are read back.
    for metric in METRICS:data[metric]=pd.to_numeric(data[metric]).astype(float)
    suffix='_partial' if args.partial else ''
    data.to_csv(OUT/f'all_scores{suffix}.csv',index=False)
    main=data[data.condition.eq('main')]
    summaries=[]
    for method,d in main.groupby('method'):
        if method != 'm2oe_choice':summaries+=summarize(d,method,'main')
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/f'main_summary{suffix}.csv',index=False)
    summary.pivot(index='method',columns='metric',values='estimate').to_csv(OUT/f'main_table{suffix}.csv')
    if args.partial:
        if all(statuses[model]['main_complete'] for model in MODELS):
            preview=[]
            for left,right in [*itertools.product(MODELS,RANKERS),*itertools.combinations(MODELS,2)]:
                for metric in PRIMARY:
                    preview.append(paired(main[main.method.eq(left)],main[main.method.eq(right)],metric,left,right))
            pd.DataFrame(holm(preview)).to_csv(OUT/'main_paired_tests_partial.csv',index=False)
        print(json.dumps(statuses,indent=2));return
    comparisons=[]
    for left,right in [*itertools.product(MODELS,RANKERS),*itertools.combinations(MODELS,2)]:
        for metric in PRIMARY:
            effect=paired(main[main.method.eq(left)],main[main.method.eq(right)],metric,left,right)
            table_difference=(summary[(summary.method==left)&(summary.metric==metric)].estimate.iloc[0]
                              -summary[(summary.method==right)&(summary.metric==metric)].estimate.iloc[0])
            assert np.isclose(effect['difference'],table_difference,atol=1e-12)
            comparisons.append(effect)
    pd.DataFrame(holm(comparisons)).to_csv(OUT/'main_paired_tests.csv',index=False)
    panel=set(json.loads((OUT/'protocol.json').read_text())['panel'])
    panel_data=data[data.generator_seed.eq(20261106)&data.source_case_id.isin(panel)]
    summaries=[];comparisons=[]
    for method in [*MODELS,*RANKERS,'first_three_display']:
        for condition in ['main','front','back','order1002','order1003','order1004','refs8','refs16','full1024','minimal512']:
            d=panel_data[panel_data.method.eq(method)&panel_data.condition.eq(condition)]
            if len(d)==96:summaries+=summarize(d,method,condition)
    pd.DataFrame(summaries).to_csv(OUT/'panel_summary.csv',index=False)
    # The two principal endpoints are tested as separate prespecified families
    # for position, reference amount, and output burden.
    for name,methods,pairs in [('order',MODELS,[('main','front'),('back','main')]),
                               ('references',MODELS,[('refs16','refs8'),('main','refs16'),('main','refs8')]),
                               ('output',MODELS,[('full1024','main'),('minimal512','main')])]:
        rows=[]
        for method,(left,right),metric in itertools.product(methods,pairs,PRIMARY):
            d=panel_data[panel_data.method.eq(method)]
            row=paired(d[d.condition.eq(left)],d[d.condition.eq(right)],metric,left,right);row['method']=method;rows.append(row)
        pd.DataFrame(holm(rows)).to_csv(OUT/f'{name}_paired_tests.csv',index=False)
    # Full output-state counts retain every inference attempt and cap hit.
    failures=[]
    for r in raw_records:
        c=by_id[r['case_id']];p=r.get('parsed_prediction');error=r.get('parse_or_inference_error') or ''
        state=('input_limit' if 'exceeding 16384' in error else 'inference_failure' if error.startswith('inference failed') else
               'no_json' if 'no JSON' in error else 'json_decode' if 'JSON decode' in error else 'schema_invalid' if p is None else
               'abstain' if p.get('abstain') else 'empty' if not p.get('explanations') or not any(x['features'] for x in p['explanations']) else 'answered')
        strict=False
        try:
            obj=json.loads(r['raw_output'].strip())
            if c.get('output_contract','full')=='minimal':
                strict=(p is not None and set(obj)=={'explanations','abstain'} and all(set(x)=={'features'} for x in obj['explanations']))
            else:
                strict=(p is not None and set(obj)=={'task_label','mechanism_feature_ids','explanations','abstain'} and
                        all(set(x)=={'features','evidence','confidence'} and len(x['evidence'].split())<=20 for x in obj['explanations']))
        except (ValueError,TypeError,KeyError):pass
        failures.append({'case_id':r['case_id'],'method':r['method'],'condition':c['condition'],'generator_seed':c['generator_seed'],
                         'response_status':state,'prompt_tokens':r.get('prompt_tokens'),'output_tokens':r.get('output_tokens'),
                         'strict_contract':strict,
                         # Post-hoc syntax diagnostic only: no repair or
                         # rescoring of the archived model output is performed.
                         'json_terminal_candidate_brace_pattern':bool(state=='json_decode' and re.search(r'\]\s*\]\s*,\s*"abstain"\s*:',r.get('raw_output',''))),
                         'output_cap':r.get('max_new_tokens',512),'cap_reached':r.get('output_tokens')==r.get('max_new_tokens',512),
                         'batch_size':r.get('batch_size',1),'reused':bool(r.get('reused_from'))})
    failure=pd.DataFrame(failures);failure.to_csv(OUT/'failure_per_case.csv',index=False)
    failure.groupby(['method','condition','response_status']).size().rename('n').to_csv(OUT/'failure_counts.csv')
    manifest={'complete':True,'n_frozen_conditions_per_model':1728,'n_main_cases':864,'n_templates':38,
              'estimator':'case mean within seed/template; available-seed mean within template; equal template mean',
              'interval':'2,000 template bootstrap replicates; conditional measures use numerator/coverage within each draw',
              'tests':'20,000 paired template sign flips, Holm main 18; order 12; references 18; output 12 as separate questions',
              'scorer_sha256':sha(HERE/'evaluate_benchmark.py'),'analysis_sha256':sha(__file__),'inputs_sha256':sha(inputs_path),
              'parser_sha256':sha(HERE/'annotate_v2.py'),'raw_annotation_sha256':{str(p):sha(p) for model in MODELS for p in [OUT/f'reused_{model}.jsonl',OUT/f'new_{model}.jsonl']},
              'gold_sha256':sha(gold_path),'table_sha256':sha(OUT/'main_table.csv'),'source_predictions':{k:sha(p) for k,p in sources.items()}}
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Complete repaired main and panel summaries written',flush=True)

if __name__=='__main__':main()
