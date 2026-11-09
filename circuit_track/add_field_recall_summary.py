#!/usr/bin/env python3
"""Add field-level recall and subset-size diagnostics to the circuit pilot."""
from __future__ import annotations
import json, random
from collections import defaultdict
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_zero_noise_pilot'
MODELS = ('qwen35_9b','deepseek_v2_lite_chat','llama31_8b_instruct')
BASELINES = ('gower_knn','gower_lof','conditional_output_surprisal','joint_output_surprisal')
def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def candidate_sets(pred):
    if not isinstance(pred, dict): return []
    xs=pred.get('explanations')
    if not isinstance(xs,list): return []
    return [set(x.get('features',[])) for x in xs if isinstance(x,dict) and isinstance(x.get('features'),list)]
def main():
    truth={x['case_id']:set(x['gold_features']) for x in rows(HERE/'ground_truth_zero_noise_pilot.jsonl')}
    pred_sources={}
    for model in MODELS:
        latest={}
        for x in rows(HERE/f'annotations_{model}.jsonl'):
            if x.get('event')=='annotation' and x.get('protocol_id')=='harvard_circuit_zero_noise_output_specific_pilot_v1': latest[x['case_id']]=x
        if set(latest)!=set(truth): raise RuntimeError(f'{model}: incomplete annotations')
        pred_sources[model]={k:candidate_sets(v.get('parsed_prediction')) for k,v in latest.items()}
    for name in BASELINES:
        pred_sources[name]={x['case_id']:candidate_sets(x) for x in rows(HERE/'baseline_predictions'/f'predictions_{name}.jsonl')}
    base=pd.read_csv(OUT/'per_case.csv').drop_duplicates(['method','case_id']).set_index(['method','case_id'])
    per=[]
    for method,source in pred_sources.items():
        for case_id,gold in truth.items():
            candidates=source.get(case_id,[])
            top=candidates[0] if candidates else set()
            union=set().union(*candidates[:4]) if candidates else set()
            scored=base.loc[(method,case_id)]
            per.append({'method':method,'case_id':case_id,'source_table':str(scored['source_table']),
                'valid':int(bool(scored['valid'])),'candidate_list_n':len(candidates),
                'top1_subset_size':len(top),'gold_subset_size':len(gold),
                'top1_field_recall':len(top&gold)/len(gold) if gold else 0.0,
                'field_recall_at_4':len(union&gold)/len(gold) if gold else 0.0})
    frame=pd.DataFrame(per)
    frame.to_csv(OUT/'field_recall_per_case.csv',index=False)
    import numpy as np
    result=[]
    rng=np.random.default_rng(20261106)
    metric_names=['top1_field_recall','field_recall_at_4']
    for method,group in frame.groupby('method'):
        cluster_groups=[g for _,g in sorted(group.groupby('source_table'))]
        summaries={m:float(group[m].mean()) for m in ['valid','candidate_list_n','top1_subset_size','gold_subset_size',*metric_names]}
        counts=np.array([len(g) for g in cluster_groups],dtype=float)
        for metric in metric_names:
            sums=np.array([g[metric].sum() for g in cluster_groups],dtype=float)
            draw_ids=rng.integers(0,len(cluster_groups),size=(20000,len(cluster_groups)))
            draws=sums[draw_ids].sum(axis=1)/counts[draw_ids].sum(axis=1)
            lo,hi=np.quantile(draws,[.025,.975])
            summaries[metric+'_ci_low']=float(lo); summaries[metric+'_ci_high']=float(hi)
        result.append({'method':method,**summaries})
    out=pd.DataFrame(result).sort_values('method')
    out.to_csv(OUT/'field_recall_summary.csv',index=False)
    print(out.to_string(index=False))
if __name__=='__main__': main()
