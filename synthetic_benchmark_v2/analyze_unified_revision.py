"""Score frozen joint-configuration runs; refuse incomplete publication exports."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_unified_revision import OUT,HERE,read
from analyze_acceptance_revision import metrics,parsed,holm

MODELS=['qwen35_9b','deepseek_v2_lite_chat','llama31_8b_instruct']
BASELINES=['sample_minimal','gower_knn','gower_lof']

def summarize(df):
    rows=[]
    for (track,condition,method),d in df.groupby(['track','evidence_condition','method']):
        cols=['exact','hit4','answered','sufficient','valid']
        units=d.groupby('aggregation_group')[cols].mean()
        row={'track':track,'condition':condition,'method':method,'n':len(d),'n_units':len(units)}
        for field in cols:row[field]=float(units[field].mean());row[field+'_sd']=float(units[field].std(ddof=1))
        rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
    prefix='partial_' if a.partial else ''
    cs=read(OUT/'inputs.jsonl');gold={g['case_id']:g for g in read(OUT/'ground_truth.jsonl')};bycase={c['case_id']:c for c in cs}
    complete=True;preds={};status={}
    for m in MODELS+BASELINES:
        f=OUT/(f'annotations_{m}.jsonl' if m in MODELS else f'predictions_{m}.jsonl')
        r=read(f) if f.exists() else [];idx={x['case_id']:x for x in r}
        assert len(r)==len(idx) and set(idx)<=set(bycase),(m,'duplicates or unexpected queries')
        if m in MODELS:
            assert all(x['prompt_sha256']==bycase[x['case_id']]['prompt_sha256'] for x in r)
        done=len(idx)==len(cs);complete &= done;status[m]={'n':len(idx),'expected':len(cs),'complete':done};preds[m]=idx
    (OUT/'analysis_readiness.json').write_text(json.dumps(status,indent=2)+'\n')
    if not a.partial and not complete:raise RuntimeError('Incomplete generation; --partial is diagnostic only.')
    rows=[];coverage=[]
    for m,ps in preds.items():
        for c in cs:
            cid=c['case_id']
            if cid not in ps:continue
            g=gold[cid];gt=g['reference_support_sufficient_minimal_subspaces'];pred=parsed(ps[cid],c)
            me=metrics(pred,gt,c['candidate_size_cap']);first=set(me['first'].split('+')) if me['first'] else set()
            absent=bool(first) and not any(all(r[f]==g['target_row'][f] for f in first) for r in g['normal_reference_rows'])
            kind=('exact' if me['exact'] else 'unavailable' if not me['answered'] else
                  'population_absent_nonminimal' if me['sufficient'] else 'sample_only_absent' if absent else 'supported_in_visible_references')
            row={'case_id':cid,'source_case_id':c['source_case_id'],'track':c['track'],
              'evidence_condition':c['evidence_condition'],'aggregation_group':c['aggregation_group'],
              'target_id':g.get('target_id',c['source_case_id']),'method':m,'min_truth_size':min(map(len,gt)),
              'truth_alternatives':len(gt),'error_type':kind,'sample_absent':int(absent),**me}
            if m in MODELS:
                e=ps[cid].get('parse_or_inference_error') or ''
                row['input_limit']=int('input' in e.lower() and 'exceed' in e.lower())
                row['output_tokens']=ps[cid].get('output_tokens')
            rows.append(row)
            if m=='sample_minimal':
                candidates={frozenset(s) for s in ps[cid]['all_sample_minima']};truth={frozenset(s) for s in gt}
                coverage.append({'case_id':cid,'track':c['track'],'condition':c['evidence_condition'],'aggregation_group':c['aggregation_group'],
                    'candidate_count':len(candidates),'extra_minima':len(candidates-truth),
                    'truth_retention':len(candidates&truth)/len(truth),'any_truth_available':int(bool(candidates&truth)),
                    'exact_family_match':int(candidates==truth)})
                if c['evidence_condition']=='complete':assert candidates==truth and me['exact']==me['hit4']==1
    df=pd.DataFrame(rows);df.to_csv(OUT/f'{prefix}per_case.csv',index=False)
    summary=summarize(df);pd.DataFrame(summary).to_csv(OUT/f'{prefix}summary.csv',index=False)
    # Identical feasible target intersection; output errors remain misses.
    excluded=set(df.loc[df.get('input_limit',pd.Series(0,index=df.index)).fillna(0).eq(1),'case_id'])
    feasible=df[~df.case_id.isin(excluded)]
    pd.DataFrame(summarize(feasible)).to_csv(OUT/f'{prefix}common_feasible_summary.csv',index=False)
    cov=pd.DataFrame(coverage);cov.to_csv(OUT/f'{prefix}sample_minimal_audit.csv',index=False)
    covsummary=[]
    for (t,c),d in cov.groupby(['track','condition']):
        fields=['candidate_count','extra_minima','truth_retention','any_truth_available','exact_family_match']
        covsummary.append({'track':t,'condition':c,'n':len(d),**d.groupby('aggregation_group')[fields].mean().mean().to_dict()})
    (OUT/f'{prefix}sample_minimal_summary.json').write_text(json.dumps(covsummary,indent=2)+'\n')
    # Joint evidence comparisons on exactly the24 paired new-rule targets.
    subset={g['target_id'] for g in gold.values() if g['track']=='new_rules' and g['evidence_condition']=='complete'}
    repeated=df[(df.track=='new_rules')&df.target_id.isin(subset)]
    pd.DataFrame(summarize(repeated)).to_csv(OUT/f'{prefix}repeated_summary.csv',index=False)
    stability=[]
    for method,d in repeated[repeated.evidence_condition!='complete'].groupby('method'):
        for target,z in d.groupby('target_id'):
            if len(z)!=3:continue
            stability.append({'method':method,'target_id':target,
                'first_set_unchanged':int(z['first'].nunique()==1),
                'exact_unchanged':int(z.exact.nunique()==1),
                'min_exact':int(z.exact.min()),'max_exact':int(z.exact.max())})
    pd.DataFrame(stability).to_csv(OUT/f'{prefix}reference_stability.csv',index=False)
    audit= pd.DataFrame(coverage)[['case_id','candidate_count','extra_minima','truth_retention']]
    strata=df.merge(audit,on='case_id',how='left')
    strata['ambiguity_bin']=pd.cut(strata.extra_minima,[-1,0,3,float('inf')],labels=['none','1-3','4+']).astype(str)
    strata['alternative_bin']=np.where(strata.truth_alternatives==1,'one','multiple')
    stratified=[]
    for field in ['min_truth_size','alternative_bin','ambiguity_bin']:
        for (track,cond,method,value),z in strata.groupby(['track','evidence_condition','method',field]):
            units=z.groupby('aggregation_group')[['exact','hit4']].mean()
            stratified.append({'track':track,'condition':cond,'method':method,'stratifier':field,'value':value,
                               'n':len(z),'n_units':len(units),**units.mean().to_dict()})
    pd.DataFrame(stratified).to_csv(OUT/f'{prefix}stratified_summary.csv',index=False)
    complete_errors=repeated[repeated.evidence_condition=='complete'].groupby(['method','error_type']).size().unstack(fill_value=0)
    complete_errors.to_csv(OUT/f'{prefix}complete_error_counts.csv')
    # One predeclared cluster-bootstrap contrast family; diagnostic p-values
    # are descriptive and do not undo prior exposure to original targets.
    tests=[]
    if complete:
        panel=df[df.track=='paired96'];rng=np.random.default_rng(20261109)
        for m in MODELS:
            for baseline in BASELINES:
                for endpoint in ['exact','hit4']:
                    left=panel[panel.method==m].set_index('case_id')
                    right=panel[panel.method==baseline].set_index('case_id')
                    delta=left[[endpoint,'aggregation_group']].join(right[[endpoint]],rsuffix='_b')
                    delta['diff']=delta[endpoint]-delta[endpoint+'_b'];v=delta.groupby('aggregation_group')['diff'].mean().to_numpy()
                    boot=v[rng.integers(0,len(v),(2000,len(v)))].mean(1)
                    signs=(rng.choice([-1,1],size=(20000,len(v)))*v).mean(1)
                    tests.append({'method':m,'baseline':baseline,'endpoint':endpoint,'difference':float(v.mean()),
                      'lo':float(np.quantile(boot,.025)),'hi':float(np.quantile(boot,.975)),
                      'p_raw':float((1+(abs(signs)>=abs(v.mean())-1e-14).sum())/20001),'n_clusters':len(v)})
        pd.DataFrame(holm(tests)).to_csv(OUT/'paired_comparisons.csv',index=False)
    # Secondary original864 sample-minimal diagnostic uses original aggregation.
    oldgold={g['case_id']:g for g in read(HERE/'runs/ccfa_revision/ground_truth_all_conditions.jsonl')}
    oldrows=[]
    for pred in read(OUT/'predictions_sample_minimal_original864.jsonl'):
        g=oldgold[pred['case_id']]
        oldrows.append({'template':g['template_id'],'seed':g['generator_seed'],**metrics(pred,g['reference_support_sufficient_minimal_subspaces'],3)})
    z=pd.DataFrame(oldrows);units=z.groupby(['template','seed'])[['exact','hit4','answered']].mean().groupby('template').mean()
    oldsummary={'n':len(z),'n_templates':len(units),'mean':units.mean().to_dict(),'sd':units.std(ddof=1).to_dict()}
    (OUT/f'{prefix}original864_sample_minimal.json').write_text(json.dumps(oldsummary,indent=2)+'\n')
    car=df[df.track=='car_decision'].copy();car['size_bin']=np.where(car.min_truth_size==1,'singleton','multi-field')
    car.groupby(['method','size_bin']).agg(n=('exact','size'),exact=('exact','mean'),hit4=('hit4','mean')).to_csv(OUT/f'{prefix}car_strata.csv')
    manifest={'complete':complete,'publication_export':not a.partial,'expected_model_records':1080,
       'observed_model_records':sum(status[m]['n'] for m in MODELS),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'inputs_sha256':hashlib.sha256((OUT/'inputs.jsonl').read_bytes()).hexdigest()}
    (OUT/f'{prefix}analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
