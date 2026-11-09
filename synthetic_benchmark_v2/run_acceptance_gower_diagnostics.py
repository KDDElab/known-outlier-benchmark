"""Paired search/prior and invoice mechanism diagnostics; no model tuning."""
from pathlib import Path
import itertools, json, gzip, hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from run_gower_subspace_baseline import extract_case_data, feature_distance_matrix, robust_contrast_batch, lof_batch

HERE=Path(__file__).resolve().parent
OUT=HERE/'runs/ccfa_acceptance_revision'
def read(p): return [json.loads(x) for x in Path(p).open() if x.strip()]
def minimal(target,refs,ids):
 out=[]
 for size in range(1,len(ids)+1):
  for sub in itertools.combinations(ids,size):
   if any(set(m)<=set(sub) for m in out):continue
   if not any(all(r[f]==target[f] for f in sub) for r in refs):out.append(sub)
 return out
def score_all(case,ids=None,ks=(3,)):
 data=extract_case_data(case['user_prompt']);ids=sorted(ids or case['feature_ids'])
 refs=data['normal_reference_rows'];n=len(refs);schema={s['feature_id']:s for s in data['feature_schema']}
 matrices=np.asarray([feature_distance_matrix([data['target_row'][f]]+[r[f] for r in refs],schema[f]['value_type'],n) for f in ids])
 subs=[s for l in range(1,len(ids)+1) for s in itertools.combinations(ids,l)]
 # Same canonical feature-addition order as the frozen baseline.
 dist=np.zeros((len(subs),n+1,n+1))
 for i,s in enumerate(subs):
  for f in s:dist[i]+=matrices[ids.index(f)]
  dist[i]/=len(s)
 scores={}
 for k in ks:
  ke=min(k,n-1)
  scores[('knn',k)]=robust_contrast_batch(dist,n,ke)
  scores[('lof',k)]=lof_batch(dist,n,ke)
 return data,subs,dist,scores
def summarize(case,gold,subs,dist,scores,cap,method,k,condition,truth_key='rule_minimal_subspaces'):
 eligible=[i for i,s in enumerate(subs) if len(s)<=cap]
 order=sorted(eligible,key=lambda i:(-float(scores[i]),len(subs[i]),subs[i]))
 truth={frozenset(s) for s in gold[truth_key]};first=order[0]
 truth_idx=[i for i in order if frozenset(subs[i]) in truth]
 row={'case_id':case['case_id'],'generator_seed':case.get('generator_seed',0),'template':gold['template_id'],'method':method,'k':k,'cap':cap,'condition':condition,
 'exact':int(frozenset(subs[first]) in truth),'hit4':int(any(frozenset(subs[i]) in truth for i in order[:4])),
 'first':'+'.join(subs[first]),'first_size':len(subs[first]),'first_score':float(scores[first]),
 'gold_rank':order.index(truth_idx[0])+1 if truth_idx else None,
 'best_gold_score':float(scores[truth_idx[0]]) if truth_idx else None,
 'score_ties_at_top':sum(float(scores[i])==float(scores[first]) for i in order),
 'target_zero_fraction':float(np.mean(dist[first,0,1:]==0)),
 'normal_zero_pair_fraction':float(np.mean(dist[first,1:,1:][np.triu_indices(dist.shape[1]-1,1)]==0)),
 'top4':json.dumps([subs[i] for i in order[:4]]),'min_truth_size':min(map(len,truth)),'truth_alternatives':len(truth)}
 return row
def main():
 OUT.mkdir(exist_ok=True)
 manifest=OUT/'gower_diagnostic_protocol.json'
 if manifest.exists():raise RuntimeError('Frozen run exists; use recorded outputs')
 manifest.write_text(json.dumps({'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
 'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'base_code_sha256':hashlib.sha256((HERE/'run_gower_subspace_baseline.py').read_bytes()).hexdigest(),
 'invoice_grid':{'cap':[1,2,3,4,8],'k':[1,3,5,10],'fields':['all','roles_only_oracle']},
 'additional_diagnostic':'At k=3 and all candidates, remove duplicate reference projections separately in each subspace; keep target. Changes empirical density weights; not a proposed replacement method.',
 'score_record':'Raw score for every candidate plus projected duplicate and zero-distance frequencies.',
 'd6':'All original 288 width-six cases, cap3 versus cap6; unchanged scores and tie rules.',
 'stats':'Descriptive paired sweeps; no best-setting selection for headline results.'},indent=2)+'\n')
 allrows=[]
 invoice=HERE/'runs/ccfa_constraint_transfer'
 gold={r['case_id']:r for r in read(invoice/'ground_truth.jsonl')}
 with gzip.open(OUT/'invoice_candidate_scores.jsonl.gz','wt') as raw:
  for case in read(invoice/'inputs.jsonl'):
   g=gold[case['case_id']]
   for cond,ids in [('all',case['feature_ids']),('roles_only_oracle',list(g['rule_role_to_feature'].values()))]:
    data,subs,dist,scores=score_all(case,ids,(1,3,5,10))
    for (method,k),values in scores.items():
     for cap in [1,2,3,4,8]:allrows.append(summarize(case,g,subs,dist,values,cap,method,k,cond))
    for i,s in enumerate(subs):
     projections=[tuple(r[f] for f in s) for r in data['normal_reference_rows']]
     raw.write(json.dumps({'case_id':case['case_id'],'condition':cond,'subset':s,
       'gold':list(s) in [sorted(x) for x in g['rule_minimal_subspaces']],
       'unique_reference_projections':len(set(projections)),
       'target_zero_fraction':float(np.mean(dist[i,0,1:]==0)),
       'normal_zero_pair_fraction':float(np.mean(dist[i,1:,1:][np.triu_indices(len(projections),1)]==0)),
       'scores':{f'{m}_k{k}':float(v[i]) for (m,k),v in scores.items()}})+'\n')
    if cond=='all':
     unique_scores={'knn':[],'lof':[]}
     for i,s in enumerate(subs):
      seen=set();keep=[0]
      for j,r in enumerate(data['normal_reference_rows'],1):
       p=tuple(r[f] for f in s)
       if p not in seen:seen.add(p);keep.append(j)
      n=len(keep)-1;dd=dist[i][np.ix_(keep,keep)][None]
      if n<2:
       # No density ratio is defined for a singleton normal support; a
       # neutral score is explicit, never silently propagated as NaN.
       unique_scores['knn'].append(0.);unique_scores['lof'].append(1.)
      else:
       unique_scores['knn'].append(float(robust_contrast_batch(dd,n,min(3,n-1))[0]))
       unique_scores['lof'].append(float(lof_batch(dd,n,min(3,n-1))[0]))
     for m,v in unique_scores.items():allrows.append(summarize(case,g,subs,dist,v,8,m,3,'unique_projection_diagnostic'))
   if len(allrows)%1968==0: print('invoice',case['case_id'],flush=True)
 pd.DataFrame(allrows).to_csv(OUT/'invoice_sweep.csv',index=False)
 df=pd.DataFrame(allrows)
 df.groupby(['condition','method','k','cap'])[['exact','hit4','first_size','normal_zero_pair_fraction','target_zero_fraction']].mean().to_csv(OUT/'invoice_sweep_summary.csv')
 main=HERE/'runs/ccfa_revision'
 gold={r['case_id']:r for r in read(main/'ground_truth_all_conditions.jsonl')}
 rows=[]
 for case in read(main/'inputs_main.jsonl'):
  if case['feature_count']!=6:continue
  data,subs,dist,scores=score_all(case)
  for (method,k),values in scores.items():
   for cap in [3,6]:rows.append(summarize(case,gold[case['case_id']],subs,dist,values,cap,method,k,'main_d6','reference_support_sufficient_minimal_subspaces'))
 pd.DataFrame(rows).to_csv(OUT/'main_d6_search.csv',index=False)
 df=pd.DataFrame(rows)
 df.groupby(['method','cap','generator_seed','template'])[['exact','hit4']].mean().groupby(['method','cap','template']).mean().groupby(['method','cap']).mean().to_csv(OUT/'main_d6_search_summary.csv')
 print('complete',len(rows),len(allrows),flush=True)
if __name__=='__main__':main()
