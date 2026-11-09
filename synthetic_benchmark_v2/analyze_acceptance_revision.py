"""Analyze the frozen reviewer-requested runs, retaining every query.

--partial writes separate execution-inspection files. Publication exports require
all model batches and native controls; missing generations never become scores.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,itertools,math
import numpy as np
import pandas as pd
from annotate_v2 import parse_prediction
HERE=Path(__file__).resolve().parent;OUT=HERE/'runs/ccfa_acceptance_revision';OLD=HERE/'runs/ccfa_revision'
MODELS=['qwen35_9b','deepseek_v2_lite_chat','llama31_8b_instruct']
def read(p):return [json.loads(x) for x in p.open() if x.strip()] if p.exists() else []
def index(rows):return {r['case_id']:r for r in rows}
def parsed(r,case):
 if r is None:return None
 if 'raw_output' not in r:return r
 if not r.get('raw_output'):return None
 return parse_prediction(r['raw_output'],set(case['feature_ids']),case.get('output_contract','full'))[0]
def metrics(pred,truth,cap=None):
 valid=isinstance(pred,dict) and 'explanations' in pred
 choices=[frozenset(e['features']) for e in pred['explanations']] if valid else []
 violation=bool(valid and cap is not None and any(not s or len(s)>cap for s in choices))
 # A shared candidate contract rejects illegal sizes, never clips or repairs.
 compliant=valid and not violation
 answered=bool(compliant and not pred.get('abstain') and choices and any(choices))
 candidates=list(dict.fromkeys(choices)) if answered else []
 gold={frozenset(s) for s in truth};first=candidates[0] if candidates else frozenset()
 return {'valid':int(valid),'size_violation':int(violation),'answered':int(answered),
  'exact':int(bool(first) and first in gold),'hit4':int(any(s in gold for s in candidates[:4])),
  'sufficient':int(bool(first) and any(g<=first for g in gold)),
  'first':'+'.join(sorted(first)),'first_size':len(first) if answered else np.nan}
def macro(df,group=['template']):
 fields=['valid','size_violation','answered','exact','hit4','sufficient']
 return df.groupby(group)[fields].mean().mean().to_dict()
def paired(df,a,b,model,metric):
 x=df[(df.method==model)&(df.condition==a)][['paired_case_id','template',metric]]
 y=df[(df.method==model)&(df.condition==b)][['paired_case_id',metric]]
 z=x.merge(y,on='paired_case_id',suffixes=('_a','_b'),validate='one_to_one');assert len(z)==96
 z['diff']=z[metric+'_a']-z[metric+'_b'];v=z.groupby('template')['diff'].mean().to_numpy();rng=np.random.default_rng(20261107)
 null=(rng.choice([-1,1],size=(20000,len(v)))*v).mean(1);boot=v[rng.integers(0,len(v),(2000,len(v)))].mean(1)
 lo,hi=np.quantile(boot,[.025,.975]);return {'method':model,'left':a,'right':b,'metric':metric,'difference':float(v.mean()),'lo':float(lo),'hi':float(hi),'p_raw':float((1+(np.abs(null)>=abs(v.mean())-1e-14).sum())/20001),'n_templates':len(v),'n_pairs':len(z)}
def holm(rows):
 prev=0.
 for j,i in enumerate(sorted(range(len(rows)),key=lambda i:rows[i]['p_raw'])):
  prev=max(prev,min(1.,rows[i]['p_raw']*(len(rows)-j)));rows[i]['p_holm']=prev
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');args=p.parse_args();prefix='partial_' if args.partial else ''
 inputs={b:read(OUT/f'inputs_{b}.jsonl') for b in ['prior3','new_rules','structured','real_cells','real_cells_structured']}
 status={};records={}
 for b,cs in inputs.items():
  for m in MODELS:
   rows=read(OUT/f'annotations_{b}_{m}.jsonl');by=index(rows);known=index(cs)
   assert len(by)==len(rows),f'duplicates {b} {m}'
   assert set(by)<=set(known)
   assert all(r['prompt_sha256']==known[cid]['prompt_sha256'] for cid,r in by.items())
   status[f'{b}/{m}']={'required':len(cs),'observed':len(rows),'complete':len(rows)==len(cs)}
   if len(rows)==len(cs):records[(b,m)]=by
 status['native_m2oe']={'observed':len(read(OUT/'native_m2oe_results.jsonl')),'required':20,'complete':len(read(OUT/'native_m2oe_results.jsonl'))==20}
 status['native_sod']={'observed':len(read(OUT/'native_sod_results.jsonl')),'required':20,'complete':len(read(OUT/'native_sod_results.jsonl'))==20}
 (OUT/'analysis_readiness.json').write_text(json.dumps(status,indent=2)+'\n')
 if not args.partial and not all(s['complete'] for s in status.values()):raise RuntimeError('Incomplete run; publication exports blocked. Use --partial only for execution inspection.')
 oldinputs=index(read(OLD/'inputs_all_conditions.jsonl'));oldgold=index(read(OLD/'ground_truth_all_conditions.jsonl'))
 oldrec={m:index(read(OLD/f'reused_{m}.jsonl')+read(OLD/f'new_{m}.jsonl')) for m in MODELS}
 rows=[]
 for m in MODELS:
  if ('prior3',m) not in records:continue
  for c in inputs['prior3']:
   cid=c['paired_case_id'];g=oldgold[cid];truth=g['reference_support_sufficient_minimal_subspaces']
   for condition,r,case,cap in [('original_reported',oldrec[m][cid],oldinputs[cid],None),('original_same_cap',oldrec[m][cid],oldinputs[cid],3),('shared_prior3',records[('prior3',m)][c['case_id']],c,3)]:
    rows.append({'method':m,'condition':condition,'case_id':case['case_id'],'paired_case_id':cid,'template':g['template_id'],**metrics(parsed(r,case),truth,cap)})
 for method in ['gower_knn_subspace','gower_lof_subspace']:
  preds=index(read(OLD/f'gower/predictions_{method}.jsonl'))
  for c in inputs['prior3']:
   cid=c['paired_case_id'];g=oldgold[cid]
   rows.append({'method':method,'condition':'gower_cap3','case_id':cid,'paired_case_id':cid,'template':g['template_id'],**metrics(preds[cid],g['reference_support_sufficient_minimal_subspaces'],3)})
 prior=pd.DataFrame(rows);prior.to_csv(OUT/f'{prefix}prior_per_case.csv',index=False)
 prior_summary=[{'method':m,'condition':c,**macro(d),'n_cases':len(d)} for (m,c),d in prior.groupby(['method','condition'])]
 token_audit=read(OUT/'prompt_token_audit.jsonl');prior_feasible=[]
 for m in MODELS:
  if ('prior3',m) not in records:continue
  token_rows=[r for r in token_audit if r['model']==m and r['batch'] in ['original_full','prior3']]
  bad={r['paired_case_id'] for r in token_rows if not r['input_feasible']}
  selected=prior[(prior.method.isin([m,'gower_knn_subspace','gower_lof_subspace']))&(~prior.paired_case_id.isin(bad))]
  for (method,cond),d in selected.groupby(['method','condition']):
   prior_feasible.append({'model_feasibility':m,'method':method,'condition':cond,'n_cases':len(d),**macro(d)})
 tests=[]
 for m in MODELS:
  if ('prior3',m) in records:
   for metric in ['exact','hit4']:tests.append(paired(prior,'shared_prior3','original_same_cap',m,metric))
 if len(tests)==6:tests=holm(tests)
 pd.DataFrame(tests).to_csv(OUT/f'{prefix}prior_paired_tests.csv',index=False)
 # Structured versus unconstrained: same minimal prompt, no field-count cap.
 rows=[]
 for m in MODELS:
  if ('structured',m) not in records:continue
  for c in inputs['structured']:
   cid=c['paired_case_id'];g=oldgold[cid]
   for cond,r,case in [('minimal_unconstrained',oldrec[m][cid],oldinputs[cid]),('minimal_structured',records[('structured',m)][c['case_id']],c)]:
    row={'method':m,'condition':cond,'case_id':case['case_id'],'paired_case_id':cid,'template':g['template_id'],**metrics(parsed(r,case),g['reference_support_sufficient_minimal_subspaces'])}
    row['input_limit']=int('input' in (r.get('parse_or_inference_error') or '').lower() and 'exceed' in (r.get('parse_or_inference_error') or ''))
    rows.append(row)
 structured=pd.DataFrame(rows);structured.to_csv(OUT/f'{prefix}structured_per_case.csv',index=False)
 structured_summary=[{'method':m,'condition':c,**macro(d),'n_cases':len(d),'conditional_exact':d.exact.sum()/d.answered.sum() if d.answered.sum() else None} for (m,c),d in structured.groupby(['method','condition'])] if len(structured) else []
 structured_feasible=[]
 if len(structured):
  for (m,cond),d in structured[structured.input_limit==0].groupby(['method','condition']):
   structured_feasible.append({'method':m,'condition':cond,'n_cases':len(d),**macro(d)})
 # New independent compositions, including repeated and complete references.
 ng=index(read(OUT/'ground_truth_new_rules.jsonl'));rows=[]
 methods=[m for m in MODELS if ('new_rules',m) in records]+['gower_knn_subspace','gower_lof_subspace']
 for m in methods:
  preds=records[('new_rules',m)] if m in MODELS else index(read(OUT/f'new_rules_baselines/predictions_{m}.jsonl'))
  for c in inputs['new_rules']:
   g=ng[c['case_id']];me=metrics(parsed(preds[c['case_id']],c),g['rule_minimal_subspaces']);first=set(me['first'].split('+')) if me['first'] else set()
   sample_absent=bool(first) and not any(all(r[f]==g['target_row'][f] for f in first) for r in g['normal_reference_rows'])
   error_type=('exact' if me['exact'] else 'unavailable' if not me['answered'] else
               'population_absent_nonminimal' if me['sufficient'] else
               'sample_only_absent' if sample_absent else 'supported_in_visible_references')
   rows.append({'method':m,'case_id':c['case_id'],'target_id':g['target_id'],'template':g['template_id'],'condition':g['condition'],
    'sample_absent':int(sample_absent),'error_type':error_type,'extra_sample_minima':g['extra_sample_minima'],'truth_retention':g['truth_retention'],'min_truth_size':g['min_truth_size'],'truth_alternatives':g['truth_alternatives'],**me})
 new=pd.DataFrame(rows);new.to_csv(OUT/f'{prefix}new_rules_per_case.csv',index=False)
 new_summary=[{'method':m,'condition':c,**macro(d),'n_cases':len(d)} for (m,c),d in new.groupby(['method','condition'])]
 paired_targets={g['target_id'] for g in ng.values() if g['condition']=='complete'}
 newpairs=new[new.target_id.isin(paired_targets)]
 repeat_summary=[{'method':m,'condition':c,**macro(d),'n_cases':len(d),'mean_extra_minima':float(d.extra_sample_minima.mean()),'mean_retention':float(d.truth_retention.mean())} for (m,c),d in newpairs.groupby(['method','condition'])]
 error_categories=['exact','unavailable','population_absent_nonminimal','sample_only_absent','supported_in_visible_references']
 error_summary=[]
 for (m,c),d in newpairs.groupby(['method','condition']):
  rates={k:float(d.assign(indicator=(d.error_type==k).astype(float)).groupby('template').indicator.mean().mean()) for k in error_categories}
  assert abs(sum(rates.values())-1)<1e-12
  if c=='complete':assert rates['sample_only_absent']==0
  error_summary.append({'method':m,'condition':c,'n_cases':len(d),**rates})
 pd.DataFrame(error_summary).to_csv(OUT/f'{prefix}new_rule_error_summary.csv',index=False)
 stability=[]
 for m,d in newpairs[newpairs.condition!='complete'].groupby('method'):
  matches=[];available=[];all_same=[]
  for tid,grp in d.groupby('target_id'):
   r=grp.to_dict('records');assert len(r)==3
   all_same.append(int(all(x['answered'] for x in r) and len({x['first'] for x in r})==1))
   for a,b in itertools.combinations(r,2):
    avail=bool(a['answered'] and b['answered']);available.append(int(avail))
    if avail:matches.append(int(a['first']==b['first']))
  stability.append({'method':m,'pair_availability':float(np.mean(available)),'conditional_pair_agreement':float(np.mean(matches)) if matches else None,'all_three_available_same':sum(all_same),'n_targets':len(all_same)})
 strata=[]
 for field,bins,labels in [('min_truth_size',[0,2,3,6],['1-2','3','4-6']),('truth_alternatives',[0,1,4,999],['1','2-4','5+']),('extra_sample_minima',[-1,0,3,999],['0','1-3','4+'])]:
  d=new[new.condition=='draw0'].copy();d['bin']=pd.cut(d[field],bins,labels=labels)
  for m in methods:
   for label in labels:
    sub=d[(d.method==m)&(d.bin==label)]
    strata.append({'method':m,'factor':field,'bin':label,'n':len(sub),'n_templates':sub.template.nunique(),**(macro(sub) if len(sub) else {'exact':None,'hit4':None})})
 pd.DataFrame(strata).to_csv(OUT/f'{prefix}new_rule_strata.csv',index=False)
 # Auditable real corrections: field-inspection rather than explanation truth.
 rg=index(read(OUT/'ground_truth_real_cells.jsonl'));realrows=[]
 real_conditions=[(m,b) for m in MODELS for b in ['real_cells','real_cells_structured'] if (b,m) in records]
 real_conditions += [(m,'real_cells') for m in ['gower_knn_subspace','gower_lof_subspace']]
 for m,b in real_conditions:
  preds=records[(b,m)] if m in MODELS else index(read(OUT/f'real_cells_baselines/predictions_{m}.jsonl'))
  method=m+':schema' if b=='real_cells_structured' else m
  for c in inputs[b]:
   g=rg[c.get('paired_case_id',c['case_id'])];pr=parsed(preds[c['case_id']],c);valid=pr is not None
   choices=[x['features'] for x in pr['explanations']] if valid else []
   compliant=valid and all(len(x)==1 for x in choices);answered=bool(compliant and choices and not pr['abstain'])
   ranking=list(dict.fromkeys(x[0] for x in choices)) if answered else [];gold=set(g['corrected_fields']);top=ranking[:3]
   first=next((j+1 for j,f in enumerate(ranking) if f in gold),c['feature_count']+1)
   realrows.append({'method':method,'case_id':c['case_id'],'paired_case_id':g['case_id'],'condition':b,'dataset':g['dataset'],'valid':int(valid),'answered':int(answered),'ranking':'+'.join(ranking),
    'field_hit1':int(bool(ranking) and ranking[0] in gold),'field_hit3':int(bool(set(top)&gold)),
    'precision_at3':len(set(top)&gold)/len(top) if top else 0.,'recall_at3':len(set(top)&gold)/len(gold),
    'first_correction_inspections':first,'first_found':int(first<=len(ranking)),'gold_fields':len(gold),'total_fields':c['feature_count']})
 # Analytic uniform ordering of up to four distinct fields, with the same
 # failure penalty as methods. This is an expectation, not sampled predictions.
 def comb(n,k):return math.comb(n,k) if 0<=k<=n else 0
 for c in inputs['real_cells']:
  g=rg[c['case_id']];d=c['feature_count'];k=len(g['corrected_fields']);limit=min(4,d);top=min(3,d)
  nohit=lambda z:comb(d-k,z)/comb(d,z)
  inspection=sum(j*nohit(j-1)*k/(d-j+1) for j in range(1,limit+1))+(d+1)*nohit(limit)
  realrows.append({'method':'uniform_random_field','case_id':c['case_id'],'paired_case_id':c['case_id'],
   'condition':'analytic_uniform_order','dataset':g['dataset'],'valid':1,'answered':1,'ranking':'analytic_expectation',
   'field_hit1':k/d,'field_hit3':1-nohit(top),'precision_at3':k/d,'recall_at3':top/d,
   'first_correction_inspections':inspection,'first_found':1-nohit(limit),'gold_fields':k,'total_fields':d})
 real=pd.DataFrame(realrows);real.to_csv(OUT/f'{prefix}real_cells_per_case.csv',index=False)
 fields=['valid','answered','field_hit1','field_hit3','precision_at3','recall_at3','first_correction_inspections','first_found']
 real_summary=[{'method':m,**d.groupby('dataset')[fields].mean().mean().to_dict(),'n_cases':len(d)} for m,d in real.groupby('method')]
 real.groupby(['dataset','method'])[fields].mean().to_csv(OUT/f'{prefix}real_cells_by_dataset.csv')
 # Native controls preserve adverse outcomes at the larger training budget.
 native=[]
 for epochs in [30,300]:
  r=[x for x in read(OUT/'native_m2oe_results.jsonl') if x['epochs']==epochs]
  if len(r)==10:
   native.append({'method':'m2oe','epochs':epochs,'n':len(r),'any_exact':float(np.mean([x['any_exact'] for x in r])),
    'any_full_recall':float(np.mean([x['any_full_recall'] for x in r])),
    'mean_choice_precision':float(np.mean([np.mean([p['precision'] for p in x['outputs']]) if x['outputs'] else 0 for x in r])),
    'mean_choice_recall':float(np.mean([np.mean([p['recall'] for p in x['outputs']]) if x['outputs'] else 0 for x in r])),
    'mean_best_cube_reduction':float(np.mean([max([p['outside_cube_reduction'] for p in x['outputs']],default=0) for x in r]))})
 sr=read(OUT/'native_sod_results.jsonl')
 if len(sr)==20:native.append({'method':'sod','n':20,**{k:float(np.mean([r[k] for r in sr])) for k in ['precision','recall','exact','score_increases']}})
 summary={'complete':all(s['complete'] for s in status.values()),'analyzed_at_utc':datetime.now(timezone.utc).isoformat(),
  'prior':prior_summary,'prior_common_feasible':prior_feasible,'prior_tests':tests,'structured':structured_summary,'structured_common_feasible':structured_feasible,'new_rules':new_summary,'repeated_reference':repeat_summary,
  'reference_stability':stability,'new_rule_errors':error_summary,'real_cells':real_summary,'native_controls':native,
  'invoice_mechanism':json.loads((OUT/'invoice_lof_mechanism_summary.json').read_text()),
  'main_d6':pd.read_csv(OUT/'main_d6_search_summary.csv').to_dict('records'),'readiness':status}
 (OUT/f'{prefix}summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'complete':summary['complete'],'completed_batches':len(records),'total_batches':len(inputs)*len(MODELS),'export':str(OUT/f'{prefix}summary.json')}))
if __name__=='__main__':main()
