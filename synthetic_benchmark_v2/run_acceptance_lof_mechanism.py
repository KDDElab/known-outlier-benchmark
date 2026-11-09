"""Exploratory decomposition of the frozen invoice LOF scores."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import pandas as pd
from run_acceptance_gower_diagnostics import read,score_all,summarize
HERE=Path(__file__).resolve().parent;OUT=HERE/'runs/ccfa_acceptance_revision'
def components(dist,k=3):
 n=dist.shape[1]-1;tr=dist[:,0,1:];tn=np.argpartition(tr,k-1,axis=1)[:,:k]
 ref=dist[:,1:,1:].copy();ref[:,np.arange(n),np.arange(n)]=np.inf
 ni=np.argpartition(ref,k-1,axis=2)[:,:,:k];nd=np.take_along_axis(ref,ni,axis=2);kd=nd.max(2)
 nkd=np.take_along_axis(kd,ni.reshape(len(kd),-1),axis=1).reshape(ni.shape)
 rr=np.maximum(nd,nkd).mean(2);rho=1/np.maximum(rr,1e-9)
 td=np.take_along_axis(tr,tn,axis=1);tkd=np.take_along_axis(kd,tn,axis=1)
 reach=np.maximum(td,tkd).mean(1);target_rho=1/np.maximum(reach,1e-9)
 neighbor_rho=np.take_along_axis(rho,tn,axis=1)
 raw=neighbor_rho.mean(1)/np.maximum(target_rho,1e-9)
 ref_neighbor_rho=np.take_along_axis(rho,ni.reshape(len(rho),-1),axis=1).reshape(ni.shape)
 ref_lof=ref_neighbor_rho.mean(2)/rho
 percentile=np.mean(ref_lof<raw[:,None],axis=1)+.5*np.mean(ref_lof==raw[:,None],axis=1)
 return raw,percentile,reach,target_rho,neighbor_rho,tn,kd,rr,ref_lof
def main():
 path=OUT/'lof_mechanism_protocol.json'
 if path.exists():raise RuntimeError('Already frozen')
 path.write_text(json.dumps({'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
 'status':'Exploratory follow-up after invoice sweep revealed very large LOF scores; not a confirmatory new method.',
 'inputs':'Same96 invoice queries,255 candidates,k3; no changed target/reference/schema.',
 'decomposition':'Trace reference reachability means, density floor, target density and target-neighbor densities for first candidate and best-ranked gold candidate. Gold only selects explanation examples, never ranks predictions.',
 'calibration':'Midrank percentile of target LOF among in-sample reference LOF scores in each subspace. Canonical size/ID ties unchanged. Reference LOF uses the same reference graph; not an independent calibration sample.',
 'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
 gold={g['case_id']:g for g in read(HERE/'runs/ccfa_constraint_transfer/ground_truth.jsonl')};rows=[];traces=[]
 for case in read(HERE/'runs/ccfa_constraint_transfer/inputs.jsonl'):
  data,subs,dist,scores=score_all(case);g=gold[case['case_id']]
  raw,cal,reach,rho,nrho,tn,kd,rr,rlof=components(dist)
  assert np.allclose(raw,scores[('lof',3)],rtol=1e-12,atol=1e-12)
  order=sorted(range(len(subs)),key=lambda i:(-float(raw[i]),len(subs[i]),subs[i]));truth={frozenset(s) for s in g['rule_minimal_subspaces']}
  indices=[order[0],next(i for i in order if frozenset(subs[i]) in truth)]
  for kind,i in zip(['first','best_gold'],indices):
   nn=tn[i];traces.append({'case_id':case['case_id'],'kind':kind,'subset':subs[i],'raw_lof':float(raw[i]),'calibrated_percentile':float(cal[i]),
    'target_reachability_mean':float(reach[i]),'target_lrd':float(rho[i]),'neighbor_lrd':nrho[i].tolist(),
    'neighbor_reference_indices':nn.tolist(),'neighbor_kdist':kd[i,nn].tolist(),'neighbor_mean_reachability':rr[i,nn].tolist(),
    'floored_neighbor_count':int(np.sum(rr[i,nn]<1e-9)),'normal_lof_min':float(rlof[i].min()),'normal_lof_max':float(rlof[i].max()),
    'has_distractor':bool(set(subs[i])-set(g['rule_role_to_feature'].values()))})
  rows.append(summarize(case,g,subs,dist,cal,8,'lof_reference_percentile',3,'exploratory_calibration'))
 pd.DataFrame(rows).to_csv(OUT/'invoice_lof_calibration.csv',index=False)
 (OUT/'invoice_lof_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in traces))
 summary={'calibrated_exact':float(np.mean([r['exact'] for r in rows])),'calibrated_hit4':float(np.mean([r['hit4'] for r in rows])),
  'top_has_floored_neighbor':sum(r['floored_neighbor_count']>0 for r in traces if r['kind']=='first'),
  'top_has_distractor':sum(r['has_distractor'] for r in traces if r['kind']=='first'),
  'raw_top_range':[min(r['raw_lof'] for r in traces if r['kind']=='first'),max(r['raw_lof'] for r in traces if r['kind']=='first')]}
 (OUT/'invoice_lof_mechanism_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
if __name__=='__main__':main()
