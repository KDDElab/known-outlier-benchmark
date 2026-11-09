"""Low-variance axis-aligned continuous positive-control for SOD adapter."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from run_sod_cop_adaptations_v2 import shared_neighbor_subspace
HERE=Path(__file__).resolve().parent;OUT=HERE/'runs/ccfa_acceptance_revision'
def score(matrix):
 n,d=matrix.shape;k=min(20,n-1);r=min(10,k-1)
 ind=NearestNeighbors(n_neighbors=k).fit(matrix).kneighbors(return_distance=False)
 shared=np.array([len(set(ind[0])&set(x)) for x in ind]);shared[0]=10**9
 ref=matrix[np.argsort(shared)[::-1][1:r+1]];sel=shared_neighbor_subspace(matrix)
 return sel,float(np.linalg.norm((matrix[0]-ref.mean(0))[sel])/len(sel)) if sel else 0.
def main():
 p=OUT/'native_sod_protocol.json'
 if p.exists():raise RuntimeError('Already frozen')
 p.write_text(json.dumps({'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'seeds':list(range(29201,29221)),
 'scenario':'200 normal continuous rows x20 independent zero-mean Gaussian fields; std0.01 on first3 fields, std1 on other17. Same held-out normal target and target with +1 on first3; no categorical expansion.',
 'parameters':'Existing shared-neighbor SOD adapter, k20/reference10/alpha0.8. No tuning.',
 'metrics':'Selected-subspace precision/recall/exact against three low-variance shifted fields; anomalous vs matched normal SOD distance.',
 'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
 out=[]
 for seed in range(29201,29221):
  rng=np.random.default_rng(seed);x=rng.normal(size=(201,20));x[:,:3]*=.01;normal=x[0].copy();bad=normal.copy();bad[:3]+=1
  sn,vn=score(np.vstack([normal,x[1:]]));sb,vb=score(np.vstack([bad,x[1:]]));truth={0,1,2};s=set(sb)
  out.append({'seed':seed,'selected':sb,'precision':len(s&truth)/len(s) if s else 0,'recall':len(s&truth)/3,'exact':int(s==truth),
   'normal_score':vn,'anomaly_score':vb,'score_increases':int(vb>vn),'target':bad.tolist(),'normal_target':normal.tolist(),'references':x[1:].tolist()})
 (OUT/'native_sod_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in out))
 print({k:float(np.mean([r[k] for r in out])) for k in ['precision','recall','exact','score_increases']})
if __name__=='__main__':main()
