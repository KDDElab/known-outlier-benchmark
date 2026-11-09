"""Native continuous M2OE README scenario; retain both choices and repairs."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,random,subprocess,os
import numpy as np
import tensorflow as tf
from M2OE.explainers.TabularExplainer import TabularExplainer
import M2OE.explainers.TabularExplainer as module
HERE=Path(__file__).resolve().parent;OUT=HERE/'runs/ccfa_acceptance_revision'
def main():
 manifest=OUT/'native_m2oe_protocol.json'
 if manifest.exists():raise RuntimeError('Already frozen')
 seeds=list(range(19031,19041))
 manifest.write_text(json.dumps({'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
 'source':'https://github.com/AIDALab-DIMES/M2OE','source_revision':'925c96ec0abca1e4cfd934d486f1217623184c7f',
 'scenario':'README example: 100x20 uniform[0,1) float32 matrix, row0 dimensions4,5,6 replaced by2, row0 target and other99 normal references. Ten fixed seeds.',
 'seeds':seeds,'epochs':[30,300],'other_parameters':{'loss_weights':[1.,1.2,.3],'lr':.001,'bs':16,'neighbors':30,'num_tries':3,'threshold':.5},
 'metrics':'Exact/precision/recall of emitted choices against injected fields; any-choice recovery separately; patch distance outside unit cube and nearest-reference distance. These are native recovery/repair diagnostics, not finite-support minimal truth.',
 'environment':'CPU TensorFlow existing isolated /tmp/m2oe_tf_env with existing compatibility source /tmp/icde_m2oe_tf20; no mixed-type encoding.',
 'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'tensorflow':tf.__version__},indent=2)+'\n')
 tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
 try:tf.config.experimental.enable_op_determinism()
 except Exception:pass
 with (OUT/'native_m2oe_results.jsonl').open('w') as stream:
  for seed in seeds:
   rng=np.random.RandomState(seed);X=rng.rand(100,20).astype(np.float32);X[0,[4,5,6]]=2.;target=X[:1];refs=X[1:];truth={4,5,6}
   for epochs in [30,300]:
    random.seed(seed);np.random.seed(seed);tf.keras.utils.set_random_seed(seed);start=time.perf_counter()
    exp=TabularExplainer(loss_weights=[1.,1.2,.3],lr=.001,epochs=epochs,bs=16,threshold=.5)
    raw=exp.compute_explanation(target,refs,n_neigh=30,num_tries=3)
    outputs=[]
    for dims,patched in raw:
     chosen=set(np.asarray(dims,dtype=int).reshape(-1).tolist());patch=np.asarray(patched,dtype=float).reshape(-1)
     outside=float(np.linalg.norm(np.maximum(patch-1,0)+np.minimum(patch,0)))
     outputs.append({'fields':sorted(chosen),'patch':patch.tolist(),'exact':int(chosen==truth),'precision':len(chosen&truth)/len(chosen) if chosen else 0.,'recall':len(chosen&truth)/3,
      'outside_cube_l2':outside,'outside_cube_reduction':1-outside/np.sqrt(3),'nearest_reference_l2':float(np.min(np.linalg.norm(refs-patch,axis=1)))})
    row={'seed':seed,'epochs':epochs,'outputs':outputs,'seconds':time.perf_counter()-start,
      'target':target.tolist(),'references':refs.tolist(),'any_exact':int(any(x['exact'] for x in outputs)),
      'any_full_recall':int(any(x['recall']==1 for x in outputs))}
    stream.write(json.dumps(row)+'\n');stream.flush();print(seed,epochs,len(outputs),row['any_exact'],row['any_full_recall'],flush=True)
    tf.keras.backend.clear_session()
if __name__=='__main__':main()
