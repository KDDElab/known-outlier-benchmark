"""Visible-input baselines for the frozen unified support comparison."""
from pathlib import Path
import hashlib,itertools,json,time
import numpy as np
from prepare_unified_revision import read,write,minima,OUT,HERE
from run_gower_subspace_baseline import extract_case_data,feature_distance_matrix,robust_contrast_batch,lof_batch,baseline_record

def predict(case):
    data=extract_case_data(case['user_prompt']);target=data['target_row'];refs=data['normal_reference_rows']
    ids=sorted(target);cap=case['candidate_size_cap'];n=len(refs);k=min(3,n-1)
    schema={s['feature_id']:s for s in data['feature_schema']}
    m=minima(target,refs,cap)
    sample=baseline_record(case['case_id'],m[:4])
    sample['candidate_count']=len(m);sample['all_sample_minima']=m
    sample['ranking']='cardinality, then canonical feature-ID tuple'
    matrices=np.array([feature_distance_matrix([target[f]]+[r[f] for r in refs],schema[f]['value_type'],n) for f in ids])
    subsets=[s for z in range(1,cap+1) for s in itertools.combinations(range(len(ids)),z)]
    scores={'gower_knn':[],'gower_lof':[]}
    for start in range(0,len(subsets),128):
        batch=subsets[start:start+128]
        d=np.array([matrices[list(s)].sum(0)/len(s) for s in batch])
        scores['gower_knn'].extend(robust_contrast_batch(d,n,k).tolist())
        scores['gower_lof'].extend(lof_batch(d,n,k).tolist())
    out={'sample_minimal':sample}
    for name,values in scores.items():
        order=sorted(range(len(subsets)),key=lambda i:(-values[i],len(subsets[i]),subsets[i]))
        r=baseline_record(case['case_id'],[[ids[j] for j in subsets[i]] for i in order[:4]])
        r['scores']=[values[i] for i in order[:4]];out[name]=r
    return out

def main():
    start=time.time();allrows={n:[] for n in ['sample_minimal','gower_knn','gower_lof']}
    for i,c in enumerate(read(OUT/'inputs.jsonl')):
        for name,r in predict(c).items():allrows[name].append(r)
        if (i+1)%40==0:print('baseline',i+1,flush=True)
    for name,rows in allrows.items():write(OUT/f'predictions_{name}.jsonl',rows)
    # Original864 is a secondary diagnostic; no new LLM outputs are substituted.
    old=read(HERE/'runs/ccfa_revision/inputs_all_conditions.jsonl')
    main_cases=[c for c in old if c.get('condition')=='main']
    assert len(main_cases)==864,len(main_cases)
    extra=[]
    for c in main_cases:
        d=extract_case_data(c['user_prompt']);m=minima(d['target_row'],d['normal_reference_rows'],3)
        r=baseline_record(c['case_id'],m[:4]);r.update(candidate_count=len(m),all_sample_minima=m);extra.append(r)
    write(OUT/'predictions_sample_minimal_original864.jsonl',extra)
    (OUT/'baseline_manifest.json').write_text(json.dumps({'complete':True,'n_queries':360,'original_diagnostic':864,
      'elapsed_seconds':time.time()-start,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'inputs_sha256':hashlib.sha256((OUT/'inputs.jsonl').read_bytes()).hexdigest(),
      'uses_hidden_labels':False,'max_candidates':4,'tie_order':'size then canonical feature IDs'},indent=2)+'\n')
if __name__=='__main__':main()
