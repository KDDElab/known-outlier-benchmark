#!/usr/bin/env python3
"""Paper-guided Beam search with the existing mixed-type Gower-LOF scorer.

This is an explicit adaptation, not execution of the EDBT Java implementation.
Its same-size exhaustive control isolates pruning from the exclusion of 1-D sets.
Parameters are literature/default fixed before inspecting its benchmark scores.
"""
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_ccfa_revision import OUT, read, write, sha
from run_gower_subspace_baseline import extract_case_data,feature_distance_matrix,lof_batch,robust_contrast_batch,baseline_record

def main():
    root=OUT/'beam_comparison';root.mkdir(exist_ok=True)
    contract={'identity':'paper-guided Beam with shared Gower-LOF scoring; not a canonical reproduction',
      'paper':'https://openproceedings.org/2021/conf/edbt/p4.pdf',
      'inspected_upstream':'https://github.com/anomaly-detection-macrobase-benchmark/macrobase/tree/alexp-vmu/alexp',
      'upstream_commit':'6b31396096923295e4501f1baa8c59e9e5ecb4d6',
      'beam_width':100,'minimum_order':2,'maximum_order':3,'candidate_budget':4,
      'stage_search':'all pairs, retain top 100, expand by one field; global rank includes pairs and visited triples',
      'same_size_control':'exhaustive pairs and triples with identical scorer and tie rule',
      'tie_rule':'descending score, ascending size, canonical feature-ID tuple',
      'k':'min(3, reference_count-1), shared with main Gower-LOF',
      'adaptations':['mixed types via original-field Gower groups','reference-only novelty LOF','four outputs','no gold dimension supplied'],
      'selection':'literature beam width 100; existing fixed scorer; no development/test score-based tuning',
      'inference':'exploratory supplement outside frozen 18-test main family',
      'inputs_sha256':sha(OUT/'inputs_main.jsonl'),'source_sha256':sha(Path(__file__)),'frozen_unix':time.time()}
    assert not (root/'manifest.json').exists(),'Preserve completed run; choose a new version for reruns'
    (root/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    outputs={'beam_gower_lof_adaptation':[],'exhaustive_pairs_triples_gower_lof':[]};diagnostics=[]
    cases=read(OUT/'inputs_main.jsonl')
    for x in cases:
        payload=extract_case_data(x['user_prompt']);ids=sorted(x['feature_ids']);refs=payload['normal_reference_rows'];n=len(refs)
        schema={s['feature_id']:s for s in payload['feature_schema']};k=min(3,n-1)
        matrices=[feature_distance_matrix([payload['target_row'][f]]+[r[f] for r in refs],schema[f]['value_type'],n) for f in ids]
        subsets=[s for size in range(1,4) for s in itertools.combinations(range(len(ids)),size)]
        lof={};knn={}
        for start in range(0,len(subsets),128):
            batch=subsets[start:start+128];distances=np.zeros((len(batch),n+1,n+1))
            for j,s in enumerate(batch):
                for i in s:distances[j]+=matrices[i]
                distances[j]/=len(s)
            lof.update(zip(batch,lof_batch(distances,n,k).tolist()))
            knn.update(zip(batch,robust_contrast_batch(distances,n,k).tolist()))
        sortkey=lambda s:(-lof[s],len(s),s)
        pairs=sorted((s for s in subsets if len(s)==2),key=sortkey)
        visited_triples={tuple(sorted((*s,i))) for s in pairs[:100] for i in range(len(ids)) if i not in s}
        visited=set(pairs)|visited_triples
        for method,universe in [('beam_gower_lof_adaptation',visited),('exhaustive_pairs_triples_gower_lof',{s for s in subsets if len(s)>=2})]:
            ranked=sorted(universe,key=sortkey)
            p=baseline_record(x['case_id'],[[ids[i] for i in s] for s in ranked[:4]])
            for c in p['explanations']:c['evidence']='Ranks original-field subsets with Gower-LOF; declared search strategy in run manifest.'
            outputs[method].append(p)
        for method,scores in [('gower_lof_subspace',lof),('gower_knn_subspace',knn)]:
            ranked=sorted(scores,key=lambda s:(-scores[s],len(s),s));values=np.asarray([scores[s] for s in ranked])
            diagnostics.append({'case_id':x['case_id'],'generator_seed':x['generator_seed'],'source_case_id':x['source_case_id'],
              'method':method,'top1_exact_score_ties':int((values==values[0]).sum()),
              'rank4_exact_score_ties':int((values==values[3]).sum()),'all_candidates':len(subsets),
              'beam_visited_pairs_triples':len(visited),'exhaustive_pairs_triples':sum(len(s)>=2 for s in subsets)})
    for method,preds in outputs.items():write(root/f'predictions_{method}.jsonl',preds)
    pd.DataFrame(diagnostics).to_csv(root/'search_and_tie_diagnostics.csv',index=False)
    (root/'manifest.json').write_text(json.dumps({**contract,'n_cases':len(cases),'completed_unix':time.time(),
       'outputs':{m:sha(root/f'predictions_{m}.jsonl') for m in outputs}},indent=2)+'\n')
    print('Completed two Beam applicability comparators and Gower tie diagnostics:',len(cases))

if __name__=='__main__':main()
