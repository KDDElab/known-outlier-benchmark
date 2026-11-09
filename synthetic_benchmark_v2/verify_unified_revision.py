"""Verify changed critical paths and population semantics, without new inference."""
import hashlib,itertools,json,subprocess,sys,tempfile
import numpy as np
from prepare_unified_revision import OUT,HERE,read,minima
from run_gower_subspace_baseline import extract_case_data,feature_distance_matrix
from run_unified_lof_diagnostics import inclusive_scores

def scalar_inclusive(d,k,eps):
    n=len(d)-1;kd=[];nn=[]
    for i in range(1,n+1):
        ds=sorted(d[i,j] for j in range(1,n+1) if j!=i);kd.append(ds[k-1])
        nn.append([j for j in range(1,n+1) if j!=i and d[i,j]<=ds[k-1]+1e-12])
    rho=[]
    for i,neigh in enumerate(nn,1):rho.append(1/max(sum(max(d[i,j],kd[j-1]) for j in neigh)/len(neigh),eps))
    boundary=sorted(d[0,1:])[k-1];neigh=[j for j in range(1,n+1) if d[0,j]<=boundary+1e-12]
    target_reach=sum(max(d[0,j],kd[j-1]) for j in neigh)/len(neigh)
    return sum(rho[j-1] for j in neigh)/len(neigh)*max(target_reach,eps)

def main():
    cases=read(OUT/'inputs.jsonl');gold={g['case_id']:g for g in read(OUT/'ground_truth.jsonl')}
    populations={p['template_id']:p['complete_normal_population'] for p in read(HERE/'runs/ccfa_acceptance_revision/populations_new_rules.jsonl')}
    car={p['case_id']:p['normal_rows'] for p in read(OUT/'car_populations.jsonl')}
    checked=0
    for c in cases:
        g=gold[c['case_id']];data=extract_case_data(c['user_prompt'])
        assert data['normal_reference_rows']==g['normal_reference_rows']
        if c['track'] not in ['new_rules','car_decision']:continue
        pop=car[c['case_id']] if c['track']=='car_decision' else populations[g['template_id']]
        # Independent sufficient-set predicate followed by all proper subsets;
        # confirms the shortcut used by the production enumerator.
        ids=sorted(data['target_row']);sufficient=[]
        for size in range(1,len(ids)+1):
            for s in itertools.combinations(ids,size):
                if all(any(row[f]!=data['target_row'][f] for f in s) for row in pop):sufficient.append(frozenset(s))
        exact={s for s in sufficient if not any(t<s for t in sufficient)}
        assert exact=={frozenset(s) for s in g['reference_support_sufficient_minimal_subspaces']},c['case_id']
        if c['evidence_condition']=='complete':
            assert {json.dumps(r,sort_keys=True) for r in data['normal_reference_rows']}=={json.dumps(r,sort_keys=True) for r in pop}
        checked+=1
    # Existing statistical predictions must be unchanged by the new LLM text.
    original_roots={'paired96':HERE/'runs/ccfa_revision/gower',
                    'new_rules':HERE/'runs/ccfa_acceptance_revision/new_rules_baselines',
                    'invoices':HERE/'runs/ccfa_constraint_transfer/baselines'}
    compare_count=0
    for short,oldname in [('gower_knn','gower_knn_subspace'),('gower_lof','gower_lof_subspace')]:
        new={p['case_id']:p for p in read(OUT/f'predictions_{short}.jsonl')}
        for track,root in original_roots.items():
            path=root/f'predictions_{oldname}.jsonl'
            if not path.exists() and track=='invoices':
                path=HERE/'runs/ccfa_constraint_transfer/gower'/f'predictions_{oldname}.jsonl'
            old={p['case_id']:p for p in read(path)}
            for c in cases:
                if c['track']!=track:continue
                a=[e['features'] for e in new[c['case_id']]['explanations']]
                b=[e['features'] for e in old[c['source_case_id']]['explanations']]
                assert a==b,(c['case_id'],short,a,b)
                compare_count+=1
    scalar_checks=0
    inv=[c for c in cases if c['track']=='invoices']
    for c in [inv[0],inv[24]]:
        data=extract_case_data(c['user_prompt']);refs=data['normal_reference_rows'];target=data['target_row'];schema={s['feature_id']:s for s in data['feature_schema']};n=len(refs)
        ids=sorted(target);mats={f:feature_distance_matrix([target[f]]+[r[f] for r in refs],schema[f]['value_type'],n) for f in ids}
        for size in [1,2,3]:
            for s in itertools.combinations(ids,size):
                d=sum(mats[f] for f in s)/len(s)
                for eps in [1e-6,1e-9,1e-12]:
                    assert np.isclose(inclusive_scores(d[None],3,eps)[0],scalar_inclusive(d,3,eps),rtol=1e-12,atol=1e-12)
                    scalar_checks+=1
    result={'all_checks_passed':True,'population_labels_independently_checked':checked,
      'unchanged_gower_lists_checked':compare_count,'inclusive_LOF_scalar_checks':scalar_checks,
      'inputs_sha256':hashlib.sha256((OUT/'inputs.jsonl').read_bytes()).hexdigest(),
      'effective_gold_sha256':hashlib.sha256((OUT/'ground_truth.jsonl').read_bytes()).hexdigest(),
      'runner_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()}
    from pathlib import Path
    with tempfile.TemporaryDirectory(prefix='unified-reconstruction-') as tmp:
        subprocess.run([sys.executable,str(HERE/'prepare_unified_revision.py'),'--outdir',tmp],check=True,capture_output=True)
        hashes={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()==hashlib.sha256((Path(tmp)/name).read_bytes()).hexdigest()
                for name in ['inputs.jsonl','car_populations.jsonl']}
        equal=read(OUT/'ground_truth.jsonl')==read(Path(tmp)/'ground_truth.jsonl')
        assert all(hashes.values()) and equal
        reconstruction={'all_checks_passed':True,'byte_identical_files':hashes,
            'effective_ground_truth_json_equal':equal,
            'note':'The documented label-mapping correction appended JSON keys in a different order; all effective gold values reconstruct exactly. Freeze timestamps and script fingerprints are execution-specific.'}
        (OUT/'reconstruction_verification.json').write_text(json.dumps(reconstruction,indent=2)+'\n')
        result['protocol_reconstruction_passed']=True
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
