"""Fixed invoice diagnostics: density floor, boundary ties, and duplicates.

Scores are reference-only novelty scores; target never enters the fitted graph.
Inclusive neighborhoods use all distances <= the kth distance + 1e-12.
Projection collapse changes empirical weights and is explicitly a variant.
Distinct-radius neighborhoods instead retain original multiplicities, choosing
a radius reaching k different non-identical reference coordinate groups.
"""
from pathlib import Path
import hashlib,itertools,json,time
import numpy as np
import pandas as pd
from prepare_unified_revision import read,write,OUT,HERE
from run_gower_subspace_baseline import extract_case_data,feature_distance_matrix,lof_batch

def fixed_scores(d,k,eps,mode='original',ranks=None):
    n=d.shape[1]-1;tr=d[:,0,1:];ref=d[:,1:,1:].copy();ref[:,np.arange(n),np.arange(n)]=np.inf
    if mode=='original':
        ni=np.argpartition(ref,k-1,axis=2)[:,:,:k];tn=np.argpartition(tr,k-1,axis=1)[:,:k]
    else:
        tie=np.asarray(ranks);tie=tie if mode=='ascending' else -tie
        ni=np.lexsort((np.broadcast_to(tie,ref.shape),ref),axis=2)[:,:,:k]
        tn=np.lexsort((np.broadcast_to(tie,tr.shape),tr),axis=1)[:,:k]
    nd=np.take_along_axis(ref,ni,axis=2);kd=nd.max(2)
    nkd=np.take_along_axis(kd,ni.reshape(len(kd),-1),axis=1).reshape(ni.shape)
    rr=np.maximum(nd,nkd).mean(2);rho=1/np.maximum(rr,eps)
    td=np.take_along_axis(tr,tn,axis=1);tkd=np.take_along_axis(kd,tn,axis=1)
    reach=np.maximum(td,tkd).mean(1);target_rho=1/np.maximum(reach,eps)
    nrho=np.take_along_axis(rho,tn,axis=1)
    return nrho.mean(1)/target_rho

def inclusive_scores(d,k,eps):
    n=d.shape[1]-1;tr=d[:,0,1:];ref=d[:,1:,1:].copy();ref[:,np.arange(n),np.arange(n)]=np.inf
    kd=np.partition(ref,k-1,axis=2)[:,:,k-1];mask=ref<=kd[:,:,None]+1e-12
    reach=np.maximum(ref,kd[:,None,:]);reach=np.where(mask,reach,0)
    rho=1/np.maximum(reach.sum(2)/mask.sum(2),eps)
    tkd=np.partition(tr,k-1,axis=1)[:,k-1];tm=tr<=tkd[:,None]+1e-12
    treach=(np.maximum(tr,kd)*tm).sum(1)/tm.sum(1)
    return (rho*tm).sum(1)/tm.sum(1)*np.maximum(treach,eps)

def distinct_radius_score(d,groups,k=3,eps=1e-9):
    # Reference multiplicities remain in density means. Select a radius using
    # distinct coordinates different from the center, then include boundary ties.
    ref=d[1:,1:].copy();n=len(ref);np.fill_diagonal(ref,np.inf)
    representatives=np.unique(groups,return_index=True)[1]
    kd=[]
    for i in range(n):
        values=sorted(d[i+1,j+1] for j in representatives if groups[j]!=groups[i])
        kd.append(values[min(k,len(values))-1] if values else 0.)
    kd=np.array(kd);mask=ref<=kd[:,None]+1e-12
    # A fully constant projection has only identical neighbors; preserve them.
    if np.any(mask.sum(1)==0):raise RuntimeError('empty distinct-radius neighborhood')
    rho=1/np.maximum(np.where(mask,np.maximum(ref,kd[None,:]),0).sum(1)/mask.sum(1),eps)
    values=sorted(d[0,j+1] for j in representatives if d[0,j+1]>1e-12)
    radius=values[min(k,len(values))-1] if values else 0.
    tm=d[0,1:]<=radius+1e-12
    tr=np.maximum(d[0,1:],kd)[tm].mean()
    return float(rho[tm].mean()*max(tr,eps))

def main():
    started=time.time();cases=read(HERE/'runs/ccfa_constraint_transfer/inputs.jsonl')
    truth={g['case_id']:g for g in read(HERE/'runs/ccfa_constraint_transfer/ground_truth.jsonl')}
    score_dir=OUT/'lof_candidate_scores';score_dir.mkdir(exist_ok=True)
    rows=[];traces=[]
    variants=[(mode,eps) for mode in ['original','ascending','descending','inclusive'] for eps in [1e-6,1e-9,1e-12]]
    (OUT/'lof_implementation.json').write_text(json.dumps({'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'variants':variants,'additional':['unique_inclusive','distinct_radius_multiplicity'],
      'distinct_radius':'k distinct non-identical reference coordinate groups; all boundary ties; original point weights retained; k capped to available groups.',
      'ties_tolerance':1e-12,'fitted_population':'reference only'},indent=2)+'\n')
    for ci,c in enumerate(cases):
        data=extract_case_data(c['user_prompt']);ids=sorted(c['feature_ids']);refs=data['normal_reference_rows'];target=data['target_row'];n=len(refs)
        schema={s['feature_id']:s for s in data['feature_schema']}
        matrices=np.array([feature_distance_matrix([target[f]]+[r[f] for r in refs],schema[f]['value_type'],n) for f in ids])
        subs=[s for z in range(1,9) for s in itertools.combinations(ids,z)]
        lookup={f:i for i,f in enumerate(ids)}
        dist=np.array([matrices[[lookup[f] for f in s]].sum(0)/len(s) for s in subs])
        # Canonical full-record ordering makes tie interventions independent of
        # the arbitrary input order; reverse order changes only tied choices.
        order=sorted(range(n),key=lambda i:(json.dumps(refs[i],sort_keys=True),i))
        ranks=np.empty(n,int);ranks[order]=np.arange(n)
        scores={}
        for mode,eps in variants:
            name=f'{mode}_eps{eps:.0e}'
            scores[name]=inclusive_scores(dist,3,eps) if mode=='inclusive' else fixed_scores(dist,3,eps,mode,ranks)
        assert np.allclose(scores['original_eps1e-09'],lof_batch(dist,n,3),rtol=1e-12,atol=1e-12)
        unique=[];distinct=[]
        for si,s in enumerate(subs):
            projections=[tuple(r[f] for f in s) for r in refs]
            unique_keys=sorted(set(projections),key=repr);mapping={v:i for i,v in enumerate(unique_keys)}
            groups=np.array([mapping[v] for v in projections]);indices=[projections.index(v)+1 for v in unique_keys]
            if len(indices)==1:
                # Same constant-subspace novelty convention as the density floor.
                unique.append(float(dist[si,0,indices[0]]/1e-9) if dist[si,0,indices[0]] else 1.)
            else:
                ix=[0]+indices;dd=dist[si][np.ix_(ix,ix)][None,:,:]
                unique.append(float(inclusive_scores(dd,min(3,len(indices)-1),1e-9)[0]))
            distinct.append(distinct_radius_score(dist[si],groups))
        scores['unique_inclusive']=np.array(unique);scores['distinct_radius_multiplicity']=np.array(distinct)
        gold={frozenset(s) for s in truth[c['case_id']]['reference_support_sufficient_minimal_subspaces']}
        original_top=None
        for name,v in scores.items():
            assert np.isfinite(v).all(),(c['case_id'],name)
            ranked=sorted(range(len(subs)),key=lambda i:(-float(v[i]),len(subs[i]),subs[i]));first=ranked[0]
            chosen=[subs[i] for i in ranked[:4]]
            if name=='original_eps1e-09':original_top=subs[first]
            best_gold=next(j+1 for j,i in enumerate(ranked) if frozenset(subs[i]) in gold)
            rows.append({'case_id':c['case_id'],'relation':truth[c['case_id']]['template_id'],'variant':name,
              'exact':int(frozenset(subs[first]) in gold),'hit4':int(any(frozenset(s) in gold for s in chosen)),
              'first':'+'.join(subs[first]),'first_score':float(v[first]),'best_gold_rank':best_gold,
              'first_normal_supported':int(np.any(dist[first,0,1:]<=1e-12))})
            if ci==0 or c['case_id']=='constraint_tax_total_000':
                traces.append({'case_id':c['case_id'],'variant':name,'top4':[{'features':list(subs[i]),'score':float(v[i])} for i in ranked[:4]],'best_gold_rank':best_gold})
        np.savez_compressed(score_dir/f"{c['case_id']}.npz",**scores)
        if (ci+1)%16==0:print('LOF',ci+1,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'lof_sensitivity_per_case.csv',index=False)
    summary=[]
    original=df[df.variant=='original_eps1e-09'].set_index('case_id')
    for name,d in df.groupby('variant'):
        units=d.groupby('relation')[['exact','hit4','first_normal_supported']].mean()
        z=d.set_index('case_id').join(original[['first']],rsuffix='_original')
        summary.append({'variant':name,**units.mean().to_dict(),
          'exact_sd':float(units.exact.std(ddof=1)),'hit4_sd':float(units.hit4.std(ddof=1)),
          'top1_changed':int((z['first']!=z.first_original).sum()),
          'score_min':float(d.first_score.min()),'score_max':float(d.first_score.max())})
    pd.DataFrame(summary).to_csv(OUT/'lof_sensitivity_summary.csv',index=False);write(OUT/'lof_traces.jsonl',traces)
    (OUT/'lof_manifest.json').write_text(json.dumps({'complete':True,'n_queries':96,'n_variants':len(summary),
      'n_candidate_scores':96*255*len(summary),'elapsed_seconds':time.time()-started},indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
