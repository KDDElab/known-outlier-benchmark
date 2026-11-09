#!/usr/bin/env python3
"""Matched unordered budget, stable-order outputs, strata, and feasibility."""
import itertools
import json
import numpy as np
import pandas as pd
from prepare_ccfa_revision import OUT, MODELS, read
from analyze_ccfa_revision import RANKERS, PRIMARY, random_metrics, template_values, summarize, paired, holm
from run_gower_subspace_baseline import extract_case_data

def matched_choices(inputs, gold, prediction_root, destination):
    # Matched at most three choices. M2OE's native choice order has no ranking
    # interpretation, so only permutation-invariant metrics are exported.
    matched=[]
    for method in [*MODELS,'m2oe_choice']:
        predictions=read(prediction_root/method/'predictions.jsonl')
        if method=='m2oe_choice':
            assert max(len(p.get('explanations',[])) for p in predictions)<=3, 'Native choice count exceeds frozen matched-three budget; do not truncate unordered choices silently'
        for p in predictions:
            cid=p['case_id'];x=inputs[cid]
            if x['condition']!='main':continue
            valid=bool(p.get('explanations') is not None and not p.get('abstain'))
            m=random_metrics({'explanations':p['explanations'][:3] if valid else []},gold[cid],x['feature_count'])
            matched.append({'method':method,'case_id':cid,'source_case_id':x['source_case_id'],'generator_seed':x['generator_seed'],
                            'template_id':gold[cid]['template_id'],'hit_at3':m['support_any_minimal_hit'],'jaccard_at3':m['support_best_jaccard']})
    m=pd.DataFrame(matched);m.to_csv(destination/'matched_three_per_case.csv',index=False)
    rows=[];tests=[]
    for method,d in m.groupby('method'):
        t=template_values(d,['hit_at3','jaccard_at3']);rng=np.random.default_rng(20261107)
        boot=t.to_numpy()[rng.integers(0,len(t),(2000,len(t)))].mean(axis=1)
        for i,metric in enumerate(t):
            lo,hi=np.quantile(boot[:,i],[.025,.975]);rows.append({'method':method,'metric':metric,'estimate':t[metric].mean(),'lo':lo,'hi':hi})
    for model,metric in itertools.product(MODELS,['hit_at3','jaccard_at3']):
        tests.append(paired(m[m.method.eq(model)],m[m.method.eq('m2oe_choice')],metric,model,'m2oe_choice'))
    pd.DataFrame(rows).to_csv(destination/'matched_three_summary.csv',index=False)
    pd.DataFrame(holm(tests)).to_csv(destination/'matched_three_tests.csv',index=False)

def prediction_support_audit(inputs, gold, prediction_root, destination):
    audit_rows=[]
    for method in [*MODELS,*RANKERS,'rlax_reference_adaptation','sod_subspace_adaptation','cop_error_vector_adaptation']:
        for p in read(prediction_root/method/'predictions.jsonl'):
            cid=p['case_id'];x=inputs[cid]
            if x['condition']!='main':continue
            payload=extract_case_data(x['user_prompt']);g=gold[cid]
            candidates=[] if p.get('abstain') else [set(c['features']) for c in p.get('explanations',[])]
            absent=lambda c: not any(all(r.get(f)==payload['target_row'].get(f) for f in c) for r in payload['normal_reference_rows'])
            support=[set(s) for s in g['reference_support_sufficient_minimal_subspaces']]
            a=bool(candidates and candidates[0] and absent(candidates[0]))
            sufficient=bool(candidates and any(s<=candidates[0] for s in support))
            row={'method':method,'case_id':cid,'generator_seed':x['generator_seed'],'template_id':g['template_id'],
                 'top1_sample_absent':float(a),'any_sample_absent':float(any(c and absent(c) for c in candidates)),
                 'top1_sample_absent_not_population_minimal':float(a and candidates[0] not in support),
                 'top1_sample_absent_population_insufficient':float(a and not sufficient),
                 'top1_population_sufficient_nonminimal':float(sufficient and candidates[0] not in support)}
            assert row['top1_sample_absent_not_population_minimal']==(
                row['top1_sample_absent_population_insufficient']+row['top1_population_sufficient_nonminimal'])
            for field in ['modified_feature_ids','value_different_from_baseline_feature_ids','mechanism_feature_ids']:
                row['top1_equals_'+field]=float(bool(candidates) and candidates[0]==set(g[field]))
            audit_rows.append(row)
    prediction_audit=pd.DataFrame(audit_rows);prediction_audit.to_csv(destination/'prediction_support_provenance_per_case.csv',index=False)
    metrics=[c for c in prediction_audit if c.startswith('top1_') or c=='any_sample_absent']
    aggregate=[]
    for method,d in prediction_audit.groupby('method'):
        t=template_values(d,metrics)
        aggregate.append({'method':method,**t.mean().to_dict()})
    pd.DataFrame(aggregate).to_csv(destination/'prediction_support_provenance_summary.csv',index=False)

def main():
    assert json.loads((OUT/'analysis_manifest.json').read_text())['complete']
    data=pd.read_csv(OUT/'all_scores.csv')
    inputs={r['case_id']:r for r in read(OUT/'inputs_all_conditions.jsonl')}
    gold={r['case_id']:r for r in read(OUT/'ground_truth_all_conditions.jsonl')}
    panel=set(json.loads((OUT/'protocol.json').read_text())['panel'])
    annotations={}
    for model in MODELS:
        # Use the same uniformly reparsed predictions as every scored metric,
        # including when the original runtime parser differed.
        annotations[model]={r['case_id']:r for r in read(OUT/'evaluation'/model/'predictions.jsonl')}
    matched_choices(inputs, gold, OUT/'evaluation', OUT)
    # Six pairwise agreements among four random orders; invalid and abstained
    # outputs never become stable empty-set answers.
    stability=[]
    conditions=['main','order1002','order1003','order1004']
    for model in MODELS:
        for source in sorted(panel):
            predictions=[]
            for condition in conditions:
                cid=f's20261106_{condition}_{source}'
                p=annotations[model][cid]
                selected=set(p['explanations'][0]['features']) if not p.get('abstain') and p.get('explanations') and p['explanations'][0]['features'] else None
                predictions.append(selected)
            pairs=list(itertools.combinations(predictions,2));eligible=[(a,b) for a,b in pairs if a is not None and b is not None]
            agreement=sum(a==b for a,b in eligible)
            stability.append({'method':model,'source_case_id':source,'template_id':gold[f's20261106_main_{source}']['template_id'],
                              'pair_count':6,'both_top1_available_pairs':len(eligible),'equal_top1_pairs':agreement,
                              'all_four_top1_available':all(p is not None for p in predictions),
                              'all_four_same_if_available':bool(all(p is not None for p in predictions) and all(p==predictions[0] for p in predictions))})
    stable=pd.DataFrame(stability);stable.to_csv(OUT/'random_order_stability_per_case.csv',index=False)
    results=[]
    for model,d in stable.groupby('method'):
        t=d.groupby('template_id')[['both_top1_available_pairs','equal_top1_pairs','all_four_top1_available','all_four_same_if_available']].mean().astype(float)
        means=t.mean();results.append({'method':model,'both_top1_available_pair_rate':means.both_top1_available_pairs/6,
            'conditional_equal_top1_rate':means.equal_top1_pairs/means.both_top1_available_pairs if means.both_top1_available_pairs else None,
            'all_four_top1_available_rate':means.all_four_top1_available,'all_four_same_allquery_rate':means.all_four_same_if_available})
    pd.DataFrame(results).to_csv(OUT/'random_order_stability_summary.csv',index=False)
    # Every Gower candidate list must be invariant under pure display changes.
    invariance=[]
    for method in RANKERS:
        preds={r['case_id']:r for r in read(OUT/'evaluation'/method/'predictions.jsonl')}
        count=0
        for source in sorted(panel):
            reference=[c['features'] for c in preds[f's20261106_main_{source}']['explanations']]
            for condition in ['front','back','order1002','order1003','order1004']:
                candidate=[c['features'] for c in preds[f's20261106_{condition}_{source}']['explanations']]
                assert reference==candidate,(method,source,condition)
                count+=1
        invariance.append({'method':method,'matched_candidate_lists':count,'required':480})
    (OUT/'gower_order_invariance.json').write_text(json.dumps(invariance,indent=2)+'\n')
    # Descriptive strata use the same template estimator, with actual template
    # counts recorded. They are not treated as paired interventions.
    summary=[]
    main=data[data.condition.eq('main')&~data.method.eq('m2oe_choice')]
    for field in ['task_label_gold','nuisance_profile','feature_count','reference_count']:
        for (method,level),d in main.groupby(['method',field],dropna=True):
            for r in summarize(d,method,'main'):
                summary.append({**r,'grouping':field,'level':level})
    pd.DataFrame(summary).to_csv(OUT/'main_stratified_summary.csv',index=False)
    # Tokenizer-based common feasibility, independent of parse success or scores.
    lengths=pd.read_csv(OUT/'token_lengths_all_conditions.csv')
    common=[];counts=[]
    for model in MODELS:
        l=lengths[lengths.method.eq(model)&lengths.generator_seed.eq(20261106)&lengths.source_case_id.isin(panel)&lengths.condition.isin(['main','refs8','refs16'])]
        eligible=l.groupby('source_case_id').input_feasible.all()
        ids=set(eligible[eligible].index)
        counts.append({'method':model,'n_panel':96,'n_all_three_feasible':len(ids)})
        for condition in ['main','refs8','refs16']:
            d=data[data.method.eq(model)&data.condition.eq(condition)&data.generator_seed.eq(20261106)&data.source_case_id.isin(ids)]
            common+=summarize(d,model,condition)
    pd.DataFrame(counts).to_csv(OUT/'nested_common_feasibility_counts.csv',index=False)
    pd.DataFrame(common).to_csv(OUT/'nested_common_feasible_summary.csv',index=False)
    # State whether template-level support/rule label changes concentrate the
    # difference in method scores; edit recovery is a provenance diagnostic.
    labels=pd.read_csv(OUT/'label_relationships_per_case.csv')
    joined=main.merge(labels[['case_id','support_rule_equal','support_alternatives','support_min_size','support_max_size','direct_mechanism_equal']],on='case_id',validate='many_to_one')
    grouped=[]
    for field in ['support_rule_equal','support_alternatives','support_min_size','support_max_size','direct_mechanism_equal']:
        for (method,level),d in joined.groupby(['method',field]):
            for r in summarize(d,method,'main'):grouped.append({**r,'grouping':field,'level':level})
    pd.DataFrame(grouped).to_csv(OUT/'label_conditioned_scores.csv',index=False)
    # Per-seed summaries expose split variability; pooled primary estimates still
    # weight each of the 38 distinct templates once after within-template pooling.
    by_seed=[]
    for (method,seed),d in main.groupby(['method','generator_seed']):
        by_seed += [{**r,'generator_seed':seed} for r in summarize(d,method,'main')]
    pd.DataFrame(by_seed).to_csv(OUT/'main_per_seed_summary.csv',index=False)
    # Reuse the value-invariant absence audit, joined by generator seed and
    # original case ID. Bins are fixed here before observing model subgroup scores.
    audit=pd.read_csv(OUT.parent/'reference_sample_audit_multiseed/per_case_test.csv')
    audit=audit.rename(columns={'case_id':'source_case_id','template_group_id':'template_id'})
    audit['ambiguity_bin']=pd.cut(audit.spurious_sample_minimal_count,[-1,0,10,50,float('inf')],labels=['0','1-10','11-50','>50'])
    merged=main.merge(audit[['generator_seed','source_case_id','ambiguity_bin','gold_candidate_recall']],
                      on=['generator_seed','source_case_id'],validate='many_to_one')
    strata=[]
    for (method,level),d in merged.groupby(['method','ambiguity_bin'],observed=True):
        strata += [{**r,'ambiguity_bin':str(level)} for r in summarize(d,method,'main')]
    pd.DataFrame(strata).to_csv(OUT/'ambiguity_conditioned_scores.csv',index=False)
    prediction_support_audit(inputs, gold, OUT/'evaluation', OUT)
    # A larger token cap should be interpreted with the recorded continuation,
    # rather than assuming that every cap hit truncated an unfinished answer.
    continuation=[]
    for model in MODELS:
        raw={r['case_id']:r for r in read(OUT/f'reused_{model}.jsonl')+read(OUT/f'new_{model}.jsonl')}
        for source in sorted(panel):
            left=raw[f's20261106_main_{source}'];right=raw[f's20261106_full1024_{source}']
            available=left.get('output_tokens') is not None and right.get('output_tokens') is not None
            capped=bool(available and left['output_tokens']==512)
            continuation.append({'method':model,'source_case_id':source,'generation_metadata_available':available,
                'tokens_512':left.get('output_tokens'),'tokens_1024':right.get('output_tokens'),
                'cap512_reached':capped,'early_stopped':bool(available and not capped),
                'same_raw_text':bool(available and left['raw_output']==right['raw_output']),
                'raw_prefix_preserved':bool(available and right['raw_output'].startswith(left['raw_output']))})
    continuity=pd.DataFrame(continuation)
    continuity.to_csv(OUT/'output_budget_continuation_per_case.csv',index=False)
    summaries=[]
    for model,d in continuity.groupby('method'):
        summaries.append({'method':model,'n_panel':len(d),'n_evaluable':int(d.generation_metadata_available.sum()),
            'early_stopped':int(d.early_stopped.sum()),'early_stopped_same_raw':int((d.early_stopped & d.same_raw_text).sum()),
            'cap512_reached':int(d.cap512_reached.sum()),'cap512_prefix_preserved':int((d.cap512_reached & d.raw_prefix_preserved).sum())})
    pd.DataFrame(summaries).to_csv(OUT/'output_budget_continuation_summary.csv',index=False)
    print('Secondary comparisons, invariance and feasibility analysis completed')

if __name__=='__main__':main()
