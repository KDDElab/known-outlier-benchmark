#!/usr/bin/env python3
"""Audit completed measurements and exported estimators before manuscript release."""
import json
import hashlib
import numpy as np
import pandas as pd
from prepare_ccfa_revision import OUT,MODELS,read,sha

def main():
    manifest=json.loads((OUT/'analysis_manifest.json').read_text());assert manifest['complete']
    inputs={r['case_id']:r for r in read(OUT/'inputs_all_conditions.jsonl')}
    primary=json.loads((OUT/'primary_complete_snapshot.json').read_text())
    lengths=pd.read_csv(OUT/'token_lengths_all_conditions.csv')
    status=pd.read_csv(OUT/'failure_per_case.csv')
    reports=[]
    for model in MODELS:
        reused=read(OUT/f'reused_{model}.jsonl');new=read(OUT/f'new_{model}.jsonl');records=reused+new
        assert len(reused)==673 and len(new)==1055 and len(records)==len({r['case_id'] for r in records})==1728
        assert {r['case_id'] for r in records}==set(inputs)
        main_records=sorted([r for r in records if inputs[r['case_id']]['condition']=='main'],key=lambda r:r['case_id'])
        main_hash=hashlib.sha256(json.dumps(main_records,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        assert main_hash==primary['main_raw_sha256'][model],'Primary raw predictions changed during auxiliary execution'
        for r in records:
            x=inputs[r['case_id']]
            assert r['prompt_sha256']==x['prompt_sha256']
            assert r.get('max_new_tokens',512)==x.get('max_new_tokens',512)
            assert r.get('max_input_tokens')==16384 and r.get('do_sample') is False
            assert r.get('batch_size',1) in {0,1}
        failures=status[status.method.eq(model)]
        infeasible=set(lengths.loc[lengths.method.eq(model)&~lengths.input_feasible,'case_id'])
        observed=set(failures.loc[failures.response_status.eq('input_limit'),'case_id'])
        assert observed==infeasible,(model,'input-limit mismatch',infeasible^observed)
        assert not any('out of memory' in (r.get('parse_or_inference_error') or '').lower() for r in records),'Resolve any recorded resource failure explicitly before release'
        reports.append({'model':model,'conditions':len(records),'reused':len(reused),'new':len(new),'input_limit_cases':len(observed),
                        'new_file_sha256':sha(OUT/f'new_{model}.jsonl'),'reused_file_sha256':sha(OUT/f'reused_{model}.jsonl')})
    data=pd.read_csv(OUT/'all_scores.csv');eps=1e-12
    for model in MODELS:
        observed=data[data.method.eq(model)]
        assert len(observed)==observed.case_id.nunique()==1728
        assert observed.condition.value_counts().to_dict()=={'main':864,**{c:96 for c in ['front','back','order1002','order1003','order1004','refs8','refs16','full1024','minimal512']}}
    assert (data.support_top1_minimal<=data.support_top1_sufficient+eps).all()
    assert (data.support_top1_minimal<=data.support_any_minimal_hit+eps).all()
    assert (data.support_any_minimal_hit<=data.support_best_jaccard+eps).all()
    assert (data.support_top1_sufficient<=data.answered+eps).all()
    assert (data.answered<=data.structurally_valid+eps).all()
    assert data.loc[data.answered.eq(0),'top1_fraction'].isna().all()
    for metric in ['support_top1_minimal','support_top1_sufficient','support_any_minimal_hit','support_best_jaccard','answered','structurally_valid']:
        assert data[metric].between(-eps,1+eps).all(),metric
    table=pd.read_csv(OUT/'main_table.csv').set_index('method');tests=pd.read_csv(OUT/'main_paired_tests.csv')
    assert len(tests)==18 and len(table)==12
    for name,digest in primary['files'].items():
        assert sha(OUT/name)==digest,'Completed primary estimates changed after auxiliary conditions finished'
    for name,digest in primary.get('secondary_files',{}).items():
        assert sha(OUT/name)==digest,'Completed primary secondary analysis changed after auxiliary conditions finished'
    for r in tests.itertuples():
        assert abs(table.loc[r.left,r.metric]-table.loc[r.right,r.metric]-r.difference)<eps
    for model in MODELS:
        bound=table.loc[model,'support_top1_minimal']+1-table.loc[model,'answered']
        assert abs(bound-primary['delivery_accounting'][model])<eps
    for family,count in [('order',12),('references',18),('output',12),('matched_three',6),('encoding_ablation',6),('beam',2)]:
        name='matched_three_tests.csv' if family=='matched_three' else f'{family}_paired_tests.csv'
        comparisons=pd.read_csv(OUT/name)
        assert len(comparisons)==count,(family,len(comparisons))
        assert comparisons.p_raw.between(0,1).all() and comparisons.p_holm.between(0,1).all()
        assert (comparisons.p_holm+eps>=comparisons.p_raw).all()
        assert (comparisons.lo<=comparisons.hi).all()
    matched=pd.read_csv(OUT/'matched_three_per_case.csv')
    for model in MODELS:
        d=data[data.method.eq(model)&data.condition.eq('main')].merge(matched[matched.method.eq(model)],on='case_id',validate='one_to_one')
        assert len(d)==864
        assert (d.hit_at3<=d.support_any_minimal_hit+eps).all()
        assert (d.jaccard_at3<=d.support_best_jaccard+eps).all()
    m2oe=read(OUT/'m2oe/predictions_m2oe_choice.jsonl')
    assert len(m2oe)==864 and max(len(r['explanations']) for r in m2oe)<=3
    prediction_audit=pd.read_csv(OUT/'prediction_support_provenance_summary.csv').set_index('method')
    for model in MODELS:
        row=prediction_audit.loc[model]
        assert abs(row.top1_sample_absent-table.loc[model,'support_top1_minimal']-row.top1_sample_absent_not_population_minimal)<eps
        assert abs(row.top1_sample_absent_not_population_minimal-row.top1_sample_absent_population_insufficient-row.top1_population_sufficient_nonminimal)<eps
    invariance=json.loads((OUT/'gower_order_invariance.json').read_text())
    assert len(invariance)==2 and all(r['matched_candidate_lists']==480 for r in invariance)
    checkpoints=json.loads((OUT/'checkpoint_reverification.json').read_text())
    assert all(r['matches_recorded_checkpoint'] for r in checkpoints)
    report={'status':'completed_measurements_verified','models':reports,'main_methods':len(table),'main_paired_tests':len(tests),
            'table_test_estimators_agree':True,'metric_ordering_invariants':True,'all_infeasible_inputs_preserved':True,
            'matched_choice_budget_verified':True,'gower_order_invariant_lists':960,
            'checkpoint_files_rehashed':len(checkpoints),'independent_machine_reproduction':False,
            'primary_snapshot_preserved':True,
            'intervention_test_family_sizes_verified':True,'prediction_absence_decomposition_verified':True,
            'audit_sha256':sha(__file__)}
    (OUT/'completion_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
