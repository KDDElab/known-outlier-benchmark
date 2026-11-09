#!/usr/bin/env python3
"""Descriptive revision deltas, with old raw responses rescored consistently.

No new inference or hypothesis-test family is introduced. Historical role-prefix
responses retain their source hashes and are never substituted into revised runs.
"""
import argparse
import json
import pandas as pd
from prepare_ccfa_revision import HERE, OUT, MODELS, read, sha
from annotate_v2 import parse_prediction
from analyze_ccfa_revision import evaluate, summarize
from run_expanded_random_order_probe_v2 import prompt_payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-primary',action='store_true')
    args=parser.parse_args()
    if args.completed_primary:
        main_path=OUT/'main_table_partial.csv'
        snapshot=json.loads((OUT/'primary_complete_snapshot.json').read_text())
        assert sha(main_path)==snapshot['files']['main_table.csv']
    else:
        assert json.loads((OUT/'analysis_manifest.json').read_text())['complete']
        main_path=OUT/'main_table.csv'
    revised=pd.read_csv(main_path).set_index('method')
    frozen={r['case_id']:r for r in read(OUT/'inputs_main.jsonl')}
    selected=json.loads((HERE/'runs/multiseed/run_selection_manifest.json').read_text())
    metadata=pd.DataFrame([{k:r[k] for k in ['case_id','source_case_id','generator_seed','condition']} for r in frozen.values()])
    summaries=[];sources=[];details=[]
    for model in MODELS:
        predictions=[]
        for seed in (20261106,20261107,20261108):
            oldroot=HERE/'runs/multiseed'/('seed_20261106_rt522' if seed==20261106 else f'seed_{seed}')
            input_path=HERE/'inputs.jsonl' if seed==20261106 else oldroot/'inputs.jsonl'
            old_inputs={r['case_id']:r for r in read(input_path) if r['split']=='test'}
            choice=selected['runs'][str(seed)][model]
            path=oldroot/f'annotations_{model}.jsonl'
            assert sha(path)==choice['annotation_sha256']
            records=[r for r in read(path) if r.get('event')=='annotation' and r.get('protocol_id')==choice['protocol_id']]
            assert len(records)==len({r['case_id'] for r in records})==288
            assert {r['case_id'] for r in records}==set(old_inputs)
            for r in records:
                original=old_inputs[r['case_id']];cid=f"s{seed}_main_{r['case_id']}";current=frozen[cid]
                assert r['prompt_sha256']==original['prompt_sha256']
                assert r['run_id'] in choice['authoritative_run_ids']
                assert r.get('do_sample') is False and r['max_new_tokens']==512 and r['max_input_tokens']==16384
                prefix_before,before=prompt_payload(original['user_prompt']);prefix_after,after=prompt_payload(current['user_prompt'])
                assert prefix_before==prefix_after and original['system_prompt']==current['system_prompt']
                assert before['target_row']==after['target_row']
                assert before['normal_reference_rows']==after['normal_reference_rows']
                assert {v['feature_id']:v for v in before['feature_schema']}=={v['feature_id']:v for v in after['feature_schema']}
                parsed,error=parse_prediction(r.get('raw_output',''),set(original['feature_ids']))
                predictions.append({'case_id':cid,**(parsed or {})})
            sources.append({'seed':seed,'method':model,'path':str(path.relative_to(HERE)),'sha256':sha(path),'records':len(records)})
        d=evaluate('legacy_prefix_'+model,predictions,OUT/'inputs_all_conditions.jsonl',OUT/'ground_truth_all_conditions.jsonl')
        d=d[d.case_id.isin(frozen)].merge(metadata,on='case_id',validate='one_to_one')
        assert len(d)==864
        d['method']=model;details.append(d)
        summaries+=summarize(d,model,'legacy_prefix_rescored')
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/'legacy_prefix_rescored_summary.csv',index=False)
    pd.concat(details).to_csv(OUT/'legacy_prefix_rescored_per_case.csv',index=False)
    delta=[]
    for r in summary.itertuples():
        new=float(revised.loc[r.method,r.metric])
        delta.append({'method':r.method,'metric':r.metric,'legacy_prefix_rescored':r.estimate,
                      'revised_independent_display':new,'revised_minus_legacy':new-r.estimate})
    pd.DataFrame(delta).to_csv(OUT/'legacy_revision_descriptive_deltas.csv',index=False)
    (OUT/'legacy_comparison_manifest.json').write_text(json.dumps({
        'scope':'Post-hoc descriptive comparison with historical raw outputs; no new significance tests',
        'common_scoring':'Current parser, explicit abstention semantics, and primary template estimator in both conditions',
        'inputs':'Identical schema/value/reference content verified; display order differs',
        'limitation':'Historical execution versus revision execution; not an independent randomized method-capability experiment',
        'sources':sources,'analysis_sha256':sha(__file__)},indent=2)+'\n')
    print('Historical 2,592 responses rescored; descriptive revision deltas exported.')


if __name__=='__main__':main()
