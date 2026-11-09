#!/usr/bin/env python3
"""Analyze fixed labels and external records under the repaired denominator."""
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_ccfa_revision import HERE, OUT, MODELS, read
from run_expanded_random_order_probe_v2 import prompt_payload
from audit_reference_sample_identifiability import audit_case

ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "circuit_track"))
from evaluate_zero_noise_pilot import score_case

def family(sets): return {frozenset(s) for s in sets}

def labels():
    inputs = read(OUT / "inputs_all_conditions.jsonl")
    truth = {r["case_id"]:r for r in read(OUT / "ground_truth_all_conditions.jsonl")}
    rows=[]
    for x in inputs:
        if x["condition"]!="main":continue
        g=truth[x["case_id"]]
        support=family(g["reference_support_sufficient_minimal_subspaces"])
        rule=family(g["rule_minimal_subspaces"])
        mechanism=set(g["mechanism_feature_ids"]);direct=set(g["modified_feature_ids"])
        value=set(g["value_different_from_baseline_feature_ids"])
        rows.append({"case_id":x["case_id"],"generator_seed":x["generator_seed"],"template_id":g["template_id"],
                     "family":g["task_label"],"nuisance_profile":g["nuisance_profile"],
                     "support_rule_equal":support==rule,"support_alternatives":len(support),"rule_alternatives":len(rule),
                     "support_min_size":min(map(len,support)),"support_max_size":max(map(len,support)),
                     "direct_mechanism_equal":direct==mechanism,"value_direct_equal":value==direct,
                     "mechanism_is_support_minimal":frozenset(mechanism) in support,
                     "direct_is_support_minimal":frozenset(direct) in support,
                     "support_in_display_first3":any(s<=set(x["feature_ids"][:3]) for s in support),
                     "mechanism_in_display_first3":mechanism<=set(x["feature_ids"][:3])})
    data=pd.DataFrame(rows);data.to_csv(OUT/"label_relationships_per_case.csv",index=False)
    numeric=[c for c in data if c not in ["case_id","generator_seed","template_id","family","nuisance_profile"]]
    def summarize(frame):
        macro=frame.groupby(["generator_seed","template_id"])[numeric].mean().groupby("template_id").mean().mean()
        return {"n_cases":len(frame),"n_templates":frame.template_id.nunique(),**macro.to_dict()}
    result={"overall":summarize(data),"by_family":{k:summarize(v) for k,v in data.groupby("family")},
            "by_nuisance":{k:summarize(v) for k,v in data.groupby("nuisance_profile")},
            "raw_counts":{c:int(data[c].sum()) for c in ["support_rule_equal","direct_mechanism_equal","value_direct_equal","support_in_display_first3","mechanism_in_display_first3"]}}
    (OUT/"label_relationships.json").write_text(json.dumps(result,indent=2)+"\n")
    audit=[]
    for x in inputs:
        if x["generator_seed"]!=20261106 or x["condition"] not in {"main","refs8","refs16"}:continue
        if x["condition"]=="main" and x["reference_count"]!=32:continue
        _,p=prompt_payload(x["user_prompt"])
        g={**truth[x["case_id"]],**p,"reference_count":x["reference_count"]}
        r=audit_case(g,3)
        r.update(source_case_id=x["source_case_id"],condition=x["condition"])
        audit.append(r)
    pd.DataFrame(audit).to_csv(OUT/"nested_reference_audit.csv",index=False)
    full=pd.read_csv(HERE/'runs/reference_sample_audit_multiseed/per_case_test.csv')
    metrics=['spurious_sample_minimal_count','gold_candidate_recall','sample_minimal_set_exact_match']
    def aggregate(frame):
        t=frame.groupby(['generator_seed','template_group_id'])[metrics].mean().groupby('template_group_id').mean().astype(float)
        rng=np.random.default_rng(20261107)
        boot=t.to_numpy()[rng.integers(0,len(t),(2000,len(t)))].mean(axis=1)
        bounds=np.quantile(boot,[.025,.975],axis=0)
        return [{'metric':metric,'estimate':t[metric].mean(),'lo':bounds[0,i],'hi':bounds[1,i],
                 'n_cases':len(frame),'n_templates':len(t)} for i,metric in enumerate(metrics)]
    audit_summary=[{**r,'group':'overall'} for r in aggregate(full)]
    for (fields,refs),frame in full.groupby(['feature_count','reference_count']):
        audit_summary.extend({**r,'group':f'd{fields}_r{refs}'} for r in aggregate(frame))
    pd.DataFrame(audit_summary).to_csv(OUT/'population_sample_audit_summary.csv',index=False)
    nuisance=[]
    for (profile,fields,refs),frame in full.groupby(['nuisance_profile','feature_count','reference_count']):
        nuisance.extend({**r,'nuisance_profile':profile,'feature_count':fields,'reference_count':refs}
                        for r in aggregate(frame))
    pd.DataFrame(nuisance).to_csv(OUT/'population_sample_audit_by_nuisance.csv',index=False)
    print('Label relationships and 288 nested-reference audits completed',flush=True)

def circuit():
    root=ROOT/"circuit_track"
    truth={r["case_id"]:r for r in read(root/"ground_truth_zero_noise_pilot.jsonl")}
    rows=[]
    for condition,path,input_path in [("all",root,root/"inputs_zero_noise_pilot.jsonl"),
                                      ("focus",root/"matched_information",root/"matched_information/inputs_focus_output_only.jsonl")]:
        inputs={r["case_id"]:r for r in read(input_path)}
        preds={}
        for model in MODELS:
            raw=read(path/f"annotations_{model}.jsonl")
            assert len(raw)==len({r['case_id'] for r in raw})==86
            assert all(r['prompt_sha256']==inputs[r['case_id']]['prompt_sha256'] for r in raw)
            preds[model]={r['case_id']:r.get('parsed_prediction') for r in raw}
        for p in (OUT/f"circuit_{condition}_baselines").glob('predictions_*.jsonl'):
            pred=read(p); assert len(pred)==86
            preds[p.stem.removeprefix('predictions_')]={r['case_id']:r for r in pred}
        assert len(preds)==7
        for method,records in preds.items():
            for case_id,g in truth.items():
                parsed=records[case_id]
                metric=score_case(set(g['gold_features']),parsed)
                candidates=[] if not parsed or parsed.get('abstain') else [set(x['features']) for x in parsed['explanations']]
                metric['field_recall_at4']=max((len(c&set(g['gold_features']))/len(g['gold_features']) for c in candidates),default=0.)
                rows.append({'condition':condition,'method':method,'case_id':case_id,'circuit':g['circuit'],
                             'source_table':g['source_table'],'source_target':case_id.rsplit('_',1)[0],**metric})
    data=pd.DataFrame(rows);data.to_csv(OUT/'circuit_per_case.csv',index=False)
    metrics=['valid','answered','abstain','exact_at_1','hit_at_4','best_jaccard_at_4','field_recall_at4']
    summaries=[]
    rng=np.random.default_rng(20261107)
    for (condition,method),d in data.groupby(['condition','method']):
        source=d.groupby(['circuit','source_table','source_target'])[metrics].mean()
        pairs=source.groupby(['circuit','source_table']).mean()
        estimate=pairs.groupby('circuit').mean().mean()
        draws=[]
        for _ in range(2000):
            vals=[]
            for _,group in pairs.groupby('circuit'):
                vals.append(group.iloc[rng.integers(0,len(group),len(group))].mean().to_numpy())
            draws.append(np.mean(vals,axis=0))
        bounds=np.quantile(draws,[.025,.975],axis=0)
        for i,metric in enumerate(metrics):
            summaries.append({'condition':condition,'method':method,'metric':metric,'estimate':estimate[metric],
                              'lo':bounds[0,i],'hi':bounds[1,i],'n_queries':len(d),'n_pairs':len(pairs),'n_circuits':4})
    pd.DataFrame(summaries).to_csv(OUT/'circuit_hierarchical_summary.csv',index=False)
    by_circuit=data.groupby(['condition','method','circuit','source_table','source_target'])[metrics].mean().groupby(['condition','method','circuit','source_table']).mean().groupby(['condition','method','circuit']).mean()
    by_circuit.to_csv(OUT/'circuit_by_circuit.csv')
    # Paired all-output minus focus-only effects, preserving target and circuit
    # hierarchy. These nine tests form their own exploratory family.
    effects=[]
    for model,metric in itertools.product(MODELS,['exact_at_1','hit_at_4','best_jaccard_at_4']):
        d=data[data.method.eq(model)]
        grouped=d.groupby(['condition','circuit','source_table','source_target'])[metric].mean().groupby(['condition','circuit','source_table']).mean()
        delta=(grouped.loc['all']-grouped.loc['focus'])
        values=delta.to_numpy();weights=np.array([1/(4*len(delta.loc[circuit])) for circuit,_ in delta.index])
        estimate=float(values@weights);rng=np.random.default_rng(20261107)
        null=(rng.choice([-1,1],size=(20000,len(values)))*values*weights).sum(axis=1)
        p=(1+np.sum(np.abs(null)>=abs(estimate)-1e-14))/20001
        draws=np.zeros(2000)
        for circuit,group in delta.groupby(level='circuit'):
            v=group.to_numpy();draws+=v[rng.integers(0,len(v),(2000,len(v)))].mean(axis=1)/4
        lo,hi=np.quantile(draws,[.025,.975])
        effects.append({'method':model,'metric':metric,'contrast':'all_minus_focus','difference':estimate,'lo':lo,'hi':hi,'p_raw':p,'n_pairs':len(delta)})
    previous=0.
    for rank,i in enumerate(sorted(range(len(effects)),key=lambda i:effects[i]['p_raw'])):
        previous=max(previous,min(1.,effects[i]['p_raw']*(len(effects)-rank)));effects[i]['p_holm']=previous
    pd.DataFrame(effects).to_csv(OUT/'circuit_paired_context_tests.csv',index=False)
    (OUT/'circuit_analysis_contract.json').write_text(json.dumps({'estimator':'focus queries within source target; source targets within table pair; pairs within circuit; equal average over four fixed circuits',
        'uncertainty':'2,000 resamples of table pairs within each fixed circuit; no claim of four circuits sampling a wider population',
        'information':'all: LLM and joint-output surprisal see all outputs; focus: LLM and focus/joint surprisal see focus only; Gower uses inputs only in both conditions',
        'new_inference':False,'baseline_rerun':True,'abstain_scores_zero':True},indent=2)+'\n')
    print('Circuit existing raw outputs verified and rescored; baselines rerun under both contexts',flush=True)

def natural():
    data=pd.read_csv(ROOT/'reports/leave_one_model_out_by_case.csv')
    data['group']=np.where(data.dataset_key.str.startswith('ADB_'),'ADBench','UCI Lymphography')
    data['schema_valid']=~data.status.eq('held-out output missing or invalid')
    data['nonabstain']=data.schema_valid & ~data.status.eq('held-out annotator abstained')
    data['eligible']=data.status.eq('scored against peer strict-majority reference')
    summaries=[]
    for (group,model),d in data.groupby(['group','held_out_annotator_id']):
        eligible=d[d.eligible]
        summaries.append({'group':group,'model':model,'n_queries':len(d),'schema_valid':int(d.schema_valid.sum()),
                          'nonabstain':int(d.nonabstain.sum()),'eligible':len(eligible),
                          'coverage_among_nonabstain':len(eligible)/d.nonabstain.sum() if d.nonabstain.sum() else None,
                          'coverage_all_queries':len(eligible)/len(d),'jaccard':eligible.jaccard.mean()})
    pd.DataFrame(summaries).to_csv(OUT/'natural_separate_groups.csv',index=False)
    data.to_csv(OUT/'natural_per_case.csv',index=False)
    # Preserve per-dataset eligibility and agreement instead of mixing the two sources.
    rows=[]
    for (dataset,model),d in data.groupby(['dataset_key','held_out_annotator_id']):
        rows.append({'dataset':dataset,'model':model,'n_queries':len(d),'schema_valid':int(d.schema_valid.sum()),
                     'nonabstain':int(d.nonabstain.sum()),'eligible':int(d.eligible.sum()),
                     'jaccard':d.loc[d.eligible,'jaccard'].mean()})
    pd.DataFrame(rows).to_csv(OUT/'natural_by_dataset.csv',index=False)
    import ast
    pairs=[]
    for group,frame in data.groupby('group'):
        common=set.intersection(*(set(d.loc[d.nonabstain,'case_id']) for _,d in frame.groupby('held_out_annotator_id')))
        models=sorted(frame.held_out_annotator_id.unique())
        chosen={(r.held_out_annotator_id,r.case_id):set(ast.literal_eval(r.held_out_features)) for r in frame[frame.nonabstain].itertuples()}
        for left,right in itertools.combinations(models,2):
            vals=[len(chosen[left,c]&chosen[right,c])/len(chosen[left,c]|chosen[right,c]) if chosen[left,c]|chosen[right,c] else 1. for c in common]
            pairs.append({'group':group,'left':left,'right':right,'n_common_queries_all_five':len(common),'jaccard':np.mean(vals) if vals else None})
    pd.DataFrame(pairs).to_csv(OUT/'natural_common_query_pairwise.csv',index=False)
    print('Natural-table ADBench 70 / UCI 8 split completed',flush=True)

if __name__=='__main__':
    labels(); circuit(); natural()
