#!/usr/bin/env python3
"""Budget sensitivity and fixed-input encoding ablations from actual runs."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_ccfa_revision import OUT, HERE, read, write, sha
from analyze_ccfa_revision import evaluate, random_metrics, paired, holm, summarize, template_values

def scores(path, inputs, gold):
    rows=read(path); xs={r['case_id']:r for r in read(inputs)}; gs={r['case_id']:r for r in read(gold)}
    assert len(rows)==len(xs)==len({r['case_id'] for r in rows})
    out=[]
    for r in rows:
        cid=r['case_id'];x=xs[cid];g=gs[cid]
        # These adapters emit the validated common schema. Suppress abstention
        # and empty predictions before computing the shared localization metrics.
        p={**r,'explanations':[] if r.get('abstain') else r['explanations']}
        metric=random_metrics(p,g,x['feature_count'])
        metric['answered']=float(not r.get('abstain') and bool(p['explanations']) and any(t['features'] for t in p['explanations']))
        metric['top1_fraction']=metric['top1_fraction'] if metric['answered'] else np.nan
        out.append({'case_id':cid,'source_case_id':x['source_case_id'],'generator_seed':x['generator_seed'],
                    'template_id':g['template_id'],**metric})
    return pd.DataFrame(out)

def main():
    rows=[]; details=[]
    devin=OUT/'inputs_dev_budget.jsonl';devgold=OUT/'ground_truth_dev_budget.jsonl'
    for name in ['rlax','m2oe']:
        settings=[(steps,seed) for steps in (256,1024,4096) for seed in (20261106,20261117,20261128)] if name=='rlax' else [(ep,20261106) for ep in (30,90)]
        for budget,seed in settings:
            p=OUT/f'rlax_dev_{budget}_{seed}/predictions_rlax_reference_adaptation.jsonl' if name=='rlax' else OUT/f'm2oe_dev_epochs{budget}/predictions_m2oe_choice.jsonl'
            d=scores(p,devin,devgold);d['method']=name;d['budget']=budget;d['optimizer_seed']=seed;details.append(d)
            mean=d.groupby('template_id')[['support_top1_minimal','support_any_minimal_hit','support_best_jaccard','answered']].mean().mean()
            rows.append({'method':name,'budget':budget,'optimizer_seed':seed,'n_cases':len(d),'n_templates':d.template_id.nunique(),**mean.to_dict()})
    pd.DataFrame(rows).to_csv(OUT/'development_budget_by_seed.csv',index=False)
    pd.concat(details).to_csv(OUT/'development_budget_per_case.csv',index=False)
    pd.DataFrame(rows).groupby(['method','budget']).mean(numeric_only=True).to_csv(OUT/'development_budget_summary.csv')
    # Encoding effects hold all visible input, case seed, epochs and candidates
    # fixed; new encoding outputs are taken from the corresponding main run.
    inputs=read(OUT/'inputs_panel.jsonl');ids={r['case_id'] for r in inputs}
    alltruth=read(OUT/'ground_truth_all_conditions.jsonl')
    gold_path=OUT/'ground_truth_panel.jsonl';write(gold_path,[r for r in alltruth if r['case_id'] in ids])
    ablation=[];encoding_tests=[];encoding_details=[]
    for method,new_path,old_path in [
        ('m2oe_choice',OUT/'m2oe/predictions_m2oe_choice.jsonl',OUT/'m2oe_legacy_panel/predictions_m2oe_choice.jsonl'),
        ('sod_subspace_adaptation',OUT/'sod_cop/predictions_sod_subspace_adaptation.jsonl',OUT/'sod_cop_legacy_panel/predictions_sod_subspace_adaptation.jsonl'),
        ('cop_error_vector_adaptation',OUT/'sod_cop/predictions_cop_error_vector_adaptation.jsonl',OUT/'sod_cop_legacy_panel/predictions_cop_error_vector_adaptation.jsonl')]:
        if not new_path.exists():continue
        new=OUT/f'{method}_encoding_panel.jsonl';write(new,[r for r in read(new_path) if r['case_id'] in ids])
        frames={}
        for encoding,path in [('schema_domain_v1',new),('legacy_reference_clip',old_path)]:
            d=scores(path,OUT/'inputs_panel.jsonl',gold_path)
            frames[encoding]=d
            encoding_details.append(d.assign(method=method,encoding=encoding))
            for metric in ['support_any_minimal_hit','support_best_jaccard','answered']:
                values=d.groupby('template_id')[metric].mean().to_numpy();rng=np.random.default_rng(20261107)
                boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
                ablation.append({'method':method,'encoding':encoding,'metric':metric,'estimate':values.mean(),'lo':lo,'hi':hi,'n_cases':len(d)})
        for metric in ['support_any_minimal_hit','support_best_jaccard']:
            effect=paired(frames['schema_domain_v1'],frames['legacy_reference_clip'],metric,'schema_domain_v1','legacy_reference_clip')
            effect['method']=method;encoding_tests.append(effect)
    pd.DataFrame(ablation).to_csv(OUT/'encoding_ablation.csv',index=False)
    pd.concat(encoding_details).to_csv(OUT/'encoding_ablation_per_case.csv',index=False)
    pd.DataFrame(holm(encoding_tests)).to_csv(OUT/'encoding_ablation_paired_tests.csv',index=False)
    beam_root=OUT/'beam_comparison'
    if (beam_root/'manifest.json').exists():
        main_inputs=OUT/'inputs_main.jsonl';main_gold=OUT/'ground_truth_main_for_beam.jsonl'
        main_ids={r['case_id'] for r in read(main_inputs)}
        write(main_gold,[r for r in alltruth if r['case_id'] in main_ids])
        comparisons={};summary=[]
        for method in ['beam_gower_lof_adaptation','exhaustive_pairs_triples_gower_lof']:
            d=scores(beam_root/f'predictions_{method}.jsonl',main_inputs,main_gold)
            d['method']=method;comparisons[method]=d
            summary+=summarize(d,method,'main')
        pd.DataFrame(summary).to_csv(OUT/'beam_summary.csv',index=False)
        pd.concat(comparisons.values()).to_csv(OUT/'beam_per_case.csv',index=False)
        tests=[paired(comparisons['beam_gower_lof_adaptation'],comparisons['exhaustive_pairs_triples_gower_lof'],m,
                      'beam_gower_lof_adaptation','exhaustive_pairs_triples_gower_lof')
               for m in ['support_top1_minimal','support_any_minimal_hit']]
        pd.DataFrame(holm(tests)).to_csv(OUT/'beam_paired_tests.csv',index=False)
        ties=pd.read_csv(beam_root/'search_and_tie_diagnostics.csv')
        lookup={r['case_id']:r['template_id'] for r in alltruth}
        ties['template_id']=ties.case_id.map(lookup)
        ties['top1_has_tie']=(ties.top1_exact_score_ties>1).astype(float)
        ties['rank4_has_tie']=(ties.rank4_exact_score_ties>1).astype(float)
        tie_summary=[]
        for method,d in ties.groupby('method'):
            metrics=['top1_has_tie','rank4_has_tie','top1_exact_score_ties','rank4_exact_score_ties','beam_visited_pairs_triples','exhaustive_pairs_triples']
            tie_summary.append({'method':method,**template_values(d,metrics).mean().to_dict()})
        pd.DataFrame(tie_summary).to_csv(OUT/'gower_tie_summary.csv',index=False)
    (OUT/'supplement_analysis_manifest.json').write_text(json.dumps({'development_cases':24,'development_templates':8,
        'RLAX_optimizer_seeds':[20261106,20261117,20261128],'RLAX_steps':[256,1024,4096],'RLAX_PPO_epochs':4,
        'M2OE_epochs':[30,90],'M2OE_budget_optimizer_seeds':[20261106],'encoding_panel_cases':96,
        'development_setting':'seed-20261106 only; descriptive budget response, not an unseen-template generalization claim',
        'analysis_sha256':sha(__file__)},indent=2)+'\n')
    print('Development budget analysis and available encoding ablations completed')

if __name__=='__main__':main()
