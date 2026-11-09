#!/usr/bin/env python3
"""Paired template-cluster comparisons over available seed replicates.

Six ranked-candidate baselines use four localization measures. M²OE is treated
as an unordered set of alternatives and uses only two order-invariant measures.
Each held-out template contributes one paired cluster average after averaging
its available seed-specific means.
"""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
SEEDS=(20261106,20261107,20261108)
LLMS=("qwen35_9b","deepseek_v2_lite_chat","llama31_8b_instruct")
RANKED_BASELINES=("gower_knn_subspace","gower_lof_subspace","top_univariate","sod_subspace_adaptation","cop_error_vector_adaptation","rlax_reference_adaptation")
SET_BASELINES=("m2oe_choice_matched3",)
ALL_BASELINES=(*RANKED_BASELINES,*SET_BASELINES)
METRICS={
    "sufficiency@1":"support_top1_sufficient",
    "minimal_exact@1":"support_top1_minimal",
    "Hit@4":"support_any_minimal_hit",
    "best_Jaccard":"support_best_jaccard",
}
SET_METRICS={"matched_budget_Hit@3":"support_any_minimal_hit","matched_budget_best_Jaccard@3":"support_best_jaccard"}

def sha256_file(path:Path)->str|None:
    if not path.is_file():
        return None
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()

def root_for(seed:int)->Path:
    return HERE/"runs"/"multiseed_analysis"/str(seed)

def holm(pvalues:list[float])->list[float]:
    order=np.argsort(pvalues); adjusted=np.empty(len(pvalues)); running=0.0; m=len(pvalues)
    for rank,index in enumerate(order):
        value=min(1.0,(m-rank)*pvalues[index]); running=max(running,value); adjusted[index]=running
    return adjusted.tolist()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=HERE/"runs"/"multiseed_analysis"/"paired_tests.json")
    parser.add_argument("--permutations",type=int,default=20000)
    parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=20261106)
    args=parser.parse_args()
    rng=np.random.default_rng(args.seed)
    cache={}
    for seed in SEEDS:
        for method in (*LLMS,*ALL_BASELINES):
            path=root_for(seed)/method/"per_case.csv"
            if path.exists():
                frame=pd.read_csv(path).query("split == 'test'")
                cache[(seed,method)]=frame
        for method in (*LLMS,"m2oe_choice"):
            path=root_for(seed)/f"{method}_matched3"/"per_case.csv"
            if path.exists():
                frame=pd.read_csv(path).query("split == 'test'")
                cache[(seed,f"{method}_matched3")]=frame
    tests=[]
    planned=[]
    for llm in LLMS:
        for base in RANKED_BASELINES:
            planned.extend((llm,llm,base,metric,field,"ranked_candidates") for metric,field in METRICS.items())
        # Compare order-invariant sets under the same maximum candidate budget.
        for metric,field in SET_METRICS.items():
            planned.append((llm,f"{llm}_matched3","m2oe_choice_matched3",metric,field,"matched_unordered_three_candidate_sets"))
    if len(planned) != 78:
        raise RuntimeError(f"Internal planned contrast family changed unexpectedly: {len(planned)} != 78")
    for llm, left_method, base, metric, field, comparison_type in planned:
        per_template={}
        for seed in SEEDS:
            left=cache.get((seed,left_method)); right=cache.get((seed,base))
            if left is None or right is None:
                continue
            a=left[["case_id","template_id",field]]
            b=right[["case_id",field]]
            paired=a.merge(b,on="case_id",suffixes=("_llm","_baseline"),validate="one_to_one")
            paired["difference"]=paired[f"{field}_llm"].astype(float)-paired[f"{field}_baseline"].astype(float)
            for template,group in paired.groupby("template_id"):
                per_template.setdefault(template,[]).append(float(group["difference"].mean()))
        clusters=np.asarray([float(np.mean(values)) for values in per_template.values()],dtype=float)
        if len(clusters)<2:
            continue
        obs=float(clusters.mean())
        signs=rng.choice(np.array([-1.0,1.0]),size=(args.permutations,len(clusters)))
        permuted=np.mean(signs*clusters[None,:],axis=1)
        p=float((1+np.sum(np.abs(permuted)>=abs(obs)))/(args.permutations+1))
        boot=np.empty(args.bootstrap)
        for idx in range(args.bootstrap):
            boot[idx]=rng.choice(clusters,size=len(clusters),replace=True).mean()
        tests.append({"llm":llm,"baseline":base,"metric":metric,"comparison_type":comparison_type,"n_template_clusters":len(clusters),
            "llm_minus_baseline":obs,"template_cluster_bootstrap_95_ci":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
            "two_sided_template_sign_flip_p":p})
    if len(tests) != len(planned):
        raise RuntimeError(f"Incomplete paired contrast family: completed {len(tests)}/{len(planned)}; check per-seed inputs and matched M²OE outputs")
    corrected=holm([row["two_sided_template_sign_flip_p"] for row in tests])
    for row,p in zip(tests,corrected): row["holm_adjusted_p_within_planned_contrast_family"]=p
    result={"seeds":list(SEEDS),"llms":list(LLMS),"ranked_baselines":list(RANKED_BASELINES),"set_baselines":list(SET_BASELINES),
        "ranked_metrics":list(METRICS),"set_metrics":list(SET_METRICS),
        "planned_contrast_count":len(planned),"completed_contrast_count":len(tests),
        "cluster_unit":"held-out task template; average seed-specific paired differences within each template",
        "m2oe_note":"M²OE and the LLM's first three candidates are treated as unordered sets with a matched maximum candidate budget of three; no M²OE top-one or ranked-list comparison is made.",
        "inference":"two-sided paired sign-flip test over template-cluster mean differences; Holm family-wise adjustment",
        "permutations":args.permutations,"bootstrap_replicates":args.bootstrap,"seed":args.seed,"contrasts":tests,
        "runtime":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__},
        "test_script_sha256":sha256_file(Path(__file__).resolve()),
        "aggregate_metrics_json_sha256":sha256_file(HERE/"runs"/"multiseed_analysis"/"multiseed_metrics.json"),
        "paired_input_sha256":{
            str(seed):{
                **{method:sha256_file(root_for(seed)/method/"per_case.csv") for method in (*LLMS,*ALL_BASELINES)},
                **{f"{method}_matched3":sha256_file(root_for(seed)/f"{method}_matched3"/"per_case.csv") for method in (*LLMS,"m2oe_choice")},
            }
            for seed in SEEDS
        }}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"completed_contrasts":len(tests),"out":str(args.out)},indent=2))

if __name__=="__main__": main()
