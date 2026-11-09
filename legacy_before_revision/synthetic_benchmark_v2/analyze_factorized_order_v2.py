#!/usr/bin/env python3
"""Score row-order-only and field-order-only prompt perturbations."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
RUNS=HERE/"runs"
PROBE=RUNS/"factorized_order_probe"
MODELS=("qwen35_9b","deepseek_v2_lite_chat","llama31_8b_instruct")
CONDITIONS=("field_order_only","reference_row_order_only")

def load_jsonl(path:Path): return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def export_predictions(source:Path, output:Path, protocol:str|None=None):
    records=load_jsonl(source); latest={}
    for row in records:
        if row.get("event")!="annotation": continue
        if protocol is not None and row.get("protocol_id")!=protocol: continue
        latest[row["case_id"]]=row.get("parsed_prediction")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text("".join(json.dumps({"case_id":cid,**pred} if isinstance(pred,dict) else {"case_id":cid},ensure_ascii=False)+"\n" for cid,pred in sorted(latest.items())),encoding="utf-8")
    return len(latest)

def eval_one(pred:Path, inputs:Path, outdir:Path):
    subprocess.run([sys.executable,str(HERE/"evaluate_benchmark.py"),str(pred),"--inputs-path",str(inputs),"--ground-truth-path",str(PROBE/"ground_truth_96.jsonl"),"--outdir",str(outdir)],check=True,stdout=subprocess.DEVNULL)
    return pd.read_csv(outdir/"per_case.csv").query("split == 'test'").set_index("case_id")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations",type=int,default=20000)
    parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=20261106)
    parser.add_argument("--models",nargs="+",choices=MODELS,default=list(MODELS))
    parser.add_argument("--out",type=Path,default=PROBE/"factorized_order_summary.json")
    args=parser.parse_args()
    models=tuple(args.models)
    inputs={"control":PROBE/"inputs_control.jsonl","field_order_only":PROBE/"inputs_field_order_only.jsonl","reference_row_order_only":PROBE/"inputs_reference_row_order_only.jsonl"}
    annotations={}
    for model in models:
        for condition in ("control",*CONDITIONS):
            if model=="qwen35_9b" and condition=="control":
                source=RUNS/"annotations_qwen35_9b_order_probe_original.jsonl"
                protocol="known_anomaly_subspace_benchmark_v2_order_probe_original_v1"
            else:
                source=PROBE/f"annotations_{model}_{condition}.jsonl"
                protocol=None
            output=PROBE/f"predictions_{model}_{condition}.jsonl"
            n=export_predictions(source,output,protocol)
            if n!=96: raise RuntimeError(f"Expected 96 predictions for {model}/{condition}, found {n} at {source}")
            annotations[(model,condition)]=eval_one(output,inputs[condition],PROBE/f"evaluation_{model}_{condition}")

    rng=np.random.default_rng(args.seed); tests=[]; details={}
    for model in models:
        base=annotations[(model,"control")]
        for condition in CONDITIONS:
            variant=annotations[(model,condition)].reindex(base.index)
            shared=base.index.intersection(variant.index)
            template=base.loc[shared,"template_id"]
            details[f"{model}/{condition}"]={}
            for metric in ("support_any_minimal_hit","support_best_jaccard"):
                diff=variant.loc[shared,metric].astype(float)-base.loc[shared,metric].astype(float)
                clusters={key:diff[template.eq(key)].to_numpy() for key in sorted(template.unique())}
                cluster_means=np.asarray([values.mean() for values in clusters.values()])
                observed=float(cluster_means.mean())
                signs=rng.choice(np.array([-1.0,1.0]),size=(args.permutations,len(cluster_means)))
                permuted=(signs*cluster_means[None,:]).mean(axis=1)
                p=float((1+np.sum(np.abs(permuted)>=abs(observed)))/(args.permutations+1))
                boots=np.empty(args.bootstrap)
                keys=list(clusters)
                for b in range(args.bootstrap):
                    picked=rng.choice(keys,size=len(keys),replace=True)
                    boots[b]=np.concatenate([clusters[key] for key in picked]).mean()
                item={"control_mean":float(base.loc[shared,metric].astype(float).mean()),
                      "condition_mean":float(variant.loc[shared,metric].astype(float).mean()),
                      "paired_difference":observed,
                      "template_cluster_bootstrap_95_ci":[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],
                      "two_sided_template_sign_flip_p":p,"n_cases":len(shared),"n_templates":len(clusters)}
                details[f"{model}/{condition}"][metric]=item
                tests.append({"model":model,"condition":condition,"metric":metric,**item})
            valid=base["structurally_valid"].astype(str).str.lower().eq("true") & variant["structurally_valid"].astype(str).str.lower().eq("true")
            left=base["explanations_predicted"].map(json.loads); right=variant["explanations_predicted"].map(json.loads)
            valid &= left.map(lambda x: bool(x) and bool(x[0])) & right.map(lambda x: bool(x) and bool(x[0]))
            jaccards=[]
            for cid in shared[valid.reindex(shared).fillna(False).to_numpy()]:
                a=set(left.loc[cid][0]); b=set(right.loc[cid][0]); jaccards.append(len(a&b)/len(a|b) if a|b else 1.0)
            details[f"{model}/{condition}"]["top1_prediction_stability"]={"n_valid_pairs":len(jaccards),"mean_jaccard":float(np.mean(jaccards)) if jaccards else None}
    # Holm correction is applied across all requested model/condition/metric contrasts.
    order=np.argsort([item["two_sided_template_sign_flip_p"] for item in tests]); adjusted=np.empty(len(tests)); running=0.0
    for rank,index in enumerate(order):
        current=min(1.0,(len(tests)-rank)*tests[index]["two_sided_template_sign_flip_p"])
        running=max(running,current); adjusted[index]=running
    for item,p in zip(tests,adjusted): item["holm_adjusted_p_12_contrast_family"]=float(p)
    for item in tests:
        item["holm_adjusted_p_contrast_family"]=item.pop("holm_adjusted_p_12_contrast_family")
        details[f"{item['model']}/{item['condition']}"][item["metric"]]["holm_adjusted_p_contrast_family"]=item["holm_adjusted_p_contrast_family"]
    result={"n_cases_per_condition":96,"n_templates":24,"conditions":list(CONDITIONS),"models":list(models),
        "seed":args.seed,"permutations":args.permutations,"bootstrap_replicates":args.bootstrap,
        "input_change":"field-order condition reverses schema and within-row field order, preserving reference row order; row-order condition reverses only the reference-row list",
        "inference":f"paired template-cluster bootstrap and two-sided template sign-flip tests, Holm-adjusted across {len(tests)} requested metric contrasts",
        "results":details,"contrast_rows":tests}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"out":str(args.out),"n_contrasts":len(tests)},indent=2))

if __name__=="__main__": main()
