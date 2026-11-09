#!/usr/bin/env python3
"""Evaluate and summarize the primary plus independent v2 seed replicates."""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = (20261106, 20261107, 20261108)
MODELS = ("qwen35_9b", "deepseek_v2_lite_chat", "llama31_8b_instruct")
METRICS = (
    "support_top1_sufficiency_rate", "support_top1_minimal_exact_hit_rate",
    "support_any_minimal_hit_at_4", "support_mean_recall_at_4",
    "support_mean_reciprocal_rank", "support_mean_best_jaccard",
    "mean_predicted_subspace_fraction",
    "rule_top1_sufficiency_rate", "rule_top1_minimal_exact_hit_rate",
    "rule_any_minimal_hit_at_4",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paths(seed: int) -> tuple[Path, Path, Path]:
    if seed == 20261106:
        root = HERE / "runs" / "multiseed" / "seed_20261106_rt522"
        return HERE / "inputs.jsonl", HERE / "ground_truth.jsonl", root
    root = HERE / "runs" / "multiseed" / f"seed_{seed}"
    return root / "inputs.jsonl", root / "ground_truth.jsonl", root


def annotations_to_predictions(source: Path, output: Path, protocol: str) -> Path | None:
    if not source.exists():
        return None
    latest = {}
    for row in read_jsonl(source):
        if row.get("event") == "annotation" and row.get("protocol_id") == protocol:
            latest[row["case_id"]] = row.get("parsed_prediction")
    if not latest:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for case_id, pred in sorted(latest.items()):
            stream.write(json.dumps({"case_id": case_id, **pred} if isinstance(pred, dict) else {"case_id": case_id}, ensure_ascii=False) + "\n")
    return output


def evaluate(pred_path: Path, inputs: Path, truth: Path, outdir: Path) -> dict | None:
    if not pred_path.exists():
        return None
    expected = {row["case_id"] for row in read_jsonl(inputs) if row.get("split") == "test"}
    observed = {row["case_id"] for row in read_jsonl(pred_path) if row.get("case_id")}
    if observed != expected:
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / "evaluator.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run([
            sys.executable, str(HERE / "evaluate_benchmark.py"), str(pred_path),
            "--inputs-path", str(inputs), "--ground-truth-path", str(truth), "--outdir", str(outdir),
        ], stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Evaluator failed for {pred_path}; see {log}")
    summary_path = outdir / "summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))["test"]


def random_prior(case_truth: dict, random_dir: Path) -> tuple[dict, dict]:
    truths = {row["case_id"]: row for row in case_truth}
    files = sorted(random_dir.glob("predictions_random_*.jsonl"))
    if not files:
        return {}, {}
    by_rep = {metric: [] for metric in (
        "support_top1_sufficiency_rate", "support_any_minimal_hit_at_4",
        "support_top1_minimal_exact_hit_rate", "support_mean_best_jaccard",
        "rule_top1_sufficiency_rate", "rule_any_minimal_hit_at_4",
        "rule_top1_minimal_exact_hit_rate", "mean_predicted_subspace_fraction",
    )}
    case_replicate_scores: dict[str, dict[str, list[float]]] = {}
    for path in files:
        rows = read_jsonl(path)
        suff, hit, exact, jac, rule_suff, rule_hit, rule_exact, size = [], [], [], [], [], [], [], []
        for pred in rows:
            truth = truths[pred["case_id"]]
            gold = {frozenset(x) for x in truth["reference_support_sufficient_minimal_subspaces"]}
            rule_gold = {frozenset(x) for x in truth["rule_minimal_subspaces"]}
            candidates = [frozenset(x["features"]) for x in pred.get("explanations", [])]
            suff.append(float(bool(candidates) and any(gold_set.issubset(candidates[0]) for gold_set in gold)))
            hit.append(float(any(candidate in gold for candidate in candidates)))
            exact.append(float(bool(candidates) and candidates[0] in gold))
            jac.append(max((len(a & b) / len(a | b) if a | b else 1.0 for a in candidates for b in gold), default=0.0))
            rule_suff.append(float(bool(candidates) and any(gold_set.issubset(candidates[0]) for gold_set in rule_gold)))
            rule_hit.append(float(any(candidate in rule_gold for candidate in candidates)))
            rule_exact.append(float(bool(candidates) and candidates[0] in rule_gold))
            size.append(float(np.mean([len(candidate) / truths[pred["case_id"]]["feature_count"] for candidate in candidates])) if candidates else 0.0)
            entry = case_replicate_scores.setdefault(pred["case_id"], {
                "suff": [], "hit": [], "exact": [], "jaccard": [],
                "rule_suff": [], "rule_hit": [], "rule_exact": [], "size": [],
            })
            entry["suff"].append(suff[-1]); entry["hit"].append(hit[-1]); entry["exact"].append(exact[-1]); entry["jaccard"].append(jac[-1])
            entry["rule_suff"].append(rule_suff[-1]); entry["rule_hit"].append(rule_hit[-1]); entry["rule_exact"].append(rule_exact[-1]); entry["size"].append(size[-1])
        for metric, values in zip(by_rep, (suff, hit, exact, jac, rule_suff, rule_hit, rule_exact, size)):
            by_rep[metric].append(float(np.mean(values)))
    truth_by_id = truths
    per_case = {}
    for key, value in case_replicate_scores.items():
        per_case[key] = {
            "case_id": key,
            "template_id": truth_by_id[key]["template_id"],
            "support_top1_sufficiency_rate": float(np.mean(value["suff"])),
            "support_any_minimal_hit_at_4": float(np.mean(value["hit"])),
            "support_top1_minimal_exact_hit_rate": float(np.mean(value["exact"])),
            "support_mean_best_jaccard": float(np.mean(value["jaccard"])),
            "rule_top1_sufficiency_rate": float(np.mean(value["rule_suff"])),
            "rule_any_minimal_hit_at_4": float(np.mean(value["rule_hit"])),
            "rule_top1_minimal_exact_hit_rate": float(np.mean(value["rule_exact"])),
            "mean_predicted_subspace_fraction": float(np.mean(value["size"])),
        }
    summary = {
        metric: {"mean_over_random_replicates": float(np.mean(values)), "sd_over_random_replicates": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
        for metric, values in by_rep.items()
    }
    summary["n_random_replicates"] = len(files)
    return summary, per_case


def cluster_bootstrap(per_case_by_seed: list[dict], metric: str, n_boot: int, seed: int) -> list[float] | None:
    rng = np.random.default_rng(seed)
    per_seed: list[dict[str, np.ndarray]] = []
    column_map = {
        "support_top1_sufficiency_rate": "support_top1_sufficient",
        "support_top1_minimal_exact_hit_rate": "support_top1_minimal",
        "support_any_minimal_hit_at_4": "support_any_minimal_hit",
        "support_mean_recall_at_4": "support_recall_at_4",
        "support_mean_best_jaccard": "support_best_jaccard",
        "rule_top1_sufficiency_rate": "rule_top1_sufficient",
        "rule_top1_minimal_exact_hit_rate": "rule_top1_minimal",
        "rule_any_minimal_hit_at_4": "rule_any_minimal_hit",
    }
    for df in per_case_by_seed:
        if metric == "support_mean_reciprocal_rank":
            if "support_first_hit_rank" not in df:
                return None
            values = df["support_first_hit_rank"].map(
                lambda rank: 1.0 / float(rank) if str(rank).strip() not in {"", "nan", "None"} and float(rank) > 0 else 0.0
            ).to_numpy(dtype=float)
        else:
            column = metric if metric in df else column_map.get(metric, metric)
            if column not in df:
                return None
            values = df[column].to_numpy(dtype=float)
        working = df[["template_id"]].copy()
        working["_metric"] = values
        # Compactness is undefined for an invalid/abstaining response and is
        # intentionally summarized conditional on a structured prediction.
        # Keep templates with at least one defined observation; drop all-NaN
        # clusters before the template bootstrap.
        grouped = {
            key: valid.to_numpy(dtype=float)
            for key, group in working.groupby("template_id")
            if len(valid := group["_metric"].dropna())
        }
        per_seed.append(grouped)
    # A rule template can occur in the test split of more than one rotated
    # seed. Resample the union of template IDs once per replicate so those
    # repeated appearances receive the same cluster multiplicity across seeds.
    template_ids = sorted({template for grouped in per_seed for template in grouped})
    values = []
    for _ in range(n_boot):
        picked = rng.choice(template_ids, size=len(template_ids), replace=True)
        multiplicity = Counter(picked)
        seed_means = []
        for grouped in per_seed:
            sample_parts = [
                grouped[template]
                for template, count in multiplicity.items()
                if template in grouped
                for _ in range(int(count))
            ]
            if not sample_parts:
                continue
            sample = np.concatenate(sample_parts)
            seed_means.append(float(sample.mean()))
        if not seed_means:
            continue
        values.append(float(np.mean(seed_means)))
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=HERE / "runs" / "multiseed_analysis")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    # M²OE exposes an unordered choice set, not a ranked list. Keep it out of
    # ranked metrics here; its matched-budget, order-invariant comparison is
    # produced by analyze_m2oe_matched3_v2.py.
    methods = [*MODELS, "all_features", "top_univariate", "gower_knn_subspace", "gower_lof_subspace", "sod_subspace_adaptation", "cop_error_vector_adaptation", "rlax_reference_adaptation"]
    seed_summaries: dict[str, dict[int, dict]] = {method: {} for method in methods}
    per_case: dict[str, dict[int, object]] = {method: {} for method in methods}

    for seed in SEEDS:
        inputs, truth, root = paths(seed)
        truths = read_jsonl(truth)
        seed_root = root if seed != 20261106 else root
        eval_root = HERE / "runs" / "multiseed_analysis" / str(seed)
        for model in MODELS:
            source = seed_root / f"annotations_{model}.jsonl"
            protocol = f"known_anomaly_subspace_v2_seed_{seed}_rt522" if seed == 20261106 else f"known_anomaly_subspace_v2_seed_{seed}"
            pred = annotations_to_predictions(source, eval_root / f"predictions_{model}.jsonl", protocol)
            pred = pred or Path("/__not_available__")
            summary = evaluate(pred, inputs, truth, eval_root / model)
            if summary is not None:
                seed_summaries[model][seed] = summary
                import pandas as pd
                frame = pd.read_csv(eval_root / model / "per_case.csv").query("split == 'test'")
                per_case[model][seed] = frame

        fixed = {
            "all_features": (HERE / "runs" / "baseline_predictions" / "predictions_all_features.jsonl") if seed == 20261106 else root / "baselines" / "predictions_all_features.jsonl",
            "top_univariate": (HERE / "runs" / "baseline_predictions" / "predictions_top_univariate.jsonl") if seed == 20261106 else root / "baselines" / "predictions_top_univariate.jsonl",
            "gower_knn_subspace": (HERE / "runs" / "baseline_predictions" / "predictions_gower_knn_subspace.jsonl") if seed == 20261106 else root / "gower" / "predictions_gower_knn_subspace.jsonl",
            "gower_lof_subspace": (HERE / "runs" / "baseline_predictions" / "predictions_gower_lof_subspace.jsonl") if seed == 20261106 else root / "gower" / "predictions_gower_lof_subspace.jsonl",
            "sod_subspace_adaptation": (HERE / "runs" / "sod_cop_adaptations" / "predictions_sod_subspace_adaptation.jsonl") if seed == 20261106 else root / "sod_cop" / "predictions_sod_subspace_adaptation.jsonl",
            "cop_error_vector_adaptation": (HERE / "runs" / "sod_cop_adaptations" / "predictions_cop_error_vector_adaptation.jsonl") if seed == 20261106 else root / "sod_cop" / "predictions_cop_error_vector_adaptation.jsonl",
            "rlax_reference_adaptation": (HERE / "runs" / "rlax_reference_adaptation" / "predictions_rlax_reference_adaptation.jsonl") if seed == 20261106 else root / "rlax" / "predictions_rlax_reference_adaptation.jsonl",
        }
        for method, pred in fixed.items():
            summary = evaluate(pred, inputs, truth, eval_root / method)
            if summary is not None:
                seed_summaries[method][seed] = summary
                import pandas as pd
                per_case[method][seed] = pd.read_csv(eval_root / method / "per_case.csv").query("split == 'test'")
        random_dir = (HERE / "runs" / "baseline_predictions") if seed == 20261106 else root / "baselines"
        rand_summary, rand_case = random_prior(truths, random_dir)
        if rand_summary:
            seed_summaries.setdefault("random_size_prior", {})[seed] = {
                metric: value["mean_over_random_replicates"]
                for metric, value in rand_summary.items()
                if metric != "n_random_replicates"
            }
            seed_summaries["random_size_prior"][seed]["random_replicate_sd"] = rand_summary["support_any_minimal_hit_at_4"]["sd_over_random_replicates"]
            import pandas as pd
            per_case.setdefault("random_size_prior", {})[seed] = pd.DataFrame(list(rand_case.values()))

    import pandas as pd
    rows = []
    for method, by_seed in seed_summaries.items():
        metrics = [m for m in METRICS if any(m in v for v in by_seed.values())]
        for metric in metrics:
            seed_values = [float(by_seed[s][metric]) for s in SEEDS if s in by_seed and metric in by_seed[s]]
            frames = [per_case[method][s] for s in SEEDS if s in per_case.get(method, {})]
            ci = cluster_bootstrap(frames, metric, args.bootstrap, 20261106) if frames else None
            rows.append({
                "method": method, "metric": metric,
                "n_seed_replicates": len(seed_values),
                "per_seed_values": json.dumps({str(s): by_seed[s].get(metric) for s in SEEDS if s in by_seed and metric in by_seed[s]}),
                "mean_over_seeds": float(np.mean(seed_values)) if seed_values else None,
                "sd_over_seeds": float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else None,
                "template_cluster_bootstrap_95_ci_macro_seed": json.dumps(ci) if ci else "",
            })
    out_csv = args.outdir / "multiseed_metrics.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    result = {
        "seeds": list(SEEDS),
        "seed_template_counts": {
            str(seed): json.loads(((HERE / "manifest.json") if seed == 20261106 else (paths(seed)[2] / "manifest.json")).read_text())["split_template_counts"]
            for seed in SEEDS
        },
        "methods_with_results": {method: sorted(seed_summaries.get(method, {})) for method in seed_summaries},
        "n_template_clusters_bootstrap": "resample the union of held-out template IDs once per replicate; apply shared template multiplicities to all seed-specific splits, then macro-average seed means",
        "bootstrap_replicates": args.bootstrap,
        "metrics_csv": str(out_csv.resolve()),
        "per_seed_summaries": seed_summaries,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "analysis_sha256": sha256_file(Path(__file__).resolve()),
        "evaluator_sha256": sha256_file(HERE / "evaluate_benchmark.py"),
        "seed_artifact_sha256": {
            str(seed): {
                "inputs": sha256_file(paths(seed)[0]),
                "ground_truth": sha256_file(paths(seed)[1]),
                "run_log": sha256_file(paths(seed)[2] / "model_runs.jsonl"),
                "model_annotations": {
                    model: sha256_file(paths(seed)[2] / f"annotations_{model}.jsonl")
                    for model in MODELS
                },
                "baseline_predictions": {
                    method: sha256_file(pred)
                    for method, pred in {
                        "all_features": (HERE / "runs" / "baseline_predictions" / "predictions_all_features.jsonl") if seed == 20261106 else paths(seed)[2] / "baselines" / "predictions_all_features.jsonl",
                        "top_univariate": (HERE / "runs" / "baseline_predictions" / "predictions_top_univariate.jsonl") if seed == 20261106 else paths(seed)[2] / "baselines" / "predictions_top_univariate.jsonl",
                        "gower_knn_subspace": (HERE / "runs" / "baseline_predictions" / "predictions_gower_knn_subspace.jsonl") if seed == 20261106 else paths(seed)[2] / "gower" / "predictions_gower_knn_subspace.jsonl",
                        "gower_lof_subspace": (HERE / "runs" / "baseline_predictions" / "predictions_gower_lof_subspace.jsonl") if seed == 20261106 else paths(seed)[2] / "gower" / "predictions_gower_lof_subspace.jsonl",
                        "sod_subspace_adaptation": (HERE / "runs" / "sod_cop_adaptations" / "predictions_sod_subspace_adaptation.jsonl") if seed == 20261106 else paths(seed)[2] / "sod_cop" / "predictions_sod_subspace_adaptation.jsonl",
                        "cop_error_vector_adaptation": (HERE / "runs" / "sod_cop_adaptations" / "predictions_cop_error_vector_adaptation.jsonl") if seed == 20261106 else paths(seed)[2] / "sod_cop" / "predictions_cop_error_vector_adaptation.jsonl",
                        "m2oe_choice": (HERE / "runs" / "m2oe" / "predictions_m2oe_choice.jsonl") if seed == 20261106 else paths(seed)[2] / "m2oe" / "predictions_m2oe_choice.jsonl",
                        "rlax_reference_adaptation": (HERE / "runs" / "rlax_reference_adaptation" / "predictions_rlax_reference_adaptation.jsonl") if seed == 20261106 else paths(seed)[2] / "rlax" / "predictions_rlax_reference_adaptation.jsonl",
                    }.items()
                },
            }
            for seed in SEEDS
        },
    }
    (args.outdir / "multiseed_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"metrics_csv": str(out_csv), "methods_with_results": result["methods_with_results"]}, indent=2))


if __name__ == "__main__":
    main()
