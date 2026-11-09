#!/usr/bin/env python3
"""Cluster-bootstrap baseline summaries and exploratory model comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
MODEL_NAMES = ("qwen35_9b", "deepseek_v2_lite_chat", "llama31_8b_instruct")
BASELINE_NAMES = ("gower_knn_subspace", "gower_lof_subspace", "top_univariate")
METRICS = {
    "support_sufficiency_at_1": "support_top1_sufficient",
    "minimal_exact_at_1": "support_top1_minimal",
    "minimal_hit_at_4": "support_any_minimal_hit",
    "best_jaccard": "support_best_jaccard",
    "compactness": "mean_predicted_subspace_fraction",
}
PAIRED_METRICS = {key: value for key, value in METRICS.items() if key != "compactness"}
SECONDARY_RANK_METRICS = {
    "mean_reciprocal_rank": "support_first_hit_rank",
    "recall_at_4": "support_recall_at_4",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row.get("split") == "test"]


def val(row: dict, field: str) -> float:
    value = row.get(field)
    if field == "support_first_hit_rank":
        return 1.0 / float(value) if value not in (None, "") else 0.0
    if value is None or value == "":
        return 0.0
    if str(value).lower() == "true":
        return 1.0
    if str(value).lower() == "false":
        return 0.0
    return float(value)


def grouped(rows: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        result[row["template_id"]].append(row)
    return dict(result)


def cluster_bootstrap(rows: list[dict], field: str, rng: random.Random, replicates: int) -> dict:
    groups = grouped(rows)
    templates = sorted(groups)

    def average(sampled_templates: list[str]) -> float:
        sample = [row for template in sampled_templates for row in groups[template]]
        return sum(val(row, field) for row in sample) / len(sample)

    draws = [average(rng.choices(templates, k=len(templates))) for _ in range(replicates)]
    draws.sort()
    return {
        "estimate": average(templates),
        "ci95": [draws[int(.025 * (replicates - 1))], draws[int(.975 * (replicates - 1))]],
        "n_template_clusters": len(templates),
    }


def signflip(differences: list[float], rng: random.Random, permutations: int) -> float:
    observed = abs(sum(differences) / len(differences))
    exceed = 0
    for _ in range(permutations):
        draw = abs(sum(value * rng.choice((-1.0, 1.0)) for value in differences) / len(differences))
        exceed += draw >= observed - 1e-15
    return (exceed + 1) / (permutations + 1)


def holm(p_values: dict[str, float]) -> dict[str, float]:
    order = sorted(p_values, key=p_values.get)
    result: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[key]))
        result[key] = running
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--out", type=Path, default=RUNS / "baseline_comparison_gower.json")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    all_rows: dict[str, dict[str, list[dict]]] = {"models": {}, "baselines": {}}
    for model in MODEL_NAMES:
        path = RUNS / f"evaluation_{model}_known_anomaly_subspace_benchmark_v2_compact_json_v1" / "per_case.csv"
        all_rows["models"][model] = read_csv(path)
    for baseline in (*BASELINE_NAMES, "top_univariate", "all_features"):
        path = RUNS / "baseline_evaluations" / baseline / "per_case.csv"
        all_rows["baselines"][baseline] = read_csv(path)

    summaries = {}
    for family, methods in all_rows.items():
        summaries[family] = {}
        for method, rows in methods.items():
            summaries[family][method] = {
                metric: cluster_bootstrap(rows, field, rng, args.bootstrap)
                for metric, field in METRICS.items()
            }

    # Ranking metrics are reported descriptively with template-cluster CIs.
    # They are secondary and do not expand the prespecified 36-test Holm family.
    rank_rng = random.Random(args.seed + 104729)
    ranking_summaries = {}
    for family, methods in all_rows.items():
        ranking_summaries[family] = {}
        for method, rows in methods.items():
            ranking_summaries[family][method] = {
                metric: cluster_bootstrap(rows, field, rank_rng, args.bootstrap)
                for metric, field in SECONDARY_RANK_METRICS.items()
            }
    random_summary_path = RUNS / "baseline_evaluations" / "summary.json"
    if random_summary_path.exists():
        random_summary = json.loads(random_summary_path.read_text(encoding="utf-8"))
        random_ci = random_summary["random_subspace"]["template_cluster_ci95_on_seed_averaged_cases"]
        random_metric_sources = {
            "mean_reciprocal_rank": "support_mean_reciprocal_rank",
            "recall_at_4": "support_mean_recall_at_4",
        }
        ranking_summaries["baselines"]["random_subspace"] = {
            metric: {
                "estimate": random_ci[random_metric_sources[metric]]["estimate"],
                "ci95": [random_ci[random_metric_sources[metric]]["ci95_low"], random_ci[random_metric_sources[metric]]["ci95_high"]],
                "n_template_clusters": random_ci[random_metric_sources[metric]]["n_template_clusters"],
            }
            for metric in SECONDARY_RANK_METRICS
        }

    comparisons = {}
    raw_p = {}
    for baseline in BASELINE_NAMES:
        base_by_id = {row["case_id"]: row for row in all_rows["baselines"][baseline]}
        for model in MODEL_NAMES:
            model_by_id = {row["case_id"]: row for row in all_rows["models"][model]}
            if model_by_id.keys() != base_by_id.keys():
                raise RuntimeError(f"case mismatch: {model} vs {baseline}")
            template_for = {row["case_id"]: row["template_id"] for row in all_rows["models"][model]}
            groups: dict[str, list[str]] = defaultdict(list)
            for case_id in model_by_id:
                groups[template_for[case_id]].append(case_id)
            comparisons[f"{model}_minus_{baseline}"] = {}
            for metric, field in PAIRED_METRICS.items():
                per_case_diff = {
                    case_id: val(model_by_id[case_id], field) - val(base_by_id[case_id], field)
                    for case_id in model_by_id
                }
                templates = sorted(groups)
                template_diff = [
                    sum(per_case_diff[cid] for cid in groups[t]) / len(groups[t])
                    for t in templates
                ]
                draws = []
                for _ in range(args.bootstrap):
                    sampled = rng.choices(templates, k=len(templates))
                    sampled_ids = [cid for t in sampled for cid in groups[t]]
                    draws.append(sum(per_case_diff[cid] for cid in sampled_ids) / len(sampled_ids))
                draws.sort()
                key = f"{model}_minus_{baseline}:{metric}"
                raw_p[key] = signflip(template_diff, rng, args.permutations)
                comparisons[f"{model}_minus_{baseline}"][metric] = {
                    "difference": sum(per_case_diff.values()) / len(per_case_diff),
                    "ci95": [draws[int(.025 * (args.bootstrap - 1))], draws[int(.975 * (args.bootstrap - 1))]],
                    "cluster_signflip_p_raw": raw_p[key],
                    "n_cases": len(per_case_diff),
                    "n_templates": len(templates),
                }
    adjusted = holm(raw_p)
    for key, value in adjusted.items():
        pair, metric = key.split(":", 1)
        comparisons[pair][metric]["holm_adjusted_p_36_model_vs_statistical_baselines"] = value

    result = {
        "benchmark_split": "test",
        "n_cases": 288,
        "bootstrap_replicates": args.bootstrap,
        "permutations": args.permutations,
        "uncertainty_unit": "template group; 24 groups, retaining all cases per sampled group",
        "comparisons_are_exploratory": True,
        "multiple_testing": "Holm correction over 36 model-vs-statistical-baseline contrasts (3 models x 3 baselines x 4 metrics)",
        "primary_comparison_baselines": list(BASELINE_NAMES),
        "secondary_ranking_metrics": "MRR and recall@4 are descriptive with template-cluster intervals; no additional hypothesis tests are performed",
        "summaries": summaries,
        "secondary_ranking_metric_summaries": ranking_summaries,
        "paired_comparisons": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "n_comparisons": len(raw_p)}, indent=2))


if __name__ == "__main__":
    main()
