#!/usr/bin/env python3
"""Score fixed v2 baselines and aggregate random-subspace replicates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import random
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASELINES = HERE / "runs" / "baseline_predictions"
EVALUATIONS = HERE / "runs" / "baseline_evaluations"
SCORER = HERE / "evaluate_benchmark.py"
METRICS = (
    "structured_output_rate",
    "support_top1_sufficiency_rate",
    "support_top1_minimal_exact_hit_rate",
    "support_any_minimal_hit_at_4",
    "support_mean_recall_at_4",
    "support_mean_reciprocal_rank",
    "support_mean_best_jaccard",
    "rule_top1_sufficiency_rate",
    "rule_any_minimal_hit_at_4",
    "mean_predicted_subspace_fraction",
)


def score(path: Path, name: str) -> tuple[dict, list[dict]]:
    destination = EVALUATIONS / name
    subprocess.run(
        [sys.executable, str(SCORER), str(path), "--outdir", str(destination)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    with (destination / "per_case.csv").open(encoding="utf-8", newline="") as stream:
        per_case = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    return summary["test"], per_case


def ci95_by_template(case_values: dict[str, tuple[str, float]], seed: int, n_bootstrap: int = 2000) -> dict:
    grouped: dict[str, list[float]] = defaultdict(list)
    for template_id, value in case_values.values():
        grouped[template_id].append(value)
    template_ids = sorted(grouped)
    point = sum(sum(grouped[key]) for key in template_ids) / sum(len(grouped[key]) for key in template_ids)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_bootstrap):
        sampled = rng.choices(template_ids, k=len(template_ids))
        values = [value for template in sampled for value in grouped[template]]
        draws.append(sum(values) / len(values))
    draws.sort()
    return {
        "estimate": point,
        "ci95_low": draws[int(0.025 * (n_bootstrap - 1))],
        "ci95_high": draws[int(0.975 * (n_bootstrap - 1))],
        "n_template_clusters": len(template_ids),
        "n_bootstrap": n_bootstrap,
    }


def numeric(row: dict, key: str) -> float:
    value = row.get(key)
    if key == "support_first_hit_rank":
        return 1.0 / float(value) if value else 0.0
    if value is None or value == "":
        return 0.0
    if str(value).lower() in {"true", "false"}:
        return float(str(value).lower() == "true")
    return float(value)


def with_template_intervals(name: str, score_summary: dict, case_rows: list[dict], seed: int) -> dict:
    fields = {
        "structured_output_rate": "structurally_valid",
        "support_top1_sufficiency_rate": "support_top1_sufficient",
        "support_top1_minimal_exact_hit_rate": "support_top1_minimal",
        "support_any_minimal_hit_at_4": "support_any_minimal_hit",
        "support_mean_recall_at_4": "support_recall_at_4",
        "support_mean_reciprocal_rank": "support_first_hit_rank",
        "support_mean_best_jaccard": "support_best_jaccard",
        "mean_predicted_subspace_fraction": "mean_predicted_subspace_fraction",
    }
    result = dict(score_summary)
    result["template_cluster_ci95"] = {}
    for offset, (metric, field) in enumerate(fields.items()):
        values = {
            row["case_id"]: (row["template_id"], numeric(row, field))
            for row in case_rows
        }
        result["template_cluster_ci95"][metric] = ci95_by_template(values, seed + offset)
    result["template_cluster_ci95_method"] = "resample 24 templates with replacement; retain all cases per sampled template"
    return result


def stats(values: list[float]) -> dict:
    values = sorted(values)
    if not values:
        return {"mean": None, "sd": None, "p05": None, "p50": None, "p95": None}
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0

    def quantile(q: float) -> float:
        position = q * (len(values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    return {"mean": mean, "sd": sd, "p05": quantile(0.05), "p50": quantile(0.5), "p95": quantile(0.95)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=100)
    args = parser.parse_args()
    results = {}
    for name in ("all_features", "top_univariate", "oracle_reference_support", "oracle_rule"):
        summary, case_rows = score(BASELINES / f"predictions_{name}.jsonl", name)
        results[name] = with_template_intervals(name, summary, case_rows, 20261105)

    random_results = []
    for replicate in range(args.replicates):
        path = BASELINES / f"predictions_random_{replicate:03d}.jsonl"
        random_results.append(score(path, f"random_{replicate:03d}"))
    random_case_values: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for _, case_rows in random_results:
        for row in case_rows:
            random_case_values[row["case_id"]].append((row["template_id"], row))
    random_fields = {
        "structured_output_rate": "structurally_valid",
        "support_top1_sufficiency_rate": "support_top1_sufficient",
        "support_top1_minimal_exact_hit_rate": "support_top1_minimal",
        "support_any_minimal_hit_at_4": "support_any_minimal_hit",
        "support_mean_recall_at_4": "support_recall_at_4",
        "support_mean_reciprocal_rank": "support_first_hit_rank",
        "support_mean_best_jaccard": "support_best_jaccard",
        "mean_predicted_subspace_fraction": "mean_predicted_subspace_fraction",
    }
    per_case_means = {}
    for metric, field in random_fields.items():
        per_case_means[metric] = {
            case_id: (
                rows[0][0],
                sum(numeric(row, field) for _, row in rows) / len(rows),
            )
            for case_id, rows in random_case_values.items()
        }
    random_template_ci = {
        metric: ci95_by_template(values, 20261105 + offset)
        for offset, (metric, values) in enumerate(per_case_means.items())
    }
    random_seed_summaries = [summary for summary, _ in random_results]
    results["random_subspace"] = {
        "n_replicates": args.replicates,
        "n_cases_per_replicate": random_seed_summaries[0]["n_cases"] if random_seed_summaries else 0,
        "metrics": {
            metric: stats([float(row[metric]) for row in random_seed_summaries if isinstance(row.get(metric), (int, float))])
            for metric in METRICS
        },
        "template_cluster_ci95_on_seed_averaged_cases": random_template_ci,
        "template_cluster_ci95_method": "average each case over random seeds, then resample 24 templates with replacement",
    }
    output = EVALUATIONS / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "random_replicates": args.replicates}, ensure_ascii=False))


if __name__ == "__main__":
    main()
