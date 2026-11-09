#!/usr/bin/env python3
"""Summarize descriptive slices across rotated three-seed held-out templates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEEDS = (20261106, 20261107, 20261108)
METHODS = (
    "qwen35_9b", "deepseek_v2_lite_chat", "llama31_8b_instruct",
    "gower_knn_subspace", "gower_lof_subspace", "top_univariate",
    "sod_subspace_adaptation", "cop_error_vector_adaptation", "rlax_reference_adaptation",
    "all_features",
)
FACTORS = ("task_label_gold", "feature_count", "reference_count", "nuisance_profile")
METRICS = {
    "structured_output_rate": "structurally_valid",
    "support_sufficiency_at_1": "support_top1_sufficient",
    "minimal_exact_at_1": "support_top1_minimal",
    "minimal_Hit_at_4": "support_any_minimal_hit",
    "best_Jaccard_at_4": "support_best_jaccard",
}


def root(seed: int) -> Path:
    return HERE / "runs" / "multiseed_analysis" / str(seed)


def cluster_summary(frame: pd.DataFrame, factor: str, metric: str, n_boot: int, rng: np.random.Generator) -> list[dict]:
    rows = []
    for (level, method), group in frame.groupby([factor, "method"], sort=True):
        by_template = group.groupby("template_id")[metric].mean()
        values = by_template.to_numpy(dtype=float)
        if not len(values):
            continue
        draws = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
        rows.append({
            "factor": factor,
            "level": str(level),
            "method": method,
            "metric": metric,
            "estimate_equal_weighted_over_templates": float(values.mean()),
            "template_cluster_bootstrap_95_ci": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "n_cases_across_seed_test_splits": int(len(group)),
            "n_unique_template_clusters": int(len(values)),
            "seed_ids_contributing": sorted(int(x) for x in group["seed"].unique()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--outdir", type=Path, default=HERE / "runs" / "multiseed_analysis" / "subgroups")
    args = parser.parse_args()
    frames = []
    for seed in SEEDS:
        eval_root = root(seed)
        for method in METHODS:
            path = eval_root / method / "per_case.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path).query("split == 'test'").copy()
            if len(frame) != 288:
                raise RuntimeError(f"{seed}/{method}: expected 288 test cases, found {len(frame)}")
            frame["seed"] = seed
            frame["method"] = method
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    rng = np.random.default_rng(args.seed)
    results = []
    for factor in FACTORS:
        for metric, column in METRICS.items():
            if column not in combined:
                raise KeyError(f"Expected metric column {column} in per-case files")
            values = combined.copy()
            values[column] = values[column].astype(str).str.lower().map({"true": 1.0, "false": 0.0}) if column == "structurally_valid" else pd.to_numeric(values[column], errors="raise")
            results.extend(cluster_summary(values, factor, column, args.bootstrap, rng))
    args.outdir.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(results)
    output.to_csv(args.outdir / "multiseed_subgroup_metrics.csv", index=False, float_format="%.6f")
    payload = {
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "factors": list(FACTORS),
        "metrics": list(METRICS),
        "n_input_test_cases_per_seed_method": 288,
        "n_unique_template_groups_across_test_splits": int(combined["template_id"].nunique()),
        "summary_unit": "For each method and subgroup, average within each template across every seed where it is held out; macro-average templates and bootstrap templates.",
        "analysis_scope": "Descriptive subgroup analysis only; no subgroup hypothesis tests. Some templates contribute more than one seed-level case block.",
        "bootstrap_replicates": args.bootstrap,
        "seed": args.seed,
        "rows": results,
    }
    (args.outdir / "multiseed_subgroup_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(args.outdir / 'multiseed_subgroup_metrics.csv'), "rows": len(results), "templates": payload["n_unique_template_groups_across_test_splits"]}, indent=2))


if __name__ == "__main__":
    main()
