#!/usr/bin/env python3
"""Jointly Holm-adjust the four paired prompt/order sensitivity conditions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
METRICS = ("support_any_minimal_hit", "support_best_jaccard")
CONDITIONS = {
    "joint_field_and_row_reversal": (
        RUNS / "order_probe_original_evaluation" / "per_case.csv",
        RUNS / "order_probe_reverse_evaluation" / "per_case.csv",
    ),
    "field_order_only_reversal": (
        RUNS / "factorized_order_probe" / "evaluation_qwen35_9b_control" / "per_case.csv",
        RUNS / "factorized_order_probe" / "evaluation_qwen35_9b_field_order_only" / "per_case.csv",
    ),
    "reference_row_order_only_reversal": (
        RUNS / "factorized_order_probe" / "evaluation_qwen35_9b_control" / "per_case.csv",
        RUNS / "factorized_order_probe" / "evaluation_qwen35_9b_reference_row_order_only" / "per_case.csv",
    ),
    "opaque_feature_id_permutation": (
        RUNS / "id_permutation_probe" / "evaluation_control" / "per_case.csv",
        RUNS / "id_permutation_probe" / "evaluation_id_permutation" / "per_case.csv",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    tests = []
    for condition, (control_path, variant_path) in CONDITIONS.items():
        control = pd.read_csv(control_path).query("split == 'test'").set_index("case_id")
        variant = pd.read_csv(variant_path).query("split == 'test'").set_index("case_id")
        shared = sorted(control.index.intersection(variant.index))
        if len(shared) != 96:
            raise RuntimeError(f"{condition}: expected 96 paired cases, got {len(shared)}")
        if not control.loc[shared, "template_id"].equals(variant.loc[shared, "template_id"]):
            raise RuntimeError(f"{condition}: paired cases do not have identical template assignments")
        templates = control.loc[shared, "template_id"]
        for metric in METRICS:
            diff = variant.loc[shared, metric].astype(float) - control.loc[shared, metric].astype(float)
            clusters = np.asarray([diff[templates.eq(name)].mean() for name in sorted(templates.unique())], dtype=float)
            observed = float(clusters.mean())
            signs = rng.choice(np.array([-1.0, 1.0]), size=(args.permutations, len(clusters)))
            permuted = np.mean(signs * clusters[None, :], axis=1)
            p_value = float((1 + np.sum(np.abs(permuted) >= abs(observed))) / (args.permutations + 1))
            draws = np.asarray([rng.choice(clusters, size=len(clusters), replace=True).mean() for _ in range(args.bootstrap)])
            tests.append({
                "condition": condition,
                "metric": metric,
                "control_mean": float(control.loc[shared, metric].astype(float).mean()),
                "condition_mean": float(variant.loc[shared, metric].astype(float).mean()),
                "condition_minus_control": observed,
                "template_cluster_bootstrap_95_ci": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
                "two_sided_template_sign_flip_p": p_value,
                "n_cases": len(shared),
                "n_templates": int(templates.nunique()),
            })
    adjusted = holm([row["two_sided_template_sign_flip_p"] for row in tests])
    for row, value in zip(tests, adjusted):
        row["holm_adjusted_p_all_eight_sensitivity_contrasts"] = value
    result = {
        "model": "Qwen3.5-9B",
        "conditions": list(CONDITIONS),
        "metrics": list(METRICS),
        "planned_contrast_count": 8,
        "completed_contrast_count": len(tests),
        "cluster_unit": "24 held-out templates; four paired cases per template",
        "tests": "Two-sided template-cluster sign flips; Holm correction across both primary metrics and all four prompt/order conditions.",
        "bootstrap_replicates": args.bootstrap,
        "permutations": args.permutations,
        "seed": args.seed,
        "preregistration": "Not preregistered. The feature-ID condition and analysis code were frozen before that condition's inference; the combined family is reported to account for related post-hoc sensitivity contrasts.",
        "sha256": {
            "analysis_script": sha256(Path(__file__).resolve()),
            **{
                f"{condition}_{side}": sha256(path)
                for condition, (control_path, variant_path) in CONDITIONS.items()
                for side, path in (("control", control_path), ("variant", variant_path))
            },
        },
        "contrasts": tests,
    }
    out = RUNS / "prompt_sensitivity_combined_summary.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "contrasts": len(tests)}, indent=2))


if __name__ == "__main__":
    main()
