#!/usr/bin/env python3
"""Create descriptive test-split slices by task family and design factor.

The case-level score files already count invalid/missing model outputs as misses.
This script preserves that denominator and intentionally does not calculate
family-level confidence intervals: there are only three held-out templates in
each task family. Outputs are descriptive CSVs for the manuscript/artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
METHODS = {
    "Qwen3.5-9B": RUNS / "evaluation_qwen35_9b_known_anomaly_subspace_benchmark_v2_compact_json_v1/per_case.csv",
    "DeepSeek-V2-Lite-Chat": RUNS / "evaluation_deepseek_v2_lite_chat_known_anomaly_subspace_benchmark_v2_compact_json_v1/per_case.csv",
    "Llama-3.1-8B-Instruct": RUNS / "evaluation_llama31_8b_instruct_known_anomaly_subspace_benchmark_v2_compact_json_v1/per_case.csv",
    "Gower-LOF": RUNS / "baseline_evaluations/gower_lof_subspace/per_case.csv",
    "Gower-kNN": RUNS / "baseline_evaluations/gower_knn_subspace/per_case.csv",
}
METRICS = [
    "support_top1_sufficient",
    "support_top1_minimal",
    "support_any_minimal_hit",
    "support_best_jaccard",
]


def load_method(name: str, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["split"].eq("test")].copy()
    if len(frame) != 288:
        raise ValueError(f"{name}: expected 288 test cases, found {len(frame)}")
    frame["method"] = name
    for col in METRICS:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    frame["structurally_valid"] = frame["structurally_valid"].astype(str).str.lower().eq("true")
    frame["task_label_correct"] = frame["task_label_correct"].astype(str).str.lower().eq("true")
    return frame


def summarize(frames: pd.DataFrame, group_col: str) -> pd.DataFrame:
    result = (
        frames.groupby([group_col, "method"], sort=True)
        .agg(
            n_cases=("case_id", "size"),
            n_templates=("template_id", "nunique"),
            structured_output_rate=("structurally_valid", "mean"),
            support_sufficiency_at_1=(METRICS[0], "mean"),
            minimal_exact_at_1=(METRICS[1], "mean"),
            minimal_hit_at_4=(METRICS[2], "mean"),
            best_jaccard_at_4=(METRICS[3], "mean"),
        )
        .reset_index()
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=RUNS / "subgroup_analysis_v2")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat([load_method(name, path) for name, path in METHODS.items()], ignore_index=True)
    for field, filename in [
        ("task_label_gold", "by_task_family.csv"),
        ("feature_count", "by_feature_count.csv"),
        ("reference_count", "by_reference_count.csv"),
        ("nuisance_profile", "by_nuisance_profile.csv"),
    ]:
        summary = summarize(combined, field)
        summary.to_csv(args.outdir / filename, index=False, float_format="%.6f")

    model_names = list(METHODS)[:3]
    label_rows = []
    for (family, method), group in combined.loc[combined["method"].isin(model_names)].groupby(["task_label_gold", "method"], sort=True):
        tp = fp = fn = 0
        for row in group.itertuples(index=False):
            predicted = set(json.loads(row.mechanism_predicted))
            gold = set(json.loads(row.mechanism_gold))
            tp += len(predicted & gold)
            fp += len(predicted - gold)
            fn += len(gold - predicted)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        label_rows.append({
            "task_label_gold": family,
            "method": method,
            "n_cases": int(len(group)),
            "n_templates": int(group["template_id"].nunique()),
            "task_label_accuracy_invalid_as_wrong": float(group["task_label_correct"].mean()),
            "mechanism_micro_f1_invalid_as_empty": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        })
    pd.DataFrame(label_rows).to_csv(args.outdir / "by_family_task_and_mechanism.csv", index=False, float_format="%.6f")
    print(f"Wrote descriptive subgroup tables to {args.outdir}")


if __name__ == "__main__":
    main()
