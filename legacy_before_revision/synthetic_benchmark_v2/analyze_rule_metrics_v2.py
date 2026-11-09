#!/usr/bin/env python3
"""Summarize rule-sufficient subspace metrics with template-cluster CIs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "rule_metrics_template_bootstrap.json"
METRICS = ["rule_top1_sufficient", "rule_top1_minimal", "rule_any_minimal_hit"]
N_BOOT = 2_000
SEED = 20261105


def read_test(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.loc[frame["split"] == "test", ["case_id", "template_id", *METRICS]].copy()


def summarize(frame: pd.DataFrame, rng: np.random.Generator) -> dict:
    frame = frame.copy()
    for metric in METRICS:
        frame[metric] = frame[metric].fillna(False).astype(float)
    templates = sorted(frame["template_id"].unique())
    grouped = {key: frame.loc[frame.template_id == key, METRICS].to_numpy() for key in templates}
    point = frame[METRICS].mean().to_dict()
    draws = {metric: np.empty(N_BOOT) for metric in METRICS}
    for i in range(N_BOOT):
        chosen = rng.choice(templates, size=len(templates), replace=True)
        sampled = np.concatenate([grouped[key] for key in chosen], axis=0)
        for j, metric in enumerate(METRICS):
            draws[metric][i] = sampled[:, j].mean()
    return {
        "n_cases": int(len(frame)),
        "n_templates": int(len(templates)),
        "metrics": {
            metric: {
                "estimate": float(point[metric]),
                "ci95_low": float(np.quantile(draws[metric], 0.025)),
                "ci95_high": float(np.quantile(draws[metric], 0.975)),
            }
            for metric in METRICS
        },
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    methods: dict[str, pd.DataFrame] = {}
    for model, run_id in [
        ("Qwen3.5-9B", "evaluation_qwen35_9b_known_anomaly_subspace_benchmark_v2_compact_json_v1"),
        ("DeepSeek-V2-Lite-Chat", "evaluation_deepseek_v2_lite_chat_known_anomaly_subspace_benchmark_v2_compact_json_v1"),
        ("Llama-3.1-8B-Instruct", "evaluation_llama31_8b_instruct_known_anomaly_subspace_benchmark_v2_compact_json_v1"),
    ]:
        methods[model] = read_test(RUNS / run_id / "per_case.csv")
    for method, folder in [
        ("Gower-LOF", "gower_lof_subspace"),
        ("Gower-kNN", "gower_knn_subspace"),
        ("Top-four univariate", "top_univariate"),
        ("All features", "all_features"),
    ]:
        methods[method] = read_test(RUNS / "baseline_evaluations" / folder / "per_case.csv")

    random_files = sorted((RUNS / "baseline_evaluations").glob("random_[0-9][0-9][0-9]/per_case.csv"))
    if len(random_files) != 100:
        raise RuntimeError(f"Expected 100 random replicates; found {len(random_files)}")
    random_frames = [read_test(path).set_index("case_id") for path in random_files]
    random_stack = pd.concat(random_frames, keys=range(len(random_frames)), names=["replicate", "case_id"])
    random_mean = random_stack.groupby(level="case_id")[METRICS].mean().reset_index()
    template_lookup = random_frames[0]["template_id"]
    random_mean["template_id"] = random_mean["case_id"].map(template_lookup)
    methods["Random (100-seed case mean)"] = random_mean

    payload = {
        "description": "Rule-sufficient subspace metrics; 2,000 bootstrap resamples of the 24 held-out templates. Random predictions are averaged per case across 100 seeds before bootstrap.",
        "seed": SEED,
        "bootstrap_resamples": N_BOOT,
        "methods": {name: summarize(frame, rng) for name, frame in methods.items()},
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["methods"], indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
