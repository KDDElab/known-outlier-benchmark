#!/usr/bin/env python3
"""Plot cluster-bootstrap localization results for LLMs and baselines."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "runs" / "baseline_comparison_gower.json"
SIMPLE = HERE / "runs" / "baseline_evaluations" / "summary.json"
OUTPUT = HERE / "figures" / "v2_baseline_comparison.pdf"


def main() -> None:
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    simple = json.loads(SIMPLE.read_text(encoding="utf-8"))
    methods = [
        ("Qwen3.5-9B", "models", "qwen35_9b"),
        ("DeepSeek-V2-Lite", "models", "deepseek_v2_lite_chat"),
        ("Llama-3.1-8B", "models", "llama31_8b_instruct"),
        ("Gower LOF", "baselines", "gower_lof_subspace"),
        ("Gower kNN", "baselines", "gower_knn_subspace"),
        ("Top-4 univariate", "baselines", "top_univariate"),
        ("Random (100 runs)", "baselines", "random_subspace"),
        ("All features", "baselines", "all_features"),
    ]
    metrics = [
        ("support_sufficiency_at_1", "Sufficiency@1"),
        ("minimal_exact_at_1", "Minimal exact@1"),
        ("minimal_hit_at_4", "Minimal Hit@4"),
        ("best_jaccard", "Best Jaccard"),
    ]
    rows = []
    for display, family, name in methods:
        if name == "random_subspace":
            result = {}
            random_data = simple[name]["template_cluster_ci95_on_seed_averaged_cases"]
            metric_map = {
                "support_sufficiency_at_1": "support_top1_sufficiency_rate",
                "minimal_exact_at_1": "support_top1_minimal_exact_hit_rate",
                "minimal_hit_at_4": "support_any_minimal_hit_at_4",
                "best_jaccard": "support_mean_best_jaccard",
            }
            for key, source in metric_map.items():
                result[key] = {
                    "estimate": random_data[source]["estimate"],
                    "ci95": [random_data[source]["ci95_low"], random_data[source]["ci95_high"]],
                }
        else:
            result = analysis["summaries"][family][name]
        rows.append((display, family, result))

    colors = {"models": "#3768A6", "baselines": "#D17B28"}
    y = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.6), sharey="row")
    for axis, (metric, title) in zip(axes.flat, metrics):
        for row_idx, (_, family, result) in enumerate(rows):
            estimate = result[metric]["estimate"]
            lo, hi = result[metric]["ci95"]
            axis.errorbar(
                estimate,
                y[row_idx],
                xerr=[[estimate - lo], [hi - estimate]],
                fmt="o",
                color=colors[family],
                markersize=4.2,
                capsize=2,
                linewidth=1.0,
            )
        axis.set_title(title, fontsize=9)
        axis.set_xlim(-0.02, 1.02)
        axis.set_xticks([0, .25, .50, .75, 1.0])
        axis.grid(axis="x", color="#D9DEE7", linewidth=.55)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelsize=7)
        axis.tick_params(axis="y", labelsize=7)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#AAB3C0")
    labels = [row[0] for row in rows]
    axes[0, 0].set_yticks(y, labels)
    axes[1, 0].set_yticks(y, labels)
    axes[0, 0].invert_yaxis()
    axes[1, 0].invert_yaxis()
    axes[0, 0].set_ylabel("Method", fontsize=8)
    axes[1, 0].set_ylabel("Method", fontsize=8)
    for axis in (axes[0, 1], axes[1, 1]):
        axis.tick_params(axis="y", left=False, labelleft=False)
    fig.legend(
        handles=[Patch(facecolor=colors["models"], label="LLM"), Patch(facecolor=colors["baselines"], label="Statistical/simple baseline")],
        loc="upper center", bbox_to_anchor=(.57, .99), ncol=2, frameon=False, fontsize=7,
    )
    fig.text(.5, .015, "Score (95% template-cluster bootstrap interval)", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .93), h_pad=.8, w_pad=.45)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
