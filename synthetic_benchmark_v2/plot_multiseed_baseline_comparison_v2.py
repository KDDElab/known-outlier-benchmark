#!/usr/bin/env python3
"""Plot multi-seed support-localization results from the frozen analysis CSV."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
CSV = HERE / "runs" / "multiseed_analysis" / "multiseed_metrics.csv"
OUTPUT = HERE / "figures" / "v2_baseline_comparison.pdf"
METHODS = [
    ("Qwen3.5-9B", "qwen35_9b", "LLM"),
    ("DeepSeek-V2-Lite", "deepseek_v2_lite_chat", "LLM"),
    ("Llama-3.1-8B", "llama31_8b_instruct", "LLM"),
    ("Gower-LOF", "gower_lof_subspace", "Baseline"),
    ("Gower-kNN", "gower_knn_subspace", "Baseline"),
    ("RLAX adaptation", "rlax_reference_adaptation", "Baseline"),
    ("M²OE choice", "m2oe_choice", "Baseline"),
    ("SOD adaptation", "sod_subspace_adaptation", "Baseline"),
    ("COP adaptation", "cop_error_vector_adaptation", "Baseline"),
    ("Top-4 univariate", "top_univariate", "Baseline"),
    ("Random size prior", "random_size_prior", "Baseline"),
    ("All features", "all_features", "Baseline"),
]
METRICS = [
    ("support_top1_sufficiency_rate", "Sufficiency@1"),
    ("support_top1_minimal_exact_hit_rate", "Minimal exact@1"),
    ("support_any_minimal_hit_at_4", "Minimal Hit@4"),
    ("support_mean_best_jaccard", "Best Jaccard@4"),
]

def main() -> None:
    frame = pd.read_csv(CSV)
    lookup = {(row.method, row.metric): row for row in frame.itertuples(index=False)}
    rows = []
    for label, method, family in METHODS:
        values = {}
        for metric, _ in METRICS:
            record = lookup.get((method, metric))
            if record is None or pd.isna(record.mean_over_seeds):
                values[metric] = None
            else:
                ci = json.loads(record.template_cluster_bootstrap_95_ci_macro_seed)
                values[metric] = (float(record.mean_over_seeds), ci)
        rows.append((label, family, values))
    colors = {"LLM": "#3768A6", "Baseline": "#D17B28"}
    y = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.4), sharey="row")
    for axis, (metric, title) in zip(axes.flat, METRICS):
        for row_idx, (_, family, values) in enumerate(rows):
            result = values[metric]
            if result is None:
                continue
            estimate, ci = result
            lo, hi = ci
            axis.errorbar(estimate, y[row_idx], xerr=[[estimate-lo], [hi-estimate]], fmt="o",
                          color=colors[family], markersize=3.8, capsize=1.8, linewidth=.9)
        axis.set_title(title, fontsize=8.5)
        axis.set_xlim(-.02, 1.02); axis.set_xticks([0,.25,.50,.75,1.0])
        axis.grid(axis="x", color="#D9DEE7", linewidth=.5); axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelsize=6.5); axis.tick_params(axis="y", labelsize=6.4)
        axis.spines[["top","right","left"]].set_visible(False); axis.spines["bottom"].set_color("#AAB3C0")
    labels = [row[0] for row in rows]
    for axis in (axes[0,0], axes[1,0]):
        axis.set_yticks(y, labels); axis.invert_yaxis(); axis.set_ylabel("Method", fontsize=7)
    for axis in (axes[0,1], axes[1,1]): axis.tick_params(axis="y", left=False, labelleft=False)
    fig.legend(handles=[Patch(facecolor=colors["LLM"],label="LLM"),Patch(facecolor=colors["Baseline"],label="Statistical/explanation baseline")],
               loc="upper center",bbox_to_anchor=(.57,.99),ncol=2,frameon=False,fontsize=7)
    fig.text(.5,.012,"Macro mean across three generator seeds; bars: template-cluster bootstrap 95% CI",ha="center",fontsize=7.3)
    fig.tight_layout(rect=(0,.045,1,.93),h_pad=.8,w_pad=.4)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True); fig.savefig(OUTPUT,bbox_inches="tight"); plt.close(fig)
    print(OUTPUT)

if __name__ == "__main__": main()
