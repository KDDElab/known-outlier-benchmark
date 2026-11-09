#!/usr/bin/env python3
"""Create vector figures from frozen manuscript tables and per-case artifacts."""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
ART = WORK / "artifact_release_staging" / "synthetic_benchmark_v2"
OUT = WORK / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.2,
    "axes.titlesize": 8.2,
    "axes.labelsize": 7.2,
    "legend.fontsize": 6.3,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "Qwen3.5-9B": "#0072B2",
    "DeepSeek-V2-Lite-Chat": "#D55E00",
    "Llama-3.1-8B-Instruct": "#009E73",
    "Gower-LOF": "#6A3D9A",
    "Gower-kNN": "#CC79A7",
    "RLAX adaptation": "#E69F00",
    "Top-four univariate": "#56B4E9",
    "SOD adaptation": "#999999",
    "COP adaptation": "#7F7F7F",
    "Random size prior": "#8C564B",
    "All features": "#222222",
}
MODELS = ["Qwen3.5-9B", "DeepSeek-V2-Lite-Chat", "Llama-3.1-8B-Instruct"]


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", metadata={"Creator": "make_review_revision_figures.py"})
    plt.close(fig)


def figure1_overall_framework() -> None:
    """Draw a flat, three-stage overview with a shared query and separate evidence tracks."""
    from matplotlib.patches import Circle, Rectangle

    fig, ax = plt.subplots(figsize=(7.16, 3.82))
    ax.set_axis_off()
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    fig.subplots_adjust(left=.012, right=.988, bottom=.02, top=.985)

    ink, muted = "#273642", "#596975"
    orange, orange_dark, orange_pale = "#F4BD77", "#C86D32", "#FFF9F1"
    blue, blue_pale = "#4E7FC7", "#F1F6FD"
    purple, purple_pale = "#7757B8", "#F5F0FC"
    green, green_pale = "#57966C", "#F1F8F1"
    grey, grey_pale = "#76838C", "#F4F6F7"
    light_border = "#D4DCE1"

    def text(x, y, s, size=6.2, color=ink, weight="normal", ha="left", va="center"):
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                ha=ha, va=va, linespacing=1.15, zorder=5)

    def box(x, y, w, h, face, edge=light_border, lw=.7, radius=.10, z=1):
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle=f"round,pad=0.025,rounding_size={radius}",
                               facecolor=face, edgecolor=edge, linewidth=lw, zorder=z)
        ax.add_patch(patch)
        return patch

    def card(x, y, w, h, title, body, color, face="#FFFFFF",
             title_size=6.1, body_size=5.55, centered=False):
        box(x, y, w, h, face, light_border, .68, .09, 2)
        align = "center" if centered else "left"
        left = x + w / 2 if centered else x + .15
        text(left, y + h - .18, title, title_size, color, "bold", align, "top")
        text(left, y + h - .43, body, body_size, ink, "normal", align, "top")

    def panel(x, y, w, h, title):
        box(x, y, w, h, orange_pale, "#E8A45E", .82, .14, 0)
        box(x + .10, y + h - .43, w - .20, .32, "#FADBAE", "none", 0, .09, 1)
        text(x + w / 2, y + h - .27, title, 7.0, "#71451F", "bold", "center")

    def arrow(x1, y1, x2, y2, color=blue, width=.95):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=8, linewidth=width, color=color,
                                     shrinkA=0, shrinkB=0, zorder=4))

    def table_icon(cx, cy, color=green):
        """Small original vector glyph for a schema and row sample."""
        ax.add_patch(Rectangle((cx - .25, cy - .22), .50, .43,
                               facecolor="white", edgecolor=color, linewidth=.9, zorder=4))
        ax.add_patch(Rectangle((cx - .25, cy + .08), .50, .13,
                               facecolor=color, edgecolor=color, linewidth=.5, zorder=4))
        for xx in (cx - .08, cx + .08):
            ax.plot([xx, xx], [cy - .21, cy + .08], color=color, lw=.55, zorder=5)
        for yy in (cy - .08, cy - .17):
            ax.plot([cx - .24, cx + .24], [yy, yy], color="#A8C6AE", lw=.5, zorder=5)
        ax.scatter([cx + .34], [cy + .20], marker="*", s=34, color=orange_dark, zorder=5)

    def chip_icon(cx, cy, color=purple):
        """Generic model glyph; avoids using a vendor logo."""
        ax.add_patch(Rectangle((cx - .17, cy - .14), .34, .28,
                               facecolor="white", edgecolor=color, linewidth=.9, zorder=4))
        ax.add_patch(Circle((cx, cy), .075, facecolor="#E7DDF5", edgecolor=color, lw=.65, zorder=5))
        ax.scatter([cx - .045, cx + .045, cx], [cy + .01, cy + .01, cy - .045],
                   s=4, color=color, zorder=6)
        for offset in (-.09, 0, .09):
            ax.plot([cx + offset, cx + offset], [cy + .14, cy + .20], color=color, lw=.7, zorder=4)
            ax.plot([cx + offset, cx + offset], [cy - .20, cy - .14], color=color, lw=.7, zorder=4)
            ax.plot([cx - .23, cx - .17], [cy + offset, cy + offset], color=color, lw=.7, zorder=4)
            ax.plot([cx + .17, cx + .23], [cy + offset, cy + offset], color=color, lw=.7, zorder=4)

    def rank_icon(cx, cy, color=blue):
        for j, h in enumerate((.12, .22, .31)):
            x = cx - .20 + j * .16
            ax.add_patch(Rectangle((x, cy - .18), .10, h,
                                   facecolor=("#A9C4EA" if j < 2 else color),
                                   edgecolor="none", zorder=4))

    # Three aligned stages, matching the benchmark's actual flow.
    panel(.12, .96, 4.20, 6.85, "1. TASK AND LABELS")
    panel(4.46, .96, 5.10, 6.85, "2. METHODS UNDER ONE QUERY")
    panel(9.70, .96, 5.18, 6.85, "3. EVALUATION TRACKS")

    # Stage 1: task input, prediction contract, and distinct targets.
    box(.36, 6.02, 3.70, .88, "#FFFFFF", "#B7D2BE", .75, .09, 2)
    table_icon(.76, 6.47)
    text(1.17, 6.66, "QUERY INPUT", 6.0, green, "bold", va="top")
    text(1.17, 6.40, "Schema $T$  ·  known outlier $x^*$\nNormal reference sample $R$",
         5.8, ink, va="top")
    arrow(2.21, 5.91, 2.21, 5.67, green)
    card(.36, 4.87, 3.70, .70, "PREDICTION", "Rank up to four field subsets  $S_1,\\ldots,S_4$",
         green, green_pale, title_size=6.0, body_size=5.5)
    text(.40, 4.47, "LABEL TARGETS ARE KEPT SEPARATE", 5.55, muted, "bold")
    target_cards = [
        (.36, 3.35, "Support", "Absent in full\nclean population", blue, blue_pale),
        (2.25, 3.35, "Rule", "Minimal rule-\nsufficient fields", orange_dark, "#FFF3E2"),
        (.36, 2.42, "Mechanism", "Fields affecting\nthe rule", green, green_pale),
        (2.25, 2.42, "Edit / value", "Hidden changed\nfields", grey, grey_pale),
    ]
    for x, y, title, body, color, face in target_cards:
        box(x, y, 1.80, .76, face, light_border, .6, .08, 2)
        text(x + .12, y + .56, title, 5.8, color, "bold", va="top")
        text(x + .12, y + .33, body, 5.0, ink, va="top")
    box(.36, 1.34, 3.70, .63, "#FFF1E2", "#E7B581", .65, .08, 2)
    text(2.21, 1.65, "No detector score or paired clean row is supplied.",
         5.25, "#70441C", "bold", "center")

    # Stage 2: every method receives the same case; two method families return ranked sets.
    box(4.74, 6.10, 4.54, .78, blue_pale, "#AFC5E5", .72, .09, 2)
    text(7.01, 6.65, "COMMON QUERY", 6.0, blue, "bold", "center", "top")
    text(7.01, 6.39, "Same schema, target, and normal rows for each method",
         5.4, ink, "normal", "center", "top")
    arrow(4.08, 6.48, 4.68, 6.48, green)
    ax.plot([7.01, 7.01], [5.99, 5.78], color=blue, lw=.85, zorder=3)
    ax.plot([5.78, 8.24], [5.78, 5.78], color=blue, lw=.85, zorder=3)
    arrow(5.78, 5.78, 5.78, 5.58, blue)
    arrow(8.24, 5.78, 8.24, 5.58, blue)

    box(4.74, 4.37, 2.22, 1.18, purple_pale, "#C8B8E2", .72, .09, 2)
    chip_icon(5.17, 5.14, purple)
    text(5.53, 5.25, "PROMPTED LMs", 5.65, purple, "bold", va="top")
    text(5.85, 4.82, "Qwen  ·  DeepSeek\nLlama", 5.5, ink, "normal", "center", "top")
    box(7.20, 4.37, 2.08, 1.18, blue_pale, "#B5C8E2", .72, .09, 2)
    rank_icon(7.62, 5.04, blue)
    text(7.94, 5.25, "RANKERS + CONTROLS", 5.15, blue, "bold", va="top")
    text(8.24, 4.82, "Gower · RLAX · SOD/COP\nM$^2$OE · rarity · random · all",
         4.9, ink, "normal", "center", "top")

    ax.plot([5.85, 5.85], [4.25, 3.92], color=purple, lw=.8, zorder=3)
    ax.plot([8.24, 8.24], [4.25, 3.92], color=blue, lw=.8, zorder=3)
    ax.plot([5.85, 8.24], [3.92, 3.92], color=blue, lw=.8, zorder=3)
    arrow(7.05, 3.92, 7.05, 3.68, blue)
    box(4.94, 2.79, 4.22, .84, "#FFFFFF", "#AAC2E2", .75, .09, 2)
    rank_icon(5.43, 3.20, blue)
    text(5.86, 3.38, "RANKED FIELD SUBSETS", 6.0, blue, "bold", va="top")
    text(5.86, 3.09, "Top four candidates evaluated with shared metrics",
         5.3, ink, va="top")
    arrow(9.20, 3.20, 9.83, 3.20, blue)
    box(5.01, 1.40, 4.08, .76, "#FFFFFF", light_border, .65, .09, 2)
    text(7.05, 1.88, "PRIMARY TRACK", 5.7, blue, "bold", "center", "top")
    text(7.05, 1.62, "40 templates  ·  3 seeds  ·  864 test cases",
         5.35, ink, "normal", "center", "top")

    # Stage 3: evidence sources and claims remain visibly separate.
    track_rows = [
        (5.20, green, green_pale, "A", "SYNTHETIC BENCHMARK",
         "864 cases  ·  3 seeds\nGold: exhaustive population and rule labels\nScores: sufficiency, exact, Hit@4, Jaccard"),
        (3.55, blue, blue_pale, "B", "CIRCUIT PILOT",
         "86 queries  ·  four circuit pairs\nGold: gate-negation input fields\nScores: best Jaccard and field recall"),
        (1.90, grey, grey_pale, "C", "REAL-TABLE PILOT",
         "78 queries  ·  ten datasets\nReference: leave-one-model-out consensus\nClaim: peer agreement, not accuracy"),
    ]
    for y, color, face, badge, title, body in track_rows:
        box(10.04, y, 4.53, 1.34, face, light_border, .72, .09, 2)
        ax.add_patch(Circle((10.37, y + 1.01), .15, facecolor=color,
                            edgecolor="none", zorder=4))
        text(10.37, y + 1.01, badge, 5.7, "white", "bold", "center")
        text(10.64, y + 1.15, title, 5.8, color, "bold", va="top")
        text(10.25, y + .78, body, 5.15, ink, va="top")
    ax.plot([9.91, 9.91], [2.55, 5.86], color=blue, lw=.85, zorder=3)
    for y, color, _, _, _, _ in track_rows:
        arrow(9.91, y + .67, 10.00, y + .67, color)

    box(.12, .18, 14.76, .58, "#FFF1E2", "#E4A96F", .7, .09, 2)
    box(.12, .18, 1.83, .58, orange_dark, orange_dark, .2, .09, 3)
    text(1.035, .47, "EVIDENCE SCOPE", 5.8, "white", "bold", "center")
    text(2.18, .47,
         "Exhaustive synthetic labels  ·  fault-specific circuit labels  ·  peer agreement only on real tables",
         5.55, "#70441C", "bold")

    out_pdf = OUT / "figure1_overall_framework.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", metadata={"Creator": "make_review_revision_figures.py"})
    fig.savefig(OUT / "figure1_overall_framework.svg", bbox_inches="tight")
    plt.close(fig)

def figure_intro_reference_sample() -> None:
    """Draw a two-panel example of how finite references alter apparent support."""
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.99, bottom=.25, top=.77, wspace=.24)

    blue = "#1769AA"
    red = "#C94F45"
    grey = "#9AA4AA"
    grid = "#DDE3E7"
    ink = "#26343D"

    clean_full = [(0, 0), (0, 1), (1, 0)]
    clean_sample = [(0, 0), (0, 1)]
    target = (1, 1)
    titles = ["(a) Complete clean population", "(b) Reference sample seen by method"]

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=8.2, fontweight="bold", color=ink, pad=7, loc="left")
        ax.set_xlim(-.28, 1.28)
        ax.set_ylim(-.28, 1.28)
        ax.set_xticks([0, 1], labels=["Retail", "Personal"])
        ax.set_yticks([0, 1], labels=["Low", "High"])
        ax.grid(True, color=grid, linewidth=.65)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=7.0, length=0, pad=4, colors=ink)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#73808A")
            ax.spines[side].set_linewidth(.65)

    # In the full population, each target value occurs on its own, but the pair does not.
    axes[0].scatter(*zip(*clean_full), s=43, marker="o", color=blue,
                    edgecolors="white", linewidths=.55, zorder=3)
    axes[0].scatter([target[0]], [target[1]], s=112, marker="*", color=red,
                    edgecolors="white", linewidths=.55, zorder=4)

    # The supplied sample omits the clean Personal/Low combination.
    axes[1].scatter(*zip(*clean_sample), s=43, marker="o", color=blue,
                    edgecolors="white", linewidths=.55, zorder=3)
    axes[1].scatter([1], [0], s=48, marker="o", facecolors="none",
                    edgecolors=grey, linewidths=1.15, zorder=3)
    axes[1].scatter([target[0]], [target[1]], s=112, marker="*", color=red,
                    edgecolors="white", linewidths=.55, zorder=4)

    axes[0].set_ylabel("Amount band", fontsize=7.1, color=ink, labelpad=6)
    axes[0].set_xlabel("Account type", fontsize=7.1, color=ink, labelpad=5)
    axes[1].set_xlabel("Account type", fontsize=7.1, color=ink, labelpad=5)

    # Dashed guides make the marginal values visible while the target combination is absent.
    axes[0].plot([0, 1], [1, 1], color=grey, linewidth=.7, linestyle=(0, (2, 2)), zorder=2)
    axes[0].plot([1, 1], [0, 1], color=grey, linewidth=.7, linestyle=(0, (2, 2)), zorder=2)
    axes[1].plot([0, 1], [1, 1], color=grey, linewidth=.7, linestyle=(0, (2, 2)), zorder=2)
    axes[1].plot([1, 1], [0, 1], color=grey, linewidth=.7, linestyle=(0, (2, 2)), zorder=2)

    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5.3,
               markerfacecolor=blue, markeredgecolor="white", label="clean normal row"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=5.3,
               markerfacecolor="white", markeredgecolor=grey, label="clean row not sampled"),
        Line2D([0], [0], marker="*", linestyle="none", markersize=8.0,
               markerfacecolor=red, markeredgecolor="white", label="known outlier"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.54, .015),
               ncol=3, frameon=False, fontsize=6.6, handletextpad=.35,
               columnspacing=1.25)
    fig.savefig(OUT / "figure_intro_reference_sample.pdf", bbox_inches="tight",
                metadata={"Creator": "make_review_revision_figures.py"})
    fig.savefig(OUT / "figure_intro_reference_sample.svg", bbox_inches="tight")
    plt.close(fig)


def figure2_pareto_and_validity() -> None:
    fig = plt.figure(figsize=(7.16, 4.02))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.15, .85],
                            hspace=.52, wspace=.34)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_title("(a) Sufficiency and compactness", loc="left", pad=4, fontweight="bold")
    rows = [
        ("Qwen3.5-9B", .207, .186, .228, .714, .617, .798),
        ("DeepSeek-V2-Lite-Chat", .371, .344, .397, .429, .384, .470),
        ("Llama-3.1-8B-Instruct", .199, .195, .204, .394, .327, .457),
        ("Gower-LOF", .181, .175, .187, .545, .421, .660),
        ("Gower-kNN", .229, .220, .237, .608, .500, .707),
        ("RLAX adaptation", .234, .226, .242, .507, .376, .628),
        ("Top-four univariate", .097, .097, .097, .155, .086, .231),
        ("SOD adaptation", .558, .529, .587, .512, .364, .668),
        ("COP adaptation", .242, .241, .243, .005, .000, .011),
        ("Random size prior", .162, .162, .162, .117, .087, .146),
        ("All features", 1.000, 1.000, 1.000, 1.000, 1.000, 1.000),
    ]
    valid_rates = {"Qwen3.5-9B": .991, "DeepSeek-V2-Lite-Chat": .806, "Llama-3.1-8B-Instruct": .817}
    offsets = {
        "Qwen3.5-9B": (-12, 12), "DeepSeek-V2-Lite-Chat": (7, -15), "Llama-3.1-8B-Instruct": (7, -12),
        "Gower-LOF": (-30, 12), "Gower-kNN": (7, 10), "RLAX adaptation": (-26, -15),
        "Top-four univariate": (6, 8), "SOD adaptation": (5, 8), "COP adaptation": (7, 6),
        "Random size prior": (-10, -18), "All features": (-48, -17),
    }
    short_names = {
        "Qwen3.5-9B": "Qwen", "DeepSeek-V2-Lite-Chat": "DeepSeek", "Llama-3.1-8B-Instruct": "Llama",
        "Gower-LOF": "G-LOF", "Gower-kNN": "G-kNN", "RLAX adaptation": "RLAX",
        "Top-four univariate": "Univariate", "SOD adaptation": "SOD", "COP adaptation": "COP",
        "Random size prior": "Random", "All features": "All features (ceiling)",
    }
    for name, compact, compact_lo, compact_hi, suff, suff_lo, suff_hi in rows:
        size = 25 + 42 * valid_rates.get(name, 1.0)
        marker = "^" if name in MODELS else ("s" if name == "All features" else "o")
        ax.errorbar(compact, suff,
                    xerr=[[compact-compact_lo], [compact_hi-compact]],
                    yerr=[[suff-suff_lo], [suff_hi-suff]],
                    fmt="none", ecolor=COLORS[name], alpha=.42, elinewidth=.65, capsize=1.2, zorder=2)
        ax.scatter(compact, suff, s=size, marker=marker, color=COLORS[name], edgecolor="white", linewidth=.5, zorder=3)
        ax.annotate(short_names[name], (compact, suff), xytext=offsets[name], textcoords="offset points",
                    fontsize=4.9, color=COLORS[name])
    ax.plot([.181, .207, 1.0], [.545, .714, 1.0], color="#333333", lw=.6, ls="--", alpha=.65, zorder=1)
    ax.axhline(1, color="#999999", lw=.5, ls=":"); ax.axvline(1, color="#999999", lw=.5, ls=":")
    ax.set_xlim(0, 1.08); ax.set_ylim(-.035, 1.08)
    ax.set_xlabel("Selected-field fraction (valid outputs only)"); ax.set_ylabel("Support sufficiency@1")
    ax.grid(axis="both", color="#E5E5E5", linewidth=.4, zorder=0)

    frame = pd.read_csv(ART.parent / "manuscript_figures" / "parseable_decomposition.csv")
    metric_pairs = [("sufficiency@1", "Suff.@1"), ("exact@1", "Exact@1"),
                    ("Hit@4", "Hit@4"), ("best Jaccard@4", "Best Jac.@4")]
    subgrid = grid[0, 1].subgridspec(2, 2, hspace=.52, wspace=.42)
    axes = [fig.add_subplot(subgrid[i, j]) for i in range(2) for j in range(2)]
    for axis, (col, label) in zip(axes, metric_pairs):
        for idx, model in enumerate(MODELS):
            row = frame.loc[frame.model == model].iloc[0]
            x0, x1 = idx - .10, idx + .10
            y0, y1 = float(row[f"{col}_deployed"]), float(row[f"{col}_parseable"])
            axis.plot([x0, x1], [y0, y1], color=COLORS[model], lw=.8, alpha=.75)
            axis.scatter(x0, y0, s=10, color=COLORS[model], marker="o", zorder=3)
            axis.scatter(x1, y1, s=12, color=COLORS[model], marker="D", zorder=3)
        axis.set_title(label, fontsize=6.2, loc="left", pad=1)
        axis.set_ylim(-.02, .9); axis.set_xlim(-.4, 2.4)
        axis.set_xticks(range(3), ["Qwen", "DS", "Llama"])
        axis.tick_params(axis="both", labelsize=5.5, pad=1)
        axis.grid(axis="y", color="#E6E6E6", lw=.35)
        if axis in (axes[0], axes[2]): axis.set_ylabel("Score", fontsize=5.7, labelpad=1)
        if axis in (axes[0], axes[1]): axis.tick_params(axis="x", labelbottom=False)

    valid_ax = fig.add_subplot(grid[1, 0])
    valid_ax.set_title("(c) Output validity (864 queries/model)", loc="left", pad=3, fontweight="bold", fontsize=7.2)
    models = ["Qwen3.5-9B", "DeepSeek-V2-Lite-Chat", "Llama-3.1-8B-Instruct"]
    parseable = np.array([856, 696, 706]) / 864
    strict = np.array([854, 334, 571]) / 864
    y = np.arange(3)
    valid_ax.barh(y + .14, parseable, height=.25, color=[COLORS[m] for m in models], alpha=.9)
    valid_ax.barh(y - .14, strict, height=.25, color=[COLORS[m] for m in models], alpha=.38)
    valid_ax.set_yticks(y, ["Qwen", "DeepSeek", "Llama"]); valid_ax.invert_yaxis()
    valid_ax.set_xlim(0, 1.10); valid_ax.set_xlabel("Rate", fontsize=6.0, labelpad=1)
    valid_ax.tick_params(axis="both", labelsize=5.7, pad=1)
    valid_ax.grid(axis="x", color="#E8E8E8", lw=.35)
    for i, (a, b) in enumerate(zip(parseable, strict)):
        valid_ax.text(a + .012, i + .14, f"{a:.3f}", va="center", fontsize=5.2)
        valid_ax.text(b + .012, i - .14, f"{b:.3f}", va="center", fontsize=5.2)

    latency_ax = fig.add_subplot(grid[1, 1])
    latency_ax.set_title("(d) Latency per query", loc="left", pad=3, fontweight="bold", fontsize=7.2)
    mean = np.array([11.98, 17.38, 7.20]); p95 = np.array([17.83, 36.90, 15.09]); gpu = [1, 2, 1]
    for i, model in enumerate(models):
        latency_ax.plot([mean[i], p95[i]], [i, i], color=COLORS[model], lw=2.0, alpha=.55, solid_capstyle="round")
        latency_ax.scatter(mean[i], i, marker="o", s=19, color=COLORS[model], edgecolor="white", linewidth=.45, zorder=3)
        latency_ax.scatter(p95[i], i, marker="|", s=30, color=COLORS[model], zorder=3)
        latency_ax.text(p95[i] + .45, i, f"{mean[i]:.1f}/{p95[i]:.1f}s; {gpu[i]} GPU{'s' if gpu[i] > 1 else ''}",
                        va="center", fontsize=5.1)
    latency_ax.set_yticks(y, ["Qwen", "DeepSeek", "Llama"]); latency_ax.invert_yaxis()
    latency_ax.set_xlim(0, 55); latency_ax.set_xlabel("Seconds (mean to p95)", fontsize=6.0, labelpad=1)
    latency_ax.tick_params(axis="both", labelsize=5.7, pad=1)
    latency_ax.grid(axis="x", color="#E8E8E8", lw=.35)
    save(fig, "figure2_sufficiency_compactness_validity.pdf")


def figure3_reference_audit() -> None:
    raw = pd.read_csv(ART / "runs" / "reference_sample_audit_multiseed" / "per_case_test.csv")
    universes = {6: 41, 12: 298, 24: 2324}
    raw["candidate_universe"] = raw.feature_count.map(universes)
    raw["sample_absent_fraction"] = raw.sample_absent_candidate_count / raw.candidate_universe
    cells = []
    for (width, refs), group in raw.groupby(["feature_count", "reference_count"]):
        t = group.groupby("template_group_id").agg(extra=("spurious_sample_minimal_count", "mean"), retention=("gold_candidate_recall", "mean"), absent=("sample_absent_fraction", "mean"))
        cells.append({"feature_count": int(width), "reference_count": int(refs), "n": len(group), "n_templates": len(t),
                      "extra_mean": t.extra.mean(), "retention_mean": t.retention.mean(), "absent_mean": t.absent.mean()})
    summary = pd.DataFrame(cells)
    summary.to_csv(ART.parent / "manuscript_figures" / "reference_audit_width_reference_recomputed.csv", index=False)
    nuisance_rows = []
    # Equal-weight feature widths after within-template aggregation; descriptive only.
    for (width, refs, profile), group in raw.groupby(["feature_count", "reference_count", "nuisance_profile"]):
        values = group.groupby("template_group_id").agg(extra=("spurious_sample_minimal_count", "mean"), retention=("gold_candidate_recall", "mean"))
        nuisance_rows.append({"feature_count": int(width), "reference_count": int(refs), "nuisance_profile": profile,
                              "extra_mean": values.extra.mean(), "retention_mean": values.retention.mean(), "n_templates": len(values)})
    nuisance = pd.DataFrame(nuisance_rows)
    balanced = nuisance.groupby(["nuisance_profile", "reference_count"]).agg(
        extra_mean=("extra_mean", "mean"), retention_mean=("retention_mean", "mean"),
        feature_widths=("feature_count", "nunique"),
    ).reset_index()
    balanced.to_csv(ART.parent / "manuscript_figures" / "reference_audit_nuisance_width_balanced.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45), gridspec_kw={"width_ratios": [1.0, 1.0, 1.28], "wspace": .46})
    widths = [6, 12, 24]; refs = [8, 16, 32]
    matrix = np.array([[float(summary.query("feature_count == @w and reference_count == @r").extra_mean.iloc[0]) for r in refs] for w in widths])
    im = axes[0].imshow(matrix, cmap="magma", norm=LogNorm(vmin=4, vmax=250), aspect="auto")
    axes[0].set_title("(a) Extra candidates", loc="left", pad=4, fontweight="bold", fontsize=7.0)
    axes[0].set_yticks(range(3), [str(w) for w in widths]); axes[0].set_xticks(range(3), [str(r) for r in refs])
    axes[0].set_ylabel("Feature width"); axes[0].set_xlabel("Reference rows")
    for i in range(3):
        for j in range(3): axes[0].text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", color="white", fontsize=6.3)
    cb = fig.colorbar(im, ax=axes[0], fraction=.05, pad=.025)
    cb.set_label("Count", fontsize=5.6)

    axes[1].set_title("(b) Gold retention", loc="left", pad=4, fontweight="bold", fontsize=7.0)
    for width, color in zip(widths, ["#0072B2", "#D55E00", "#009E73"]):
        part = summary.query("feature_count == @width").sort_values("reference_count")
        axes[1].plot(part.reference_count, part.retention_mean, marker="o", ms=3.5, lw=1.1, label=f"{width} fields", color=color)
    axes[1].set_xlabel("Reference rows"); axes[1].set_ylabel("")
    axes[1].set_ylim(.55, 1.02); axes[1].set_xticks(refs); axes[1].grid(axis="y", color="#E6E6E6", lw=.45)
    axes[1].legend(frameon=False, fontsize=5.8, loc="lower right")

    order_path = ART / "runs" / "expanded_random_field_order_288" / "random_order_pairwise_tests.csv"
    if order_path.exists():
        tests = pd.read_csv(order_path)
        axes[2].set_title("(c) Random-order effect", loc="left", pad=4, fontweight="bold", fontsize=7.0)
        metric_order = ["support_top1_sufficient", "support_top1_minimal", "support_any_minimal_hit", "support_best_jaccard"]
        metric_short = {"support_top1_sufficient": "Suff.", "support_top1_minimal": "Exact",
                        "support_any_minimal_hit": "Hit@4", "support_best_jaccard": "Jacc."}
        model_short = {MODELS[0]: "Qwen", MODELS[1]: "DeepSeek", MODELS[2]: "Llama"}
        markers = {metric: marker for metric, marker in zip(metric_order, ["o", "s", "D", "^"])}
        labels, ys = [], []
        for model_idx, model in enumerate(MODELS):
            for metric_idx, metric in enumerate(metric_order):
                row = tests.query("model == @model and metric == @metric").iloc[0]
                y = model_idx * 4 + metric_idx
                estimate = float(row.random_minus_original)
                low, high = float(row.ci95_low), float(row.ci95_high)
                axes[2].errorbar(estimate, y, xerr=[[max(0.0, estimate-low)], [max(0.0, high-estimate)]],
                                 fmt=markers[metric], color=COLORS[model], ms=2.8,
                                 elinewidth=.7, capsize=1.1, zorder=3)
                labels.append(f"{model_short[model]}  {metric_short[metric]}"); ys.append(y)
        axes[2].axvline(0, color="#555555", lw=.65, ls="--", zorder=1)
        axes[2].set_yticks(ys, labels); axes[2].invert_yaxis()
        primary = tests[tests.metric.isin(metric_order)]
        limit = min(1.0, max(.25, 1.08 * np.max(np.abs(primary[["ci95_low", "ci95_high"]].to_numpy()))))
        axes[2].set_xlim(-limit, limit); axes[2].set_xlabel("Random orders − original", fontsize=6.0, labelpad=1)
        axes[2].tick_params(axis="y", labelsize=4.8, pad=1)
        axes[2].grid(axis="x", color="#E6E6E6", lw=.4)
    else:
        axes[2].set_title("(c) Gold retention by nuisance", loc="left", pad=4, fontweight="bold", fontsize=7.0)
        profiles = sorted(balanced.nuisance_profile.unique())
        short = {
            "independent_noise": "Independent", "high_cardinality_noise": "High-cardinality",
            "noisy_proxy": "Noisy proxy", "deterministic_proxy_preserved": "Proxy preserved",
            "deterministic_proxy_broken": "Proxy broken", "missingness_noise": "Missingness",
        }
        palette = plt.cm.tab10(np.linspace(0, .8, len(profiles)))
        for profile, color in zip(profiles, palette):
            part = balanced.query("nuisance_profile == @profile").sort_values("reference_count")
            axes[2].plot(part.reference_count, part.retention_mean, marker="o", ms=2.8, lw=.85,
                         color=color, label=short.get(profile, profile))
        axes[2].set_xlabel("Reference rows"); axes[2].set_ylim(.55, 1.02); axes[2].set_xticks(refs)
        axes[2].grid(axis="y", color="#E6E6E6", lw=.45)
        axes[2].legend(frameon=False, fontsize=5.0, ncol=2, loc="lower right")
    save(fig, "figure3_reference_and_order_robustness.pdf")


def figure4_order_probe() -> None:
    order_dir = ART / "runs" / "expanded_random_field_order_288"
    score_path = order_dir / "per_case_all_orders.csv"
    if not score_path.exists():
        print("Order-probe plot deferred; full model inference has not completed.")
        return
    variant = pd.read_csv(score_path)
    original_root = ART / "runs" / "multiseed_analysis" / "20261106"
    metric_columns = [
        ("support_top1_sufficient", "Sufficiency@1"),
        ("support_top1_minimal", "Exact@1"),
        ("support_any_minimal_hit", "Hit@4"),
        ("support_best_jaccard", "Best Jaccard@4"),
    ]
    order_seeds = [1001, 1002, 1003, 1004]
    order_ids = [0] + order_seeds
    names = ["Original", "1001", "1002", "1003", "1004"]
    all_metric_names = [metric for metric, _ in metric_columns] + [
        "rule_top1_sufficient", "rule_top1_minimal", "rule_any_minimal_hit",
    ]
    base_specs = [
        ("Gower-LOF", "gower_lof_subspace", "--", COLORS["Gower-LOF"]),
        ("Gower-kNN", "gower_knn_subspace", ":", COLORS["Gower-kNN"]),
        ("Top-4 univariate", "top_univariate", "-.", COLORS["Top-four univariate"]),
        ("SOD adaptation", "sod_subspace_adaptation", (0, (2, 1)), COLORS["SOD adaptation"]),
        ("COP adaptation", "cop_error_vector_adaptation", (0, (1, 1)), COLORS["COP adaptation"]),
        ("RLAX adaptation", "rlax_reference_adaptation", (0, (4, 1)), COLORS["RLAX adaptation"]),
    ]
    base_frames = {
        label: pd.read_csv(original_root / folder / "per_case.csv").query("split == 'test'")
        for label, folder, _, _ in base_specs
    }
    rng = np.random.default_rng(1005)
    metric_names = [name for name, _ in metric_columns]
    paired_records = []
    for model in MODELS:
        model_id = {MODELS[0]: "qwen35_9b", MODELS[1]: "deepseek_v2_lite_chat", MODELS[2]: "llama31_8b_instruct"}[model]
        orig = pd.read_csv(original_root / model_id / "per_case.csv").query("split == 'test'").copy()
        orig["case_id"] = orig["case_id"].astype(str)
        variants = variant.query("model == @model").copy()
        variants["source_case_id"] = variants["source_case_id"].astype(str)
        base = orig.set_index("case_id")
        for metric in all_metric_names:
            random_wide = variants.pivot(index="source_case_id", columns="random_order_seed", values=metric)
            shared = sorted(set(base.index) & set(random_wide.index))
            random_wide = random_wide.reindex(index=shared, columns=order_seeds)
            template_ids = base.loc[shared, "template_id"].astype(str)
            templates = sorted(template_ids.unique())
            case_groups = [np.flatnonzero(template_ids.to_numpy() == template) for template in templates]
            group_sizes = {len(group) for group in case_groups}
            if len(group_sizes) != 1:
                raise RuntimeError("Order-probe bootstrap requires equal case counts per template")
            n_templates, n_cases, n_orders = len(case_groups), len(case_groups[0]), len(order_seeds)
            random_cube = np.stack([random_wide.iloc[group].to_numpy(dtype=float) for group in case_groups])
            original_cube = np.stack([base.loc[np.asarray(shared)[group], metric].to_numpy(dtype=float) for group in case_groups])
            random_case_mean = random_cube.mean(axis=2)
            difference_by_template = (random_case_mean - original_cube).mean(axis=1)
            observed = float(difference_by_template.mean())
            sign_draws = rng.choice(np.array([-1.0, 1.0]), size=(20000, n_templates))
            permuted = np.mean(sign_draws * difference_by_template[None, :], axis=1)
            p_value = float((1 + np.count_nonzero(np.abs(permuted) >= abs(observed))) / 20001)

            # Resample template clusters and, within each case, the four independent field orders.
            selected_templates = rng.integers(0, n_templates, size=(2000, n_templates))
            selected_random = random_cube[selected_templates]
            selected_orders = rng.integers(0, n_orders, size=(2000, n_templates, n_cases, n_orders))
            order_means = np.take_along_axis(selected_random, selected_orders, axis=-1).mean(axis=-1)
            random_boot = order_means.mean(axis=(1, 2))
            original_boot = original_cube[selected_templates].mean(axis=(1, 2))
            bootstrap = random_boot - original_boot
            paired_records.append({
                "model": model, "metric": metric, "n_cases": len(shared), "n_templates": n_templates,
                "original_mean": float(original_cube.mean()),
                "random_order_mean": float(random_case_mean.mean()),
                "random_minus_original": observed,
                "ci95_low": float(np.quantile(bootstrap, .025)), "ci95_high": float(np.quantile(bootstrap, .975)),
                "p_sign_flip": p_value,
            })
    paired = pd.DataFrame(paired_records)
    order = np.argsort(paired.p_sign_flip.to_numpy())
    adjusted = np.empty(len(paired), dtype=float)
    running = 0.0
    for rank, ix in enumerate(order):
        running = max(running, min(1.0, (len(paired) - rank) * paired.iloc[ix].p_sign_flip))
        adjusted[ix] = running
    paired["p_holm_21"] = adjusted
    paired.to_csv(order_dir / "random_order_pairwise_tests.csv", index=False)
    manifest_path = order_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        analysis = manifest.setdefault("analysis", {})
        analysis.update({
            "random_order_tests_sha256": hashlib.sha256((order_dir / "random_order_pairwise_tests.csv").read_bytes()).hexdigest(),
            "pairwise_test_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "bootstrap_seed": 1005,
            "template_cluster_bootstrap_replicates": 2000,
            "within_case_order_resampling": "four independently seeded field permutations sampled with replacement within each selected case",
            "baseline_reference_lines": "canonical-order baseline scores shown for context only; non-LLM baselines were not rerun under field permutations, and no order-invariance claim is made",
            "sign_flip_draws": 20000,
            "holm_family_size": 21,
            "inferential_metrics": all_metric_names,
        })
        manifest["analysis"] = analysis
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.5), sharex=True, sharey=True)
    axes = axes.ravel()
    template_col = "template_id"
    for axis, (metric, title) in zip(axes, metric_columns):
        for model in MODELS:
            means = []; cis = []; template_series = []
            original = pd.read_csv(original_root / model.lower().replace(".", "").replace("-", "_") / "per_case.csv") if False else None
            # Resolve fixed paths from the model IDs used by the experiment.
            model_id = {MODELS[0]: "qwen35_9b", MODELS[1]: "deepseek_v2_lite_chat", MODELS[2]: "llama31_8b_instruct"}[model]
            orig = pd.read_csv(original_root / model_id / "per_case.csv").query("split == 'test'")
            conditions = [orig]
            conditions += [variant.query("model == @model and random_order_seed == @seed") for seed in order_seeds]
            for condition in conditions:
                groups = condition.groupby(template_col)[metric].mean()
                vals = groups.to_numpy(dtype=float)
                draw = np.mean(rng.choice(vals, size=(2000, len(vals)), replace=True), axis=1)
                means.append(float(vals.mean())); cis.append((float(np.quantile(draw, .025)), float(np.quantile(draw, .975))))
                template_series.append(groups)
            x = np.arange(5, dtype=float) + (MODELS.index(model) - 1) * .12
            color = COLORS[model]
            for i, groups in enumerate(template_series):
                axis.scatter(np.repeat(x[i], len(groups)) + rng.normal(0, .012, len(groups)), groups.to_numpy(),
                             s=4, color=color, alpha=.13, edgecolors="none", zorder=1)
            yerr = np.array([[m - lo for m, (lo, hi) in zip(means, cis)], [hi - m for m, (lo, hi) in zip(means, cis)]])
            axis.errorbar(x, means, yerr=yerr, color=color, marker="o", ms=3.4, lw=1.15, capsize=1.6, label=model, zorder=3)
        for base_name, frame in base_frames.items():
            base = frame.groupby(template_col)[metric].mean().mean()
            spec = next(item for item in base_specs if item[0] == base_name)
            axis.axhline(base, lw=.65, ls=spec[2], color=spec[3], alpha=.8)
        axis.set_title(title, loc="left", fontsize=7.1, pad=2)
        axis.set_xticks(range(5), names); axis.grid(axis="y", color="#E7E7E7", lw=.4)
        axis.set_ylim(-.02, 1.02)
    axes[0].set_ylabel("Case score (template mean)"); axes[2].set_ylabel("Case score (template mean)")
    for axis in axes[2:]: axis.set_xlabel("Field-order condition (order seeds)")
    handles, labels = axes[0].get_legend_handles_labels()
    base_handles = [plt.Line2D([0], [0], color=color, lw=.8, ls=style, label=f"{name} (fixed)")
                    for name, _, style, color in base_specs]
    fig.legend(handles + base_handles, labels + [h.get_label() for h in base_handles], loc="lower center",
               bbox_to_anchor=(.52, -.08), ncol=3, frameon=False, fontsize=5.3)
    fig.suptitle("Four case-specific random field orders on all 288 test cases of one held-out split", x=.02, y=1.015, ha="left", fontsize=8.3, weight="bold")
    save(fig, "figure4_random_order_robustness.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate manuscript vector figures.")
    parser.add_argument("--figure1-only", action="store_true",
                        help="regenerate only the self-contained overall framework figure")
    parser.add_argument("--intro-figure-only", action="store_true",
                        help="regenerate only the introduction example figure")
    args = parser.parse_args()
    if args.figure1_only:
        figure1_overall_framework()
        print("Wrote Figure 1 to", OUT)
        return
    if args.intro_figure_only:
        figure_intro_reference_sample()
        print("Wrote introduction figure to", OUT)
        return
    figure4_order_probe()
    figure1_overall_framework()
    figure2_pareto_and_validity()
    figure3_reference_audit()
    figure_intro_reference_sample()
    print("Wrote vector figures to", OUT)


if __name__ == "__main__":
    main()
