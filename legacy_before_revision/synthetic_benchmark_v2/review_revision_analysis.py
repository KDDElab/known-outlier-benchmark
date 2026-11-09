#!/usr/bin/env python3
"""Reproduce review-requested analyses from frozen per-case benchmark outputs."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifact_release_staging"
RUNS = ART / "synthetic_benchmark_v2" / "runs" / "multiseed_analysis"
SEEDS = (20261106, 20261107, 20261108)
MODELS = {
    "Qwen3.5-9B": "qwen35_9b",
    "DeepSeek-V2-Lite-Chat": "deepseek_v2_lite_chat",
    "Llama-3.1-8B-Instruct": "llama31_8b_instruct",
}
METRICS = {
    "sufficiency@1": "support_top1_sufficient",
    "exact@1": "support_top1_minimal",
    "Hit@4": "support_any_minimal_hit",
    "best Jaccard@4": "support_best_jaccard",
}
OUT = ART / "manuscript_figures"
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 1001
SIGN_FLIPS = 20000
SIGN_FLIP_SEED = 1002


def load_model_frames(model: str) -> list[pd.DataFrame]:
    frames = []
    for seed in SEEDS:
        path = RUNS / str(seed) / model / "per_case.csv"
        frame = pd.read_csv(path).query("split == 'test'").copy()
        frame["seed"] = seed
        frames.append(frame)
    return frames


def bootstrap_macro_seed(frames: list[pd.DataFrame], column: str, *, valid_only: bool = False,
                         reps: int = BOOTSTRAPS, seed: int = BOOTSTRAP_SEED) -> list[float]:
    """Shared-template bootstrap, macro-averaging seed-specific case means."""
    rng = np.random.default_rng(seed)
    per_seed: list[dict[str, np.ndarray]] = []
    for frame in frames:
        work = frame
        if valid_only:
            work = work.loc[work["structurally_valid"].astype(bool)]
        values = work[column].astype(float).fillna(0.0)
        grouped = {
            str(template): values.loc[indices].to_numpy(dtype=float)
            for template, indices in work.groupby("template_id").groups.items()
        }
        per_seed.append(grouped)
    templates = sorted({template for group in per_seed for template in group})
    draws = np.empty(reps, dtype=float)
    for b in range(reps):
        selected = rng.choice(templates, size=len(templates), replace=True)
        multiplicity = Counter(selected)
        seed_means = []
        for grouped in per_seed:
            parts = [grouped[t] for t, count in multiplicity.items() if t in grouped for _ in range(int(count))]
            if parts:
                seed_means.append(float(np.concatenate(parts).mean()))
        draws[b] = float(np.mean(seed_means))
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def metric_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    data = {}
    for name, method in MODELS.items():
        frames = load_model_frames(method)
        data[name] = frames
        valid_n = sum(int(f["structurally_valid"].astype(bool).sum()) for f in frames)
        total_n = sum(len(f) for f in frames)
        row = {"model": name, "valid_n": valid_n, "total_n": total_n, "valid_rate": valid_n / total_n}
        for metric, column in METRICS.items():
            deployed_values = [f[column].astype(float).fillna(0.0).mean() for f in frames]
            valid_values = [f.loc[f["structurally_valid"].astype(bool), column].astype(float).mean() for f in frames]
            row[f"{metric}_deployed"] = float(np.mean(deployed_values))
            row[f"{metric}_deployed_ci_low"], row[f"{metric}_deployed_ci_high"] = bootstrap_macro_seed(frames, column)
            row[f"{metric}_parseable"] = float(np.mean(valid_values))
            row[f"{metric}_parseable_ci_low"], row[f"{metric}_parseable_ci_high"] = bootstrap_macro_seed(frames, column, valid_only=True, seed=BOOTSTRAP_SEED + 1)
        # The published compactness diagnostic remains conditional on a structured output.
        comp_seed_means = [f["mean_predicted_subspace_fraction"].astype(float).mean() for f in frames]
        row["compactness_conditional"] = float(np.mean(comp_seed_means))
        row["compactness_conditional_ci_low"], row["compactness_conditional_ci_high"] = bootstrap_macro_seed(
            frames, "mean_predicted_subspace_fraction", valid_only=True
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "parseable_decomposition.csv", index=False)
    return summary, data


def paired_template_differences(left: list[pd.DataFrame], right: list[pd.DataFrame], column: str) -> pd.Series:
    per_seed = []
    for a, b in zip(left, right):
        merged = a[["case_id", "template_id", column]].merge(
            b[["case_id", column]], on="case_id", suffixes=("_left", "_right"), validate="one_to_one"
        )
        merged["difference"] = merged[f"{column}_left"].astype(float).fillna(0.0) - merged[f"{column}_right"].astype(float).fillna(0.0)
        per_seed.append(merged.groupby("template_id")["difference"].mean())
    all_templates = sorted(set().union(*(set(s.index) for s in per_seed)))
    values = {}
    for template in all_templates:
        seed_diffs = [float(s.loc[template]) for s in per_seed if template in s.index]
        values[template] = float(np.mean(seed_diffs))
    return pd.Series(values, dtype=float).sort_index()


def sign_flip_p(values: np.ndarray, reps: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(values)))
    null = np.abs((signs * values[None, :]).mean(axis=1))
    return float((1 + np.count_nonzero(null >= observed)) / (reps + 1))


def bootstrap_ci(values: np.ndarray, reps: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(values, size=(reps, len(values)), replace=True), axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    n = len(p_values)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (n - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def llm_pair_tests(data: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    names = list(MODELS)
    records = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            for metric_index, (metric, column) in enumerate(METRICS.items()):
                diffs = paired_template_differences(data[a_name], data[b_name], column)
                values = diffs.to_numpy(dtype=float)
                records.append({
                    "left": a_name,
                    "right": b_name,
                    "metric": metric,
                    "n_template_clusters": int(len(values)),
                    "left_minus_right": float(values.mean()),
                    "ci95_low": bootstrap_ci(values, BOOTSTRAPS, SIGN_FLIP_SEED + metric_index),
                    "p_sign_flip": sign_flip_p(values, SIGN_FLIPS, SIGN_FLIP_SEED + 20 + metric_index),
                    "template_differences": json.dumps({k: float(v) for k, v in diffs.items()}),
                })
    adjusted = holm([r["p_sign_flip"] for r in records])
    for row, p_adj in zip(records, adjusted):
        row["p_holm_12"] = p_adj
        lo, hi = row["ci95_low"]
        row["ci95_low"] = lo
        row["ci95_high"] = hi
    out = pd.DataFrame(records).drop(columns=["template_differences"])
    out.to_csv(OUT / "llm_pairwise_tests.csv", index=False)
    (OUT / "llm_pairwise_template_differences.json").write_text(
        json.dumps({f"{r['left']}|{r['right']}|{r['metric']}": json.loads(r["template_differences"]) for r in records}, indent=2) + "\n"
    )
    return out


def reference_audit_summaries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = ART / "synthetic_benchmark_v2" / "runs" / "reference_sample_audit_multiseed" / "per_case_test.csv"
    df = pd.read_csv(path)
    universe = {6: 41, 12: 298, 24: 2324}
    df["candidate_universe"] = df["feature_count"].map(universe)
    df["sample_absent_fraction"] = df["sample_absent_candidate_count"] / df["candidate_universe"]
    by_width_ref = df.groupby(["feature_count", "reference_count"]).agg(
        n=("case_id", "size"), n_templates=("template_group_id", "nunique"),
        spurious_template_macro=("spurious_sample_minimal_count", lambda x: x.mean()),
        absent_fraction_template_macro=("sample_absent_fraction", "mean"),
        gold_retention_template_macro=("gold_candidate_recall", "mean"),
    ).reset_index()
    # Recompute each cell with template-macro weights (rather than case weights).
    by_width_ref_rows = []
    for (width, refs), group in df.groupby(["feature_count", "reference_count"]):
        templ = group.groupby("template_group_id").agg(
            spurious=("spurious_sample_minimal_count", "mean"),
            absent=("sample_absent_fraction", "mean"),
            retention=("gold_candidate_recall", "mean"),
        )
        by_width_ref_rows.append({
            "feature_count": int(width), "reference_count": int(refs), "n": len(group), "n_templates": len(templ),
            "spurious_mean": float(templ["spurious"].mean()),
            "absence_fraction_mean": float(templ["absent"].mean()),
            "gold_retention_mean": float(templ["retention"].mean()),
        })
    by_width_ref = pd.DataFrame(by_width_ref_rows)
    by_nuisance_rows = []
    for (refs, profile), group in df.groupby(["reference_count", "nuisance_profile"]):
        templ = group.groupby("template_group_id").agg(
            spurious=("spurious_sample_minimal_count", "mean"),
            retention=("gold_candidate_recall", "mean"),
        )
        by_nuisance_rows.append({
            "reference_count": int(refs), "nuisance_profile": profile, "n": len(group), "n_templates": len(templ),
            "spurious_mean": float(templ["spurious"].mean()),
            "gold_retention_mean": float(templ["retention"].mean()),
        })
    by_nuisance = pd.DataFrame(by_nuisance_rows)
    by_ref_rows = []
    for refs, group in df.groupby("reference_count"):
        templ = group.groupby("template_group_id")["gold_candidate_recall"].mean()
        by_ref_rows.append({"reference_count": int(refs), "n": len(group), "n_templates": len(templ), "gold_retention_mean": float(templ.mean())})
    by_ref = pd.DataFrame(by_ref_rows)
    by_width_ref.to_csv(OUT / "reference_audit_width_reference.csv", index=False)
    by_nuisance.to_csv(OUT / "reference_audit_nuisance_reference.csv", index=False)
    by_ref.to_csv(OUT / "reference_audit_retention_by_reference.csv", index=False)
    return by_width_ref, by_nuisance, by_ref


def adaptation_sizes() -> pd.DataFrame:
    rows = []
    for method in ("sod_subspace_adaptation", "cop_error_vector_adaptation"):
        arrays = []
        for seed in SEEDS:
            frame = pd.read_csv(RUNS / str(seed) / method / "per_case.csv").query("split == 'test'")
            for _, row in frame.iterrows():
                raw = row.get("explanations_predicted")
                if not isinstance(raw, str) or not raw.strip():
                    continue
                try:
                    candidates = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidates, list):
                    arrays.extend(len(item) for item in candidates if isinstance(item, list))
        if arrays:
            rows.append({
                "method": method,
                "n_candidate_sets": len(arrays),
                "mean_fields_per_candidate": float(np.mean(arrays)),
                "median_fields_per_candidate": float(np.median(arrays)),
                "p25_fields_per_candidate": float(np.quantile(arrays, .25)),
                "p75_fields_per_candidate": float(np.quantile(arrays, .75)),
                "min_fields_per_candidate": int(np.min(arrays)),
                "max_fields_per_candidate": int(np.max(arrays)),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "adaptation_output_cardinality.csv", index=False)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, data = metric_summary()
    pairwise = llm_pair_tests(data)
    by_width, by_nuisance, by_ref = reference_audit_summaries()
    sizes = adaptation_sizes()
    summary = {
        "bootstrap_repetitions": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "llm_pair_sign_flips": SIGN_FLIPS,
        "llm_pair_sign_flip_seed": SIGN_FLIP_SEED,
        "llm_pair_adjustment": "Holm correction within the 12 model-pair by primary-metric contrasts",
        "metrics": metrics.to_dict(orient="records"),
        "llm_pairwise_tests": pairwise.to_dict(orient="records"),
        "reference_by_width_and_count": by_width.to_dict(orient="records"),
        "reference_by_nuisance_and_count": by_nuisance.to_dict(orient="records"),
        "gold_retention_by_reference_count": by_ref.to_dict(orient="records"),
        "adaptation_output_cardinality": sizes.to_dict(orient="records"),
    }
    (OUT / "review_revision_analysis.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "parseable_decomposition": metrics.to_dict(orient="records"),
        "llm_pairwise_tests": pairwise.to_dict(orient="records"),
        "gold_retention_by_reference_count": by_ref.to_dict(orient="records"),
        "sod_cop_size": sizes.to_dict(orient="records"),
    }, indent=2))


if __name__ == "__main__":
    main()
