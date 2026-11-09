#!/usr/bin/env python3
"""Compare population-gold localization with compatibility to observed rows.

This is a diagnostic, not a replacement gold label: a target projection absent
from the finite normal sample is consistent with support exclusion, but absence
alone does not establish population-level minimality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEEDS = (20261106, 20261107, 20261108)
METHODS = (
    "qwen35_9b", "deepseek_v2_lite_chat", "llama31_8b_instruct",
    "gower_knn_subspace", "gower_lof_subspace", "rlax_reference_adaptation",
    "top_univariate", "sod_subspace_adaptation", "cop_error_vector_adaptation",
    "all_features",
)


def paths(seed: int) -> tuple[Path, Path, Path]:
    analysis = HERE / "runs" / "multiseed_analysis" / str(seed)
    if seed == 20261106:
        return HERE / "inputs.jsonl", HERE / "ground_truth.jsonl", analysis
    seed_root = HERE / "runs" / "multiseed" / f"seed_{seed}"
    return seed_root / "inputs.jsonl", seed_root / "ground_truth.jsonl", analysis


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_payload(prompt: str) -> dict[str, Any]:
    marker = '"feature_schema"'
    position = prompt.find(marker)
    if position < 0:
        raise ValueError("Could not find feature_schema in frozen prompt")
    start = prompt.rfind("{", 0, position)
    payload, _ = json.JSONDecoder().raw_decode(prompt[start:])
    return payload


def absent_projection(candidate: list[str], target: dict, references: list[dict]) -> bool:
    if not candidate or any(feature not in target for feature in candidate):
        return False
    target_projection = tuple(json.dumps(target[f], sort_keys=True) for f in candidate)
    for row in references:
        reference_projection = tuple(json.dumps(row[f], sort_keys=True) for f in candidate)
        if reference_projection == target_projection:
            return False
    return True


def cluster_bootstrap(
    rows: list[dict[str, Any]], metric: str, repetitions: int, seed: int
) -> list[float]:
    by_seed_template: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    all_templates: set[str] = set()
    for row in rows:
        by_seed_template[int(row["generator_seed"])][str(row["template_id"])].append(float(row[metric]))
        all_templates.add(str(row["template_id"]))
    template_ids = sorted(all_templates)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        picked = rng.choice(template_ids, size=len(template_ids), replace=True)
        counts = defaultdict(int)
        for template_id in picked:
            counts[str(template_id)] += 1
        seed_means = []
        for template_map in by_seed_template.values():
            vals = [
                value
                for template_id, count in counts.items()
                if template_id in template_map
                for _ in range(count)
                for value in template_map[template_id]
            ]
            if vals:
                seed_means.append(float(np.mean(vals)))
        if seed_means:
            draws.append(float(np.mean(seed_means)))
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--outdir", type=Path, default=HERE / "runs" / "observability_compatibility")
    args = parser.parse_args()
    case_info: dict[tuple[int, str], dict[str, Any]] = {}
    inputs_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    truth_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    source_hashes = {}
    for seed in SEEDS:
        input_path, truth_path, _ = paths(seed)
        source_hashes[str(seed)] = {"inputs": sha256(input_path), "ground_truth": sha256(truth_path)}
        inputs = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        truths = [json.loads(line) for line in truth_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        inputs_by_seed[seed] = {row["case_id"]: row for row in inputs if row.get("split") == "test"}
        truth_by_seed[seed] = {row["case_id"]: row for row in truths if row.get("split") == "test"}
        if inputs_by_seed[seed].keys() != truth_by_seed[seed].keys():
            raise RuntimeError(f"Input/truth case mismatch for seed {seed}")
        for case_id, inp in inputs_by_seed[seed].items():
            payload = prompt_payload(inp["user_prompt"])
            case_info[(seed, case_id)] = {
                "generator_seed": seed,
                "template_id": truth_by_seed[seed][case_id]["template_group_id"],
                "target": payload["target_row"],
                "references": payload["normal_reference_rows"],
                "gold": {tuple(sorted(candidate)) for candidate in truth_by_seed[seed][case_id]["reference_support_sufficient_minimal_subspaces"]},
            }

    all_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        _, _, analysis_root = paths(seed)
        for method in METHODS:
            per_case_path = analysis_root / method / "per_case.csv"
            if not per_case_path.exists():
                raise FileNotFoundError(per_case_path)
            frame = pd.read_csv(per_case_path).query("split == 'test'")
            for row in frame.to_dict(orient="records"):
                info = case_info[(seed, row["case_id"])]
                candidates = json.loads(row["explanations_predicted"] or "[]")
                candidates = [list(dict.fromkeys(candidate)) for candidate in candidates[:4] if isinstance(candidate, list)]
                exact = bool(candidates and tuple(sorted(candidates[0])) in info["gold"])
                hit4 = any(tuple(sorted(candidate)) in info["gold"] for candidate in candidates)
                top1_absent = absent_projection(candidates[0], info["target"], info["references"]) if candidates else False
                any4_absent = any(absent_projection(c, info["target"], info["references"]) for c in candidates)
                all_rows.append({
                    "method": method,
                    "case_id": row["case_id"],
                    "generator_seed": info["generator_seed"],
                    "template_id": info["template_id"],
                    "has_prediction": bool(candidates),
                    "population_gold_exact_at_1": exact,
                    "population_gold_hit_at_4": hit4,
                    "top1_projection_absent_from_observed_references": top1_absent,
                    "any_top4_projection_absent_from_observed_references": any4_absent,
                    "top1_absent_but_not_population_gold": bool(top1_absent and not exact),
                })

    args.outdir.mkdir(parents=True, exist_ok=True)
    per_case_path = args.outdir / "observability_compatibility_per_case.csv"
    with per_case_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows = []
    metrics = (
        "population_gold_exact_at_1", "population_gold_hit_at_4",
        "top1_projection_absent_from_observed_references",
        "any_top4_projection_absent_from_observed_references",
        "top1_absent_but_not_population_gold",
    )
    for method in METHODS:
        method_rows = [row for row in all_rows if row["method"] == method]
        for metric_ix, metric in enumerate(metrics):
            by_seed = []
            for seed in SEEDS:
                seed_rows = [row for row in method_rows if row["generator_seed"] == seed]
                by_seed.append(float(np.mean([float(row[metric]) for row in seed_rows])))
            summary_rows.append({
                "method": method,
                "metric": metric,
                "macro_mean_over_seeds": float(np.mean(by_seed)),
                "shared_template_cluster_bootstrap_95_ci": json.dumps(cluster_bootstrap(method_rows, metric, args.bootstrap, args.seed + metric_ix)),
                "n_case_runs": len(method_rows),
                "n_unique_templates": len({row["template_id"] for row in method_rows}),
            })
    summary_path = args.outdir / "observability_compatibility_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    manifest = {
        "purpose": "Supplemental input-compatibility diagnostic; does not change population-support gold labels.",
        "sample_absence_definition": "The candidate target projection appears in none of the visible normal reference rows.",
        "interpretation": "Sample absence is necessary for a population-support-absent explanation under the clean-reference assumption, but is not sufficient to establish population-level minimality.",
        "seeds": list(SEEDS),
        "test_case_runs": len(case_info),
        "methods": list(METHODS),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "bootstrap_unit": "Shared template IDs resampled once across seed-specific test splits; seed estimates are macro-averaged per replicate.",
        "source_sha256": source_hashes,
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
    }
    manifest_path = args.outdir / "observability_compatibility_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"per_case": str(per_case_path), "summary": str(summary_path), "manifest": str(manifest_path), "case_runs": len(case_info)}, indent=2))


if __name__ == "__main__":
    main()
