#!/usr/bin/env python3
"""Extend finite-reference ambiguity diagnostics across all three generator seeds."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_reference_sample_identifiability import audit_case, load_jsonl

HERE = Path(__file__).resolve().parent
SEEDS = (20261106, 20261107, 20261108)
METRICS = (
    "sample_minimal_set_exact_match", "has_spurious_sample_minimal",
    "gold_candidate_recall", "spurious_sample_minimal_count",
    "sample_minimal_candidate_count",
)


def ground_truth_path(seed: int) -> Path:
    return HERE / "ground_truth.jsonl" if seed == 20261106 else HERE / "runs" / "multiseed" / f"seed_{seed}" / "ground_truth.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro_cluster_summary(rows: list[dict], repetitions: int, seed: int) -> dict:
    if not rows:
        return {"n_cases": 0, "n_templates": 0}
    result = {"n_cases": len(rows), "n_templates": len({row["template_group_id"] for row in rows})}
    rng = np.random.default_rng(seed)
    for metric in METRICS:
        clusters: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            clusters[row["template_group_id"]].append(float(row[metric]))
        names = sorted(clusters)
        means = {name: float(np.mean(clusters[name])) for name in names}
        draws = np.asarray([rng.choice(list(means.values()), len(names), replace=True).mean() for _ in range(repetitions)])
        result[metric] = float(np.mean(list(means.values())))
        result[f"{metric}_template_cluster_bootstrap_95_ci"] = [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--outdir", type=Path, default=HERE / "runs" / "reference_sample_audit_multiseed")
    args = parser.parse_args()
    per_seed = {}
    all_test = []
    hashes = {}
    for generator_seed in SEEDS:
        path = ground_truth_path(generator_seed)
        hashes[str(generator_seed)] = sha256(path)
        cases = load_jsonl(path)
        rows = [audit_case(case, args.max_order) for case in cases]
        for row in rows:
            row["generator_seed"] = generator_seed
        test = [row for row in rows if row["split"] == "test"]
        if len(test) != 288:
            raise RuntimeError(f"Expected 288 test rows for seed {generator_seed}, found {len(test)}")
        per_seed[str(generator_seed)] = macro_cluster_summary(test, args.bootstrap, args.seed + generator_seed % 1000)
        all_test.extend(test)

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "per_case_test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_test[0]))
        writer.writeheader(); writer.writerows(all_test)
    overall = macro_cluster_summary(all_test, args.bootstrap, args.seed)
    slices = {}
    for row in all_test:
        row["feature_width_x_reference_count"] = f"{row['feature_count']}x{row['reference_count']}"
    for factor in ("reference_count", "feature_count", "nuisance_profile", "feature_width_x_reference_count"):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in all_test:
            groups[str(row[factor])].append(row)
        slices[factor] = {
            value: macro_cluster_summary(rows, args.bootstrap, args.seed + index + len(factor))
            for index, (value, rows) in enumerate(sorted(groups.items()))
        }
    payload = {
        "candidate_max_order": args.max_order,
        "seeds": list(SEEDS),
        "test_cases_per_seed": 288,
        "total_test_case_instances": len(all_test),
        "n_unique_template_groups": len({row["template_group_id"] for row in all_test}),
        "summary_unit": "Macro-average within-template means across all seed-specific held-out cases; bootstrap unique template groups, not case rows.",
        "definition": "A sample-absent subset has a target projection observed in none of the sampled normal references; sample-minimal means no proper subset up to the same order is absent.",
        "overall": overall,
        "per_seed": per_seed,
        "slices": slices,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "source_ground_truth_sha256": hashes,
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
    }
    output = args.outdir / "summary.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "cases": len(all_test), "templates": payload["n_unique_template_groups"], "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
