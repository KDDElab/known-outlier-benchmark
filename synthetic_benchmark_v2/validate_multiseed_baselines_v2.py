#!/usr/bin/env python3
"""Validate one-to-one case coverage for every frozen baseline across three seeds."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (20261106, 20261107, 20261108)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locations(seed: int) -> tuple[Path, Path, dict[str, Path]]:
    if seed == 20261106:
        inputs = HERE / "inputs.jsonl"
        root = HERE / "runs" / "multiseed" / "seed_20261106_rt522"
        paths = {
            "all_features": HERE / "runs" / "baseline_predictions" / "predictions_all_features.jsonl",
            "top_univariate": HERE / "runs" / "baseline_predictions" / "predictions_top_univariate.jsonl",
            "gower_knn_subspace": HERE / "runs" / "baseline_predictions" / "predictions_gower_knn_subspace.jsonl",
            "gower_lof_subspace": HERE / "runs" / "baseline_predictions" / "predictions_gower_lof_subspace.jsonl",
            "sod_subspace_adaptation": HERE / "runs" / "sod_cop_adaptations" / "predictions_sod_subspace_adaptation.jsonl",
            "cop_error_vector_adaptation": HERE / "runs" / "sod_cop_adaptations" / "predictions_cop_error_vector_adaptation.jsonl",
            "m2oe_choice": HERE / "runs" / "m2oe" / "predictions_m2oe_choice.jsonl",
            "rlax_reference_adaptation": HERE / "runs" / "rlax_reference_adaptation" / "predictions_rlax_reference_adaptation.jsonl",
        }
        paths.update({f"random_{path.stem}": path for path in sorted((HERE / "runs" / "baseline_predictions").glob("predictions_random_*.jsonl"))})
    else:
        inputs = HERE / "runs" / "multiseed" / f"seed_{seed}" / "inputs.jsonl"
        root = HERE / "runs" / "multiseed" / f"seed_{seed}"
        paths = {
            "all_features": root / "baselines" / "predictions_all_features.jsonl",
            "top_univariate": root / "baselines" / "predictions_top_univariate.jsonl",
            "gower_knn_subspace": root / "gower" / "predictions_gower_knn_subspace.jsonl",
            "gower_lof_subspace": root / "gower" / "predictions_gower_lof_subspace.jsonl",
            "sod_subspace_adaptation": root / "sod_cop" / "predictions_sod_subspace_adaptation.jsonl",
            "cop_error_vector_adaptation": root / "sod_cop" / "predictions_cop_error_vector_adaptation.jsonl",
            "m2oe_choice": root / "m2oe" / "predictions_m2oe_choice.jsonl",
            "rlax_reference_adaptation": root / "rlax" / "predictions_rlax_reference_adaptation.jsonl",
        }
        paths.update({f"random_{path.stem}": path for path in sorted((root / "baselines").glob("predictions_random_*.jsonl"))})
    return inputs, root, paths


def main() -> None:
    seeds = {}
    for seed in SEEDS:
        inputs, _, methods = locations(seed)
        expected = {row["case_id"] for row in read_jsonl(inputs) if row.get("split") == "test"}
        if len(expected) != 288:
            raise RuntimeError(f"Seed {seed} expected test split is not 288 cases")
        seed_results = {}
        random_files = [name for name in methods if name.startswith("random_predictions_random_")]
        if len(random_files) != 100:
            raise RuntimeError(f"Seed {seed} has {len(random_files)}/100 random prior replicates")
        for name, path in sorted(methods.items()):
            rows = read_jsonl(path)
            counts = Counter(row.get("case_id") for row in rows)
            observed = set(counts)
            duplicates = sum(value != 1 for value in counts.values())
            if observed != expected or len(rows) != len(expected) or duplicates:
                raise RuntimeError(f"{seed}/{name}: expected 288 unique IDs, found rows={len(rows)}, unique={len(observed)}, duplicate_ids={duplicates}")
            seed_results[name] = {"n_rows": len(rows), "n_unique_case_ids": len(observed), "sha256": sha256(path), "path": str(path.resolve())}
        seeds[str(seed)] = {"n_test_cases": len(expected), "methods": seed_results}
    payload = {
        "seeds": list(SEEDS),
        "ranked_baselines": ["gower_knn_subspace", "gower_lof_subspace", "top_univariate", "sod_subspace_adaptation", "cop_error_vector_adaptation", "rlax_reference_adaptation"],
        "unordered_choice_baseline": "m2oe_choice; evaluated only by matched-budget order-invariant set metrics",
        "simple_controls": ["all_features", "100 random training-size-prior replicates per seed"],
        "coverage_rule": "Exactly one prediction row per each of 288 held-out case IDs for each baseline and random replicate; no missing or duplicate IDs.",
        "validation": seeds,
        "validator_sha256": sha256(Path(__file__).resolve()),
    }
    out = HERE / "runs" / "multiseed" / "baseline_validation_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "seeds": len(seeds), "baseline_files": sum(len(row["methods"]) for row in seeds.values())}, indent=2))


if __name__ == "__main__":
    main()
