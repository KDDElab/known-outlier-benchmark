#!/usr/bin/env python3
"""Run transparent subspace controls on the circuit pilot.

Gower controls use eligible binary input columns, the target input, and normal
reference inputs only. Two conditional-output surprisal controls rank the same
subsets using either the focus output or the full observed output vector. This
provides both input-only and matched all-output comparisons to the LLMs.

All nonempty input subsets are enumerated so every gold subset in this track
is inside the baseline candidate space. Run from repository root:
    python adbench_llm_reference_v1/circuit_track/run_circuit_gower_baselines.py

Dependencies: Python 3.10+, NumPy.
"""

from __future__ import annotations

import itertools
import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "synthetic_benchmark_v2"
sys.path.insert(0, str(V2))
from run_gower_subspace_baseline import (  # noqa: E402
    feature_distance_matrix,
    lof_batch,
    robust_contrast_batch,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def record(case_id: str, ranked_features: list[list[str]], method: str) -> dict:
    return {
        "case_id": case_id,
        "explanations": [
            {
                "features": features,
                "evidence": f"Ranks binary input subsets by {method} over target and normal references.",
                "confidence": 0.5,
            }
            for features in ranked_features
        ],
        "abstain": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-path", type=Path, default=HERE / "inputs_zero_noise_pilot.jsonl")
    parser.add_argument("--outdir", type=Path, default=HERE / "baseline_predictions")
    args = parser.parse_args()
    cases = read_jsonl(args.inputs_path)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, list[dict]] = {
        "gower_knn": [], "gower_lof": [], "conditional_output_surprisal": [],
        "joint_output_surprisal": []
    }
    for case in cases:
        feature_ids = sorted(case["eligible_feature_ids"])
        target = case["target_row"]
        refs = case["normal_reference_rows"]
        n_refs = len(refs)
        k = min(3, n_refs - 1)
        matrices = [
            feature_distance_matrix(
                [target[field]] + [row[field] for row in refs], "binary", n_refs
            )
            for field in feature_ids
        ]
        subsets = [
            subset
            for order in range(1, len(feature_ids) + 1)
            for subset in itertools.combinations(range(len(feature_ids)), order)
        ]
        ranked = {"gower_knn": [], "gower_lof": [], "conditional_output_surprisal": [], "joint_output_surprisal": []}
        batch_size = 256
        for start in range(0, len(subsets), batch_size):
            batch = subsets[start:start + batch_size]
            distance_sums = np.zeros((len(batch), n_refs + 1, n_refs + 1), dtype=np.float64)
            for row_ix, subset in enumerate(batch):
                for feature_ix in subset:
                    distance_sums[row_ix] += matrices[feature_ix]
            distance_sums /= np.asarray([len(subset) for subset in batch])[:, None, None]
            knn_scores = robust_contrast_batch(distance_sums, n_refs, k)
            lof_scores = lof_batch(distance_sums, n_refs, k)
            ranked["gower_knn"].extend(
                (float(score), subset) for score, subset in zip(knn_scores, batch)
            )
            ranked["gower_lof"].extend(
                (float(score), subset) for score, subset in zip(lof_scores, batch)
            )

        for subset in subsets:
            target_key = tuple(target[feature_ids[index]] for index in subset)
            matching = [
                row for row in refs
                if tuple(row[feature_ids[index]] for index in subset) == target_key
            ]
            if matching:
                focus_value = target[case["focus_output"]]
                matching_output_count = sum(row[case["focus_output"]] == focus_value for row in matching)
                probability = (matching_output_count + 1) / (len(matching) + 2)
                score = -math.log(probability)
            else:
                score = 0.0
            ranked["conditional_output_surprisal"].append((score, subset))

            output_ids = [key for key in target if key not in feature_ids]
            target_output_vector = tuple(target[key] for key in output_ids)
            joint_count = sum(tuple(row[key] for key in output_ids) == target_output_vector for row in matching)
            # Add-one smoothing over all binary output vectors, matching the full context.
            n_output_patterns = 2 ** len(output_ids)
            joint_probability = (joint_count + 1) / (len(matching) + n_output_patterns)
            ranked["joint_output_surprisal"].append((-math.log(joint_probability), subset))

        for name, scores in ranked.items():
            scores.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
            top = [sorted(feature_ids[index] for index in subset) for _, subset in scores[:4]]
            method = {
                "gower_knn": "target-versus-reference kNN distance contrast",
                "gower_lof": "novelty LOF",
                "conditional_output_surprisal": "smoothed focus-output surprisal conditional on the input subset",
                "joint_output_surprisal": "smoothed full-output-vector surprisal conditional on the input subset",
            }[name]
            predictions[name].append(record(case["case_id"], top, method))

    for name, rows in predictions.items():
        (outdir / f"predictions_{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    output_hashes = {
        name: hashlib.sha256((outdir / f"predictions_{name}.jsonl").read_bytes()).hexdigest()
        for name in predictions
    }
    manifest = {
        "track": "Harvard circuit zero-noise output-specific pilot",
        "n_queries": len(cases),
        "methods": list(predictions),
        "eligible_features_only": True,
        "uses_target_and_normal_reference_inputs": True,
        "gower_methods_use_outputs": False,
        "conditional_output_surprisal_uses_focus_output": True,
        "joint_output_surprisal_uses_all_outputs": True,
        "joint_output_smoothing": "add-one categorical smoothing over 2**n_output_fields binary vectors",
        "max_subset_order": max(len(case["eligible_feature_ids"]) for case in cases),
        "subset_search": "all nonempty eligible-input subsets for each query; no labels or gold-set sizes determine the search space",
        "top_k": 4,
        "neighbor_count": "min(3, n_references - 1)",
        "distance": "mean binary Gower/Hamming mismatch",
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "information_asymmetry": "Gower methods use input rows only; focus conditional surprisal uses input values and the focus output; joint conditional surprisal and LLMs use all observed outputs.",
        "inputs_sha256": hashlib.sha256(args.inputs_path.read_bytes()).hexdigest(),
        "inputs_path": str(args.inputs_path),
        "tie_break": "descending score, ascending size, lexicographically sorted feature IDs",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prediction_sha256": output_hashes,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
