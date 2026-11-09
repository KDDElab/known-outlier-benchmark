#!/usr/bin/env python3
"""Rank feature subsets by target-vs-reference Gower kNN separation.

This is an unsupervised, target-conditioned statistical baseline. It uses only
the benchmark prompt (schema, target, and normal references), never gold labels.
All nonempty subsets up to --max-order are enumerated and ranked by the robust
standardized kNN-distance contrast between the target and leave-one-out normal
reference rows.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def extract_case_data(prompt: str) -> dict:
    marker = '"feature_schema"'
    marker_index = prompt.rfind(marker)
    if marker_index < 0:
        raise ValueError("feature_schema missing from benchmark prompt")
    start = prompt.rfind("{", 0, marker_index)
    value, _ = json.JSONDecoder().raw_decode(prompt[start:])
    return value


def as_numeric(value: object, value_type: str) -> float | None:
    if value is None:
        return None
    if value_type == "datetime" and isinstance(value, str):
        try:
            return float(np.datetime64(value, "D").astype("int64"))
        except (TypeError, ValueError):
            return None
    if value_type in {"numeric", "ordinal"} and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def feature_distance_matrix(values: list[object], value_type: str, reference_count: int) -> np.ndarray:
    """Gower-style [0,1] distance matrix over target followed by references."""
    n = len(values)
    matrix = np.zeros((n, n), dtype=np.float64)
    numeric = [as_numeric(value, value_type) for value in values]
    if value_type in {"numeric", "ordinal", "datetime"} and all(v is not None for v in numeric):
        ref_values = [v for v in numeric[1:1 + reference_count] if v is not None]
        scale = max(ref_values) - min(ref_values) if ref_values else 0.0
        if scale <= 0:
            scale = 1.0
        arr = np.asarray(numeric, dtype=np.float64)
        matrix = np.minimum(np.abs(arr[:, None] - arr[None, :]) / scale, 1.0)
        return matrix

    for i in range(n):
        for j in range(i + 1, n):
            left, right = values[i], values[j]
            if left is None and right is None:
                distance = 0.0
            elif left is None or right is None:
                distance = 1.0
            else:
                distance = float(left != right)
            matrix[i, j] = matrix[j, i] = distance
    return matrix


def robust_contrast_batch(distances: np.ndarray, reference_count: int, k: int) -> np.ndarray:
    target_reference = distances[:, 0, 1:1 + reference_count]
    target_knn = np.partition(target_reference, k - 1, axis=1)[:, :k].mean(axis=1)
    reference = distances[:, 1:1 + reference_count, 1:1 + reference_count].copy()
    diagonal = np.arange(reference_count)
    reference[:, diagonal, diagonal] = np.inf
    normal_knn = np.partition(reference, k - 1, axis=2)[:, :, :k].mean(axis=2)
    center = np.median(normal_knn, axis=1)
    mad = np.median(np.abs(normal_knn - center[:, None]), axis=1)
    scale = 1.4826 * mad
    std = np.std(normal_knn, axis=1)
    scale = np.where(scale < 1e-9, std, scale)
    scale = np.maximum(scale, 1e-9)
    return (target_knn - center) / scale


def lof_batch(distances: np.ndarray, reference_count: int, k: int) -> np.ndarray:
    target_reference = distances[:, 0, 1:1 + reference_count]
    target_neighbors = np.argpartition(target_reference, k - 1, axis=1)[:, :k]
    reference = distances[:, 1:1 + reference_count, 1:1 + reference_count].copy()
    diagonal = np.arange(reference_count)
    reference[:, diagonal, diagonal] = np.inf
    neighbor_ids = np.argpartition(reference, k - 1, axis=2)[:, :, :k]
    neighbor_distances = np.take_along_axis(reference, neighbor_ids, axis=2)
    kdist = np.max(neighbor_distances, axis=2)
    neighbor_kdist = np.take_along_axis(
        kdist, neighbor_ids.reshape(len(kdist), -1), axis=1
    ).reshape(neighbor_ids.shape)
    ref_reachability = np.maximum(neighbor_distances, neighbor_kdist)
    ref_lrd = 1.0 / np.maximum(ref_reachability.mean(axis=2), 1e-9)

    target_dist = np.take_along_axis(target_reference, target_neighbors, axis=1)
    target_neighbor_kdist = np.take_along_axis(kdist, target_neighbors, axis=1)
    target_lrd = 1.0 / np.maximum(np.maximum(target_dist, target_neighbor_kdist).mean(axis=1), 1e-9)
    target_neighbor_lrd = np.take_along_axis(ref_lrd, target_neighbors, axis=1)
    return target_neighbor_lrd.mean(axis=1) / np.maximum(target_lrd, 1e-9)


def baseline_record(case_id: str, explanations: list[list[str]]) -> dict:
    return {
        "case_id": case_id,
        "task_label": "unknown",
        "mechanism_feature_ids": [],
        "mechanism_metric_applicable": False,
        "explanations": [
            {
                "features": features,
                "evidence": "Ranks subspaces by robust target-versus-reference kNN distance contrast.",
                "confidence": 0.5,
            }
            for features in explanations
        ],
        "abstain": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--inputs-path", type=Path, default=HERE / "inputs.jsonl")
    parser.add_argument("--outdir", type=Path, default=HERE / "runs" / "baseline_predictions")
    args = parser.parse_args()
    if args.max_order < 1 or args.top_k < 1 or args.batch_size < 1:
        raise ValueError("max-order, top-k, and batch-size must be positive")

    cases = [row for row in read_jsonl(args.inputs_path) if row["split"] == "test"]
    outputs = {"gower_knn_subspace": [], "gower_lof_subspace": []}
    for case in cases:
        data = extract_case_data(case["user_prompt"])
        # Canonical IDs determine enumeration, summation and exact-score ties;
        # the display position must not affect numeric results or ranking.
        feature_ids = sorted(case["feature_ids"])
        schema = {item["feature_id"]: item for item in data["feature_schema"]}
        refs = data["normal_reference_rows"]
        target = data["target_row"]
        r_count = len(refs)
        k = min(3, max(1, r_count - 1))
        matrices = []
        for feature_id in feature_ids:
            values = [target.get(feature_id)] + [row.get(feature_id) for row in refs]
            matrices.append(feature_distance_matrix(values, schema[feature_id]["value_type"], r_count))

        subsets = [
            subset
            for order in range(1, min(args.max_order, len(feature_ids)) + 1)
            for subset in itertools.combinations(range(len(feature_ids)), order)
        ]
        ranked_knn: list[tuple[float, tuple[int, ...]]] = []
        ranked_lof: list[tuple[float, tuple[int, ...]]] = []
        for start in range(0, len(subsets), args.batch_size):
            batch = subsets[start:start + args.batch_size]
            # Batch each feature-sum distance matrix to keep memory bounded.
            sums = np.zeros((len(batch), r_count + 1, r_count + 1), dtype=np.float64)
            for row_index, subset in enumerate(batch):
                for feature_index in subset:
                    sums[row_index] += matrices[feature_index]
                sums[row_index] /= len(subset)
            knn_scores = robust_contrast_batch(sums, r_count, k)
            lof_scores = lof_batch(sums, r_count, k)
            ranked_knn.extend((float(score), subset) for score, subset in zip(knn_scores, batch))
            ranked_lof.extend((float(score), subset) for score, subset in zip(lof_scores, batch))

        for name, ranked in (("gower_knn_subspace", ranked_knn), ("gower_lof_subspace", ranked_lof)):
            ranked.sort(key=lambda pair: (-pair[0], len(pair[1]), pair[1]))
            chosen: list[list[str]] = []
            seen: set[frozenset[str]] = set()
            for _, subset in ranked:
                candidate = frozenset(feature_ids[index] for index in subset)
                if candidate in seen:
                    continue
                seen.add(candidate)
                chosen.append(sorted(candidate))
                if len(chosen) == args.top_k:
                    break
            outputs[name].append(baseline_record(case["case_id"], chosen))

    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        output_path = args.outdir / f"predictions_{name}.jsonl"
        output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "names": ["exhaustive_gower_knn_subspace", "exhaustive_gower_lof_subspace"],
        "split": "test",
        "n_cases": len(cases),
        "max_subspace_order": args.max_order,
        "top_k": args.top_k,
        "k_neighbors": "min(3, reference_count - 1)",
        "distance": "mean per-feature Gower distance; numeric/ordinal/datetime range-scaled by references; categorical/binary/identifier mismatch; missingness mismatch",
        "score": "(target kNN distance - median leave-one-out normal kNN distance) / (1.4826 * MAD), with standard-deviation and epsilon fallbacks",
        "lof_score": "LOF novelty score computed from Gower kNN and reference-only local reachability densities",
        "uses_test_labels": False,
        "uses_only_prompt_fields": True,
        "inputs_path": str(args.inputs_path.resolve()),
        "inputs_sha256": hashlib.sha256(args.inputs_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    (args.outdir / "gower_subspace_baselines_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
