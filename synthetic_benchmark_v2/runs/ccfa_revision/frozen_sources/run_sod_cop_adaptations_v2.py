#!/usr/bin/env python3
"""Run transparent mixed-type SOD/COP adaptations on the v2 test cases.

Both methods use the same one-hot/continuous encoding documented for M²OE.
SOD returns its selected axis-parallel local subspace. COP returns a local
correlation error vector; this adapter ranks original fields by the norm of
their grouped error-vector components. These are labeled adaptations because
the benchmark offers only small sampled references and mixed feature types.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from sklearn.neighbors import NearestNeighbors

from run_m2oe_baseline_v2 import continuous_encoding, legacy_continuous_encoding, extract_payload, read_jsonl

HERE = Path(__file__).resolve().parent


def baseline_record(case_id: str, choices: list[list[str]], evidence: str) -> dict:
    return {
        "case_id": case_id,
        "task_label": "unknown",
        "mechanism_feature_ids": [],
        "mechanism_metric_applicable": False,
        "explanations": [
            {"features": fields, "evidence": evidence, "confidence": 0.5}
            for fields in choices[:4]
        ],
        "abstain": not bool(choices),
    }


def shared_neighbor_subspace(matrix: np.ndarray, alpha: float = 0.8) -> list[int]:
    """PyOD SOD's shared-neighbor and variance-threshold steps for row 0."""
    n, d = matrix.shape
    if n < 4 or d == 0:
        return []
    n_neighbors = min(20, n - 1)
    ref_set = min(10, n_neighbors - 1)
    if ref_set < 1:
        return []
    indices = NearestNeighbors(n_neighbors=n_neighbors).fit(matrix).kneighbors(return_distance=False)
    target_set = set(indices[0].tolist())
    shared = np.asarray([len(target_set.intersection(row.tolist())) for row in indices], dtype=np.int64)
    shared[0] = np.iinfo(np.int64).max
    ref_indices = np.argsort(shared)[::-1][1:ref_set + 1]
    reference = matrix[ref_indices]
    means = reference.mean(axis=0)
    var_total = float(np.square(reference - means).sum() / ref_set)
    var_expect = alpha * var_total / d
    return np.flatnonzero(np.var(reference, axis=0) < var_expect).tolist()


def sod_fields(case: dict, payload: dict, target: np.ndarray, normals: np.ndarray, groups: list[list[int]]) -> list[list[str]]:
    matrix = np.vstack((target, normals))
    selected = set(shared_neighbor_subspace(matrix))
    feature_ids = [item["feature_id"] for item in payload["feature_schema"]]
    fields = [fid for fid, dims in zip(feature_ids, groups) if selected.intersection(dims)]
    return [sorted(fields)] if fields else []


def cop_ranked_fields(payload: dict, target: np.ndarray, normals: np.ndarray, groups: list[list[int]], k: int = 30) -> list[list[str]]:
    """COP-style local PCA error vector, grouped to original fields.

    Uses ordinary local PCA (not the paper's robust-PCA options) and a
    chi-square tail score to select the correlation dimensionality. The
    returned field ordering is a benchmark adapter over the error vector.
    """
    if len(normals) < 3:
        return []
    k = min(k, len(normals))
    neighbor_ids = NearestNeighbors(n_neighbors=k).fit(normals).kneighbors(target, return_distance=False)[0]
    local = normals[neighbor_ids].astype(np.float64)
    center = local.mean(axis=0)
    centered = local - center
    # Work in the row-space to keep the computation stable when d >> k.
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular ** 2 / max(1, len(local) - 1)
    valid = eigenvalues > max(1e-12, float(eigenvalues.max(initial=0.0)) * 1e-10)
    eigenvalues, components = eigenvalues[valid], vt[valid]
    if not len(eigenvalues):
        return []
    residual = target.reshape(-1).astype(np.float64) - center
    projected = components @ residual
    best_score, best_delta = -1.0, 0
    for delta in range(len(eigenvalues)):
        tail_eigen = eigenvalues[delta:]
        tail_proj = projected[delta:]
        if not len(tail_eigen):
            continue
        d2 = float(np.sum(np.square(tail_proj) / np.maximum(tail_eigen, 1e-12)))
        score = float(chi2.cdf(d2, df=len(tail_eigen)))
        if score > best_score:
            best_score, best_delta = score, delta
    error_vector = components[best_delta:].T @ projected[best_delta:]
    feature_ids = [item["feature_id"] for item in payload["feature_schema"]]
    field_scores = []
    for fid, dims in zip(feature_ids, groups):
        score = float(np.square(error_vector[dims]).sum()) if dims else 0.0
        field_scores.append((score, fid))
    field_scores.sort(key=lambda pair: (-pair[0], pair[1]))
    ranked = [fid for score, fid in field_scores if score > 1e-15]
    return [sorted(ranked[:size]) for size in range(1, min(4, len(ranked)) + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-path", type=Path, default=HERE / "inputs.jsonl")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--encoding", choices=["schema_domain_v1", "legacy_reference_clip"], default="schema_domain_v1")
    args = parser.parse_args()
    cases = [row for row in read_jsonl(args.inputs_path) if row["split"] == args.split]
    outputs = {"sod_subspace_adaptation": [], "cop_error_vector_adaptation": []}
    start = time.perf_counter()
    for case in cases:
        payload = extract_payload(case["user_prompt"])
        encode = continuous_encoding if args.encoding == "schema_domain_v1" else legacy_continuous_encoding
        target, normals, groups = encode(payload)
        outputs["sod_subspace_adaptation"].append(baseline_record(
            case["case_id"], sod_fields(case, payload, target, normals, groups),
            "SOD shared-neighbor local subspace mapped from continuous encoded dimensions.",
        ))
        outputs["cop_error_vector_adaptation"].append(baseline_record(
            case["case_id"], cop_ranked_fields(payload, target, normals, groups),
            "COP local-PCA error vector grouped and ranked by original field.",
        ))
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        (args.outdir / f"predictions_{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    prediction_hashes = {
        name: hashlib.sha256((args.outdir / f"predictions_{name}.jsonl").read_bytes()).hexdigest()
        for name in outputs
    }
    manifest = {
        "n_cases": len(cases), "split": args.split,
        "encoding": args.encoding,
        "encoder_sha256": hashlib.sha256((HERE / "run_m2oe_baseline_v2.py").read_bytes()).hexdigest(),
        "sod": "PyOD SOD 1.1.3 shared-nearest-neighbor reference selection and alpha=0.8 variance-threshold subspace; implementation translated from official BSD-licensed source; target + sampled normals only",
        "sod_source_url": "https://github.com/yzhao062/pyod/blob/1.1.3/pyod/models/sod.py",
        "cop": "Kriegel et al. COP local-PCA error-vector adaptation; nearest sampled normals, ordinary (non-robust) PCA due small reference count; chi-square tail criterion over estimable dimensions; field ranking by grouped squared error-vector norm",
        "cop_source_doi": "10.1109/ICDM.2012.21",
        "comparability": "task-specific mixed-type adapters, not claims of canonical full-method reproduction; use outputs as explanatory candidate sets",
        "elapsed_seconds": time.perf_counter() - start,
        "inputs_sha256": hashlib.sha256(args.inputs_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prediction_sha256": prediction_hashes,
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
