#!/usr/bin/env python3
"""Score the reversed-order Qwen run and compare it with the original run."""
from __future__ import annotations

import json
import hashlib
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
RUNS = HERE / "runs"
BASELINE_PROTOCOL = "known_anomaly_subspace_benchmark_v2_order_probe_original_v1"
REVERSE_PROTOCOL = "known_anomaly_subspace_benchmark_v2_reverse_feature_and_reference_order_v1"
BASE_ANNOTATIONS = RUNS / "annotations_qwen35_9b_order_probe_original.jsonl"
REVERSE_ANNOTATIONS = RUNS / "annotations_qwen35_9b_order_probe_reverse.jsonl"
BASE_INPUTS = RUNS / "order_probe_original_inputs.jsonl"
REVERSE_INPUTS = RUNS / "order_probe_reverse_inputs.jsonl"
BASE_PREDICTIONS = RUNS / "predictions_qwen35_9b_order_probe_original.jsonl"
REVERSE_PREDICTIONS = RUNS / "predictions_qwen35_9b_order_probe_reverse.jsonl"
BASE_OUTDIR = RUNS / "order_probe_original_evaluation"
REVERSE_OUTDIR = RUNS / "order_probe_reverse_evaluation"
N_BOOT = 2_000
SEED = 20261106


def load_annotations(path: Path, protocol: str, input_path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in records if r.get("event") == "annotation" and r.get("protocol_id") == protocol]
    expected = {json.loads(line)["case_id"] for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    got = {r["case_id"] for r in records}
    if got != expected or len(records) != len(expected):
        raise RuntimeError(f"Incomplete or duplicate run for {protocol}: {len(got)}/{len(expected)} unique cases; records={len(records)}")
    return records


def export_predictions(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in records:
            prediction = row.get("parsed_prediction")
            if isinstance(prediction, dict):
                stream.write(json.dumps({"case_id": row["case_id"], **prediction}, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_metadata(protocol: str) -> dict:
    metadata_path = RUNS / "model_runs.jsonl"
    rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("protocol_id") == protocol]
    if not rows:
        return {"protocol_id": protocol, "metadata_found": False}
    row = rows[-1]
    return {
        "protocol_id": protocol,
        "run_id": row.get("run_id"),
        "event": row.get("event"),
        "model_id": row.get("model_id"),
        "checkpoint_transformers_version": row.get("checkpoint_transformers_version"),
        "inference_transformers_version": row.get("inference_transformers_version"),
        "torch_version": row.get("torch_version"),
        "gpu": row.get("gpu"),
        "started_at_utc": row.get("started_at_utc"),
        "completed_at_utc": row.get("completed_at_utc"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "annotation_records_written": row.get("annotation_records_written"),
    }


def main() -> None:
    base_records = load_annotations(BASE_ANNOTATIONS, BASELINE_PROTOCOL, BASE_INPUTS)
    reverse_records = load_annotations(REVERSE_ANNOTATIONS, REVERSE_PROTOCOL, REVERSE_INPUTS)
    export_predictions(base_records, BASE_PREDICTIONS)
    export_predictions(reverse_records, REVERSE_PREDICTIONS)
    subprocess.run([
        sys.executable, str(HERE / "evaluate_benchmark.py"), str(BASE_PREDICTIONS),
        "--outdir", str(BASE_OUTDIR), "--inputs-path", str(BASE_INPUTS),
    ], check=True)
    subprocess.run([
        sys.executable, str(HERE / "evaluate_benchmark.py"), str(REVERSE_PREDICTIONS),
        "--outdir", str(REVERSE_OUTDIR), "--inputs-path", str(REVERSE_INPUTS),
    ], check=True)
    original = pd.read_csv(BASE_OUTDIR / "per_case.csv").query("split == 'test'").set_index("case_id")
    variant = pd.read_csv(REVERSE_OUTDIR / "per_case.csv").query("split == 'test'").set_index("case_id")
    shared = original.index.intersection(variant.index)
    metrics = ["support_any_minimal_hit", "support_best_jaccard"]
    templates = sorted(original.loc[shared, "template_id"].unique())
    group_diffs: dict[str, dict[str, list[float]]] = defaultdict(dict)
    diffs = {}
    for metric in metrics:
        diffs[metric] = variant.loc[shared, metric].astype(float) - original.loc[shared, metric].astype(float)
        group_diffs[metric] = {
            key: diffs[metric].loc[original.loc[shared, "template_id"].eq(key).values].to_list()
            for key in templates
        }
    rng = np.random.default_rng(SEED)
    summaries = {}
    for metric in metrics:
        draws = np.empty(N_BOOT)
        groups = group_diffs[metric]
        for b in range(N_BOOT):
            picked = rng.choice(templates, size=len(templates), replace=True)
            values = [value for key in picked for value in groups[key]]
            draws[b] = np.mean(values)
        summaries[metric] = {
            "original": float(original.loc[shared, metric].astype(float).mean()),
            "reversed": float(variant.loc[shared, metric].astype(float).mean()),
            "paired_difference_reversed_minus_original": float(diffs[metric].mean()),
            "template_cluster_bootstrap_95_ci": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
        }
    original_lists = original.loc[shared, "explanations_predicted"].map(json.loads)
    variant_lists = variant.loc[shared, "explanations_predicted"].map(json.loads)
    valid = original.loc[shared, "structurally_valid"].astype(str).str.lower().eq("true") & variant.loc[shared, "structurally_valid"].astype(str).str.lower().eq("true")
    has_top1 = original_lists.map(lambda values: bool(values) and bool(values[0])) & variant_lists.map(lambda values: bool(values) and bool(values[0]))
    valid_for_stability = valid & has_top1
    pair_jaccard = []
    for case in shared[valid_for_stability.to_numpy()]:
        left = set(original_lists.loc[case][0]) if original_lists.loc[case] else set()
        right = set(variant_lists.loc[case][0]) if variant_lists.loc[case] else set()
        union = left | right
        pair_jaccard.append(len(left & right) / len(union) if union else 1.0)
    summary = {
        "baseline_protocol_id": BASELINE_PROTOCOL,
        "reverse_protocol_id": REVERSE_PROTOCOL,
        "n_cases": int(len(shared)),
        "n_templates": len(templates),
        "cases_per_template": int(len(shared) / len(templates)),
        "original_structurally_valid": int(original.loc[shared, "structurally_valid"].astype(str).str.lower().eq("true").sum()),
        "reversed_structurally_valid": int(variant.loc[shared, "structurally_valid"].astype(str).str.lower().eq("true").sum()),
        "original_nonvalid_or_missing_case_ids": shared[
            ~original.loc[shared, "structurally_valid"].astype(str).str.lower().eq("true").to_numpy()
        ].tolist(),
        "reversed_nonvalid_or_missing_case_ids": shared[
            ~variant.loc[shared, "structurally_valid"].astype(str).str.lower().eq("true").to_numpy()
        ].tolist(),
        "both_valid_with_nonempty_top1_for_stability": int(valid_for_stability.sum()),
        "mean_top1_prediction_set_jaccard_when_both_valid": float(np.mean(pair_jaccard)) if pair_jaccard else None,
        "metrics": summaries,
        "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": SEED,
        "runtime": {
            "analysis_python": platform.python_version(),
            "analysis_numpy": np.__version__,
            "analysis_pandas": pd.__version__,
            "original_inference": run_metadata(BASELINE_PROTOCOL),
            "reverse_inference": run_metadata(REVERSE_PROTOCOL),
        },
        "sha256": {
            str(path.relative_to(PACKAGE_ROOT)): sha256_file(path)
            for path in [
                BASE_INPUTS,
                REVERSE_INPUTS,
                BASE_ANNOTATIONS,
                REVERSE_ANNOTATIONS,
                RUNS / "order_probe_selection_manifest.json",
                HERE / "ground_truth.jsonl",
                HERE / "case_manifest.jsonl",
                HERE / "evaluate_benchmark.py",
                Path(__file__).resolve(),
                BASE_PREDICTIONS,
                REVERSE_PREDICTIONS,
                BASE_OUTDIR / "per_case.csv",
                REVERSE_OUTDIR / "per_case.csv",
            ]
        },
        "interpretation": "Exploratory paired sensitivity on a seed-selected four-case-per-template subset to simultaneous feature-order and reference-row-order reversal; not separate row-only and column-only effects.",
    }
    (REVERSE_OUTDIR / "order_perturbation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
