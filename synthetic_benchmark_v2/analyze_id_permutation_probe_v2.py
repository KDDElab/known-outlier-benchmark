#!/usr/bin/env python3
"""Evaluate the paired opaque-feature-ID permutation sensitivity probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
PROBE = RUNS / "id_permutation_probe"
CONTROL_ANNOTATIONS = RUNS / "annotations_qwen35_9b_order_probe_original.jsonl"
CONTROL_PROTOCOL = "known_anomaly_subspace_benchmark_v2_order_probe_original_v1"
VARIANT_ANNOTATIONS = PROBE / "annotations_qwen35_9b_id_permutation.jsonl"
VARIANT_PROTOCOL = "known_anomaly_subspace_v2_opaque_feature_id_permutation_v1"
CONTROL_INPUTS = RUNS / "order_probe_original_inputs.jsonl"
VARIANT_INPUTS = PROBE / "inputs_opaque_id_permutation.jsonl"
GT = HERE / "ground_truth.jsonl"
N_BOOT = 2_000
N_PERM = 20_000
SEED = 20261110


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_run_metadata(path: Path, protocol: str) -> dict:
    rows = [row for row in read_jsonl(path) if row.get("event") == "completed" and row.get("protocol_id") == protocol]
    if not rows:
        raise RuntimeError(f"No completed run metadata for {protocol} in {path}")
    return rows[-1]


def load_records(path: Path, protocol: str, expected_ids: set[str]) -> dict[str, dict]:
    records = [row for row in read_jsonl(path) if row.get("event") == "annotation" and row.get("protocol_id") == protocol]
    by_id = {row["case_id"]: row for row in records}
    if len(by_id) != len(records) or set(by_id) != expected_ids:
        raise RuntimeError(f"Incomplete/duplicate annotations for {protocol}: {len(by_id)}/{len(expected_ids)}")
    return by_id


def remap_prediction(prediction: dict | None, new_to_old: dict[str, str]) -> dict | None:
    if not isinstance(prediction, dict):
        return None
    result = dict(prediction)
    result["mechanism_feature_ids"] = [new_to_old[item] for item in result.get("mechanism_feature_ids", [])]
    explanations = []
    for item in result.get("explanations", []):
        candidate = dict(item)
        candidate["features"] = [new_to_old[feature] for feature in candidate.get("features", [])]
        explanations.append(candidate)
    result["explanations"] = explanations
    return result


def export_predictions(records: dict[str, dict], output: Path, mappings: dict[str, dict[str, dict[str, str]]] | None = None) -> None:
    rows = []
    for case_id in sorted(records):
        prediction = records[case_id].get("parsed_prediction")
        if mappings is not None:
            prediction = remap_prediction(prediction, mappings[case_id]["new_to_old"])
        rows.append({"case_id": case_id, **prediction} if isinstance(prediction, dict) else {"case_id": case_id})
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def evaluate(predictions: Path, inputs: Path, outdir: Path) -> pd.DataFrame:
    subprocess.run([
        sys.executable, str(HERE / "evaluate_benchmark.py"), str(predictions),
        "--inputs-path", str(inputs), "--ground-truth-path", str(GT), "--outdir", str(outdir),
    ], check=True, stdout=subprocess.DEVNULL)
    return pd.read_csv(outdir / "per_case.csv").query("split == 'test'").set_index("case_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=N_BOOT)
    parser.add_argument("--permutations", type=int, default=N_PERM)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    control_input_rows = read_jsonl(CONTROL_INPUTS)
    variant_input_rows = read_jsonl(VARIANT_INPUTS)
    expected = {row["case_id"] for row in control_input_rows}
    if len(expected) != 96 or {row["case_id"] for row in variant_input_rows} != expected:
        raise RuntimeError("Control and relabeled conditions must share the frozen 96 cases")
    manifest = json.loads((PROBE / "manifest.json").read_text(encoding="utf-8"))
    if manifest["n_cases"] != len(expected):
        raise RuntimeError("Unexpected probe manifest case count")
    maps = manifest["case_mappings"]

    control = load_records(CONTROL_ANNOTATIONS, CONTROL_PROTOCOL, expected)
    variant = load_records(VARIANT_ANNOTATIONS, VARIANT_PROTOCOL, expected)
    pred_control = PROBE / "predictions_qwen35_9b_control.jsonl"
    pred_variant = PROBE / "predictions_qwen35_9b_id_permutation_mapped_back.jsonl"
    export_predictions(control, pred_control)
    export_predictions(variant, pred_variant, maps)
    base = evaluate(pred_control, CONTROL_INPUTS, PROBE / "evaluation_control")
    changed = evaluate(pred_variant, CONTROL_INPUTS, PROBE / "evaluation_id_permutation")
    changed = changed.reindex(base.index)
    shared = base.index.intersection(changed.index)
    if len(shared) != 96:
        raise RuntimeError(f"Scored only {len(shared)}/96 paired cases")
    templates = base.loc[shared, "template_id"]
    rng = np.random.default_rng(args.seed)
    metric_rows = []
    for metric in ("support_any_minimal_hit", "support_best_jaccard"):
        difference = changed.loc[shared, metric].astype(float) - base.loc[shared, metric].astype(float)
        grouped = {key: difference[templates.eq(key)].to_numpy() for key in sorted(templates.unique())}
        cluster_means = np.asarray([value.mean() for value in grouped.values()], dtype=float)
        observed = float(cluster_means.mean())
        signs = rng.choice(np.array([-1.0, 1.0]), size=(args.permutations, len(cluster_means)))
        permuted = (signs * cluster_means[None, :]).mean(axis=1)
        p = float((1 + np.sum(np.abs(permuted) >= abs(observed))) / (args.permutations + 1))
        keys = list(grouped)
        boot = np.empty(args.bootstrap)
        for index in range(args.bootstrap):
            chosen = rng.choice(keys, size=len(keys), replace=True)
            boot[index] = np.concatenate([grouped[key] for key in chosen]).mean()
        metric_rows.append({
            "metric": metric,
            "control_mean": float(base.loc[shared, metric].astype(float).mean()),
            "id_permutation_mean": float(changed.loc[shared, metric].astype(float).mean()),
            "paired_difference_permutation_minus_control": observed,
            "template_cluster_bootstrap_95_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "two_sided_template_sign_flip_p": p,
            "n_cases": len(shared),
            "n_template_clusters": len(grouped),
        })
    order = np.argsort([row["two_sided_template_sign_flip_p"] for row in metric_rows])
    adjusted = np.empty(len(metric_rows)); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(metric_rows) - rank) * metric_rows[index]["two_sided_template_sign_flip_p"]))
        adjusted[index] = running
    for row, value in zip(metric_rows, adjusted):
        row["holm_adjusted_p_two_primary_metrics"] = float(value)

    left = base.loc[shared, "explanations_predicted"].map(json.loads)
    right = changed.loc[shared, "explanations_predicted"].map(json.loads)
    valid = base.loc[shared, "structurally_valid"].astype(str).str.lower().eq("true")
    valid &= changed.loc[shared, "structurally_valid"].astype(str).str.lower().eq("true")
    valid &= left.map(lambda values: bool(values) and bool(values[0]))
    valid &= right.map(lambda values: bool(values) and bool(values[0]))
    top1_jaccard = []
    for case_id in shared[valid.to_numpy()]:
        a, b = set(left.loc[case_id][0]), set(right.loc[case_id][0])
        top1_jaccard.append(len(a & b) / len(a | b) if a | b else 1.0)

    control_run = completed_run_metadata(RUNS / "model_runs.jsonl", CONTROL_PROTOCOL)
    variant_run = completed_run_metadata(PROBE / "model_runs.jsonl", VARIANT_PROTOCOL)
    comparable_fields = (
        "model_id", "model_path", "model_files", "inference_transformers_version",
        "checkpoint_transformers_version", "torch_version", "dtype", "loaded_model_type",
        "decoding",
    )
    mismatches = [field for field in comparable_fields if control_run.get(field) != variant_run.get(field)]
    if mismatches:
        raise RuntimeError(f"Control and ID-permutation model/runtime metadata differ: {mismatches}")
    control_gpu = control_run.get("gpu") or {}
    variant_gpu = variant_run.get("gpu") or {}
    if control_gpu.get("device_name") != variant_gpu.get("device_name"):
        raise RuntimeError("Control and ID-permutation runs used different GPU models")
    summary = {
        "design": manifest["design"],
        "n_cases": len(shared),
        "n_templates": int(templates.nunique()),
        "control_structurally_valid": int(base.loc[shared, "structurally_valid"].astype(str).str.lower().eq("true").sum()),
        "id_permutation_structurally_valid": int(changed.loc[shared, "structurally_valid"].astype(str).str.lower().eq("true").sum()),
        "metrics": metric_rows,
        "top1_prediction_stability": {"n_valid_pairs": len(top1_jaccard), "mean_jaccard": float(np.mean(top1_jaccard)) if top1_jaccard else None},
        "inference": "Two-sided template-level sign-flip tests and template-cluster bootstrap; Holm adjustment across the two primary metrics.",
        "bootstrap_replicates": args.bootstrap,
        "permutations": args.permutations,
        "analysis_seed": args.seed,
        "runtime": {
            "analysis_python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "control_run_metadata": control_run, "id_permutation_run_metadata": variant_run,
            "matched_model_runtime_checkpoint": True,
            "matched_gpu_model": control_gpu.get("device_name"),
            "gpu_device_index_equal": control_gpu.get("visible_devices") == variant_gpu.get("visible_devices"),
        },
        "sha256": {
            str(path.relative_to(HERE)): sha256(path)
            for path in [CONTROL_INPUTS, VARIANT_INPUTS, GT, CONTROL_ANNOTATIONS, VARIANT_ANNOTATIONS,
                         PROBE / "manifest.json", HERE / "evaluate_benchmark.py", Path(__file__).resolve(),
                         pred_control, pred_variant, PROBE / "evaluation_control" / "per_case.csv",
                         PROBE / "evaluation_id_permutation" / "per_case.csv"]
        },
        "interpretation_limit": "Tests invariance to arbitrary opaque IDs only; it does not test semantic names, natural-language evidence faithfulness, or other models.",
    }
    out = PROBE / "id_permutation_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "metrics": metric_rows}, indent=2))


if __name__ == "__main__":
    main()
