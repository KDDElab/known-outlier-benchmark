#!/usr/bin/env python3
"""Score and test paired, nested reference-size conditions for one Qwen checkpoint."""
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
PROBE = RUNS / "reference_size_probe"
SEED_ROOT = RUNS / "multiseed" / "seed_20261106_rt522"
CONTROL_PROTOCOL = "known_anomaly_subspace_v2_seed_20261106_rt522"
CONDITION_PROTOCOLS = {
    8: "known_anomaly_subspace_v2_reference_size_8_v1",
    16: "known_anomaly_subspace_v2_reference_size_16_v1",
    32: "known_anomaly_subspace_v2_reference_size_32_control_v1",
}
METRICS = {
    "sufficiency@1": "support_top1_sufficient",
    "minimal_exact@1": "support_top1_minimal",
    "Hit@4": "support_any_minimal_hit",
    "best_Jaccard": "support_best_jaccard",
}
PAIRS = ((8, 16), (16, 32), (8, 32))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_run(path: Path, protocol: str, model_id: str | None = None) -> dict:
    rows = [
        row for row in read_jsonl(path)
        if row.get("event") == "completed"
        and row.get("protocol_id") == protocol
        and (model_id is None or row.get("model_id") == model_id)
    ]
    if not rows:
        raise RuntimeError(f"No completed run metadata for {protocol}")
    return rows[-1]


def load_annotations(path: Path, protocol: str, expected: set[str]) -> dict[str, dict]:
    # The seed-level control log also contains train/dev annotations. Restrict
    # it to the explicitly selected test cases before checking completeness.
    rows = [
        row for row in read_jsonl(path)
        if row.get("event") == "annotation"
        and row.get("protocol_id") == protocol
        and row.get("case_id") in expected
    ]
    by_id = {row["case_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != expected:
        raise RuntimeError(f"Incomplete/duplicate {protocol}: {len(by_id)}/{len(expected)} unique cases")
    return by_id


def export(rows: dict[str, dict], path: Path) -> None:
    path.write_text("".join(
        json.dumps({"case_id": case_id, **row["parsed_prediction"]} if isinstance(row.get("parsed_prediction"), dict) else {"case_id": case_id}, ensure_ascii=False) + "\n"
        for case_id, row in sorted(rows.items())
    ), encoding="utf-8")


def evaluate(pred: Path, inputs: Path, truth: Path, outdir: Path) -> pd.DataFrame:
    subprocess.run([
        sys.executable, str(HERE / "evaluate_benchmark.py"), str(pred),
        "--inputs-path", str(inputs), "--ground-truth-path", str(truth), "--outdir", str(outdir),
    ], check=True, stdout=subprocess.DEVNULL)
    return pd.read_csv(outdir / "per_case.csv").query("split == 'test'").set_index("case_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    manifest = json.loads((PROBE / "manifest.json").read_text(encoding="utf-8"))
    input_paths = {size: PROBE / manifest["inputs"][str(size)]["path"] for size in (8, 16, 32)}
    input_rows = {size: read_jsonl(path) for size, path in input_paths.items()}
    ids_by_size = {size: {row["case_id"] for row in rows} for size, rows in input_rows.items()}
    expected = ids_by_size[32]
    if len(expected) != 96 or any(ids_by_size[size] != expected for size in (8, 16)):
        raise RuntimeError("Reference-size conditions do not contain the same 96 cases")

    control_annotations = load_annotations(SEED_ROOT / "annotations_qwen35_9b.jsonl", CONTROL_PROTOCOL, expected)
    # The original 32-row prompts must match the exact common-runtime control run case by case.
    hash_by_id = {row["case_id"]: row["prompt_sha256"] for row in input_rows[32]}
    if any(control_annotations[case_id].get("prompt_sha256") != hash_by_id[case_id] for case_id in expected):
        raise RuntimeError("32-reference control prompts do not match the frozen main-run annotations")
    predictions = {}
    annotation_rows = {32: control_annotations}
    predictions[32] = PROBE / "predictions_qwen35_9b_reference_size_32.jsonl"
    export(control_annotations, predictions[32])
    scored = {
        32: evaluate(predictions[32], input_paths[32], PROBE / "ground_truth_96.jsonl", PROBE / "evaluation_32")
    }
    for size in (8, 16):
        annotations = load_annotations(PROBE / f"annotations_qwen35_9b_reference_size_{size}.jsonl", CONDITION_PROTOCOLS[size], expected)
        annotation_rows[size] = annotations
        input_hash = {row["case_id"]: row["prompt_sha256"] for row in input_rows[size]}
        if any(annotations[case_id].get("prompt_sha256") != input_hash[case_id] for case_id in expected):
            raise RuntimeError(f"{size}-reference prompt hashes disagree with annotations")
        predictions[size] = PROBE / f"predictions_qwen35_9b_reference_size_{size}.jsonl"
        export(annotations, predictions[size])
        scored[size] = evaluate(predictions[size], input_paths[size], PROBE / "ground_truth_96.jsonl", PROBE / f"evaluation_{size}")

    control_meta = completed_run(SEED_ROOT / "model_runs.jsonl", CONTROL_PROTOCOL, "Qwen3.5-9B")
    run_meta = {size: completed_run(PROBE / "model_runs.jsonl", CONDITION_PROTOCOLS[size]) for size in (8, 16)}
    comparable = ("model_id", "model_path", "model_files", "inference_transformers_version", "checkpoint_transformers_version", "torch_version", "dtype", "loaded_model_type", "gpu", "decoding")
    for size, row in run_meta.items():
        mismatch = [field for field in comparable if row.get(field) != control_meta.get(field)]
        if mismatch:
            raise RuntimeError(f"Reference-size {size} run differs from the common Qwen control in {mismatch}")

    rng = np.random.default_rng(args.seed)
    cluster_summary = {}
    for size in (8, 16, 32):
        frame = scored[size]
        cluster_summary[str(size)] = {
            "n_cases": len(frame),
            "n_templates": int(frame["template_id"].nunique()),
            "valid_outputs": int(frame["structurally_valid"].astype(str).str.lower().eq("true").sum()),
            "parse_or_inference_failures": sum(bool(row.get("parse_or_inference_error")) for row in annotation_rows[size].values()),
            "input_over_cap_case_ids": sorted(
                case_id for case_id, row in annotation_rows[size].items()
                if "exceeding 16384" in str(row.get("parse_or_inference_error", ""))
            ),
            "output_cap_hits": sum(row.get("output_tokens") == 512 for row in annotation_rows[size].values()),
            "mean_input_tokens_when_recorded": float(np.mean([
                row["prompt_tokens"] for row in annotation_rows[size].values()
                if isinstance(row.get("prompt_tokens"), (int, float))
            ])) if any(isinstance(row.get("prompt_tokens"), (int, float)) for row in annotation_rows[size].values()) else None,
            "metrics": {},
        }
        for name, column in METRICS.items():
            by_template = {key: group[column].astype(float).to_numpy() for key, group in frame.groupby("template_id")}
            names = list(by_template)
            draws = np.asarray([np.concatenate([by_template[key] for key in rng.choice(names, len(names), replace=True)]).mean() for _ in range(args.bootstrap)])
            cluster_summary[str(size)]["metrics"][name] = {
                "estimate": float(frame[column].astype(float).mean()),
                "template_cluster_bootstrap_95_ci": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            }

    tests = []
    for low, high in PAIRS:
        left = scored[low].loc[sorted(expected)]
        right = scored[high].loc[sorted(expected)]
        templates = left["template_id"]
        if not templates.equals(right["template_id"]):
            raise RuntimeError(f"Template IDs changed between reference sizes {low} and {high}")
        for name, column in METRICS.items():
            differences = right[column].astype(float) - left[column].astype(float)
            clusters = np.asarray([differences[templates.eq(template)].mean() for template in sorted(templates.unique())], dtype=float)
            observed = float(clusters.mean())
            signs = rng.choice(np.array([-1.0, 1.0]), size=(args.permutations, len(clusters)))
            permuted = np.mean(signs * clusters[None, :], axis=1)
            p_value = float((1 + np.sum(np.abs(permuted) >= abs(observed))) / (args.permutations + 1))
            bootstrap = np.asarray([rng.choice(clusters, len(clusters), replace=True).mean() for _ in range(args.bootstrap)])
            tests.append({
                "comparison": f"{high}_minus_{low}_references",
                "metric": name,
                "high_minus_low": observed,
                "template_cluster_bootstrap_95_ci": [float(np.quantile(bootstrap, .025)), float(np.quantile(bootstrap, .975))],
                "two_sided_template_sign_flip_p": p_value,
                "n_cases": len(expected),
                "n_template_clusters": int(templates.nunique()),
            })
    order = np.argsort([row["two_sided_template_sign_flip_p"] for row in tests])
    adjusted = np.empty(len(tests)); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(tests) - rank) * tests[index]["two_sided_template_sign_flip_p"]))
        adjusted[index] = running
    for row, value in zip(tests, adjusted):
        row["holm_adjusted_p_12_contrast_family"] = float(value)

    run_hashes = {str(size): sha256(PROBE / "model_runs.jsonl") for size in (8, 16)}
    result = {
        "design": manifest["nested_design"],
        "selection_seed": manifest["seed"],
        "analysis_seed": args.seed,
        "model": "Qwen3.5-9B",
        "n_cases": len(expected),
        "n_templates": int(scored[32]["template_id"].nunique()),
        "condition_summaries": cluster_summary,
        "planned_contrast_count": 12,
        "completed_contrast_count": len(tests),
        "contrasts": tests,
        "inference": "Two-sided paired template-cluster sign flips; Holm adjustment across three pairwise sample-size contrasts and four primary metrics.",
        "bootstrap_replicates": args.bootstrap,
        "permutations": args.permutations,
        "preregistration": manifest["status"],
        "runtime": {"analysis_python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "control_run_metadata": control_meta, "new_run_metadata": run_meta, "run_log_sha256": run_hashes},
        "sha256": {
            "manifest": sha256(PROBE / "manifest.json"),
            "ground_truth_96": sha256(PROBE / "ground_truth_96.jsonl"),
            "control_annotations": sha256(SEED_ROOT / "annotations_qwen35_9b.jsonl"),
            **{f"inputs_{size}": sha256(path) for size, path in input_paths.items()},
            **{f"annotations_{size}": sha256(PROBE / f"annotations_qwen35_9b_reference_size_{size}.jsonl") for size in (8, 16)},
            **{f"per_case_{size}": sha256(PROBE / f"evaluation_{size}" / "per_case.csv") for size in (8, 16, 32)},
            "analysis_script": sha256(Path(__file__).resolve()),
            "evaluator": sha256(HERE / "evaluate_benchmark.py"),
        },
        "interpretation_limit": "A single model and seed on 96 selected cases. The nested conditions estimate the effect of changing available reference count under fixed token limits; prompt-length-driven validity changes are part of this deployed-protocol effect, so this is not a pure information-only effect or an estimate over arbitrary reference-sampling variability.",
    }
    out = PROBE / "reference_size_summary.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "completed_contrasts": len(tests), "conditions": cluster_summary}, indent=2))


if __name__ == "__main__":
    main()
