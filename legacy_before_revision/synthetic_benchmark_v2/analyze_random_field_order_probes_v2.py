#!/usr/bin/env python3
"""Score four random field-order variants against the frozen 96-case baseline."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import platform
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
PROBE = RUNS / "random_field_order_probe"
SEEDS = (20261121, 20261122, 20261123, 20261124)
METRICS = ("support_any_minimal_hit", "support_best_jaccard")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require_complete_annotations(path: Path, protocol: str, case_ids: set[str]) -> list[dict]:
    records = [row for row in read_jsonl(path) if row.get("event") == "annotation" and row.get("protocol_id") == protocol]
    got = [row["case_id"] for row in records]
    if len(got) != len(case_ids) or set(got) != case_ids:
        raise RuntimeError(f"Incomplete or duplicate annotations for {protocol}: {len(got)} records, {len(set(got))} cases")
    return records


def export_predictions(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in records:
            parsed = row.get("parsed_prediction")
            if isinstance(parsed, dict):
                stream.write(json.dumps({"case_id": row["case_id"], **parsed}, ensure_ascii=False) + "\n")


def evaluate(predictions: Path, inputs: Path, output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(HERE / "evaluate_benchmark.py"), str(predictions),
        "--inputs-path", str(inputs), "--ground-truth-path", str(PROBE / "ground_truth_96.jsonl"),
        "--outdir", str(output),
    ], check=True)
    return pd.read_csv(output / "per_case.csv").query("split == 'test'").set_index("case_id")


def paired_cluster_bootstrap(
    baseline: pd.DataFrame, random_mean: pd.DataFrame, template_by_case: dict[str, str], metric: str,
    repetitions: int = 2000, seed: int = 20261125,
) -> list[float]:
    case_ids = sorted(set(baseline.index) & set(random_mean.index))
    templates = sorted({template_by_case[case] for case in case_ids})
    group_values = {
        template: [float(random_mean.loc[case, metric]) - float(baseline.loc[case, metric])
                   for case in case_ids if template_by_case[case] == template]
        for template in templates
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for index in range(repetitions):
        selected = rng.choice(templates, size=len(templates), replace=True)
        values = [value for template in selected for value in group_values[str(template)]]
        draws[index] = np.mean(values)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def main() -> None:
    manifest = json.loads((PROBE / "manifest.json").read_text(encoding="utf-8"))
    input_paths = {seed: PROBE / f"inputs_{seed}.jsonl" for seed in SEEDS}
    cases_by_seed = {seed: read_jsonl(path) for seed, path in input_paths.items()}
    case_ids = {row["case_id"] for row in cases_by_seed[SEEDS[0]]}
    if any({row["case_id"] for row in rows} != case_ids for rows in cases_by_seed.values()):
        raise RuntimeError("Variant case IDs do not match")
    truth_rows = read_jsonl(PROBE / "ground_truth_96.jsonl")
    template_by_case = {row["case_id"]: row["template_group_id"] for row in truth_rows}
    baseline = pd.read_csv(RUNS / "order_probe_original_evaluation" / "per_case.csv").query("split == 'test'").set_index("case_id")

    variant_frames = {}
    variant_summaries = {}
    run_metadata = {}
    source_paths = []
    for seed in SEEDS:
        protocol = manifest["variants"][str(seed)]["protocol_id"]
        ann_path = PROBE / f"annotations_qwen35_9b_{seed}.jsonl"
        records = require_complete_annotations(ann_path, protocol, case_ids)
        expected_hashes = {row["case_id"]: row["prompt_sha256"] for row in cases_by_seed[seed]}
        if any(record.get("prompt_sha256") != expected_hashes.get(record["case_id"]) for record in records):
            raise RuntimeError(f"Prompt hash mismatch in random field-order run {seed}")
        log_path = PROBE / f"model_runs_{seed}.jsonl"
        if not log_path.exists() and seed == SEEDS[0]:
            log_path = PROBE / "model_runs.jsonl"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing run metadata for permutation {seed}: {log_path}")
        metadata_records = [row for row in read_jsonl(log_path) if row.get("protocol_id") == protocol]
        if not any(row.get("event") == "completed" for row in metadata_records):
            raise RuntimeError(f"Run metadata is incomplete for permutation {seed}")
        run_metadata[str(seed)] = metadata_records
        prediction_path = PROBE / f"predictions_qwen35_9b_{seed}.jsonl"
        export_predictions(records, prediction_path)
        eval_path = PROBE / f"evaluation_qwen35_9b_{seed}"
        frame = evaluate(prediction_path, input_paths[seed], eval_path)
        variant_frames[seed] = frame
        variant_summaries[str(seed)] = {
            "n_structurally_valid": int(frame["structurally_valid"].astype(str).str.lower().eq("true").sum()),
            "metrics": {metric: float(frame[metric].astype(float).mean()) for metric in METRICS},
        }
        source_paths.extend([ann_path, log_path, prediction_path, eval_path / "per_case.csv"])

    averaged = pd.DataFrame(index=sorted(case_ids))
    averaged["template_id"] = [template_by_case[case] for case in averaged.index]
    averaged["structurally_valid_rate"] = np.mean([
        variant_frames[seed].loc[averaged.index, "structurally_valid"].astype(str).str.lower().eq("true").to_numpy(dtype=float)
        for seed in SEEDS
    ], axis=0)
    for metric in METRICS:
        averaged[metric] = np.mean([
            variant_frames[seed].loc[averaged.index, metric].astype(float).to_numpy()
            for seed in SEEDS
        ], axis=0)

    metric_summary = {}
    for index, metric in enumerate(METRICS):
        ci = paired_cluster_bootstrap(baseline, averaged, template_by_case, metric, seed=20261125 + index)
        metric_summary[metric] = {
            "baseline_mean": float(baseline[metric].astype(float).mean()),
            "mean_over_four_random_permutations": float(np.mean([variant_summaries[str(seed)]["metrics"][metric] for seed in SEEDS])),
            "permutation_range": [
                min(variant_summaries[str(seed)]["metrics"][metric] for seed in SEEDS),
                max(variant_summaries[str(seed)]["metrics"][metric] for seed in SEEDS),
            ],
            "mean_paired_difference_random_minus_baseline": float(averaged[metric].mean() - baseline.loc[averaged.index, metric].astype(float).mean()),
            "paired_template_cluster_bootstrap_95_ci": ci,
        }
    summary = {
        "n_cases_per_variant": len(case_ids),
        "n_templates": len(set(template_by_case.values())),
        "n_random_permutations": len(SEEDS),
        "permutation_seeds": list(SEEDS),
        "inference_environment": {
            key: manifest.get(key)
            for key in ("python_version", "python_implementation", "platform")
        },
        "analysis_environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "baseline_protocol": "known_anomaly_subspace_benchmark_v2_order_probe_original_v1",
        "metrics": metric_summary,
        "variants": variant_summaries,
        "run_metadata": run_metadata,
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 20261125,
        "uncertainty_unit": "24 template clusters; per-case scores averaged across four random permutations before paired cluster resampling",
        "interpretation": "Exploratory single-checkpoint sensitivity; four transformations of the same 96 cases do not create independent case samples or establish broad prompt robustness.",
        "sha256": {str(path.relative_to(HERE.parent)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [
            PROBE / "manifest.json", PROBE / "ground_truth_96.jsonl", HERE / "evaluate_benchmark.py",
            Path(__file__).resolve(), RUNS / "order_probe_original_evaluation" / "per_case.csv", *source_paths,
        ]},
    }
    out_json = PROBE / "random_field_order_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    averaged.to_csv(PROBE / "random_field_order_per_case_average.csv", index_label="case_id")
    print(json.dumps({"summary": str(out_json), "metrics": metric_summary}, indent=2))


if __name__ == "__main__":
    main()
