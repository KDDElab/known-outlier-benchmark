#!/usr/bin/env python3
"""Score one-seed Phi-4/Gemma-3 family-sensitivity runs separately from the main cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODELS = ("phi4_mini", "gemma3_4b")
PROTOCOL = "known_anomaly_subspace_v2_seed_20261106_rt522_extra_model_family_v1"
PRIMARY_CASE_METRICS = {
    "support_sufficiency_at_1": "support_top1_sufficient",
    "support_minimal_exact_at_1": "support_top1_minimal",
    "support_Hit_at_4": "support_any_minimal_hit",
    "support_best_Jaccard": "support_best_jaccard",
}


def template_bootstrap(frame: pd.DataFrame, metric: str, seed: int, repetitions: int = 2_000) -> list[float]:
    grouped = {name: values[metric].to_numpy(dtype=float) for name, values in frame.groupby("template_id")}
    names = list(grouped)
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for index in range(repetitions):
        picked = rng.choice(names, size=len(names), replace=True)
        draws[index] = np.concatenate([grouped[name] for name in picked]).mean()
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=HERE, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=HERE / "runs" / "extra_model_family_seed06")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS),
                        help="Subset of completed model runs to score.")
    args = parser.parse_args()
    args.outdir = args.outdir.resolve()
    root = HERE / "runs" / "multiseed" / "seed_20261106_rt522"
    inputs, truth = HERE / "inputs.jsonl", HERE / "ground_truth.jsonl"
    args.outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    for model in args.models:
        annotations = root / f"annotations_{model}.jsonl"
        records = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines() if line.strip()]
        records = [row for row in records if row.get("event") == "annotation" and row.get("protocol_id") == PROTOCOL]
        expected = {row["case_id"] for row in map(json.loads, inputs.read_text(encoding="utf-8").splitlines()) if row.get("split") == "test"}
        by_case = {row["case_id"]: row for row in records}
        if set(by_case) != expected or len(records) != len(by_case):
            raise SystemExit(f"Incomplete or duplicate output for {model}: {len(by_case)}/{len(expected)} unique cases")
        predictions = args.outdir / f"predictions_{model}.jsonl"
        run([sys.executable, str(HERE / "export_predictions_v2.py"), str(annotations),
             "--protocol", PROTOCOL, "--output", str(predictions)])
        evaluated = args.outdir / model
        run([sys.executable, str(HERE / "evaluate_benchmark.py"), str(predictions),
             "--inputs-path", str(inputs), "--ground-truth-path", str(truth), "--outdir", str(evaluated)])
        per_case = pd.read_csv(evaluated / "per_case.csv").query("split == 'test'")
        confidence_intervals = {
            name: template_bootstrap(per_case, column, 20261112 + len(results) * 100 + index)
            for index, (name, column) in enumerate(PRIMARY_CASE_METRICS.items())
        }
        results[model] = {
            "n_test_cases": len(expected), "n_valid_structured": sum(row.get("parsed_prediction") is not None for row in records),
            "annotation_sha256": sha256_file(annotations), "prediction_sha256": sha256_file(predictions),
            "run_log_sha256": sha256_file(root / "additional_model_runs.jsonl"),
            "template_cluster_bootstrap_95_ci": confidence_intervals,
            "bootstrap_replicates": 2_000,
            "summary": json.loads((evaluated / "summary.json").read_text(encoding="utf-8"))["test"],
        }
    payload = {
        "protocol_id": PROTOCOL, "seed": 20261106, "models": list(args.models), "n_test_cases": 288,
        "status": "descriptive single-seed family-sensitivity cohort; excluded from three-seed confirmatory contrasts",
        "runtime": {"transformers": "5.22.1", "pytorch": "2.14.1+cu126", "gpu": "NVIDIA RTX 3090"},
        "runner_sha256": sha256_file(Path(__file__).resolve()), "evaluator_sha256": sha256_file(HERE / "evaluate_benchmark.py"),
        "results": results,
    }
    out = args.outdir / "summary.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out.resolve()), "models": list(args.models)}, indent=2))


if __name__ == "__main__":
    main()
