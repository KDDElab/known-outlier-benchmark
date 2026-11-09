#!/usr/bin/env python3
"""Create non-LLM reference predictions for the v2 synthetic benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path


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


def baseline_record(case_id: str, explanations: list[list[str]], evidence: str) -> dict:
    return {
        "case_id": case_id,
        "task_label": "unknown",
        "mechanism_feature_ids": [],
        "mechanism_metric_applicable": False,
        "explanations": [
            {"features": features, "evidence": evidence, "confidence": 0.5}
            for features in explanations
        ],
        "abstain": False,
    }


def univariate_score(schema_item: dict, target: object, references: list[dict]) -> float:
    feature_id = schema_item["feature_id"]
    values = [row.get(feature_id) for row in references]
    nonmissing = [value for value in values if value is not None]
    if target is None:
        return 1.0
    frequency = sum(value == target for value in nonmissing) / max(1, len(nonmissing))
    rarity = 1.0 - frequency
    if schema_item.get("value_type") not in {"numeric", "ordinal", "binary"}:
        return rarity
    if not isinstance(target, (int, float)) or isinstance(target, bool):
        return rarity
    numeric = [float(value) for value in nonmissing if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not numeric:
        return rarity
    median = statistics.median(numeric)
    mad = statistics.median(abs(value - median) for value in numeric)
    robust_z = abs(float(target) - median) / (1.4826 * mad) if mad > 0 else 0.0
    return max(robust_z, rarity)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--inputs-path", type=Path, default=HERE / "inputs.jsonl")
    parser.add_argument("--ground-truth-path", type=Path, default=HERE / "ground_truth.jsonl")
    parser.add_argument("--outdir", type=Path, default=HERE / "runs" / "baseline_predictions")
    args = parser.parse_args()

    if args.replicates < 1:
        raise ValueError("--replicates must be positive")
    inputs = [row for row in read_jsonl(args.inputs_path) if row["split"] == "test"]
    truths = {row["case_id"]: row for row in read_jsonl(args.ground_truth_path)}
    train_truths = [row for row in truths.values() if row["split"] == "train"]
    size_pool = [
        len(subspace)
        for truth in train_truths
        for subspace in truth["reference_support_sufficient_minimal_subspaces"]
        if subspace
    ]
    if not size_pool:
        raise ValueError("training split contains no nonempty minimal support subspaces")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    output_rows: dict[str, list[dict]] = {
        "all_features": [],
        "top_univariate": [],
        "oracle_reference_support": [],
        "oracle_rule": [],
    }
    random_rows = {seed: [] for seed in range(args.replicates)}

    for case in inputs:
        data = extract_case_data(case["user_prompt"])
        feature_ids = case["feature_ids"]
        schema = {item["feature_id"]: item for item in data["feature_schema"]}
        target = data["target_row"]
        references = data["normal_reference_rows"]

        output_rows["all_features"].append(baseline_record(
            case["case_id"], [sorted(feature_ids)], "Full feature set baseline."
        ))
        ranked = sorted(
            feature_ids,
            key=lambda fid: (-univariate_score(schema[fid], target.get(fid), references), fid),
        )
        output_rows["top_univariate"].append(baseline_record(
            case["case_id"], [[fid] for fid in ranked[:4]],
            "Ranked by a univariate robust-deviation or empirical-rarity score.",
        ))

        support_oracle = truths[case["case_id"]]["reference_support_sufficient_minimal_subspaces"]
        rule_oracle = truths[case["case_id"]]["rule_minimal_subspaces"]
        output_rows["oracle_reference_support"].append(baseline_record(
            case["case_id"], [sorted(s) for s in support_oracle[:4]], "Oracle reference-support explanation upper bound."
        ))
        output_rows["oracle_rule"].append(baseline_record(
            case["case_id"], [sorted(s) for s in rule_oracle[:4]], "Oracle rule-sufficient explanation upper bound."
        ))

        for replicate in range(args.replicates):
            rng = random.Random(args.seed + replicate * 1_000_003 + sum(map(ord, case["case_id"])))
            chosen = []
            seen = set()
            for _ in range(4):
                size = min(len(feature_ids), max(1, rng.choice(size_pool)))
                candidate = tuple(sorted(rng.sample(feature_ids, size)))
                if candidate not in seen:
                    chosen.append(list(candidate))
                    seen.add(candidate)
            random_rows[replicate].append(baseline_record(
                case["case_id"], chosen, "Uniform random feature subset; size sampled from train-split minimal-support sizes."
            ))

    for name, rows in output_rows.items():
        path = outdir / f"predictions_{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    for replicate, rows in random_rows.items():
        path = outdir / f"predictions_random_{replicate:03d}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    manifest = {
        "benchmark_protocol": "known_anomaly_subspace_benchmark_v2_compact_json_v1",
        "split": "test",
        "n_cases": len(inputs),
        "random_replicates": args.replicates,
        "random_seed_base": args.seed,
        "random_size_pool_source": "training split reference_support_sufficient_minimal_subspaces only",
        "univariate_score": "max(empirical rarity, absolute median deviation / (1.4826*MAD)); nonnumeric uses empirical rarity",
        "oracle_methods": ["reference_support_minimal", "rule_minimal"],
        "inputs_path": str(args.inputs_path.resolve()),
        "ground_truth_path": str(args.ground_truth_path.resolve()),
        "inputs_sha256": hashlib.sha256(args.inputs_path.read_bytes()).hexdigest(),
        "ground_truth_sha256": hashlib.sha256(args.ground_truth_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
