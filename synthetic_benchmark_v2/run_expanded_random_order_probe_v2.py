#!/usr/bin/env python3
"""Prepare and analyze a three-model, four-order probe on the full 288-case test split."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifact_release_staging" / "synthetic_benchmark_v2"
PROBE = ART / "runs" / "expanded_random_field_order_288"
SOURCE = ART / "inputs.jsonl"
TRUTH = ART / "ground_truth.jsonl"
SEEDS = (1001, 1002, 1003, 1004)
MODELS = {
    "qwen35_9b": "Qwen3.5-9B",
    "deepseek_v2_lite_chat": "DeepSeek-V2-Lite-Chat",
    "llama31_8b_instruct": "Llama-3.1-8B-Instruct",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ordered_row(row: dict, order: list[str]) -> dict:
    if set(row) != set(order):
        raise ValueError("Row keys differ from schema feature IDs")
    return {field: row[field] for field in order}


def prompt_payload(prompt: str) -> tuple[str, dict]:
    marker = '"feature_schema"'
    loc = prompt.find(marker)
    if loc < 0:
        raise ValueError("Prompt has no feature_schema")
    start = prompt.rfind("{", 0, loc)
    payload, _ = json.JSONDecoder().raw_decode(prompt[start:])
    return prompt[:start], payload


def transformed(row: dict, seed: int) -> dict:
    result = dict(row)
    old_id = row["case_id"]
    prefix, payload = prompt_payload(row["user_prompt"])
    base = [item["feature_id"] for item in payload["feature_schema"]]
    rng = random.Random(f"expanded-field-order:{seed}:{old_id}")
    order = base.copy()
    rng.shuffle(order)
    if order == base and len(order) > 1:
        order = order[1:] + order[:1]
    by_id = {item["feature_id"]: item for item in payload["feature_schema"]}
    payload["feature_schema"] = [by_id[field] for field in order]
    payload["target_row"] = ordered_row(payload["target_row"], order)
    payload["normal_reference_rows"] = [ordered_row(item, order) for item in payload["normal_reference_rows"]]
    result["user_prompt"] = prefix + json.dumps(payload, ensure_ascii=False, indent=2)
    result["prompt_sha256"] = hashlib.sha256(
        (result["system_prompt"] + "\0" + result["user_prompt"]).encode("utf-8")
    ).hexdigest()
    result["source_case_id"] = old_id
    result["random_order_seed"] = seed
    result["case_id"] = f"order{seed}_{old_id}"
    result["feature_ids"] = order
    result["protocol_id"] = f"expanded_random_field_order_{seed}"
    return result


def prepare() -> None:
    source = [row for row in read_jsonl(SOURCE) if row["split"] == "test"]
    truth = read_jsonl(TRUTH)
    truth_by_id = {row["case_id"]: row for row in truth}
    ids = {row["case_id"] for row in source}
    if len(source) != 288 or len(ids) != 288 or not ids.issubset(truth_by_id):
        raise RuntimeError("Expected the frozen 288-case seed-20261106 test split and complete labels")
    PROBE.mkdir(parents=True, exist_ok=True)
    transformed_rows = [transformed(row, seed) for seed in SEEDS for row in source]
    transformed_truth = []
    for seed in SEEDS:
        for case_id in sorted(ids):
            label = dict(truth_by_id[case_id])
            label["source_case_id"] = case_id
            label["random_order_seed"] = seed
            label["case_id"] = f"order{seed}_{case_id}"
            transformed_truth.append(label)
    inputs_path = PROBE / "inputs_all_orders.jsonl"
    truth_path = PROBE / "ground_truth_all_orders.jsonl"
    inputs_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in transformed_rows), encoding="utf-8")
    truth_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in transformed_truth), encoding="utf-8")
    manifest = {
        "source_inputs": str(SOURCE.relative_to(HERE.parent)),
        "source_ground_truth": str(TRUTH.relative_to(HERE.parent)),
        "n_cases_per_order": len(source),
        "n_orders": len(SEEDS),
        "n_case_runs_per_model": len(transformed_rows),
        "order_seeds": list(SEEDS),
        "models": MODELS,
        "transform": "For each case, uniformly shuffle feature-schema order and target/reference-row object key order; preserve values, reference-row order, task text, and labels.",
        "inputs_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
        "ground_truth_sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (PROBE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"probe_dir": str(PROBE), **manifest}, indent=2))


def analyze() -> None:
    import subprocess
    import sys

    inputs_path = PROBE / "inputs_all_orders.jsonl"
    truth_path = PROBE / "ground_truth_all_orders.jsonl"
    manifest_path = PROBE / "manifest.json"
    if not inputs_path.exists() or not truth_path.exists():
        raise FileNotFoundError("Run prepare before analyze")
    variant_rows = []
    for model, model_name in MODELS.items():
        annotations_path = PROBE / f"annotations_{model}.jsonl"
        records = [row for row in read_jsonl(annotations_path) if row.get("event") == "annotation"]
        unique = {row["case_id"]: row for row in records}
        expected = {row["case_id"] for row in read_jsonl(inputs_path)}
        if set(unique) != expected or len(records) != len(expected):
            raise RuntimeError(f"{model}: expected {len(expected)} unique annotations, found {len(unique)}")
        preds_path = PROBE / f"predictions_{model}.jsonl"
        preds_path.write_text("".join(
            json.dumps({"case_id": row["case_id"], **row["parsed_prediction"]}, ensure_ascii=False) + "\n"
            for row in records if isinstance(row.get("parsed_prediction"), dict)
        ), encoding="utf-8")
        out = PROBE / f"evaluation_{model}"
        subprocess.run([
            sys.executable, str(HERE / "evaluate_benchmark.py"), str(preds_path),
            "--inputs-path", str(inputs_path), "--ground-truth-path", str(truth_path),
            "--outdir", str(out),
        ], check=True)
        scored = pd.read_csv(out / "per_case.csv").query("split == 'test'")
        scored["source_case_id"] = scored["case_id"].str.replace(r"^order(?:1001|1002|1003|1004)_", "", regex=True)
        scored["random_order_seed"] = scored["case_id"].str.extract(r"^order(\d+)_").astype(int)
        scored["model"] = model_name
        variant_rows.append(scored)
    all_scores = pd.concat(variant_rows, ignore_index=True)
    metric_map = {
        "Sufficiency@1": "support_top1_sufficient",
        "Exact@1": "support_top1_minimal",
        "Hit@4": "support_any_minimal_hit",
        "Best Jaccard@4": "support_best_jaccard",
        "Rule sufficiency@1": "rule_top1_sufficient",
        "Rule exact@1": "rule_top1_minimal",
        "Rule Hit@4": "rule_any_minimal_hit",
    }
    summary = all_scores.groupby(["model", "random_order_seed"]).agg(
        n=("case_id", "size"), valid_rate=("structurally_valid", "mean"),
        **{out: (column, "mean") for out, column in metric_map.items()},
    ).reset_index()
    # Compactness is conditional on a structured prediction, matching the main table.
    compactness = {
        key: frame.loc[frame["structurally_valid"].astype(bool), "mean_predicted_subspace_fraction"].mean()
        for key, frame in all_scores.groupby(["model", "random_order_seed"])
    }
    summary["Compactness"] = [float(compactness[(row.model, row.random_order_seed)]) for row in summary.itertuples()]
    summary.to_csv(PROBE / "metrics_by_model_order.csv", index=False)
    all_scores.to_csv(PROBE / "per_case_all_orders.csv", index=False)
    original_root = ART / "runs" / "multiseed_analysis" / "20261106"
    base_rows = []
    for model, model_name in MODELS.items():
        frame = pd.read_csv(original_root / model / "per_case.csv").query("split == 'test'")
        row = {"model": model_name, "random_order_seed": 0, "n": len(frame), "valid_rate": frame["structurally_valid"].astype(bool).mean()}
        for out, column in metric_map.items():
            row[out] = frame[column].astype(float).fillna(0).mean()
        row["Compactness"] = frame.loc[frame["structurally_valid"].astype(bool), "mean_predicted_subspace_fraction"].mean()
        base_rows.append(row)
    all_summary = pd.concat([pd.DataFrame(base_rows), summary], ignore_index=True)
    all_summary["order"] = all_summary["random_order_seed"].map(lambda x: "Original" if x == 0 else f"Random {x}")
    all_summary.to_csv(PROBE / "metrics_with_original.csv", index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["analysis"] = {
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scoring_code_sha256": hashlib.sha256(Path(HERE / "evaluate_benchmark.py").read_bytes()).hexdigest(),
        "per_case_scores_sha256": hashlib.sha256((PROBE / "per_case_all_orders.csv").read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256((PROBE / "metrics_with_original.csv").read_bytes()).hexdigest(),
        "metrics": list(metric_map) + ["Compactness"],
        "compactness_basis": "conditional on structurally valid outputs, matching the main table",
        "unit": "288 cases from one held-out generator seed; four independent field-order permutations per case; invalid outputs count as misses for support/rule outcomes. The complete main-table metric set is summarized by order.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(all_summary.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "analyze"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else analyze()
