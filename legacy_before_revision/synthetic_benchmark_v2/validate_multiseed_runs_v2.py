#!/usr/bin/env python3
"""Validate complete model outputs and record authoritative run selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (20261106, 20261107, 20261108)
MODELS = ("qwen35_9b", "deepseek_v2_lite_chat", "llama31_8b_instruct")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paths(seed: int) -> tuple[Path, Path, str]:
    if seed == 20261106:
        root = HERE / "runs" / "multiseed" / "seed_20261106_rt522"
        return HERE / "inputs.jsonl", root, f"known_anomaly_subspace_v2_seed_{seed}_rt522"
    root = HERE / "runs" / "multiseed" / f"seed_{seed}"
    return root / "inputs.jsonl", root, f"known_anomaly_subspace_v2_seed_{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "runs" / "multiseed" / "run_selection_manifest.json")
    args = parser.parse_args()
    selection = {}
    for seed in SEEDS:
        input_path, root, protocol = paths(seed)
        expected = {row["case_id"] for row in read_jsonl(input_path) if row.get("split") == "test"}
        selection[str(seed)] = {}
        log_path = root / "model_runs.jsonl"
        logs = read_jsonl(log_path) if log_path.exists() else []
        for model in MODELS:
            annotation_path = root / f"annotations_{model}.jsonl"
            rows = [row for row in read_jsonl(annotation_path) if row.get("event") == "annotation" and row.get("protocol_id") == protocol]
            by_case: dict[str, dict] = {}
            duplicates = Counter()
            for row in rows:
                duplicates[row["case_id"]] += 1
                by_case[row["case_id"]] = row
            observed = set(by_case)
            if observed != expected or any(n != 1 for n in duplicates.values()):
                raise SystemExit(f"incomplete or duplicate {seed}/{model}: expected={len(expected)}, observed={len(observed)}, duplicate_cases={sum(n != 1 for n in duplicates.values())}")
            run_ids = sorted({row.get("run_id") for row in rows})
            completed = [row for row in logs if row.get("event") == "completed" and row.get("run_id") in run_ids]
            valid = sum(row.get("parsed_prediction") is not None for row in rows)
            errors = sum(bool(row.get("parse_or_inference_error")) for row in rows)
            latencies = [float(row["latency_seconds"]) for row in rows if row.get("latency_seconds") is not None]
            selection[str(seed)][model] = {
                "protocol_id": protocol,
                "annotation_file": str(annotation_path.resolve()),
                "annotation_sha256": sha256_file(annotation_path),
                "n_test_cases": len(expected),
                "n_valid_structured": valid,
                "n_parse_or_inference_errors": errors,
                "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
                "authoritative_run_ids": run_ids,
                "matching_completed_run_events": len(completed),
                "run_log_sha256": sha256_file(log_path),
            }
    excluded = []
    for path in (HERE / "runs" / "multiseed" / "seed_20261108").glob("annotations_*_gpu0_oom.jsonl"):
        rows = read_jsonl(path)
        valid = sum(row.get("parsed_prediction") is not None for row in rows)
        errors = sum(bool(row.get("parse_or_inference_error")) for row in rows)
        excluded.append({
            "file": str(path.resolve()), "sha256": sha256_file(path),
            "n_cases": len(rows), "n_valid_structured": valid, "n_inference_or_parse_errors": errors,
            "reason": "excluded failed GPU run; all records contain inference errors and the matched GPU-2 rerun is authoritative",
        })
    payload = {
        "seeds": list(SEEDS), "models": list(MODELS),
        "common_inference_runtime": {"transformers": "5.22.1", "pytorch": "2.14.1+cu126", "gpu": "NVIDIA RTX 3090"},
        "selection_rule": "one annotation per expected test case under the frozen protocol; invalid model outputs remain included as misses",
        "runs": selection, "excluded_runs": excluded,
        "validator_sha256": sha256_file(Path(__file__).resolve()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "validated_seed_model_pairs": sum(map(len, selection.values())), "excluded_runs": len(excluded)}, indent=2))


if __name__ == "__main__":
    main()
