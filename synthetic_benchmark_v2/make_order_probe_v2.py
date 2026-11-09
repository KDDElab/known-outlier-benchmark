#!/usr/bin/env python3
"""Select a balanced 96-case, template-stratified order-sensitivity probe."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs.jsonl"
MANIFEST = HERE / "case_manifest.jsonl"
RUNS = HERE / "runs"
SEED = 20261106
N_PER_TEMPLATE = 4
BASELINE_PROTOCOL = "known_anomaly_subspace_benchmark_v2_order_probe_original_v1"
REVERSE_PROTOCOL = "known_anomaly_subspace_benchmark_v2_reverse_feature_and_reference_order_v1"


def reverse_order(case: dict) -> dict:
    prompt = case["user_prompt"]
    marker = "\n{\n  \"feature_schema\""
    offset = prompt.find(marker)
    if offset < 0:
        raise ValueError(f"Could not locate JSON payload in {case['case_id']}")
    prefix, payload_text = prompt[: offset + 1], prompt[offset + 1 :]
    payload = json.loads(payload_text)
    payload["feature_schema"] = list(reversed(payload["feature_schema"]))
    payload["target_row"] = dict(reversed(list(payload["target_row"].items())))
    payload["normal_reference_rows"] = [
        dict(reversed(list(row.items())))
        for row in reversed(payload["normal_reference_rows"])
    ]
    case["user_prompt"] = prefix + json.dumps(payload, ensure_ascii=False, indent=2)
    case["prompt_sha256"] = hashlib.sha256(
        (case["system_prompt"] + "\0" + case["user_prompt"]).encode("utf-8")
    ).hexdigest()
    case["protocol_id"] = REVERSE_PROTOCOL
    return case


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    inputs = [json.loads(line) for line in INPUTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    input_by_id = {row["case_id"]: row for row in inputs}
    manifests = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in manifests:
        if row["split"] == "test":
            grouped[row["template_id"]].append(row)
    rng = np.random.default_rng(SEED)
    selected = []
    for template in sorted(grouped):
        cases = sorted(grouped[template], key=lambda row: row["case_id"])
        indices = sorted(rng.choice(len(cases), size=N_PER_TEMPLATE, replace=False))
        selected.extend(cases[index] for index in indices)
    original_rows = []
    reverse_rows = []
    for item in selected:
        case = input_by_id[item["case_id"]]
        original = dict(case)
        original["protocol_id"] = BASELINE_PROTOCOL
        original_rows.append(original)
        reverse_rows.append(reverse_order(dict(case)))
    if len(original_rows) != 96 or len({row["case_id"] for row in original_rows}) != 96:
        raise RuntimeError("Expected exactly 96 unique selected cases")
    RUNS.mkdir(parents=True, exist_ok=True)
    original_path = RUNS / "order_probe_original_inputs.jsonl"
    reverse_path = RUNS / "order_probe_reverse_inputs.jsonl"
    write_jsonl(original_path, original_rows)
    write_jsonl(reverse_path, reverse_rows)

    prior_path = RUNS / "annotations_qwen35_9b_reverse_order.jsonl"
    probe_annotations = RUNS / "annotations_qwen35_9b_order_probe_reverse.jsonl"
    selected_ids = {row["case_id"] for row in reverse_rows}
    prior = [json.loads(line) for line in prior_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reuse = [row for row in prior if row.get("event") == "annotation" and row.get("case_id") in selected_ids and row.get("prompt_sha256") == next(x["prompt_sha256"] for x in reverse_rows if x["case_id"] == row["case_id"])]
    write_jsonl(probe_annotations, reuse)

    selection = {
        "seed": SEED,
        "cases_per_template": N_PER_TEMPLATE,
        "n_cases": len(selected),
        "n_test_templates": len(grouped),
        "selected_case_ids": [row["case_id"] for row in original_rows],
        "feature_count_distribution": dict(Counter(str(row["feature_count"]) for row in selected)),
        "reference_count_distribution": dict(Counter(str(row["reference_count"]) for row in selected)),
        "nuisance_distribution": dict(Counter(row["nuisance_profile"] for row in selected)),
        "field_name_distribution": dict(Counter(row["field_name_mode"] for row in selected)),
        "reversed_annotations_reused": len(reuse),
        "baseline_protocol_id": BASELINE_PROTOCOL,
        "reverse_protocol_id": REVERSE_PROTOCOL,
    }
    (RUNS / "order_probe_selection_manifest.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
