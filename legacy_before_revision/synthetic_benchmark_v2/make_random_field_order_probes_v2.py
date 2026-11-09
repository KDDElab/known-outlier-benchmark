#!/usr/bin/env python3
"""Create repeated, deterministic random field-order variants of the 96-case probe."""
from __future__ import annotations

import hashlib
import json
import platform
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SOURCE = RUNS / "order_probe_original_inputs.jsonl"
OUT = RUNS / "random_field_order_probe"
SEEDS = (20261121, 20261122, 20261123, 20261124)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_payload(prompt: str) -> tuple[str, dict]:
    marker = '"feature_schema"'
    location = prompt.find(marker)
    if location < 0:
        raise ValueError("feature_schema marker not found")
    start = prompt.rfind("{", 0, location)
    payload, _ = json.JSONDecoder().raw_decode(prompt[start:])
    return prompt[:start], payload


def ordered_row(row: dict, feature_order: list[str]) -> dict:
    if set(row) != set(feature_order):
        raise ValueError("feature keys do not match the schema")
    return {feature: row[feature] for feature in feature_order}


def make_variant(case: dict, seed: int) -> dict:
    variant = dict(case)
    prefix, payload = prompt_payload(case["user_prompt"])
    base_order = [item["feature_id"] for item in payload["feature_schema"]]
    rng = random.Random(f"{seed}:{case['case_id']}")
    feature_order = base_order.copy()
    rng.shuffle(feature_order)
    if feature_order == base_order and len(feature_order) > 1:
        feature_order = feature_order[1:] + feature_order[:1]

    schema_by_id = {item["feature_id"]: item for item in payload["feature_schema"]}
    payload["feature_schema"] = [schema_by_id[feature] for feature in feature_order]
    payload["target_row"] = ordered_row(payload["target_row"], feature_order)
    payload["normal_reference_rows"] = [
        ordered_row(row, feature_order) for row in payload["normal_reference_rows"]
    ]

    variant["feature_ids"] = feature_order
    variant["user_prompt"] = prefix + json.dumps(payload, ensure_ascii=False, indent=2)
    variant["prompt_sha256"] = hashlib.sha256(
        (variant["system_prompt"] + "\0" + variant["user_prompt"]).encode("utf-8")
    ).hexdigest()
    variant["protocol_id"] = f"known_anomaly_subspace_v2_random_field_order_{seed}"
    return variant


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    source_rows = read_jsonl(SOURCE)
    truth_rows = read_jsonl(HERE / "ground_truth.jsonl")
    truth_by_id = {row["case_id"]: row for row in truth_rows}
    if len(source_rows) != 96 or len({row["case_id"] for row in source_rows}) != 96:
        raise RuntimeError("Expected the frozen 96-case, four-cases-per-template probe")
    if not {row["case_id"] for row in source_rows}.issubset(truth_by_id):
        raise RuntimeError("The selected cases are missing from ground_truth.jsonl")
    OUT.mkdir(parents=True, exist_ok=True)
    selected_ids = {row["case_id"] for row in source_rows}
    write_jsonl(OUT / "ground_truth_96.jsonl", [row for row in truth_rows if row["case_id"] in selected_ids])
    manifest = {
        "source": str(SOURCE.relative_to(HERE)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "n_cases_per_permutation": len(source_rows),
        "n_templates": len({truth_by_id[row["case_id"]]["template_group_id"] for row in source_rows}),
        "ground_truth_96_sha256": hashlib.sha256((OUT / "ground_truth_96.jsonl").read_bytes()).hexdigest(),
        "seeds": list(SEEDS),
        "transformation": "For each case independently, shuffle schema fields and target/reference object key order with a deterministic seed; preserve all values, row order, task text, and labels.",
        "variants": {},
    }
    for seed in SEEDS:
        path = OUT / f"inputs_{seed}.jsonl"
        write_jsonl(path, [make_variant(row, seed) for row in source_rows])
        manifest["variants"][str(seed)] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "protocol_id": f"known_anomaly_subspace_v2_random_field_order_{seed}",
        }
    manifest["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outdir": str(OUT), "seeds": list(SEEDS), "cases_per_variant": len(source_rows)}, indent=2))


if __name__ == "__main__":
    main()
