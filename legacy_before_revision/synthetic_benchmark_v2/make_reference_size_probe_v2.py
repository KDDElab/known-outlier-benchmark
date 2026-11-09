#!/usr/bin/env python3
"""Create nested, paired 8/16/32-reference prompts for the same 96 targets."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "reference_size_probe"
SOURCE = HERE / "inputs.jsonl"
TRUTH = HERE / "ground_truth.jsonl"
MANIFEST = HERE / "case_manifest.jsonl"
SEED = 20261113
PROTOCOLS = {
    8: "known_anomaly_subspace_v2_reference_size_8_v1",
    16: "known_anomaly_subspace_v2_reference_size_16_v1",
    32: "known_anomaly_subspace_v2_reference_size_32_control_v1",
}
PAYLOAD_MARKER = '\n{\n  "feature_schema"'


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def payload(prompt: str) -> tuple[str, dict]:
    offset = prompt.find(PAYLOAD_MARKER)
    if offset < 0:
        raise ValueError("Embedded feature-schema payload not found")
    return prompt[: offset + 1], json.loads(prompt[offset + 1 :])


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = [row for row in read_jsonl(SOURCE) if row.get("split") == "test" and row.get("reference_count") == 32]
    if len(source) != 96 or len({row["case_id"] for row in source}) != 96:
        raise RuntimeError(f"Expected 96 unique 32-reference test cases, got {len(source)}")
    case_templates = {row["case_id"]: row["template_id"] for row in read_jsonl(MANIFEST)}
    if len({case_templates[row["case_id"]] for row in source}) != 24:
        raise RuntimeError("Expected four selected cases from each of 24 held-out templates")

    rng = random.Random(SEED)
    by_size: dict[int, list[dict]] = {size: [] for size in (8, 16, 32)}
    selected_indices: dict[str, dict[str, list[int]]] = {}
    for original in source:
        prefix, base_payload = payload(original["user_prompt"])
        references = base_payload["normal_reference_rows"]
        if len(references) != 32:
            raise RuntimeError(f"Reference rows disagree with metadata for {original['case_id']}")
        order = list(range(32))
        rng.shuffle(order)
        pick8 = set(order[:8])
        pick16 = set(order[:16])
        selected_indices[original["case_id"]] = {
            "n8_original_positions": sorted(pick8),
            "n16_original_positions": sorted(pick16),
            "n32_original_positions": list(range(32)),
        }
        if not pick8.issubset(pick16):
            raise RuntimeError("Nested reference selection invariant failed")
        for size in (8, 16, 32):
            row = dict(original)
            row["protocol_id"] = PROTOCOLS[size]
            if size < 32:
                p = json.loads(json.dumps(base_payload))
                keep = pick8 if size == 8 else pick16
                p["normal_reference_rows"] = [item for index, item in enumerate(references) if index in keep]
                row["user_prompt"] = prefix + json.dumps(p, ensure_ascii=False, indent=2)
                row["prompt_sha256"] = hashlib.sha256((row["system_prompt"] + "\0" + row["user_prompt"]).encode()).hexdigest()
            row["reference_count"] = size
            by_size[size].append(row)

    paths = {}
    for size, rows in by_size.items():
        path = OUT / f"inputs_reference_size_{size}.jsonl"
        write_jsonl(path, rows)
        paths[str(size)] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n_cases": len(rows)}
    truth_by_id = {row["case_id"]: row for row in read_jsonl(TRUTH)}
    truth_rows = [truth_by_id[row["case_id"]] for row in source]
    truth_path = OUT / "ground_truth_96.jsonl"
    write_jsonl(truth_path, truth_rows)
    payload_out = {
        "selection": "96 test cases with 32 normal references each; four cases per each of 24 held-out templates",
        "nested_design": "For each target, choose a seeded random ordering of the 32 reference-row positions. The 8-row set is a subset of the 16-row set, which is a subset of all 32 rows; selected rows preserve their original relative order. Greedy decoding, 16,384-token input cap without truncation, and 512-token output cap are fixed. Prompt length changes with reference count, so measured system effects include any validity changes caused by the input cap.",
        "seed": SEED,
        "model": "Qwen3.5-9B; 32-row condition is the exact frozen seed-20261106 common-runtime main-run prompt/output",
        "primary_metrics": ["support_top1_sufficient", "support_top1_minimal", "support_any_minimal_hit", "support_best_jaccard"],
        "planned_comparisons": "8 vs 16, 16 vs 32, and 8 vs 32 for each primary metric; template sign-flip tests and Holm across 12 contrasts",
        "status": "Post-hoc controlled sensitivity; case selection, nested reference subsets, metrics, and analysis family fixed before new 8/16-reference inference; not preregistered.",
        "templates": sorted({case_templates[row["case_id"]] for row in source}),
        "selection_indices_by_case": selected_indices,
        "inputs": paths,
        "ground_truth_sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
        "source_inputs_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "manifest.json").write_text(json.dumps(payload_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"cases_per_condition": {size: len(rows) for size, rows in by_size.items()}, "templates": len(payload_out["templates"]), "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
