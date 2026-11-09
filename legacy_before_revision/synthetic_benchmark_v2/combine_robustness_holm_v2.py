#!/usr/bin/env python3
"""Apply a single Holm family across all planned prompt/order/ID/reference tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def main() -> None:
    prompt_path = RUNS / "prompt_sensitivity_combined_summary.json"
    reference_path = RUNS / "reference_size_probe" / "reference_size_summary.json"
    out_path = RUNS / "robustness_family_holm_summary.json"
    prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    prompt_tests = prompt.get("contrasts", [])
    reference_tests = reference.get("contrasts", [])
    planned = int(prompt.get("planned_contrast_count", -1)) + int(reference.get("planned_contrast_count", -1))
    if len(prompt_tests) != 8 or len(reference_tests) != 12 or planned != 20:
        raise RuntimeError(
            "Expected 8 prompt/order/ID and 12 reference-count contrasts; "
            f"found {len(prompt_tests)} and {len(reference_tests)} (planned={planned})."
        )
    tests = []
    for family, rows in (("prompt_order_id", prompt_tests), ("reference_count", reference_tests)):
        for row in rows:
            p_value = row.get("two_sided_template_sign_flip_p")
            if not isinstance(p_value, (int, float)) or not 0.0 <= p_value <= 1.0:
                raise RuntimeError(f"Invalid p-value in {family} contrast: {row}")
            tests.append({"family": family, **row})
    adjusted = holm([float(row["two_sided_template_sign_flip_p"]) for row in tests])
    for row, p_adj in zip(tests, adjusted):
        row["holm_adjusted_p_across_20_robustness_contrasts"] = p_adj
    result = {
        "planned_contrast_count": 20,
        "completed_contrast_count": len(tests),
        "families": {"prompt_order_id": 8, "reference_count": 12},
        "multiplicity_rule": "Holm step-down family-wise correction across the 8 prompt/order/ID and 12 nested reference-count contrasts together.",
        "tests": tests,
        "sha256": {
            "prompt_sensitivity_summary": sha256(prompt_path),
            "reference_size_summary": sha256(reference_path),
            "analysis_script": sha256(Path(__file__).resolve()),
        },
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "completed_contrasts": len(tests)}, indent=2))


if __name__ == "__main__":
    main()
