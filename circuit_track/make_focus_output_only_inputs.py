#!/usr/bin/env python3
"""Create a matched circuit prompt condition exposing only the known focus output."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from prepare_zero_noise_pilot import prompt_for

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "inputs_zero_noise_pilot.jsonl"
OUTDIR = HERE / "matched_information"
OUT = OUTDIR / "inputs_focus_output_only.jsonl"

def main() -> None:
    cases = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    outputs = []
    for case in cases:
        input_ids = list(case["eligible_feature_ids"])
        output_ids = [key for key in case["target_row"] if key not in set(input_ids)]
        focus = case["focus_output"]
        target = {key: str(case["target_row"][key]) for key in (*input_ids, focus)}
        refs = [{key: str(row[key]) for key in (*input_ids, focus)} for row in case["normal_reference_rows"]]
        user = prompt_for(input_ids, [focus], target, focus, refs)
        row = {
            "case_id": case["case_id"],
            "system_prompt": case["system_prompt"],
            "user_prompt": user,
            "eligible_feature_ids": input_ids,
            "target_row": target,
            "normal_reference_rows": refs,
            "focus_output": focus,
            "prompt_sha256": hashlib.sha256((case["system_prompt"] + "\n" + user).encode()).hexdigest(),
            "condition": "focus_output_only",
        }
        if (set(output_ids) - {focus}) & set(row["target_row"]):
            raise AssertionError("A non-focus output leaked into the reduced target")
        outputs.append(row)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs), encoding="utf-8")
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    manifest = {
        "condition": "focus_output_only",
        "n_queries": len(outputs),
        "input_fields": "all eligible binary input columns",
        "output_fields": "the known focus output only",
        "system_prompt": "identical to all-output condition; only user payload visibility changes",
        "ground_truth_used_in_prompt": False,
        "source_inputs_sha256": source_hash,
        "focus_only_inputs_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "case_ids_match_all_output_condition": [row["case_id"] for row in outputs] == [row["case_id"] for row in cases],
    }
    (OUTDIR / "focus_output_only_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
