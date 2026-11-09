#!/usr/bin/env python3
"""Create a test-only input with feature and normal-row order both reversed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "inputs.jsonl"
OUTPUT = HERE / "runs" / "order_perturbation_inputs.jsonl"
PROTOCOL = "known_anomaly_subspace_benchmark_v2_reverse_feature_and_reference_order_v1"


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    perturbed = []
    for row in rows:
        if row["split"] != "test":
            continue
        prompt = row["user_prompt"]
        marker = "\n{\n  \"feature_schema\""
        offset = prompt.find(marker)
        if offset < 0:
            raise ValueError(f"Could not locate JSON payload in {row['case_id']}")
        prefix, payload_text = prompt[: offset + 1], prompt[offset + 1 :]
        payload = json.loads(payload_text)
        payload["feature_schema"] = list(reversed(payload["feature_schema"]))
        payload["target_row"] = dict(reversed(list(payload["target_row"].items())))
        payload["normal_reference_rows"] = [
            dict(reversed(list(ref.items())))
            for ref in reversed(payload["normal_reference_rows"])
        ]
        row["user_prompt"] = prefix + json.dumps(payload, ensure_ascii=False, indent=2)
        row["prompt_sha256"] = hashlib.sha256(
            (row["system_prompt"] + "\0" + row["user_prompt"]).encode("utf-8")
        ).hexdigest()
        row["protocol_id"] = PROTOCOL
        perturbed.append(row)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in perturbed),
        encoding="utf-8",
    )
    print(f"Wrote {len(perturbed)} test prompts to {OUTPUT}")


if __name__ == "__main__":
    main()
