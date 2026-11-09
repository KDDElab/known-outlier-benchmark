#!/usr/bin/env python3
"""Flatten annotate_v2 JSONL records into the evaluator's prediction format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--protocol",
        default="known_anomaly_subspace_benchmark_v2_compact_json_v1",
        help="retain only annotation rows from this frozen prompt/generation protocol",
    )
    args = parser.parse_args()

    latest: dict[str, dict] = {}
    with args.annotations.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("event") != "annotation"
                or not row.get("case_id")
                or row.get("protocol_id") != args.protocol
            ):
                continue
            latest[row["case_id"]] = row

    predictions = []
    for case_id, row in sorted(latest.items()):
        prediction = row.get("parsed_prediction")
        # Preserve failed generations as present but structurally invalid records;
        # the benchmark scorer counts them in the denominator.
        if not isinstance(prediction, dict):
            prediction = {"case_id": case_id}
        else:
            prediction = {"case_id": case_id, **prediction}
        predictions.append(prediction)

    output = args.output or args.annotations.with_name(
        args.annotations.stem.replace("annotations_", "predictions_") + ".jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "n_latest_annotation_records": len(latest),
        "n_structured_predictions": sum(len(row) > 1 for row in predictions),
        "n_failed_or_unparsed": sum(len(row) == 1 for row in predictions),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
