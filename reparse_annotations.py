#!/usr/bin/env python3
"""Re-parse preserved raw answers after a documented parser normalization."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from annotate import append_jsonl, jsonl_read, parse_annotation


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--annotator-id", required=True)
    args = parser.parse_args()

    name = Path(args.annotation_file)
    if name.name != args.annotation_file or not name.name.startswith("annotations"):
        parser.error("--annotation-file must be a JSONL filename in runs/")

    annotation_path = ROOT / "runs" / name
    payloads = {
        row["case_id"]: row
        for row in jsonl_read(ROOT / "runs" / "input_payloads.jsonl")
    }
    records = jsonl_read(annotation_path)
    revised_count = 0
    for row in records:
        if (
            row.get("event") != "annotation"
            or row.get("annotator_id") != args.annotator_id
            or row.get("parser_revision")
            or str(row.get("parse_or_inference_error", "")).startswith(
                "inference failed:"
            )
        ):
            continue
        payload = payloads.get(row.get("case_id"))
        if (
            payload is None
            or row.get("prompt_sha256") != payload.get("prompt_sha256")
        ):
            continue
        allowed = set(
            json.loads(payload["user_prompt"])["target_row"]["features"].keys()
        )
        parsed, error = parse_annotation(row.get("raw_output", ""), allowed)
        if parsed == row.get("parsed_annotation") and error == row.get(
            "parse_or_inference_error"
        ):
            continue
        revised = dict(row)
        revised["parsed_annotation"] = parsed
        revised["parse_or_inference_error"] = error
        revised["parser_revision"] = "unicode_tokenizer_whitespace_v1"
        revised["parser_reprocessed_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds")
        append_jsonl(annotation_path, revised)
        revised_count += 1

    print(f"Appended {revised_count} re-parsed records to {annotation_path.name}.")


if __name__ == "__main__":
    main()
