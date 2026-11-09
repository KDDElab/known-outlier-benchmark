#!/usr/bin/env python3
"""Audit zero-noise Harvard circuit rows against originals and explanations.

The archive is read in place. This script never sends labels or ground-truth
files to an evaluated model; it only checks row/output alignment.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = HERE.parent / "data/circuit/harvard_dataset_10.7910_DVN_W4FPPN.zip"
INPUT_RE = re.compile(r"i\d+\Z")
OUTPUT_RE = re.compile(r"o\d+\Z")


def read_rows(zf: zipfile.ZipFile, path: str, delimiter: str) -> list[dict[str, str]]:
    with zf.open(path) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        return list(csv.DictReader(text, delimiter=delimiter))


def row_map(rows: list[dict[str, str]], keys: list[str], file_name: str):
    mapped = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in mapped:
            raise ValueError(f"duplicate input row in {file_name}: {key}")
        mapped[key] = row
    return mapped


def audit(archive: Path) -> dict:
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    pairs = []
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        modified_paths = sorted(
            n for n in names
            if n.startswith("Truth_Tables/") and n.endswith("_0noise_in_labeled.tab")
        )
        for modified_path in modified_paths:
            circuit = modified_path.split("/")[1]
            filename = Path(modified_path).name
            stem = filename.removesuffix("_0noise_in_labeled.tab")
            original_path = f"Truth_Tables/{circuit}/{circuit}_0noise_out.tab"
            ground_truth_path = f"Ground_Truth/{circuit}/{stem}_ground_truth_0noise.csv"
            if original_path not in names or ground_truth_path not in names:
                raise ValueError(f"unpaired zero-noise table: {modified_path}")

            modified = read_rows(zf, modified_path, "\t")
            original = read_rows(zf, original_path, "\t")
            ground_truth = read_rows(zf, ground_truth_path, ",")
            input_columns = [k for k in modified[0] if INPUT_RE.fullmatch(k)]
            output_columns = [k for k in modified[0] if OUTPUT_RE.fullmatch(k)]
            if not input_columns or not output_columns or "label" not in modified[0]:
                raise ValueError(f"unexpected schema in {modified_path}")
            if any(k not in original[0] for k in input_columns + output_columns):
                raise ValueError(f"modified/original schema mismatch: {modified_path}")
            if any(k not in ground_truth[0] for k in input_columns + output_columns):
                raise ValueError(f"modified/ground-truth schema mismatch: {modified_path}")

            modified_by_input = row_map(modified, input_columns, modified_path)
            original_by_input = row_map(original, input_columns, original_path)
            gold_by_input = row_map(ground_truth, input_columns, ground_truth_path)
            if modified_by_input.keys() != original_by_input.keys():
                raise ValueError(f"modified/original input combinations differ: {modified_path}")

            positive_inputs = set()
            changed_output_counts = {name: 0 for name in output_columns}
            anomalous_rows = 0
            for key, modified_row in modified_by_input.items():
                original_row = original_by_input[key]
                changed = {
                    output for output in output_columns
                    if modified_row[output] != original_row[output]
                }
                for output in changed:
                    changed_output_counts[output] += 1
                if modified_row["label"] == "1":
                    anomalous_rows += 1
                    positive_inputs.add(key)
                    if not changed:
                        raise ValueError(f"positive label without output delta: {modified_path} {key}")
                elif changed:
                    raise ValueError(f"output delta without positive label: {modified_path} {key}")

            if positive_inputs != set(gold_by_input):
                raise ValueError(
                    f"label/ground-truth row mismatch in {modified_path}: "
                    f"labels={len(positive_inputs)}, ground_truth={len(gold_by_input)}"
                )

            output_alignment_errors = 0
            non_input_features = 0
            for key, gold_row in gold_by_input.items():
                modified_row = modified_by_input[key]
                original_row = original_by_input[key]
                changed = {
                    output for output in output_columns
                    if modified_row[output] != original_row[output]
                }
                explanation = ast.literal_eval(gold_row["explanation for output"])
                if not isinstance(explanation, dict) or set(explanation) != changed:
                    output_alignment_errors += 1
                    continue
                if any(
                    not isinstance(features, list)
                    or any(feature not in input_columns for feature in features)
                    for features in explanation.values()
                ):
                    non_input_features += 1
            if output_alignment_errors or non_input_features:
                raise ValueError(
                    f"ground-truth mismatch in {modified_path}: "
                    f"outputs={output_alignment_errors}, noninputs={non_input_features}"
                )

            pairs.append({
                "circuit": circuit,
                "modified_table": modified_path,
                "original_table": original_path,
                "ground_truth": ground_truth_path,
                "n_table_rows": len(modified),
                "n_input_features": len(input_columns),
                "n_output_features": len(output_columns),
                "n_anomalous_rows": anomalous_rows,
                "n_ground_truth_rows": len(ground_truth),
                "changed_output_row_counts": changed_output_counts,
                "output_specific_ground_truth_aligned": True,
            })

    per_circuit = {}
    for circuit in sorted({p["circuit"] for p in pairs}):
        items = [p for p in pairs if p["circuit"] == circuit]
        per_circuit[circuit] = {
            "zero_noise_pairs": len(items),
            "labeled_anomalous_rows": sum(p["n_anomalous_rows"] for p in items),
        }
    return {
        "archive": str(archive),
        "archive_sha256": archive_sha,
        "condition": "zero noise",
        "pair_count": len(pairs),
        "total_labeled_anomalous_rows": sum(p["n_anomalous_rows"] for p in pairs),
        "per_circuit": per_circuit,
        "alignment_errors": 0,
        "notes": [
            "Keys are complete input-feature vectors.",
            "An anomalous row is identified by the source label and verified by at least one output differing from the matching original row.",
            "The set of changed outputs is checked against the output keys in the source explanation dictionary.",
            "This is a data-alignment audit, not an LLM evaluation.",
        ],
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=HERE / "zero_noise_alignment_audit.json")
    args = parser.parse_args()
    report = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "condition", "pair_count", "total_labeled_anomalous_rows", "per_circuit", "alignment_errors"
    )}, indent=2))


if __name__ == "__main__":
    main()
