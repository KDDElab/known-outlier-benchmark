#!/usr/bin/env python3
"""Prepare a leakage-controlled, output-specific zero-noise circuit pilot."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import random
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = HERE.parent / "data/circuit/harvard_dataset_10.7910_DVN_W4FPPN.zip"
SEED = 20261105
N_TARGETS_PER_PAIR = 4
N_REFERENCES = 16
IN_RE = re.compile(r"i\d+\Z")
OUT_RE = re.compile(r"o\d+\Z")

SYSTEM_PROMPT = """You are given a table, one target row already known to be anomalous, a focus output known to be affected, and clean reference rows from the same circuit. Identify compact subsets of input features that best localize the anomalous behavior of the focus output. Only input features are eligible for explanations; output fields are observed context, not candidates. Use the supplied rows as evidence and do not infer or name a fault location. Return JSON only with this schema: {\"explanations\":[{\"features\":[\"i1\"],\"evidence\":\"short evidence from the rows\",\"confidence\":0.5}],\"abstain\":false}. Return at most four ranked candidates. Feature names must be copied exactly from the eligible input fields; do not include output fields. Keep each evidence string to at most 20 words. If evidence is insufficient, return an empty explanations list and set abstain true."""


def read_rows(zf: zipfile.ZipFile, path: str, delimiter: str) -> list[dict[str, str]]:
    with zf.open(path) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        return list(csv.DictReader(text, delimiter=delimiter))


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def prompt_for(input_columns, output_columns, target, focus, references) -> str:
    schema = [f"{x}: binary circuit input, eligible candidate" for x in input_columns]
    schema += [f"{x}: observed binary circuit output, not eligible as a candidate" for x in output_columns]
    row_columns = input_columns + output_columns
    render = lambda row: "\t".join(row[k] for k in row_columns)
    return (
        "Table schema:\n" + "\n".join(schema)
        + f"\n\nKnown anomalous target row (column order: {', '.join(row_columns)}):\n"
        + render(target)
        + f"\n\nFocus output known to be affected: {focus}"
        + f"\n\nClean normal reference rows (same column order; {len(references)} rows):\n"
        + "\n".join(render(row) for row in references)
        + "\n\nReturn the required JSON object."
    )


def build(archive: Path) -> tuple[list[dict], list[dict], dict]:
    inputs, labels, pair_report = [], [], []
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        modified_paths = sorted(
            n for n in names
            if n.startswith("Truth_Tables/") and n.endswith("_0noise_in_labeled.tab")
        )
        for pair_no, modified_path in enumerate(modified_paths, 1):
            circuit = modified_path.split("/")[1]
            filename = Path(modified_path).name
            stem = filename.removesuffix("_0noise_in_labeled.tab")
            original_path = f"Truth_Tables/{circuit}/{circuit}_0noise_out.tab"
            gold_path = f"Ground_Truth/{circuit}/{stem}_ground_truth_0noise.csv"
            modified = read_rows(zf, modified_path, "\t")
            original = read_rows(zf, original_path, "\t")
            gold_rows = read_rows(zf, gold_path, ",")
            input_columns = [k for k in modified[0] if IN_RE.fullmatch(k)]
            output_columns = [k for k in modified[0] if OUT_RE.fullmatch(k)]
            key = lambda row: tuple(row[k] for k in input_columns)
            original_by_input = {key(row): row for row in original}
            gold_by_input = {key(row): row for row in gold_rows}
            if len(original_by_input) != len(original) or len(gold_by_input) != len(gold_rows):
                raise ValueError(f"duplicate input keys in {modified_path} or its references")

            anomaly_candidates = []
            for row in modified:
                k = key(row)
                changed_outputs = [o for o in output_columns if row[o] != original_by_input[k][o]]
                if row["label"] == "1":
                    if not changed_outputs or k not in gold_by_input:
                        raise ValueError(f"unmatched anomaly row in {modified_path}: {k}")
                    anomaly_candidates.append((k, row, changed_outputs))
                elif changed_outputs:
                    raise ValueError(f"unlabeled output change in {modified_path}: {k}")

            rng = random.Random(stable_seed(modified_path + str(SEED)))
            sampled = sorted(rng.sample(anomaly_candidates, min(N_TARGETS_PER_PAIR, len(anomaly_candidates))), key=lambda x: x[0])
            case_count = 0
            for sample_no, (k, target, changed_outputs) in enumerate(sampled, 1):
                for focus in changed_outputs:
                    case_id = f"CIR_{pair_no:02d}_{sample_no:02d}_{focus}"
                    ref_pool = [r for r in original if key(r) != k]
                    ref_rng = random.Random(stable_seed(case_id + str(SEED)))
                    refs = sorted(ref_rng.sample(ref_pool, min(N_REFERENCES, len(ref_pool))), key=key)
                    explanation_map = ast.literal_eval(gold_by_input[k]["explanation for output"])
                    if focus not in explanation_map:
                        raise ValueError(f"missing output-specific gold for {case_id}")
                    gold_features = explanation_map[focus]
                    if not isinstance(gold_features, list) or any(f not in input_columns for f in gold_features):
                        raise ValueError(f"invalid input-feature gold for {case_id}")
                    user_prompt = prompt_for(input_columns, output_columns, target, focus, refs)
                    prompt_hash = hashlib.sha256((SYSTEM_PROMPT + "\n" + user_prompt).encode()).hexdigest()
                    inputs.append({
                        "case_id": case_id,
                        "system_prompt": SYSTEM_PROMPT,
                        "user_prompt": user_prompt,
                        "eligible_feature_ids": input_columns,
                        # Structured copies support transparent non-LLM baselines;
                        # the rendered model prompt and its hash are unchanged.
                        "target_row": {k: int(target[k]) for k in input_columns + output_columns},
                        "normal_reference_rows": [
                            {k: int(row[k]) for k in input_columns + output_columns} for row in refs
                        ],
                        "focus_output": focus,
                        "prompt_sha256": prompt_hash,
                    })
                    labels.append({
                        "case_id": case_id,
                        "source_table": modified_path,
                        "circuit": circuit,
                        "target_input": dict(zip(input_columns, k)),
                        "focus_output": focus,
                        "gold_features": gold_features,
                        "n_references": len(refs),
                    })
                    case_count += 1
            pair_report.append({
                "pair_id": pair_no,
                "circuit": circuit,
                "modified_table": modified_path,
                "ground_truth_file": gold_path,
                "labeled_anomalous_rows": len(anomaly_candidates),
                "sampled_target_rows": len(sampled),
                "output_specific_queries": case_count,
            })

    manifest = {
        "track": "Harvard circuit zero-noise output-specific pilot",
        "source_archive": str(archive),
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "selection_seed": SEED,
        "target_rows_per_table_pair": N_TARGETS_PER_PAIR,
        "reference_rows_per_query": N_REFERENCES,
        "zero_noise_pairs": len(pair_report),
        "sampled_target_rows": sum(p["sampled_target_rows"] for p in pair_report),
        "output_specific_queries": len(inputs),
        "input_contains_labels_or_gate_metadata": False,
        "outputs_in_context_but_not_candidates": True,
        "candidate_features": "input columns only",
        "score_unit": "target-row/focus-output query",
        "pairs": pair_report,
    }
    return inputs, labels, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    inputs, labels, manifest = build(args.archive)
    (HERE / "inputs_zero_noise_pilot.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in inputs), encoding="utf-8"
    )
    (HERE / "ground_truth_zero_noise_pilot.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in labels), encoding="utf-8"
    )
    (HERE / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in (
        "zero_noise_pairs", "sampled_target_rows", "output_specific_queries", "input_contains_labels_or_gate_metadata"
    )}, indent=2))


if __name__ == "__main__":
    main()
