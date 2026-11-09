#!/usr/bin/env python3
"""Summarize annotator disagreement and leave-one-model-out stability."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
PAYLOAD_PATH = RUNS / "input_payloads.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def strict_majority(sets: list[set[str]]) -> set[str]:
    if not sets:
        return set()
    votes = Counter(feature for selected in sets for feature in selected)
    threshold = len(sets) // 2 + 1
    return {feature for feature, count in votes.items() if count >= threshold}


def set_scores(predicted: set[str], reference: set[str]) -> dict:
    intersection = len(predicted & reference)
    precision = intersection / len(predicted) if predicted else 0.0
    recall = intersection / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = predicted | reference
    jaccard = intersection / len(union) if union else 1.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "jaccard": round(jaccard, 6),
    }


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def mean_binary_entropy(sets: list[set[str]], universe: set[str]) -> float | None:
    if not sets or not universe:
        return None
    total = len(sets)
    values = []
    for feature in universe:
        p = sum(feature in selected for selected in sets) / total
        if p in (0.0, 1.0):
            values.append(0.0)
        else:
            values.append(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))
    return sum(values) / len(values)


def main() -> None:
    annotation_files = sorted(RUNS.glob("annotations*.jsonl"))
    annotations = [
        row
        for path in annotation_files
        for row in read_jsonl(path)
    ]
    payloads = read_jsonl(PAYLOAD_PATH)
    if not payloads:
        raise FileNotFoundError("Run prepare_data.py before leave-one-model-out.")
    payload_by_case = {row["case_id"]: row for row in payloads}
    # Evaluate only current-prompt records; the last matching record is
    # authoritative after a retry.
    latest: dict[tuple[str, str], dict] = {}
    model_names: dict[str, str] = {}
    model_families: dict[str, str] = {}
    stale_prompt_records_excluded = 0
    for row in annotations:
        if row.get("event") != "annotation":
            continue
        current_payload = payload_by_case.get(row.get("case_id"))
        if (
            current_payload is None
            or row.get("prompt_sha256") != current_payload.get("prompt_sha256")
        ):
            stale_prompt_records_excluded += 1
            continue
        key = (row.get("annotator_id"), row.get("case_id"))
        latest[key] = row
        model_names[row.get("annotator_id")] = row.get("model_id", row.get("annotator_id"))
        model_families[row.get("annotator_id")] = row.get("model_family", "unknown")

    annotators = sorted(model_names)
    if not annotators:
        raise RuntimeError("No annotation records found.")
    by_case: dict[str, list[dict]] = defaultdict(list)
    for (annotator_id, case_id), row in latest.items():
        by_case[case_id].append(row)

    disagreement_rows = []
    pair_stats: dict[tuple[str, str], list[float]] = defaultdict(list)
    lomo_rows = []
    reference_rows = []
    for case_id, payload in payload_by_case.items():
        records = {row["annotator_id"]: row for row in by_case.get(case_id, [])}
        parsed_records = {}
        invalid_count = 0
        abstain_count = 0
        for annotator_id, record in records.items():
            parsed = record.get("parsed_annotation")
            if record.get("parse_or_inference_error") or not isinstance(parsed, dict):
                invalid_count += 1
                continue
            if parsed.get("abstain") is True:
                abstain_count += 1
                continue
            features = parsed.get("feature_ids_ranked")
            if not isinstance(features, list):
                invalid_count += 1
                continue
            parsed_records[annotator_id] = {
                "features": set(features),
                "ranked": features,
                "annotation": parsed,
            }
        model_sets = {
            model: item["features"] for model, item in parsed_records.items()
        }
        models = sorted(model_sets)
        sets = [model_sets[model] for model in models]
        universe = set(
            json.loads(payload["user_prompt"])["target_row"]["features"].keys()
        )
        vote_counts = Counter(feature for selected in sets for feature in selected)
        # A single valid vote is not inter-model consensus.
        consensus = strict_majority(sets) if len(sets) >= 2 else set()
        pairwise = []
        for i, model_a in enumerate(models):
            for model_b in models[i + 1:]:
                score = jaccard(model_sets[model_a], model_sets[model_b])
                pair_stats[(model_a, model_b)].append(score)
                pairwise.append(score)
        entropy = mean_binary_entropy(sets, universe)
        reference_rows.append({
            "reference_label": "LLM-generated reference annotation",
            "case_id": case_id,
            "dataset_key": payload["dataset_key"],
            "annotators_with_valid_nonabstain_output": models,
            "feature_vote_counts": dict(sorted(vote_counts.items())),
            "strict_majority_features": sorted(consensus),
            "strict_majority_threshold": (
                len(sets) // 2 + 1 if len(sets) >= 2 else None
            ),
            "has_majority_reference": len(sets) >= 2 and bool(consensus),
            "no_peer_ground_truth": True,
        })
        disagreement_rows.append({
            "case_id": case_id,
            "dataset_key": payload["dataset_key"],
            "models_expected": len(annotators),
            "models_with_records": len(records),
            "models_valid_nonabstain": len(models),
            "n_parse_or_inference_failures": invalid_count,
            "n_abstentions": abstain_count,
            "selected_set_sizes": {
                model: len(model_sets[model]) for model in models
            },
            "strict_majority_features": sorted(consensus),
            "strict_majority_feature_count": len(consensus),
            "has_majority_reference": len(sets) >= 2 and bool(consensus),
            "strict_majority_threshold": (
                len(sets) // 2 + 1 if len(sets) >= 2 else None
            ),
            "feature_inclusion_vote_counts": dict(sorted(vote_counts.items())),
            "mean_pairwise_jaccard": (
                round(sum(pairwise) / len(pairwise), 6) if pairwise else None
            ),
            "exact_pairwise_set_match_rate": (
                round(
                    sum(
                        model_sets[a] == model_sets[b]
                        for i, a in enumerate(models)
                        for b in models[i + 1:]
                    ) / len(pairwise),
                    6,
                )
                if pairwise else None
            ),
            "mean_binary_feature_entropy_bits": (
                round(entropy, 6) if entropy is not None else None
            ),
            "no_ground_truth_accuracy_claim": True,
        })

        for held_out in annotators:
            record = records.get(held_out)
            if not record or record.get("parse_or_inference_error"):
                lomo_rows.append({
                    "case_id": case_id,
                    "dataset_key": payload["dataset_key"],
                    "held_out_annotator_id": held_out,
                    "held_out_model_id": model_names.get(held_out, held_out),
                    "status": "held-out output missing or invalid",
                    "peer_annotators": [],
                    "peer_consensus_features": [],
                })
                continue
            parsed = record.get("parsed_annotation")
            if not isinstance(parsed, dict):
                lomo_rows.append({
                    "case_id": case_id,
                    "dataset_key": payload["dataset_key"],
                    "held_out_annotator_id": held_out,
                    "held_out_model_id": model_names.get(held_out, held_out),
                    "status": "held-out output missing or invalid",
                    "peer_annotators": [],
                    "peer_consensus_features": [],
                })
                continue
            if parsed.get("abstain") is True:
                lomo_rows.append({
                    "case_id": case_id,
                    "dataset_key": payload["dataset_key"],
                    "held_out_annotator_id": held_out,
                    "held_out_model_id": model_names.get(held_out, held_out),
                    "status": "held-out annotator abstained",
                    "peer_annotators": [],
                    "peer_consensus_features": [],
                })
                continue
            held_set = set(parsed.get("feature_ids_ranked", []))
            peers = {
                model: model_sets[model]
                for model in models
                if model != held_out
            }
            peer_sets = list(peers.values())
            peer_consensus = strict_majority(peer_sets)
            peer_union = set().union(*peer_sets) if peer_sets else set()
            if len(peer_sets) < 2:
                status = "insufficient valid peer outputs"
                scores = {}
            elif not peer_consensus:
                status = "no nonempty strict-majority peer reference"
                scores = {}
            else:
                status = "scored against peer strict-majority reference"
                scores = set_scores(held_set, peer_consensus)
            lomo_rows.append({
                "case_id": case_id,
                "dataset_key": payload["dataset_key"],
                "held_out_annotator_id": held_out,
                "held_out_model_id": model_names.get(held_out, held_out),
                "held_out_features": parsed.get("feature_ids_ranked", []),
                "peer_annotators": sorted(peers),
                "peer_consensus_features": sorted(peer_consensus),
                "peer_union_features": sorted(peer_union),
                "status": status,
                "peer_consensus_coverage": bool(peer_consensus),
                **scores,
                "jaccard_to_peer_union": (
                    round(jaccard(held_set, peer_union), 6) if peer_sets else None
                ),
                "reference_type": "weak annotator reference; no ground truth",
            })

    write_csv(REPORTS / "disagreement_by_case.csv", disagreement_rows)
    write_csv(REPORTS / "pairwise_model_agreement.csv", [
        {
            "annotator_a": a,
            "model_a": model_names.get(a, a),
            "family_a": model_families.get(a, "unknown"),
            "annotator_b": b,
            "model_b": model_names.get(b, b),
            "family_b": model_families.get(b, "unknown"),
            "n_cases": len(values),
            "mean_jaccard": round(sum(values) / len(values), 6) if values else None,
            "exact_set_match_rate": round(
                sum(
                    set(latest[(a, case)]["parsed_annotation"]["feature_ids_ranked"])
                    == set(latest[(b, case)]["parsed_annotation"]["feature_ids_ranked"])
                    for case in payload_by_case
                    if (a, case) in latest and (b, case) in latest
                    and latest[(a, case)].get("parsed_annotation")
                    and latest[(b, case)].get("parsed_annotation")
                    and not latest[(a, case)].get("parse_or_inference_error")
                    and not latest[(b, case)].get("parse_or_inference_error")
                    and latest[(a, case)]["parsed_annotation"].get("abstain") is False
                    and latest[(b, case)]["parsed_annotation"].get("abstain") is False
                ) / len(values),
                6,
            ) if values else None,
        }
        for (a, b), values in sorted(pair_stats.items())
    ])
    write_csv(REPORTS / "leave_one_model_out_by_case.csv", lomo_rows)
    with (RUNS / "reference_annotations.jsonl").open("w", encoding="utf-8") as stream:
        for item in reference_rows:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary_rows = []
    for annotator_id in annotators:
        relevant = [
            row for row in lomo_rows
            if row["held_out_annotator_id"] == annotator_id
        ]
        scored = [
            row for row in relevant
            if row["status"] == "scored against peer strict-majority reference"
        ]
        valid = sum(
            1 for row in relevant
            if row["status"] != "held-out output missing or invalid"
            and row["status"] != "held-out annotator abstained"
        )
        summary_rows.append({
            "held_out_annotator_id": annotator_id,
            "model_id": model_names.get(annotator_id, annotator_id),
            "model_family": model_families.get(annotator_id, "unknown"),
            "n_cases": len(relevant),
            "n_valid_held_out_outputs": valid,
            "n_cases_with_nonempty_peer_consensus": len(scored),
            "peer_reference_coverage": round(len(scored) / valid, 6) if valid else None,
            "mean_precision_vs_peer_consensus": (
                round(sum(row["precision"] for row in scored) / len(scored), 6)
                if scored else None
            ),
            "mean_recall_vs_peer_consensus": (
                round(sum(row["recall"] for row in scored) / len(scored), 6)
                if scored else None
            ),
            "mean_f1_vs_peer_consensus": (
                round(sum(row["f1"] for row in scored) / len(scored), 6)
                if scored else None
            ),
            "mean_jaccard_vs_peer_consensus": (
                round(sum(row["jaccard"] for row in scored) / len(scored), 6)
                if scored else None
            ),
            "interpretation": "agreement with peer weak reference; not accuracy",
        })
    write_csv(REPORTS / "leave_one_model_out_summary.csv", summary_rows)

    by_dataset = defaultdict(list)
    for row in lomo_rows:
        by_dataset[(row["dataset_key"], row["held_out_annotator_id"])].append(row)
    dataset_rows = []
    for (dataset_key, annotator_id), relevant in sorted(by_dataset.items()):
        scored = [
            row for row in relevant
            if row["status"] == "scored against peer strict-majority reference"
        ]
        valid = sum(
            row["status"] not in {
                "held-out output missing or invalid", "held-out annotator abstained"
            }
            for row in relevant
        )
        dataset_rows.append({
            "dataset_key": dataset_key,
            "held_out_annotator_id": annotator_id,
            "model_id": model_names.get(annotator_id, annotator_id),
            "n_cases": len(relevant),
            "n_valid_held_out_outputs": int(valid),
            "n_peer_reference_scored": len(scored),
            "peer_reference_coverage": round(len(scored) / valid, 6) if valid else None,
            "mean_jaccard_vs_peer_consensus": (
                round(sum(row["jaccard"] for row in scored) / len(scored), 6)
                if scored else None
            ),
            "mean_f1_vs_peer_consensus": (
                round(sum(row["f1"] for row in scored) / len(scored), 6)
                if scored else None
            ),
        })
    write_csv(REPORTS / "leave_one_model_out_by_dataset.csv", dataset_rows)

    quality_rows = []
    failure_rows = []
    for annotator_id in annotators:
        model_records = [
            row for (model, _case), row in latest.items()
            if model == annotator_id
        ]
        valid_records = [
            row for row in model_records
            if not row.get("parse_or_inference_error")
            and isinstance(row.get("parsed_annotation"), dict)
        ]
        normalized_records = [
            row for row in valid_records
            if row["parsed_annotation"].get("parse_normalization_notes")
        ]
        abstentions = sum(
            row["parsed_annotation"].get("abstain") is True
            for row in valid_records
        )
        quality_rows.append({
            "annotator_id": annotator_id,
            "model_id": model_names.get(annotator_id, annotator_id),
            "model_family": model_families.get(annotator_id, "unknown"),
            "n_cases": len(payload_by_case),
            "n_latest_outputs": len(model_records),
            "n_schema_valid_outputs": len(valid_records),
            "n_parse_or_inference_failures": len(model_records) - len(valid_records),
            "n_outputs_with_explicit_normalization": len(normalized_records),
            "n_abstentions": int(abstentions),
            "n_valid_nonabstaining_outputs": len(valid_records) - int(abstentions),
            "schema_valid_rate": round(len(valid_records) / len(payload_by_case), 6)
            if payload_by_case else None,
        })
        for row in model_records:
            if row.get("parse_or_inference_error") or not isinstance(
                row.get("parsed_annotation"), dict
            ):
                failure_rows.append({
                    "annotator_id": annotator_id,
                    "model_id": model_names.get(annotator_id, annotator_id),
                    "case_id": row.get("case_id"),
                    "dataset_key": row.get("dataset_key"),
                    "prompt_sha256": row.get("prompt_sha256"),
                    "parse_or_inference_error": row.get("parse_or_inference_error"),
                    "raw_output": row.get("raw_output", ""),
                })
    write_csv(REPORTS / "annotator_quality.csv", quality_rows)
    write_csv(REPORTS / "parse_failures.csv", failure_rows)

    success = sum(
        1 for row in latest.values()
        if not row.get("parse_or_inference_error")
        and isinstance(row.get("parsed_annotation"), dict)
    )
    summary = {
        "protocol_id": "adbench_uci_llm_reference_v1",
        "annotation_records_latest_per_model_case": len(latest),
        "stale_prompt_records_excluded": stale_prompt_records_excluded,
        "parse_valid_records": success,
        "parse_or_inference_failures": len(latest) - success,
        "outputs_with_explicit_parser_normalization": sum(
            bool(row.get("parsed_annotation", {}).get("parse_normalization_notes"))
            for row in latest.values()
            if isinstance(row.get("parsed_annotation"), dict)
            and not row.get("parse_or_inference_error")
        ),
        "annotator_quality": quality_rows,
        "annotators": [
            {
                "annotator_id": model,
                "model_id": model_names.get(model, model),
                "family": model_families.get(model, "unknown"),
            }
            for model in annotators
        ],
        "cases": len(payload_by_case),
        "ground_truth_available_for_feature_subsets": False,
        "metrics_interpretation": "inter-annotator agreement and leave-one-model-out stability only",
    }
    (REPORTS / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
