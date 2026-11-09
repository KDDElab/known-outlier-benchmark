#!/usr/bin/env python3
"""Score one JSONL file of v2 model predictions against independent labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def feature_metrics(pairs: list[tuple[set[str], set[str]]]) -> dict:
    tp = sum(len(pred & gold) for pred, gold in pairs)
    fp = sum(len(pred - gold) for pred, gold in pairs)
    fn = sum(len(gold - pred) for pred, gold in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    case_f1 = []
    for pred, gold in pairs:
        p = len(pred & gold) / len(pred) if pred else 0.0
        r = len(pred & gold) / len(gold) if gold else 0.0
        case_f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "macro_case_f1": mean(case_f1),
        "exact_set_accuracy": mean([float(pred == gold) for pred, gold in pairs]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="JSONL with one prediction per case")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--inputs-path", type=Path, default=HERE / "inputs.jsonl")
    parser.add_argument("--ground-truth-path", type=Path, default=HERE / "ground_truth.jsonl")
    parser.add_argument("--summary-only", action="store_true", help="omit the per-case CSV")
    args = parser.parse_args()

    outdir = args.outdir or args.predictions.parent / "evaluation_v2"
    outdir.mkdir(parents=True, exist_ok=True)
    inputs = {row["case_id"]: row for row in read_jsonl(args.inputs_path)}
    gold = {row["case_id"]: row for row in read_jsonl(args.ground_truth_path)}
    pred_rows = read_jsonl(args.predictions)
    predictions = {row["case_id"]: row for row in pred_rows if "case_id" in row}

    per_case = []
    explanation_metrics = {split: [] for split in ("all", "train", "dev", "test")}
    for case_id, case_input in inputs.items():
        truth = gold[case_id]
        pred = predictions.get(case_id)
        known_ids = set(case_input["feature_ids"])
        structurally_valid = False
        if isinstance(pred, dict):
            required = {"case_id", "task_label", "mechanism_feature_ids", "explanations", "abstain"}
            mechanism = pred.get("mechanism_feature_ids")
            explanations = pred.get("explanations")
            structurally_valid = (
                required.issubset(pred)
                and isinstance(pred.get("task_label"), str)
                and pred.get("task_label") in {
                    "numeric_marginal", "categorical_rare", "pairwise_interaction", "conditional_group",
                    "multiple_valid", "modified_vs_mechanism", "relational", "higher_order", "unknown",
                }
                and isinstance(mechanism, list)
                and all(isinstance(feature, str) for feature in mechanism)
                and len(mechanism) == len(set(mechanism))
                and set(mechanism).issubset(known_ids)
                and isinstance(explanations, list)
                and len(explanations) <= 4
                and isinstance(pred.get("abstain"), bool)
            )
            if structurally_valid:
                for item in explanations:
                    if not isinstance(item, dict):
                        structurally_valid = False
                        break
                    features = item.get("features")
                    confidence = item.get("confidence")
                    if (
                        not isinstance(features, list)
                        or not all(isinstance(feature, str) for feature in features)
                        or len(features) != len(set(features))
                        or not set(features).issubset(known_ids)
                        or not isinstance(confidence, (int, float))
                        or not 0 <= confidence <= 1
                        or not isinstance(item.get("evidence"), str)
                    ):
                        structurally_valid = False
                        break

        mechanism_requested = bool(pred is not None and pred.get("mechanism_metric_applicable", True))
        mechanism_applicable = bool(structurally_valid and mechanism_requested)
        pred_mechanism = set(pred.get("mechanism_feature_ids", [])) if mechanism_applicable else set()
        gold_mechanism = set(truth["mechanism_feature_ids"])
        interaction_case = len(gold_mechanism) >= 2
        candidates = [set(item["features"]) for item in pred.get("explanations", [])] if structurally_valid else []
        support_minimal = {frozenset(s) for s in truth["reference_support_sufficient_minimal_subspaces"]}
        rule_minimal = {frozenset(s) for s in truth["rule_minimal_subspaces"]}
        # Sufficiency is monotone: adding fields to a sufficient subset cannot
        # remove the evidence already present in that subset. The hidden labels
        # enumerate inclusion-minimal subsets over active mechanism/proxy fields;
        # this also lets us score predictions containing nuisance fields.
        support_hits = [candidate in support_minimal for candidate in candidates]
        support_sufficient = [any(target.issubset(candidate) for target in support_minimal) for candidate in candidates]
        rule_hits = [candidate in rule_minimal for candidate in candidates]
        rule_sufficient = [any(target.issubset(candidate) for target in rule_minimal) for candidate in candidates]
        gold_minimal_count = len(support_minimal)
        retrieved = len(set(map(frozenset, candidates)) & support_minimal)
        hit_rank = next((rank for rank, hit in enumerate(support_hits, start=1) if hit), None)
        best_jaccard = max((
            len(candidate & target) / len(candidate | target) if candidate | target else 1.0
            for candidate in candidates for target in support_minimal
        ), default=0.0)
        split = case_input["split"]
        row = {
            "case_id": case_id,
            "split": split,
            "template_id": truth["template_id"],
            "task_label_gold": truth["task_label"],
            "feature_count": case_input["feature_count"],
            "reference_count": case_input["reference_count"],
            "nuisance_profile": case_input["nuisance_profile"],
            "field_name_mode": case_input["field_name_mode"],
            "task_label_predicted": pred.get("task_label", "") if structurally_valid else "",
            "task_label_correct": bool(
                structurally_valid and pred["task_label"] != "unknown"
                and pred["task_label"] == truth["task_label"]
            ),
            "task_label_scored": bool(structurally_valid and pred["task_label"] != "unknown"),
            "prediction_present": pred is not None,
            "structurally_valid": structurally_valid,
            "abstain": pred.get("abstain", "") if structurally_valid else "",
            "mechanism_gold": json.dumps(sorted(gold_mechanism)),
            "mechanism_predicted": json.dumps(sorted(pred_mechanism)),
            "mechanism_metric_applicable": mechanism_applicable,
            "mechanism_exact": pred_mechanism == gold_mechanism if mechanism_applicable else "",
            "interaction_case": interaction_case,
            "interaction_metric_applicable": bool(interaction_case and mechanism_requested),
            "interaction_all_fields_hit": (
                bool(mechanism_applicable and pred_mechanism == gold_mechanism)
                if interaction_case and mechanism_requested else ""
            ),
            "explanations_predicted": json.dumps([sorted(s) for s in candidates]),
            "support_sufficient_candidate_count": sum(support_sufficient),
            "support_minimal_candidate_count": sum(support_hits),
            "support_any_minimal_hit": bool(any(support_hits)),
            "support_first_hit_rank": hit_rank or "",
            "support_recall_at_4": retrieved / gold_minimal_count if gold_minimal_count else 0.0,
            "support_best_jaccard": best_jaccard,
            "support_top1_sufficient": support_sufficient[0] if candidates else False,
            "support_top1_minimal": support_hits[0] if candidates else False,
            "rule_sufficient_candidate_count": sum(rule_sufficient),
            "rule_minimal_candidate_count": sum(rule_hits),
            "rule_top1_sufficient": rule_sufficient[0] if candidates else False,
            "rule_top1_minimal": rule_hits[0] if candidates else False,
            "rule_any_minimal_hit": bool(any(rule_hits)),
            "mean_predicted_subspace_fraction": mean([len(s) / case_input["feature_count"] for s in candidates]),
        }
        per_case.append(row)
        for bucket in ("all", split):
            explanation_metrics[bucket].append(row)

    if not args.summary_only:
        write_csv(outdir / "per_case.csv", per_case)
    summaries = {}
    for split, rows in explanation_metrics.items():
        valid_rows = [row for row in rows if row["structurally_valid"]]
        interaction_rows = [row for row in rows if row["interaction_metric_applicable"]]
        # Keep mechanism and task metrics split-specific as well.
        split_pairs = []
        split_task = []
        for row in valid_rows:
            pred = predictions[row["case_id"]]
            if pred.get("mechanism_metric_applicable", True):
                split_pairs.append((set(pred["mechanism_feature_ids"]), set(gold[row["case_id"]]["mechanism_feature_ids"])))
            if pred["task_label"] != "unknown":
                split_task.append(float(pred["task_label"] == gold[row["case_id"]]["task_label"]))
        summaries[split] = {
            "n_cases": len(rows),
            "n_structurally_valid": len(valid_rows),
            "structured_output_rate": mean([float(row["structurally_valid"]) for row in rows]),
            "task_label_coverage": sum(row["task_label_scored"] for row in rows) / len(rows) if rows else None,
            "mechanism_metric_coverage": sum(row["mechanism_metric_applicable"] for row in rows) / len(rows) if rows else None,
            "abstention_rate_on_valid_predictions": mean([
                float(bool(predictions[row["case_id"]].get("abstain"))) for row in valid_rows
            ]),
            "task_label_accuracy_on_valid_predictions": mean(split_task),
            "mechanism_feature_metrics": feature_metrics(split_pairs),
            "interaction_case_count": sum(row["interaction_case"] for row in rows),
            "interaction_scored_case_count": len(interaction_rows),
            "interaction_metric_coverage": (
                len(interaction_rows) / sum(row["interaction_case"] for row in rows)
                if any(row["interaction_case"] for row in rows) else None
            ),
            "interaction_all_fields_recovery_rate": mean([
                float(row["interaction_all_fields_hit"]) for row in interaction_rows
            ]),
            "support_top1_sufficiency_rate": mean([float(row["support_top1_sufficient"]) for row in rows]),
            "support_top1_minimal_exact_hit_rate": mean([float(row["support_top1_minimal"]) for row in rows]),
            "support_any_minimal_hit_at_4": mean([float(row["support_any_minimal_hit"]) for row in rows]),
            "support_mean_recall_at_4": mean([float(row["support_recall_at_4"]) for row in rows]),
            "support_mean_reciprocal_rank": mean([1 / row["support_first_hit_rank"] if row["support_first_hit_rank"] else 0.0 for row in rows]),
            "support_mean_best_jaccard": mean([float(row["support_best_jaccard"]) for row in rows]),
            "rule_top1_sufficiency_rate": mean([float(row["rule_top1_sufficient"]) for row in rows]),
            "rule_top1_minimal_exact_hit_rate": mean([float(row["rule_top1_minimal"]) for row in rows]),
            "rule_any_minimal_hit_at_4": mean([float(row["rule_any_minimal_hit"]) for row in rows]),
            "mean_predicted_subspace_fraction": mean([
                float(row["mean_predicted_subspace_fraction"])
                for row in rows if isinstance(row["mean_predicted_subspace_fraction"], (int, float))
            ]),
        }
    (outdir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"outdir": str(outdir), "summary": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
