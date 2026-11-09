#!/usr/bin/env python3
"""Audit top-one confidence calibration and literal evidence grounding.

Confidence correctness is exact support-minimal retrieval for the first ranked
candidate. Evidence grounding is intentionally narrow: it checks explicit
feature IDs, category/identifier tokens, and ISO dates against visible inputs;
it does not claim semantic or causal faithfulness.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np

from evaluate_benchmark import read_jsonl
from run_m2oe_baseline_v2 import extract_payload

VALUE_RE = re.compile(r"\b(?:c\d+|u\d+|id_\d+|level_\d+|\d{4}-\d{2}-\d{2})\b")
FEATURE_RE = re.compile(r"\bF\d{3}\b")


def ece(confidence: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> float:
    if not len(confidence):
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for index in range(n_bins):
        include = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == n_bins - 1 else confidence < edges[index + 1]
        )
        if include.any():
            total += include.mean() * abs(float(confidence[include].mean() - outcome[include].mean()))
    return float(total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--inputs-path", type=Path, required=True)
    parser.add_argument("--ground-truth-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()

    cases = {row["case_id"]: row for row in read_jsonl(args.inputs_path) if row.get("split") == args.split}
    truths = {row["case_id"]: row for row in read_jsonl(args.ground_truth_path)}
    predictions = {row["case_id"]: row for row in read_jsonl(args.predictions)}
    confidence, confidence_outcome, confidence_templates = [], [], []
    top1_outcome, top1_templates = [], []
    literal_supported = literal_total = evidence_rows = unknown_feature_ids = 0
    unsupported_examples = []

    for case_id, case in cases.items():
        pred = predictions.get(case_id)
        if not pred or not isinstance(pred.get("explanations"), list) or not pred["explanations"]:
            continue
        first = pred["explanations"][0]
        if not isinstance(first, dict) or not isinstance(first.get("features"), list):
            continue
        candidate = frozenset(first["features"])
        minimal_gold = {frozenset(s) for s in truths[case_id]["reference_support_sufficient_minimal_subspaces"]}
        hit = float(candidate in minimal_gold)
        template_id = truths[case_id]["template_id"]
        top1_outcome.append(hit)
        top1_templates.append(template_id)
        reported_confidence = first.get("confidence")
        if isinstance(reported_confidence, (int, float)) and not isinstance(reported_confidence, bool):
            reported_confidence = float(reported_confidence)
            if np.isfinite(reported_confidence) and 0.0 <= reported_confidence <= 1.0:
                confidence.append(reported_confidence)
                confidence_outcome.append(hit)
                confidence_templates.append(template_id)

        evidence = first.get("evidence", "")
        if not isinstance(evidence, str):
            continue
        evidence_rows += 1
        payload = extract_payload(case["user_prompt"])
        known_features = {item["feature_id"] for item in payload["feature_schema"]}
        feature_mentions = set(FEATURE_RE.findall(evidence))
        unknown_feature_ids += len(feature_mentions - known_features)
        visible_values = set()
        for row in [payload["target_row"], *payload["normal_reference_rows"]]:
            for value in row.values():
                if value is not None:
                    visible_values.add(str(value))
        for item in payload["feature_schema"]:
            visible_values.update(str(value) for value in item.get("allowed_values", []) if value is not None)
        mentions = VALUE_RE.findall(evidence)
        for token in mentions:
            literal_total += 1
            if token in visible_values:
                literal_supported += 1
            elif len(unsupported_examples) < 25:
                unsupported_examples.append({"case_id": case_id, "token": token, "evidence": evidence})

    c = np.asarray(confidence, dtype=float)
    y = np.asarray(confidence_outcome, dtype=float)
    cluster_ids = np.asarray(confidence_templates, dtype=object)
    all_y = np.asarray(top1_outcome, dtype=float)
    all_cluster_ids = np.asarray(top1_templates, dtype=object)
    brier = float(np.mean((c-y)**2)) if len(c) else None
    cluster_values = sorted(set(cluster_ids.tolist()))
    all_cluster_values = sorted(set(all_cluster_ids.tolist()))
    rng = np.random.default_rng(args.seed)
    draws = {"top1_exact_hit_rate_all_top1": [], "brier": [], "ece_10_bins": []}
    if all_cluster_values and args.bootstrap:
        indices_by_cluster = {key: np.flatnonzero(all_cluster_ids == key) for key in all_cluster_values}
        for _ in range(args.bootstrap):
            picked = rng.choice(all_cluster_values, size=len(all_cluster_values), replace=True)
            indices = np.concatenate([indices_by_cluster[key] for key in picked])
            draws["top1_exact_hit_rate_all_top1"].append(float(all_y[indices].mean()))
    if cluster_values and args.bootstrap:
        indices_by_cluster = {key: np.flatnonzero(cluster_ids == key) for key in cluster_values}
        for _ in range(args.bootstrap):
            picked = rng.choice(cluster_values, size=len(cluster_values), replace=True)
            indices = np.concatenate([indices_by_cluster[key] for key in picked])
            draws["brier"].append(float(np.mean((c[indices]-y[indices])**2)))
            draws["ece_10_bins"].append(ece(c[indices], y[indices]))

    def interval(values):
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if values else None

    result = {
        "n_cases": len(cases),
        "split": args.split,
        "n_top1_predictions": len(all_y),
        "top1_confidence_coverage": len(c) / len(cases) if cases else None,
        "n_top1_with_confidence": len(c),
        "top1_exact_support_minimal_hit_rate_all_top1": float(all_y.mean()) if len(all_y) else None,
        "mean_top1_confidence": float(c.mean()) if len(c) else None,
        "brier_score_conditional_on_top1_with_reported_confidence": brier,
        "ece_10_equal_width_bins_conditional_on_top1_with_reported_confidence": ece(c, y) if len(c) else None,
        "template_cluster_bootstrap_95_ci": {
            "top1_exact_hit_rate_all_top1": interval(draws["top1_exact_hit_rate_all_top1"]),
            "brier": interval(draws["brier"]),
            "ece_10_bins": interval(draws["ece_10_bins"]),
        },
        "evidence_audit": {
            "n_top1_evidence_strings": evidence_rows,
            "explicit_category_identifier_or_date_tokens": literal_total,
            "literal_token_grounding_rate": literal_supported / literal_total if literal_total else None,
            "unmatched_explicit_value_tokens": literal_total - literal_supported,
            "unknown_feature_id_mentions": unknown_feature_ids,
            "unmatched_examples_for_human_review": unsupported_examples,
            "scope": "literal visible-input matching only; not a semantic-faithfulness claim",
        },
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
