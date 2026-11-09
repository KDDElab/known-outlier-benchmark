#!/usr/bin/env python3
"""Export, score, and template-cluster-bootstrap v2 model runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
PROTOCOL = "known_anomaly_subspace_benchmark_v2_compact_json_v1"
MODELS = ("qwen35_9b", "deepseek_v2_lite_chat", "llama31_8b_instruct")
def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return float(value.lower() == "true")
        return float(value)
    except (TypeError, ValueError):
        return None


def mechanism_f1(rows: list[dict]) -> float:
    tp = fp = fn = 0
    for row in rows:
        if row["mechanism_metric_applicable"].lower() != "true":
            continue
        predicted = set(json.loads(row["mechanism_predicted"]))
        gold = set(json.loads(row["mechanism_gold"]))
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def mechanism_f1_invalid_as_empty(rows: list[dict]) -> float:
    """Score every requested case, treating an invalid/missing output as an empty set."""
    tp = fp = fn = 0
    for row in rows:
        predicted = set(json.loads(row["mechanism_predicted"]))
        gold = set(json.loads(row["mechanism_gold"]))
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def interaction_recovery(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return sum(row["interaction_all_fields_hit"].lower() == "true" for row in rows) / len(rows)


def bootstrap_ci(rows: list[dict], metric, rng: random.Random, n_bootstrap: int) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["template_id"]].append(row)
    template_ids = sorted(grouped)
    draws = []
    for _ in range(n_bootstrap):
        sampled = [row for template in rng.choices(template_ids, k=len(template_ids)) for row in grouped[template]]
        value = metric(sampled)
        if value is not None:
            draws.append(value)
    point = metric(rows)
    draws.sort()
    if not draws:
        return {"estimate": point, "ci95_low": None, "ci95_high": None, "bootstrap_replicates": 0}
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]
    return {
        "estimate": point,
        "ci95_low": low,
        "ci95_high": high,
        "bootstrap_replicates": len(draws),
        "n_template_clusters": len(template_ids),
    }


def mean_metric(field: str):
    def compute(rows: list[dict]) -> float | None:
        values = [number(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else None
    return compute


def scored_task_accuracy(rows: list[dict]) -> float | None:
    values = [row for row in rows if row["task_label_scored"].lower() == "true"]
    if not values:
        return None
    return sum(row["task_label_correct"].lower() == "true" for row in values) / len(values)


def task_accuracy_invalid_as_wrong(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return sum(row["task_label_correct"].lower() == "true" for row in rows) / len(rows)


def output_protocol_audit(rows: list[dict]) -> dict:
    """Separate recoverable JSON objects from strict compliance with JSON-only instructions."""
    required_top = {"task_label", "mechanism_feature_ids", "explanations", "abstain"}
    required_item = {"features", "evidence", "confidence"}
    parsed = [row.get("parsed_prediction") for row in rows if isinstance(row.get("parsed_prediction"), dict)]
    strict_json_only = 0
    exact_schema = 0
    evidence_word_compliant = 0
    preamble_count = 0
    trailing_count = 0
    for row in rows:
        raw = row.get("raw_output") or ""
        stripped = raw.strip()
        if stripped and not stripped.startswith("{"):
            preamble_count += 1
        object_end = None
        try:
            object_start = stripped.find("{")
            if object_start >= 0:
                _, relative_end = json.JSONDecoder().raw_decode(stripped[object_start:])
                object_end = object_start + relative_end
        except (TypeError, json.JSONDecodeError):
            pass
        if object_end is not None and stripped[object_end:].strip():
            trailing_count += 1
        prediction = row.get("parsed_prediction")
        if not isinstance(prediction, dict):
            continue
        schema_ok = set(prediction) == required_top and isinstance(prediction.get("explanations"), list)
        evidence_ok = True
        if schema_ok:
            for item in prediction["explanations"]:
                if not isinstance(item, dict) or set(item) != required_item:
                    schema_ok = False
                    break
                evidence = item.get("evidence")
                if not isinstance(evidence, str) or len(evidence.split()) > 20:
                    evidence_ok = False
        if schema_ok:
            exact_schema += 1
        if evidence_ok:
            evidence_word_compliant += 1
        if (
            schema_ok and evidence_ok and object_end == len(stripped)
            and stripped.startswith("{")
        ):
            strict_json_only += 1
    return {
        "n_test_cases": len(rows),
        "n_parseable_schema_records": len(parsed),
        "parseable_schema_rate": len(parsed) / len(rows) if rows else None,
        "strict_json_only_exact_schema_and_evidence_limit_count": strict_json_only,
        "strict_json_only_exact_schema_and_evidence_limit_rate": strict_json_only / len(rows) if rows else None,
        "exact_key_schema_count": exact_schema,
        "exact_key_schema_rate": exact_schema / len(rows) if rows else None,
        "all_candidate_evidence_within_20_words_count_among_parsed": evidence_word_compliant,
        "all_candidate_evidence_within_20_words_rate_among_parsed": evidence_word_compliant / len(parsed) if parsed else None,
        "outputs_with_non_json_prefix_count": preamble_count,
        "outputs_with_non_json_suffix_count": trailing_count,
    }


def strict_output_compliant(row: dict) -> bool:
    raw = row.get("raw_output") or ""
    stripped = raw.strip()
    prediction = row.get("parsed_prediction")
    if not stripped.startswith("{") or not isinstance(prediction, dict):
        return False
    try:
        decoded, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return False
    if end != len(stripped) or decoded != prediction:
        return False
    if set(decoded) != {"task_label", "mechanism_feature_ids", "explanations", "abstain"}:
        return False
    if not isinstance(decoded.get("explanations"), list):
        return False
    for item in decoded["explanations"]:
        if not isinstance(item, dict) or set(item) != {"features", "evidence", "confidence"}:
            return False
        if not isinstance(item.get("evidence"), str) or len(item["evidence"].split()) > 20:
            return False
    return True


def protocol_compliance_rows(rows: list[dict]) -> list[dict]:
    output = []
    decoder = json.JSONDecoder()
    for row in rows:
        raw = row.get("raw_output") or ""
        stripped = raw.strip()
        prediction = row.get("parsed_prediction")
        start = stripped.find("{")
        end = None
        if start >= 0:
            try:
                _, relative_end = decoder.raw_decode(stripped[start:])
                end = start + relative_end
            except json.JSONDecodeError:
                pass
        explanations = prediction.get("explanations", []) if isinstance(prediction, dict) else []
        long_evidence = sum(
            isinstance(item, dict)
            and isinstance(item.get("evidence"), str)
            and len(item["evidence"].split()) > 20
            for item in explanations
        )
        output.append({
            "case_id": row["case_id"],
            "parseable_schema": isinstance(prediction, dict),
            "strict_json_contract_compliant": strict_output_compliant(row),
            "non_json_prefix": bool(stripped and not stripped.startswith("{")),
            "non_json_suffix": bool(end is not None and stripped[end:].strip()),
            "evidence_fields_over_20_words": long_evidence,
            "output_token_cap_hit": (row.get("output_tokens") or 0) >= 512,
            "prompt_tokens": row.get("prompt_tokens"),
            "output_tokens": row.get("output_tokens"),
            "latency_seconds": row.get("latency_seconds"),
            "parse_or_inference_error": row.get("parse_or_inference_error"),
        })
    return output


def cluster_signflip_test(template_differences: list[float], rng: random.Random, n_permutations: int) -> float:
    observed = abs(sum(template_differences) / len(template_differences))
    exceed = 0
    for _ in range(n_permutations):
        permuted = abs(sum(value * rng.choice((-1.0, 1.0)) for value in template_differences) / len(template_differences))
        if permuted >= observed - 1e-15:
            exceed += 1
    return (exceed + 1) / (n_permutations + 1)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    n = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (n - rank) * p_values[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def ensure_model_output(model: str, n_bootstrap: int, seed: int) -> tuple[list[dict], dict]:
    annotation_path = RUNS / f"annotations_{model}.jsonl"
    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)
    rows = [row for row in read_jsonl(annotation_path) if row.get("event") == "annotation" and row.get("protocol_id") == PROTOCOL]
    cases = {row["case_id"] for row in rows}
    expected_inputs = {
        row["case_id"]: row["prompt_sha256"]
        for row in read_jsonl(HERE / "inputs.jsonl") if row["split"] == "test"
    }
    expected = set(expected_inputs)
    if cases != expected:
        missing = sorted(expected - cases)[:5]
        raise RuntimeError(f"{model} run incomplete: {len(cases)}/{len(expected)} cases; examples missing={missing}")
    if len(rows) != len(expected):
        raise RuntimeError(f"{model} has {len(rows)} protocol records for {len(expected)} cases; expected exactly one per case")
    hash_mismatches = [
        row["case_id"] for row in rows
        if row.get("prompt_sha256") != expected_inputs[row["case_id"]]
    ]
    if hash_mismatches:
        raise RuntimeError(f"{model} protocol prompt hash mismatch; examples={hash_mismatches[:5]}")

    predictions = RUNS / f"predictions_{model}_{PROTOCOL}.jsonl"
    subprocess.run(
        [sys.executable, str(HERE / "export_predictions_v2.py"), str(annotation_path), "--protocol", PROTOCOL, "--output", str(predictions)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    outdir = RUNS / f"evaluation_{model}_{PROTOCOL}"
    subprocess.run(
        [sys.executable, str(HERE / "evaluate_benchmark.py"), str(predictions), "--outdir", str(outdir)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))["test"]
    case_rows = [row for row in read_csv(outdir / "per_case.csv") if row["split"] == "test"]
    if len(case_rows) != len(expected):
        raise RuntimeError(f"Scorer returned {len(case_rows)} test rows for {model}; expected {len(expected)}")
    strict_by_case = {row["case_id"]: strict_output_compliant(row) for row in rows}
    for row in case_rows:
        row["strict_json_contract_compliant"] = str(strict_by_case[row["case_id"]]).lower()

    rng = random.Random(seed)
    metrics = {name: bootstrap_ci(case_rows, mean_metric(field), rng, n_bootstrap) for name, field in (
        ("structured_output_rate", "structurally_valid"),
        ("strict_json_contract_compliance_rate", "strict_json_contract_compliant"),
        ("support_top1_sufficiency_rate", "support_top1_sufficient"),
        ("support_top1_minimal_exact_hit_rate", "support_top1_minimal"),
        ("support_any_minimal_hit_at_4", "support_any_minimal_hit"),
        ("support_mean_best_jaccard", "support_best_jaccard"),
    )}
    # Convert first-hit rank to reciprocal rank explicitly, with misses scored 0.
    metrics["support_mean_reciprocal_rank"] = bootstrap_ci(
        case_rows,
        lambda subset: sum((1 / float(row["support_first_hit_rank"])) if row["support_first_hit_rank"] else 0.0 for row in subset) / len(subset) if subset else None,
        rng,
        n_bootstrap,
    )
    metrics["mechanism_micro_f1_on_valid_predictions"] = bootstrap_ci(case_rows, mechanism_f1, rng, n_bootstrap)
    metrics["mechanism_micro_f1_invalid_as_empty"] = bootstrap_ci(
        case_rows, mechanism_f1_invalid_as_empty, rng, n_bootstrap
    )
    interaction_rows = [row for row in case_rows if row["interaction_metric_applicable"].lower() == "true"]
    metrics["interaction_all_fields_recovery_rate"] = bootstrap_ci(interaction_rows, interaction_recovery, rng, n_bootstrap)
    metrics["interaction_all_fields_recovery_rate"]["n_interaction_cases"] = len(interaction_rows)
    metrics["interaction_all_fields_recovery_rate"]["n_template_clusters"] = len({row["template_id"] for row in interaction_rows})
    metrics["task_label_accuracy_on_scored"] = bootstrap_ci(case_rows, scored_task_accuracy, rng, n_bootstrap)
    metrics["task_label_accuracy_invalid_as_wrong"] = bootstrap_ci(
        case_rows, task_accuracy_invalid_as_wrong, rng, n_bootstrap
    )
    metrics["support_mean_recall_at_4"] = bootstrap_ci(case_rows, mean_metric("support_recall_at_4"), rng, n_bootstrap)
    metrics["mean_predicted_subspace_fraction"] = bootstrap_ci(case_rows, mean_metric("mean_predicted_subspace_fraction"), rng, n_bootstrap)

    subgroup_metrics = {}
    for field in ("task_label_gold", "feature_count", "reference_count", "nuisance_profile", "field_name_mode"):
        subgroup_metrics[field] = {}
        values = sorted({row[field] for row in case_rows})
        for value in values:
            subset = [row for row in case_rows if row[field] == value]
            subgroup_metrics[field][str(value)] = {
                "n_cases": len(subset),
                "support_top1_minimal_exact_hit_rate": mean_metric("support_top1_minimal")(subset),
                "support_any_minimal_hit_at_4": mean_metric("support_any_minimal_hit")(subset),
                "support_mean_best_jaccard": mean_metric("support_best_jaccard")(subset),
                "structured_output_rate": mean_metric("structurally_valid")(subset),
            }

    current_rows = rows
    compliance_path = outdir / "protocol_compliance.csv"
    write_csv(compliance_path, protocol_compliance_rows(current_rows))
    latency = [float(row["latency_seconds"]) for row in current_rows if row.get("latency_seconds") is not None]
    sorted_latency = sorted(latency)
    prompt_tokens = [int(row["prompt_tokens"]) for row in current_rows if row.get("prompt_tokens") is not None]
    output_tokens = [int(row["output_tokens"]) for row in current_rows if row.get("output_tokens") is not None]
    over_cap_input_tokens = []
    for row in current_rows:
        error = row.get("parse_or_inference_error") or ""
        match = re.search(r"rendered input has (\d+) tokens, exceeding", error)
        if match:
            over_cap_input_tokens.append(int(match.group(1)))
    measured_or_over_cap_inputs = prompt_tokens + over_cap_input_tokens
    valid_count = sum(row.get("parsed_prediction") is not None for row in current_rows)
    parse_failures = Counter(
        row["parse_or_inference_error"]
        for row in current_rows if row.get("parse_or_inference_error")
    )
    run_records = [
        row for row in read_jsonl(RUNS / "model_runs.jsonl")
        if row.get("protocol_id") == PROTOCOL and row.get("annotator_id") == model
    ]
    run_metadata = run_records[-1] if run_records else {}
    model_result = {
        "model": model,
        "protocol_id": PROTOCOL,
        "n_test_cases": len(expected),
        "n_template_clusters": len({row["template_id"] for row in case_rows}),
        "n_protocol_records": len(current_rows),
        "n_parsed_records": valid_count,
        "mean_latency_seconds": sum(latency) / len(latency) if latency else None,
        "n_latency_measurements": len(latency),
        "p95_latency_seconds": sorted_latency[min(len(sorted_latency) - 1, int(0.95 * len(sorted_latency)))] if sorted_latency else None,
        "mean_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None,
        "n_prompt_token_measurements": len(prompt_tokens),
        "mean_prompt_tokens_including_recorded_over_cap_inputs": (
            sum(measured_or_over_cap_inputs) / len(measured_or_over_cap_inputs)
            if measured_or_over_cap_inputs else None
        ),
        "n_recorded_input_over_cap": len(over_cap_input_tokens),
        "max_recorded_prompt_tokens_including_over_cap": max(measured_or_over_cap_inputs) if measured_or_over_cap_inputs else None,
        "mean_output_tokens": sum(output_tokens) / len(output_tokens) if output_tokens else None,
        "n_output_token_measurements": len(output_tokens),
        "total_prompt_tokens": sum(prompt_tokens),
        "total_output_tokens": sum(output_tokens),
        "parse_failure_reasons": dict(parse_failures),
        "output_protocol_audit": output_protocol_audit(current_rows),
        "run_metadata": {
            key: run_metadata.get(key)
            for key in ("run_id", "model_id", "model_path", "model_type", "dtype", "inference_transformers_version", "torch_version", "gpu", "decoding", "started_at_utc", "completed_at_utc", "elapsed_seconds", "model_files")
            if key in run_metadata
        },
        "evaluator_summary": summary,
        "cluster_bootstrap_metrics": metrics,
        "subgroups": subgroup_metrics,
        "bootstrap": {"method": "resample eligible template groups with replacement; retain all cases in each sampled group", "n_replicates": n_bootstrap, "seed": seed},
        "paths": {"predictions": str(predictions), "evaluation": str(outdir), "protocol_compliance": str(compliance_path)},
    }
    (outdir / "analysis.json").write_text(json.dumps(model_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return case_rows, model_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()

    results = {}
    case_rows = {}
    for model in args.models:
        rows, result = ensure_model_output(model, args.bootstrap, args.seed)
        case_rows[model] = {row["case_id"]: row for row in rows}
        results[model] = result

    contrasts = {}
    contrast_p_values = {}
    baseline_p_values = {}
    rng = random.Random(args.seed + 1)
    baseline_path = RUNS / "baseline_evaluations" / "top_univariate" / "per_case.csv"
    baseline_rows = {
        row["case_id"]: row for row in read_csv(baseline_path) if row["split"] == "test"
    }
    if set(baseline_rows) != set(next(iter(case_rows.values()))):
        raise RuntimeError("Top-univariate baseline and LLM analyses do not cover the same test cases")

    for model in args.models:
        model_rows = case_rows[model]
        contrast_key = f"{model}_minus_top_univariate"
        contrasts[contrast_key] = {}
        for metric_name, field in (
            ("support_top1_minimal_exact_hit_rate", "support_top1_minimal"),
            ("support_any_minimal_hit_at_4", "support_any_minimal_hit"),
            ("support_mean_best_jaccard", "support_best_jaccard"),
            ("support_top1_sufficiency_rate", "support_top1_sufficient"),
        ):
            per_case_diff = {}
            for case_id, model_row in model_rows.items():
                model_value = number(model_row[field]) or 0.0
                baseline_value = number(baseline_rows[case_id][field]) or 0.0
                per_case_diff[case_id] = model_value - baseline_value
            groups: dict[str, list[str]] = defaultdict(list)
            for case_id in per_case_diff:
                groups[model_rows[case_id]["template_id"]].append(case_id)
            template_ids = sorted(groups)
            template_differences = [
                sum(per_case_diff[case_id] for case_id in groups[template]) / len(groups[template])
                for template in template_ids
            ]
            bootstrap_estimates = []
            for _ in range(args.bootstrap):
                sampled_templates = rng.choices(template_ids, k=len(template_ids))
                sampled_cases = [case_id for template in sampled_templates for case_id in groups[template]]
                bootstrap_estimates.append(
                    sum(per_case_diff[case_id] for case_id in sampled_cases) / len(sampled_cases)
                )
            bootstrap_estimates.sort()
            test_key = f"{contrast_key}:{metric_name}"
            baseline_p_values[test_key] = cluster_signflip_test(
                template_differences, rng, args.permutations
            )
            contrasts[contrast_key][metric_name] = {
                "paired_difference": sum(per_case_diff.values()) / len(per_case_diff),
                "ci95_low": bootstrap_estimates[int(0.025 * (len(bootstrap_estimates) - 1))],
                "ci95_high": bootstrap_estimates[int(0.975 * (len(bootstrap_estimates) - 1))],
                "n_bootstrap": args.bootstrap,
                "cluster_signflip_p": baseline_p_values[test_key],
                "n_template_clusters": len(template_ids),
                "n_paired_cases": len(per_case_diff),
            }

    for left_index, left in enumerate(args.models):
        for right in args.models[left_index + 1:]:
            left_rows = case_rows[left]
            right_rows = case_rows[right]
            if left_rows.keys() != right_rows.keys():
                raise RuntimeError(f"Cannot pair different case sets: {left} vs {right}")
            contrasts[f"{left}_minus_{right}"] = {}
            for metric_name, accessor, eligible in (
                ("support_top1_minimal_exact_hit_rate", lambda row: number(row["support_top1_minimal"]) or 0.0, lambda row: True),
                ("support_any_minimal_hit_at_4", lambda row: number(row["support_any_minimal_hit"]) or 0.0, lambda row: True),
                ("support_mean_best_jaccard", lambda row: float(row["support_best_jaccard"]), lambda row: True),
                ("interaction_all_fields_recovery_rate", lambda row: number(row["interaction_all_fields_hit"]) or 0.0, lambda row: row["interaction_metric_applicable"].lower() == "true"),
            ):
                metric_case_ids = [case_id for case_id in left_rows if eligible(left_rows[case_id]) and eligible(right_rows[case_id])]
                metric_groups: dict[str, list[str]] = defaultdict(list)
                for case_id in metric_case_ids:
                    metric_groups[left_rows[case_id]["template_id"]].append(case_id)
                metric_templates = sorted(metric_groups)
                per_case_diff = {
                    case_id: accessor(left_rows[case_id]) - accessor(right_rows[case_id])
                    for case_id in metric_case_ids
                }
                estimates = []
                for _ in range(args.bootstrap):
                    sampled_templates = rng.choices(metric_templates, k=len(metric_templates))
                    sample_ids = [case_id for template in sampled_templates for case_id in metric_groups[template]]
                    estimates.append(sum(per_case_diff[case_id] for case_id in sample_ids) / len(sample_ids))
                estimates.sort()
                template_differences = [
                    sum(per_case_diff[case_id] for case_id in metric_groups[template]) / len(metric_groups[template])
                    for template in metric_templates
                ]
                contrast_key = f"{left}_minus_{right}:{metric_name}"
                contrast_p_values[contrast_key] = cluster_signflip_test(
                    template_differences, rng, args.permutations
                )
                contrasts[f"{left}_minus_{right}"][metric_name] = {
                    "paired_difference": sum(per_case_diff.values()) / len(per_case_diff),
                    "ci95_low": estimates[int(0.025 * (len(estimates) - 1))],
                    "ci95_high": estimates[int(0.975 * (len(estimates) - 1))],
                    "n_bootstrap": args.bootstrap,
                    "cluster_signflip_p": contrast_p_values[contrast_key],
                    "n_template_clusters": len(metric_templates),
                    "n_paired_cases": len(metric_case_ids),
                }

    adjusted_primary = holm_adjust(contrast_p_values)
    for contrast_key, p_adjusted in adjusted_primary.items():
        pair, metric = contrast_key.split(":", 1)
        contrasts[pair][metric]["holm_adjusted_p_primary_model_pair_family"] = p_adjusted
    adjusted_baselines = holm_adjust(baseline_p_values)
    for contrast_key, p_adjusted in adjusted_baselines.items():
        pair, metric = contrast_key.split(":", 1)
        contrasts[pair][metric]["holm_adjusted_p_exploratory_model_vs_univariate_family"] = p_adjusted

    output = RUNS / f"model_analysis_{PROTOCOL}.json"
    source_files = (
        "generate_benchmark.py", "annotate_v2.py", "export_predictions_v2.py",
        "evaluate_benchmark.py", "analyze_model_runs_v2.py",
    )
    output.write_text(json.dumps({
        "models": results,
        "paired_template_bootstrap_contrasts": contrasts,
        "evaluation_software_sha256": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in source_files
        },
        "multiple_testing": {
            "method": "Holm step-down",
            "permutation_null": "paired template-level sign flips; two-sided",
            "n_permutations": args.permutations,
            "primary_model_pair_family": {
                "interpretation": "planned comparison among the three LLMs; 4 metrics for each of 3 pairs",
                "n_tests": len(contrast_p_values),
                "test_names": sorted(contrast_p_values),
            },
            "exploratory_model_vs_top_univariate_family": {
                "interpretation": "post hoc paired comparison against the deterministic top-four univariate baseline",
                "n_tests": len(baseline_p_values),
                "test_names": sorted(baseline_p_values),
            },
        },
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "models": list(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
