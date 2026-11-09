#!/usr/bin/env python3
"""Score circuit-pilot model and baseline predictions against input-feature gold.

Missing, invalid, or abstained predictions remain in the denominator and count
as misses. Confidence intervals resample source table-pair clusters, retaining
all focus-output queries from a sampled pair.

Run from the repository root after inference and baseline generation:
    python adbench_llm_reference_v1/circuit_track/evaluate_zero_noise_pilot.py

Dependencies: Python 3.10+ standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
MODEL_NAMES = ("qwen35_9b", "deepseek_v2_lite_chat", "llama31_8b_instruct")
BASELINE_NAMES = ("gower_knn", "gower_lof", "conditional_output_surprisal", "joint_output_surprisal")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_latest_annotations(path: Path, protocol_id: str | None = None) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("event") == "annotation" and (protocol_id is None or row.get("protocol_id") == protocol_id):
            latest[row["case_id"]] = row
    return latest


def score_case(gold: set[str], parsed: dict[str, Any] | None) -> dict[str, Any]:
    candidates: list[set[str]] = []
    abstain = False
    if isinstance(parsed, dict):
        abstain = bool(parsed.get("abstain", False))
        if not abstain and isinstance(parsed.get("explanations"), list):
            candidates = [
                set(item["features"])
                for item in parsed["explanations"]
                if isinstance(item, dict) and isinstance(item.get("features"), list)
            ]
    ranks = [rank for rank, candidate in enumerate(candidates, 1) if candidate == gold]
    first_hit = min(ranks) if ranks else None
    best_jaccard = max(
        (
            len(candidate & gold) / len(candidate | gold)
            if candidate | gold else 1.0
            for candidate in candidates
        ),
        default=0.0,
    )
    top = candidates[0] if candidates else set()
    tp = len(top & gold)
    precision = tp / len(top) if top else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "valid": parsed is not None,
        "abstain": abstain,
        "answered": bool(candidates and any(candidates)),
        "n_candidates": len(candidates),
        "exact_at_1": bool(candidates and candidates[0] == gold),
        "hit_at_4": bool(ranks),
        "mrr": 1.0 / first_hit if first_hit else 0.0,
        "recall_at_4": 1.0 if ranks else 0.0,
        "top1_jaccard": len(top & gold) / len(top | gold) if top | gold else 1.0,
        "best_jaccard_at_4": best_jaccard,
        "top1_feature_f1": f1,
        "first_hit_rank": first_hit or "",
    }


def grouped(rows: list[dict[str, Any]], metric: str) -> float:
    return sum(float(row[metric]) for row in rows) / len(rows) if rows else 0.0


def bootstrap_ci(
    rows: list[dict[str, Any]], metric: str, repetitions: int, seed: int
) -> list[float]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[row["source_table"]].append(row)
    names = sorted(clusters)
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        sample_names = [rng.choice(names) for _ in names]
        sample = [row for name in sample_names for row in clusters[name]]
        draws.append(grouped(sample, metric))
    draws.sort()
    return [draws[int(.025 * (repetitions - 1))], draws[int(.975 * (repetitions - 1))]]


def summarize(rows: list[dict[str, Any]], repetitions: int, seed: int) -> dict[str, Any]:
    metrics = (
        "valid", "exact_at_1", "hit_at_4", "mrr", "recall_at_4",
        "top1_jaccard", "best_jaccard_at_4", "top1_feature_f1", "abstain",
    )
    result: dict[str, Any] = {
        "n_queries": len(rows),
        "n_source_table_clusters": len({row["source_table"] for row in rows}),
    }
    for metric_ix, metric in enumerate(metrics):
        values = [float(row[metric]) for row in rows]
        result[metric] = {
            "estimate": grouped(rows, metric),
            "ci95_table_cluster_bootstrap": bootstrap_ci(rows, metric, repetitions, seed + metric_ix),
        }
    return result


def matched_context_tests(
    method_rows: dict[str, list[dict[str, Any]]], repetitions: int, permutations: int, seed: int
) -> dict[str, Any]:
    metrics = ("exact_at_1", "hit_at_4", "best_jaccard_at_4")
    tests = []
    for model_ix, model in enumerate(MODEL_NAMES):
        full = {row["case_id"]: row for row in method_rows.get(model, [])}
        focus_name = f"{model}_focus_only"
        focus = {row["case_id"]: row for row in method_rows.get(focus_name, [])}
        shared = sorted(full.keys() & focus.keys())
        if not shared:
            continue
        for metric_ix, metric in enumerate(metrics):
            by_cluster: dict[str, list[float]] = defaultdict(list)
            for case_id in shared:
                by_cluster[full[case_id]["source_table"]].append(
                    float(full[case_id][metric]) - float(focus[case_id][metric])
                )
            clusters = [sum(values) / len(values) for _, values in sorted(by_cluster.items())]
            observed = sum(clusters) / len(clusters)
            rng = random.Random(seed + model_ix * 100 + metric_ix)
            extreme = 0
            for _ in range(permutations):
                permuted = sum(value * rng.choice((-1.0, 1.0)) for value in clusters) / len(clusters)
                extreme += abs(permuted) >= abs(observed)
            p_value = (1 + extreme) / (permutations + 1)
            draws = []
            for _ in range(repetitions):
                draws.append(sum(rng.choice(clusters) for _ in clusters) / len(clusters))
            draws.sort()
            tests.append({
                "model": model, "metric": metric, "n_queries": len(shared),
                "n_source_table_clusters": len(clusters), "all_outputs_minus_focus_only": observed,
                "table_cluster_bootstrap_95_ci": [draws[int(.025 * (repetitions - 1))], draws[int(.975 * (repetitions - 1))]],
                "two_sided_cluster_sign_flip_p": p_value,
            })
    order = sorted(range(len(tests)), key=lambda i: tests[i]["two_sided_cluster_sign_flip_p"])
    running = 0.0
    for rank, index in enumerate(order):
        current = min(1.0, (len(tests) - rank) * tests[index]["two_sided_cluster_sign_flip_p"])
        running = max(running, current)
        tests[index]["holm_adjusted_p_9_contrast_family"] = running
    return {
        "metrics": list(metrics), "planned_contrasts": len(MODEL_NAMES) * len(metrics),
        "completed_contrasts": len(tests),
        "cluster_unit": "source modified-table pair; average paired query differences within pair",
        "tests": tests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--outdir", type=Path, default=HERE / "results_zero_noise_pilot")
    parser.add_argument("--focus-only-dir", type=Path, default=None)
    parser.add_argument("--permutations", type=int, default=20000)
    args = parser.parse_args()

    input_path = HERE / "inputs_zero_noise_pilot.jsonl"
    truth_path = HERE / "ground_truth_zero_noise_pilot.jsonl"
    cases = {row["case_id"]: row for row in read_jsonl(input_path)}
    truths = {row["case_id"]: row for row in read_jsonl(truth_path)}
    if cases.keys() != truths.keys():
        raise RuntimeError("Input and ground-truth case IDs do not match")

    prediction_sources: dict[str, dict[str, Any]] = {}
    model_diagnostics: dict[str, dict[str, Any]] = {}
    expected_case_ids = {case_id for case_id in cases}
    for model in MODEL_NAMES:
        path = HERE / f"annotations_{model}.jsonl"
        latest = load_latest_annotations(path, "harvard_circuit_zero_noise_output_specific_pilot_v1")
        if set(latest) != expected_case_ids:
            raise SystemExit(f"All-output circuit run for {model} is incomplete: {len(latest)}/{len(expected_case_ids)} cases")
        prediction_sources[model] = {
            case_id: row.get("parsed_prediction")
            for case_id, row in latest.items()
        }
        failures = Counter(
            row.get("parse_or_inference_error") or "valid"
            for row in latest.values()
        )
        run_rows = [
            row for row in read_jsonl(HERE / "model_runs.jsonl")
            if row.get("annotator_id") == model
            and row.get("event") == "completed"
            and row.get("protocol_id") == "harvard_circuit_zero_noise_output_specific_pilot_v1"
        ]
        model_diagnostics[model] = {
            "n_annotation_records": len(latest),
            "n_valid_parsed": sum(row.get("parsed_prediction") is not None for row in latest.values()),
            "output_cap_hits": sum(row.get("output_tokens") == 512 for row in latest.values()),
            "parse_or_inference_errors": dict(failures),
            "completed_run_metadata": run_rows[-1] if run_rows else None,
        }
        if args.focus_only_dir is not None:
            focus_path = args.focus_only_dir / f"annotations_{model}.jsonl"
            focus_latest = load_latest_annotations(focus_path, "harvard_circuit_focus_output_only_v1")
            if set(focus_latest) != expected_case_ids:
                raise SystemExit(f"Focus-only circuit run for {model} is incomplete: {len(focus_latest)}/{len(expected_case_ids)} cases")
            prediction_sources[f"{model}_focus_only"] = {
                case_id: row.get("parsed_prediction") for case_id, row in focus_latest.items()
            }
            model_diagnostics[f"{model}_focus_only"] = {
                "n_annotation_records": len(focus_latest),
                "n_valid_parsed": sum(row.get("parsed_prediction") is not None for row in focus_latest.values()),
                "output_cap_hits": sum(row.get("output_tokens") == 512 for row in focus_latest.values()),
                "parse_or_inference_errors": dict(Counter(
                    row.get("parse_or_inference_error") or "valid" for row in focus_latest.values()
                )),
            }
    for baseline in BASELINE_NAMES:
        path = HERE / "baseline_predictions" / f"predictions_{baseline}.jsonl"
        prediction_sources[baseline] = {row["case_id"]: row for row in read_jsonl(path)}

    per_case: list[dict[str, Any]] = []
    method_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for method, predictions in prediction_sources.items():
        for case_id, case in cases.items():
            truth = truths[case_id]
            scored = score_case(set(truth["gold_features"]), predictions.get(case_id))
            row = {
                "method": method,
                "case_id": case_id,
                "circuit": truth["circuit"],
                "source_table": truth["source_table"],
                "focus_output": truth["focus_output"],
                "gold_features": json.dumps(sorted(truth["gold_features"])),
                **scored,
            }
            per_case.append(row)
            method_rows[method].append(row)

    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "per_case.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_case[0]))
        writer.writeheader()
        writer.writerows(per_case)

    summary = {
        "track": "Harvard zero-noise circuit output-specific pilot",
        "n_queries": len(cases),
        "denominator": "all 86 queries; invalid, missing, or abstained predictions count as misses",
        "uncertainty_unit": "source modified-table pair; 16 clusters, retaining focus-output queries",
        "bootstrap_repetitions": args.bootstrap,
        "bootstrap_seed": args.seed,
        "model_output_diagnostics": model_diagnostics,
        "methods": {
            method: summarize(rows, args.bootstrap, args.seed + ix * 1000)
            for ix, (method, rows) in enumerate(sorted(method_rows.items()))
        },
        "by_circuit": {},
        "limitations": [
            "The circuit corpus is a narrow binary logic and single-gate-negation setting.",
            "LLMs receive all output context; Gower controls use input rows only, focus conditional surprisal uses only the focus output, and joint conditional surprisal uses the full observed output vector.",
            "The supplementary inference used a runtime newer than the main synthetic benchmark.",
        ],
    }
    if args.focus_only_dir is not None:
        summary["matched_information_comparison"] = matched_context_tests(
            method_rows, args.bootstrap, args.permutations, args.seed
        )
    circuits = sorted({row["circuit"] for row in per_case})
    for circuit in circuits:
        summary["by_circuit"][circuit] = {
            method: summarize([row for row in rows if row["circuit"] == circuit], args.bootstrap, args.seed + 5000 + ix * 1000)
            for ix, (method, rows) in enumerate(sorted(method_rows.items()))
        }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
