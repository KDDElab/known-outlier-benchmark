#!/usr/bin/env python3
"""Audit finite-reference support ambiguity for synthetic benchmark v2.

For every case, enumerate feature subsets up to a declared maximum order and
mark a subset sample-absent when the target projection does not occur in the
provided normal reference rows. The script then compares inclusion-minimal
sample-absent subsets with the independently enumerated complete-population
support gold labels. This is a diagnostic of evidence available in the input;
it does not change benchmark labels or model scores.

Run from the repository root:
    python adbench_llm_reference_v1/synthetic_benchmark_v2/\
audit_reference_sample_identifiability.py

Dependencies: Python 3.10+ standard library only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def value_key(value: Any) -> Any:
    """Return a stable hashable representation for JSON scalar/list/dict data."""
    if isinstance(value, dict):
        return tuple(sorted((key, value_key(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(value_key(item) for item in value)
    return value


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    metric: str,
    repetitions: int,
    seed: int,
) -> list[float]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[row["template_group_id"]].append(row)
    names = sorted(clusters)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        sample = [rng.choice(names) for _ in names]
        values = [row[metric] for name in sample for row in clusters[name]]
        estimates.append(sum(values) / len(values) if values else 0.0)
    return estimates


def audit_case(case: dict[str, Any], max_order: int) -> dict[str, Any]:
    target = case["target_row"]
    references = case["normal_reference_rows"]
    features = sorted(target)
    absent: set[tuple[str, ...]] = set()

    for order in range(1, min(max_order, len(features)) + 1):
        for subset in itertools.combinations(features, order):
            target_projection = tuple(value_key(target[feature]) for feature in subset)
            observed = {
                tuple(value_key(row[feature]) for feature in subset)
                for row in references
            }
            if target_projection not in observed:
                absent.add(subset)

    sample_minimal = {
        subset
        for subset in absent
        if not any(
            tuple(feature for index, feature in enumerate(subset) if index != removed)
            in absent
            for removed in range(len(subset))
        )
    }
    gold = {
        tuple(sorted(subset))
        for subset in case["reference_support_sufficient_minimal_subspaces"]
    }
    if any(len(subset) > max_order for subset in gold):
        raise ValueError(
            f"{case['case_id']} has a gold subset larger than max_order={max_order}"
        )

    matched = sample_minimal & gold
    extras = sample_minimal - gold
    return {
        "case_id": case["case_id"],
        "split": case["split"],
        "template_group_id": case["template_group_id"],
        "task_label": case["task_label"],
        "feature_count": case["feature_count"],
        "reference_count": case["reference_count"],
        "nuisance_profile": case["nuisance_profile"],
        "max_order": max_order,
        "gold_minimal_count": len(gold),
        "sample_absent_candidate_count": len(absent),
        "sample_minimal_candidate_count": len(sample_minimal),
        "gold_minimal_candidates_observed": len(matched),
        "gold_candidate_recall": len(matched) / len(gold) if gold else 0.0,
        "spurious_sample_minimal_count": len(extras),
        "has_spurious_sample_minimal": bool(extras),
        "sample_minimal_set_exact_match": sample_minimal == gold,
        "gold_minimal_subspaces": json.dumps([list(x) for x in sorted(gold)]),
        "sample_minimal_absent_subspaces": json.dumps([list(x) for x in sorted(sample_minimal)]),
    }


def summarize(rows: list[dict[str, Any]], repetitions: int, seed: int) -> dict[str, Any]:
    if not rows:
        return {"n_cases": 0}
    summary: dict[str, Any] = {"n_cases": len(rows)}
    for metric in (
        "sample_minimal_set_exact_match",
        "has_spurious_sample_minimal",
        "gold_candidate_recall",
        "spurious_sample_minimal_count",
        "sample_minimal_candidate_count",
    ):
        vals = [float(row[metric]) for row in rows]
        summary[metric] = sum(vals) / len(vals)
        boot = cluster_bootstrap(rows, metric, repetitions, seed)
        summary[f"{metric}_template_bootstrap_95ci"] = [
            percentile(boot, 0.025),
            percentile(boot, 0.975),
        ]
    summary["n_template_clusters"] = len({row["template_group_id"] for row in rows})
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--outdir", type=Path, default=HERE / "runs" / "reference_sample_audit")
    args = parser.parse_args()

    truth_path = HERE / "ground_truth.jsonl"
    cases = load_jsonl(truth_path)
    rows = [audit_case(case, args.max_order) for case in cases]
    test_rows = [row for row in rows if row["split"] == "test"]
    by_reference_count: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_width_reference_count: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in test_rows:
        by_reference_count[str(row["reference_count"])].append(row)
        key = f"{row['feature_count']}x{row['reference_count']}"
        by_width_reference_count[key].append(row)

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "per_case.csv", rows)
    write_csv(
        args.outdir / "test_by_reference_count.csv",
        [
            {"reference_count": count, **summarize(group, args.bootstrap, args.seed)}
            for count, group in sorted(by_reference_count.items(), key=lambda item: int(item[0]))
        ],
    )
    write_csv(
        args.outdir / "test_by_feature_width_and_reference_count.csv",
        [
            {
                "feature_count": int(key.split("x")[0]),
                "reference_count": int(key.split("x")[1]),
                **summarize(group, args.bootstrap, args.seed + 1),
            }
            for key, group in sorted(
                by_width_reference_count.items(),
                key=lambda item: tuple(map(int, item[0].split("x"))),
            )
        ],
    )
    report = {
        "audit": "empirical-reference absence versus complete-population support gold",
        "candidate_max_order": args.max_order,
        "sample_absent_definition": "target projection occurs in none of the provided normal reference rows",
        "minimality_definition": "no proper subset up to candidate_max_order is sample-absent",
        "warning": "Descriptive diagnostic only; it does not redefine labels or establish model identifiability.",
        "seed": args.seed,
        "bootstrap_repetitions": args.bootstrap,
        "source_sha256": {"ground_truth.jsonl": sha256(truth_path)},
        "all_splits": summarize(rows, args.bootstrap, args.seed),
        "test": summarize(test_rows, args.bootstrap, args.seed),
        "test_by_reference_count": {
            count: summarize(group, args.bootstrap, args.seed)
            for count, group in sorted(by_reference_count.items(), key=lambda item: int(item[0]))
        },
        "test_by_feature_width_and_reference_count": {
            key: summarize(group, args.bootstrap, args.seed + 1)
            for key, group in sorted(
                by_width_reference_count.items(),
                key=lambda item: tuple(map(int, item[0].split("x"))),
            )
        },
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
