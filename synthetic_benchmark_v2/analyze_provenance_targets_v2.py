#!/usr/bin/env python3
"""Audit hidden edit-provenance labels separately from explanation targets.

This diagnostic asks whether predictions happen to recover directly edited
fields or all fields whose values changed from the hidden pre-injection row.
Those labels are deliberately absent from the main prompt, so this is not a
primary-task accuracy measure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEEDS = (20261106, 20261107, 20261108)
METHODS = (
    "qwen35_9b", "deepseek_v2_lite_chat", "llama31_8b_instruct",
    "gower_knn_subspace", "gower_lof_subspace", "top_univariate",
    "sod_subspace_adaptation", "cop_error_vector_adaptation",
    "rlax_reference_adaptation", "m2oe_choice", "all_features",
    "random_size_prior",
)
METRICS = ("modified_exact_at_1", "modified_exact_hit_at_4",
           "direct_field_recall_at_4", "changed_value_field_recall_at_4")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_candidates(value: object) -> list[frozenset[str]]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [frozenset(x) for x in parsed if isinstance(x, list) and all(isinstance(v, str) for v in x)]


def score(candidates: list[frozenset[str]], direct: frozenset[str], changed: frozenset[str]) -> dict[str, float]:
    top = candidates[:1]
    top4 = candidates[:4]
    union4 = frozenset().union(*top4) if top4 else frozenset()
    return {
        "modified_exact_at_1": float(bool(top) and top[0] == direct),
        "modified_exact_hit_at_4": float(any(candidate == direct for candidate in top4)),
        "direct_field_recall_at_4": len(union4 & direct) / len(direct) if direct else float("nan"),
        "changed_value_field_recall_at_4": len(union4 & changed) / len(changed) if changed else float("nan"),
    }


def seed_paths(seed: int) -> tuple[Path, Path, Path]:
    if seed == 20261106:
        root = HERE / "runs" / "multiseed" / "seed_20261106_rt522"
        return HERE / "inputs.jsonl", HERE / "ground_truth.jsonl", HERE / "runs" / "multiseed_analysis" / str(seed)
    root = HERE / "runs" / "multiseed" / f"seed_{seed}"
    return root / "inputs.jsonl", root / "ground_truth.jsonl", HERE / "runs" / "multiseed_analysis" / str(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--outdir", type=Path, default=HERE / "runs" / "multiseed_analysis" / "provenance")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    per_case_rows: list[dict] = []
    input_hashes: dict[str, str] = {}
    truth_hashes: dict[str, str] = {}
    prediction_hashes: dict[str, str] = {}

    for seed in SEEDS:
        inputs, truth_path, eval_root = seed_paths(seed)
        truth_rows = {row["case_id"]: row for row in read_jsonl(truth_path) if row.get("split") == "test"}
        test_ids = set(truth_rows)
        input_hashes[str(seed)] = sha256(inputs)
        truth_hashes[str(seed)] = sha256(truth_path)

        for method in METHODS:
            if method == "random_size_prior":
                baseline_dir = (HERE / "runs" / "baseline_predictions") if seed == 20261106 else HERE / "runs" / "multiseed" / f"seed_{seed}" / "baselines"
                paths = sorted(baseline_dir.glob("predictions_random_*.jsonl"))
                if len(paths) != 100:
                    raise RuntimeError(f"Expected 100 random replicates for {seed}; found {len(paths)}")
                accumulated: dict[str, list[dict[str, float]]] = {case_id: [] for case_id in test_ids}
                for prediction_path in paths:
                    prediction_hashes[f"{seed}/{prediction_path.name}"] = sha256(prediction_path)
                    preds = {row["case_id"]: row for row in read_jsonl(prediction_path)}
                    if set(preds) != test_ids:
                        raise RuntimeError(f"Random predictions do not cover test cases: {prediction_path}")
                    for case_id in test_ids:
                        pred = preds[case_id]
                        candidates = [frozenset(x.get("features", [])) for x in pred.get("explanations", []) if isinstance(x, dict)]
                        truth = truth_rows[case_id]
                        accumulated[case_id].append(score(candidates, frozenset(truth["modified_feature_ids"]), frozenset(truth["value_different_from_baseline_feature_ids"])))
                case_scores = {case_id: {metric: float(np.mean([x[metric] for x in rows])) for metric in METRICS} for case_id, rows in accumulated.items()}
            else:
                prediction_path = eval_root / method / "per_case.csv"
                if not prediction_path.is_file():
                    raise FileNotFoundError(prediction_path)
                prediction_hashes[f"{seed}/{method}"] = sha256(prediction_path)
                frame = pd.read_csv(prediction_path).query("split == 'test'")
                if set(frame.case_id) != test_ids or frame.case_id.duplicated().any():
                    raise RuntimeError(f"Per-case predictions do not exactly cover test cases: {prediction_path}")
                case_scores = {}
                for row in frame.to_dict("records"):
                    truth = truth_rows[row["case_id"]]
                    candidates = parse_candidates(row.get("explanations_predicted"))
                    case_scores[row["case_id"]] = score(candidates, frozenset(truth["modified_feature_ids"]), frozenset(truth["value_different_from_baseline_feature_ids"]))

            for case_id, values in case_scores.items():
                per_case_rows.append({"seed": seed, "method": method, "case_id": case_id,
                                      "template_id": truth_rows[case_id]["template_id"], **values})

    per_case = pd.DataFrame(per_case_rows)
    per_case_path = args.outdir / "provenance_per_case.csv"
    per_case.to_csv(per_case_path, index=False)
    summaries = []
    for method in METHODS:
        for metric in METRICS:
            if method == "m2oe_choice" and metric == "modified_exact_at_1":
                # M²OE does not define a ranking over its choices.
                continue
            seed_values = {}
            seed_clusters: dict[int, dict[str, np.ndarray]] = {}
            for seed in SEEDS:
                frame = per_case.query("seed == @seed and method == @method")
                seed_values[str(seed)] = float(frame[metric].mean())
                seed_clusters[seed] = {template: group[metric].to_numpy(dtype=float) for template, group in frame.groupby("template_id")}
            draws = []
            for _ in range(args.bootstrap):
                means = []
                for seed in SEEDS:
                    clusters = seed_clusters[seed]
                    names = list(clusters)
                    chosen = rng.choice(names, size=len(names), replace=True)
                    values = np.concatenate([clusters[name] for name in chosen])
                    means.append(float(values.mean()))
                draws.append(float(np.mean(means)))
            summaries.append({"method": method, "metric": metric, "n_seed_replicates": len(SEEDS),
                              "per_seed_values": json.dumps(seed_values, sort_keys=True),
                              "mean_over_seeds": float(np.mean(list(seed_values.values()))),
                              "sd_over_seeds": float(np.std(list(seed_values.values()), ddof=1)),
                              "template_cluster_bootstrap_95_ci_macro_seed": json.dumps([float(np.quantile(draws, .025)), float(np.quantile(draws, .975))])})
    summary_path = args.outdir / "provenance_metrics.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    manifest = {
        "design": "Hidden direct-edit and value-changed provenance targets are scored separately from rule and complete-population support targets. The pre-injection row and provenance labels are not shown to methods.",
        "seeds": SEEDS,
        "methods": METHODS,
        "metrics": METRICS,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "interpretation_limit": "This is a descriptive hidden-provenance audit, not the primary explanation task: direct edits are unidentifiable from the unpaired visible prompt by construction. Field recall uses the union of up to four output subsets; M2OE choices are treated as an unordered set.",
        "sha256": {"inputs": input_hashes, "ground_truth": truth_hashes, "predictions": prediction_hashes,
                   "per_case_csv": sha256(per_case_path), "analysis_script": sha256(Path(__file__))},
    }
    manifest_path = args.outdir / "provenance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"per_case": str(per_case_path), "metrics": str(summary_path), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
