#!/usr/bin/env python3
"""Score M²OE as an unordered choice set against a matched three-candidate budget.

LLMs contribute their first three ranked candidates, treated as a set; M²OE
contributes all choices it emits up to three. This avoids assigning an ordering
that M²OE does not claim and matches the maximum candidate budget.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (20261106, 20261107, 20261108)
MODELS = ("qwen35_9b", "deepseek_v2_lite_chat", "llama31_8b_instruct")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paths(seed: int) -> tuple[Path, Path, Path, Path]:
    if seed == 20261106:
        root = HERE / "runs" / "multiseed" / "seed_20261106_rt522"
        return HERE / "inputs.jsonl", HERE / "ground_truth.jsonl", root, HERE / "runs" / "multiseed_analysis" / str(seed)
    root = HERE / "runs" / "multiseed" / f"seed_{seed}"
    return root / "inputs.jsonl", root / "ground_truth.jsonl", root, HERE / "runs" / "multiseed_analysis" / str(seed)


def write_truncated_set(source: Path, target: Path, limit: int = 2) -> dict:
    rows = read_jsonl(source)
    counts = []
    for row in rows:
        choices = row.get("explanations")
        if not isinstance(choices, list):
            choices = []
        counts.append(len(choices))
        row["explanations"] = choices[:limit]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return {"n_rows": len(rows), "n_unique_case_ids": len({row.get("case_id") for row in rows}),
            "choice_count_min": min(counts, default=0), "choice_count_max": max(counts, default=0),
            "choice_count_mean": sum(counts) / len(counts) if counts else 0.0,
            "source_sha256": sha256(source), "truncated_sha256": sha256(target)}


def evaluate(prediction: Path, inputs: Path, truth: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "evaluator.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run([
            sys.executable, str(HERE / "evaluate_benchmark.py"), str(prediction),
            "--inputs-path", str(inputs), "--ground-truth-path", str(truth), "--outdir", str(outdir),
        ], stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Matched-budget evaluation failed for {prediction}; inspect {outdir / 'evaluator.log'}")


def main() -> None:
    manifest = {
        "candidate_budget": 3,
        "comparison": "unordered set of the LLM's first three ranked candidates versus all M²OE choices up to three",
        "M2OE_ordering_assumption": "none; choices are evaluated without ranking",
        "label_use": "test labels are used only by the evaluator, not by truncation or candidate selection",
        "per_seed": {},
    }
    for seed in SEEDS:
        inputs, truth, root, eval_root = paths(seed)
        expected = {row["case_id"] for row in read_jsonl(inputs) if row.get("split") == "test"}
        sources = {
            model: eval_root / f"predictions_{model}.jsonl"
            for model in MODELS
        }
        m2oe_source = ((HERE / "runs" / "m2oe" / "predictions_m2oe_choice.jsonl")
                       if seed == 20261106 else root / "m2oe" / "predictions_m2oe_choice.jsonl")
        sources["m2oe_choice"] = m2oe_source
        seed_record = {"inputs_sha256": sha256(inputs), "ground_truth_sha256": sha256(truth), "methods": {}}
        for method, source in sources.items():
            if not source.is_file():
                raise FileNotFoundError(source)
            rows = read_jsonl(source)
            if {row.get("case_id") for row in rows} != expected or len(rows) != len(expected):
                raise ValueError(f"{seed}/{method} predictions do not exactly cover the expected test cases")
            truncated = eval_root / f"{method}_matched3" / f"predictions_{method}_matched3.jsonl"
            counts = write_truncated_set(source, truncated, limit=3)
            evaluate(truncated, inputs, truth, truncated.parent)
            seed_record["methods"][method] = counts
        manifest["per_seed"][str(seed)] = seed_record
    out = HERE / "runs" / "multiseed_analysis" / "matched3_baseline_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest["script_sha256"] = sha256(Path(__file__).resolve())
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(out), "candidate_budget": 3, "seeds": list(SEEDS)}, indent=2))


if __name__ == "__main__":
    main()
