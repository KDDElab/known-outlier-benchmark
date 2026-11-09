#!/usr/bin/env python3
"""Export predictions and audit confidence/evidence for all common-runtime seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (20261106, 20261107, 20261108)
MODELS = ("qwen35_9b", "deepseek_v2_lite_chat", "llama31_8b_instruct")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paths(seed: int, model: str) -> tuple[Path, Path, Path, str]:
    if seed == 20261106:
        root = HERE / "runs" / "multiseed" / "seed_20261106_rt522"
        inputs, truth = HERE / "inputs.jsonl", HERE / "ground_truth.jsonl"
        protocol = f"known_anomaly_subspace_v2_seed_{seed}_rt522"
    else:
        root = HERE / "runs" / "multiseed" / f"seed_{seed}"
        inputs, truth = root / "inputs.jsonl", root / "ground_truth.jsonl"
        protocol = f"known_anomaly_subspace_v2_seed_{seed}"
    return root / f"annotations_{model}.jsonl", inputs, truth, protocol


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=HERE, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--out", type=Path, default=HERE / "runs" / "multiseed_analysis" / "confidence_evidence.json")
    args = parser.parse_args()
    results = {}
    for seed in SEEDS:
        results[str(seed)] = {}
        for model in MODELS:
            annotations, inputs, truth, protocol = paths(seed, model)
            predictions = annotations.parent / f"predictions_{model}.jsonl"
            audit = annotations.parent / f"confidence_evidence_{model}.json"
            run([sys.executable, str(HERE / "export_predictions_v2.py"), str(annotations),
                 "--protocol", protocol, "--output", str(predictions)])
            run([sys.executable, str(HERE / "audit_confidence_evidence_v2.py"), str(predictions),
                 "--inputs-path", str(inputs), "--ground-truth-path", str(truth),
                 "--out", str(audit), "--bootstrap", str(args.bootstrap), "--seed", str(args.seed)])
            results[str(seed)][model] = {
                "audit_path": str(audit.resolve()),
                "annotation_sha256": sha256_file(annotations),
                "prediction_sha256": sha256_file(predictions),
                "audit_sha256": sha256_file(audit),
                "summary": json.loads(audit.read_text(encoding="utf-8")),
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seeds": list(SEEDS), "models": list(MODELS),
        "confidence_target": "exact support-minimal top-one retrieval among valid structured predictions",
        "confidence_metrics_conditioning": "Brier/ECE use cases with a valid top-one prediction and reported confidence; coverage is reported separately",
        "evidence_scope": "literal visible-input matching only; not semantic or causal faithfulness",
        "bootstrap_replicates_per_seed": args.bootstrap, "bootstrap_seed": args.seed,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "audits": results,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "seeds": list(SEEDS), "models": list(MODELS)}, indent=2))


if __name__ == "__main__":
    main()
