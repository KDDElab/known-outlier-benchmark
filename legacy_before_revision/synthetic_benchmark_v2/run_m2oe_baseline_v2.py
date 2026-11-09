#!/usr/bin/env python3
"""Run the official M²OE single-outlier explainer on a numeric encoding.

The source repository accepts continuous tables only. This adapter uses
reference-fitted min-max encoding for numeric/ordinal/date values and one-hot
encoding for categorical/binary/identifier values, then maps selected encoded
dimensions back to original field IDs. It evaluates M²OE's emitted choice
subspaces; it does not score its counterfactual patches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_payload(prompt: str) -> dict:
    marker = '"feature_schema"'
    index = prompt.rfind(marker)
    if index < 0:
        raise ValueError("feature_schema is missing")
    start = prompt.rfind("{", 0, index)
    return json.JSONDecoder().raw_decode(prompt[start:])[0]


def numeric_value(value: object, value_type: str) -> float | None:
    if value is None:
        return None
    if value_type == "datetime":
        try:
            return float(np.datetime64(str(value), "D").astype("int64"))
        except (TypeError, ValueError):
            return None
    if value_type in {"numeric", "ordinal", "binary"} and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def continuous_encoding(payload: dict) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Return target, normals, and encoded-column groups for each source field."""
    schema = payload["feature_schema"]
    target = payload["target_row"]
    references = payload["normal_reference_rows"]
    target_columns: list[float] = []
    reference_columns: list[list[float]] = [[] for _ in references]
    groups: list[list[int]] = []

    for item in schema:
        fid, kind = item["feature_id"], item["value_type"]
        values = [target.get(fid)] + [row.get(fid) for row in references]
        nums = [numeric_value(value, kind) for value in values]
        start = len(target_columns)
        if kind in {"numeric", "ordinal", "datetime"} and all(value is not None for value in nums):
            numeric = np.asarray(nums, dtype=np.float64)
            scale = float(np.max(numeric[1:]) - np.min(numeric[1:])) if len(numeric) > 1 else 0.0
            if scale <= 0:
                normalized = np.zeros_like(numeric)
            else:
                minimum = float(np.min(numeric[1:]))
                normalized = np.clip((numeric - minimum) / scale, 0.0, 1.0)
            target_columns.append(float(normalized[0]))
            for idx in range(len(references)):
                reference_columns[idx].append(float(normalized[idx + 1]))
        elif kind in {"numeric", "ordinal", "datetime"}:
            # Mixed missingness: preserve the missing state explicitly and use
            # reference-fitted values for observed entries.
            observed = [value for value in nums[1:] if value is not None]
            low, high = (min(observed), max(observed)) if observed else (0.0, 0.0)
            scale = high - low
            def norm(value: float | None) -> float:
                return 0.0 if value is None or scale <= 0 else float(np.clip((value - low) / scale, 0.0, 1.0))
            def miss(value: float | None) -> float:
                return float(value is None)
            target_columns.extend((norm(nums[0]), miss(nums[0])))
            for idx in range(len(references)):
                reference_columns[idx].extend((norm(nums[idx + 1]), miss(nums[idx + 1])))
        else:
            domain = list(item.get("allowed_values", []))
            for value in values:
                if value is not None and value not in domain:
                    domain.append(value)
            # Stable ordering across Python versions and mixed value types.
            domain = sorted(domain, key=lambda value: (type(value).__name__, str(value)))
            domain.append("__MISSING__")
            encoded = []
            for value in values:
                key = "__MISSING__" if value is None else value
                encoded.append([float(key == category) for category in domain])
            target_columns.extend(encoded[0])
            for idx in range(len(references)):
                reference_columns[idx].extend(encoded[idx + 1])
        groups.append(list(range(start, len(target_columns))))

    return (
        np.asarray(target_columns, dtype=np.float32).reshape(1, -1),
        np.asarray(reference_columns, dtype=np.float32),
        groups,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-path", type=Path, default=HERE / "inputs.jsonl")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all selected cases")
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=30)
    args = parser.parse_args()

    # TensorFlow/M²OE is an optional isolated dependency; import after CLI parse.
    import tensorflow as tf
    import sklearn
    import mlxtend
    from M2OE.explainers.TabularExplainer import TabularExplainer

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    cases = [row for row in read_jsonl(args.inputs_path) if row["split"] == args.split]
    if args.max_cases:
        cases = cases[:args.max_cases]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    log_path = args.output_path.with_suffix(".log.jsonl")
    for index, case in enumerate(cases, start=1):
        case_seed = args.seed + sum(case["case_id"].encode("utf-8"))
        random.seed(case_seed)
        np.random.seed(case_seed % (2**32 - 1))
        tf.keras.utils.set_random_seed(case_seed)
        row = {"case_id": case["case_id"], "task_label": "unknown",
               "mechanism_feature_ids": [], "mechanism_metric_applicable": False,
               "explanations": [], "abstain": False}
        case_start = time.perf_counter()
        error = None
        try:
            payload = extract_payload(case["user_prompt"])
            target, normals, feature_groups = continuous_encoding(payload)
            n_neigh = min(args.neighbors, len(normals))
            if n_neigh < 2:
                raise ValueError("M²OE requires at least two normal references")
            explainer = TabularExplainer(
                loss_weights=[1.0, 1.2, 0.3], lr=0.001,
                epochs=args.epochs, bs=args.batch_size, threshold=0.5,
            )
            explanations = explainer.compute_explanation(
                target, normals, n_neigh=n_neigh, num_tries=3,
            )
            seen: set[tuple[str, ...]] = set()
            for encoded_dims, _patch in explanations:
                chosen = set()
                for dimension in np.asarray(encoded_dims, dtype=int).reshape(-1):
                    # Map each selected encoded column back to its source field;
                    # one-hot expansions can make encoded indices exceed D.
                    for item, dims in zip(payload["feature_schema"], feature_groups):
                        if dimension in dims:
                            chosen.add(item["feature_id"])
                            break
                candidate = tuple(sorted(chosen))
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                row["explanations"].append({
                    "features": list(candidate),
                    "evidence": "M²OE choice subspace mapped from continuous encoded dimensions.",
                    "confidence": 0.5,
                })
                if len(row["explanations"]) == 4:
                    break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"case_id": case["case_id"], "seed": case_seed,
                "elapsed_seconds": time.perf_counter() - case_start, "error": error,
                "n_explanations": len(row["explanations"])}, ensure_ascii=False) + "\n")
        if index == 1 or index % 10 == 0 or index == len(cases):
            print(f"M2OE {index}/{len(cases)} case={case['case_id']} error={error}", flush=True)

    args.output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "method": "M2OE official TabularExplainer, choice outputs only",
        "inputs_path": str(args.inputs_path.resolve()),
        "n_cases": len(cases), "split": args.split, "seed_base": args.seed,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "neighbors": args.neighbors, "elapsed_seconds": time.perf_counter() - started,
        "encoding": "numeric/ordinal/date reference-fitted min-max; categorical/binary/identifier one-hot; explicit missing indicator",
        "ranking_note": "M²OE does not define a candidate ranking score; preserves official returned order and scores Hit@4/recall only descriptively.",
        "source_repository_revision": "925c96ec0abca1e4cfd934d486f1217623184c7f",
        "environment": {"python": platform.python_version(), "tensorflow": tf.__version__,
                        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                        "mlxtend": mlxtend.__version__},
        "input_sha256": hashlib.sha256(args.inputs_path.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(args.output_path.read_bytes()).hexdigest(),
        "official_source_compatibility_patch": "Removed four unused tensorflow.python.keras.layers imports (ReLU, BatchNormalization) and tuple-wrapped four Keras Input shapes for TF 2.20/Keras 3; API compatibility only, no algorithm code changed.",
    }
    args.output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
