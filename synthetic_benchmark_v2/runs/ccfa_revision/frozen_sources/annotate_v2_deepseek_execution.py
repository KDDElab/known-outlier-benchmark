#!/usr/bin/env python3
"""Run one configured local LLM on the v2 held-out benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
sys.path.insert(0, str(PACKAGE_ROOT))

import torch
import transformers
from annotate import (
    ANNOTATORS,
    append_jsonl,
    jsonl_read,
    load_model,
    model_file_manifest,
    render_prompt,
    utc_now,
)

PROTOCOL_ID = "known_anomaly_subspace_benchmark_v2_compact_json_v1"
MAX_INPUT_TOKENS = 16384
MAX_NEW_TOKENS = 512
TASK_LABELS = {
    "numeric_marginal",
    "categorical_rare",
    "pairwise_interaction",
    "conditional_group",
    "multiple_valid",
    "modified_vs_mechanism",
    "relational",
    "higher_order",
}


def parse_prediction(raw_text: str, feature_ids: set[str], output_contract: str = "full"):
    parse_text = raw_text.replace("Ġ", " ").replace("Ċ", "\n")
    start = parse_text.find("{")
    if start < 0:
        return None, "no JSON object found"
    try:
        value, _ = json.JSONDecoder().raw_decode(parse_text[start:])
    except Exception as exc:
        return None, f"JSON decode failed: {type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return None, "top-level value must be an object"
    if output_contract == "minimal":
        if not {"explanations", "abstain"}.issubset(value):
            return None, "minimal output requires explanations and abstain"
        if not isinstance(value["abstain"], bool) or not isinstance(value["explanations"], list) or len(value["explanations"]) > 4:
            return None, "invalid minimal output structure"
        normalized = []
        for item in value["explanations"]:
            if not isinstance(item, dict) or "features" not in item:
                return None, "minimal candidate requires features"
            fields = item["features"]
            if (not isinstance(fields, list) or any(not isinstance(v, str) for v in fields)
                or len(fields) != len(set(fields)) or set(fields) - feature_ids):
                return None, "invalid minimal candidate fields"
            normalized.append({"features": fields, "evidence": "", "confidence": 0.0})
        # Auxiliary placeholders exist only for the common evaluator; they
        # are explicitly excluded from mechanism, task and calibration scores.
        return {"task_label": "unknown", "mechanism_feature_ids": [],
                "mechanism_metric_applicable": False, "confidence_metric_applicable": False,
                "explanations": normalized, "abstain": value["abstain"]}, None
    required = {"task_label", "mechanism_feature_ids", "explanations", "abstain"}
    if not required.issubset(value):
        return None, f"missing required keys: {sorted(required - set(value))}"
    if value["task_label"] not in TASK_LABELS:
        return None, "unsupported task_label"
    mechanism = value["mechanism_feature_ids"]
    if not isinstance(mechanism, list) or any(not isinstance(item, str) for item in mechanism):
        return None, "mechanism_feature_ids must be a string array"
    if len(mechanism) != len(set(mechanism)):
        return None, "mechanism_feature_ids contains duplicates"
    if set(mechanism) - feature_ids:
        return None, f"unknown mechanism feature IDs: {sorted(set(mechanism) - feature_ids)}"
    explanations = value["explanations"]
    if not isinstance(explanations, list) or len(explanations) > 4:
        return None, "explanations must be an array with at most four candidates"
    for index, item in enumerate(explanations):
        if not isinstance(item, dict):
            return None, f"explanation {index} must be an object"
        features = item.get("features")
        evidence = item.get("evidence")
        confidence = item.get("confidence")
        if not isinstance(features, list) or any(not isinstance(feature, str) for feature in features):
            return None, f"explanation {index} features must be a string array"
        if len(features) != len(set(features)):
            return None, f"explanation {index} contains duplicate feature IDs"
        if set(features) - feature_ids:
            return None, f"explanation {index} has unknown feature IDs"
        if not isinstance(evidence, str):
            return None, f"explanation {index} evidence must be a string"
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return None, f"explanation {index} confidence must be numeric"
        if not 0 <= float(confidence) <= 1:
            return None, f"explanation {index} confidence is outside [0,1]"
    if not isinstance(value["abstain"], bool):
        return None, "abstain must be boolean"
    return value, None


def generate_one(model, tokenizer, case: dict):
    rendered = render_prompt(tokenizer, case["system_prompt"], case["user_prompt"])
    tokens = tokenizer(rendered, return_tensors="pt", truncation=False)
    prompt_length = int(tokens["input_ids"].shape[1])
    if prompt_length > MAX_INPUT_TOKENS:
        raise ValueError(
            f"rendered input has {prompt_length} tokens, exceeding {MAX_INPUT_TOKENS}; "
            "input was not truncated"
        )
    input_ids = tokens["input_ids"].to(model.device)
    attention_mask = tokens.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": int(case.get("max_new_tokens", MAX_NEW_TOKENS)),
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    with torch.inference_mode():
        output_ids = model.generate(**kwargs)
    generated = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, prompt_length, int(generated.shape[0])


def find_model_manifest(annotator_id: str, model_path: Path) -> dict:
    candidates = [HERE / "runs" / "model_runs.jsonl", PACKAGE_ROOT / "runs" / "model_runs.jsonl"]
    cached = []
    for path in candidates:
        cached.extend(jsonl_read(path))
    for row in reversed(cached):
        if row.get("model_path") == str(model_path) and isinstance(row.get("model_files"), dict):
            return row["model_files"]
    return model_file_manifest(model_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[
        "qwen35_9b", "phi4_mini", "gemma3_4b",
        "deepseek_v2_lite_chat", "llama31_8b_instruct",
    ], required=True)
    parser.add_argument("--split", choices=["test", "all"], default="test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inputs-path", type=Path, default=HERE / "inputs.jsonl")
    parser.add_argument("--annotations-path", type=Path, default=None)
    parser.add_argument("--run-log-path", type=Path, default=HERE / "runs" / "model_runs.jsonl")
    parser.add_argument("--protocol-id", default=PROTOCOL_ID)
    args = parser.parse_args()

    spec = ANNOTATORS[args.model]
    model_path = Path(spec["model_path"])
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory unavailable: {model_path}")
    inputs_path = args.inputs_path
    cases = [json.loads(line) for line in inputs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.split == "test":
        cases = [case for case in cases if case["split"] == "test"]

    run_path = args.run_log_path
    run_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path = args.annotations_path or HERE / "runs" / f"annotations_{args.model}.jsonl"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    existing = jsonl_read(annotation_path)
    completed = {
        (row.get("case_id"), row.get("prompt_sha256"))
        for row in existing
        if row.get("event") == "annotation" and row.get("protocol_id") == args.protocol_id
    }

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    run_id = f"{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    metadata = {
        "event": "started",
        "protocol_id": args.protocol_id,
        "run_id": run_id,
        "annotator_id": args.model,
        "model_id": spec["model_id"],
        "model_family": spec["family"],
        "model_path": str(model_path),
        "model_type": config.get("model_type"),
        "runner_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_loader_sha256": hashlib.sha256((PACKAGE_ROOT / "annotate.py").read_bytes()).hexdigest(),
        "inputs_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
        "per_case_output_settings": sorted({(c.get("output_contract", "full"), c.get("max_new_tokens", MAX_NEW_TOKENS)) for c in cases}),
        "checkpoint_transformers_version": config.get("transformers_version"),
        "inference_transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "decoding": {
            "temperature": 0.0,
            "do_sample": False,
            "num_beams": 1,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "input_truncation": False,
        },
        "split": args.split,
        "n_cases": len(cases),
        "gpu": {
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "visible_devices": visible,
        },
        "model_files": find_model_manifest(args.model, model_path),
        "started_at_utc": utc_now(),
    }
    append_jsonl(run_path, metadata)
    print(f"[{args.model}] loading {model_path}; cases={len(cases)}; GPUs={visible}", flush=True)
    try:
        model, tokenizer, loaded_config, dtype_name = load_model(model_path)
    except Exception as exc:
        metadata.update({
            "event": "load_failed",
            "failed_at_utc": utc_now(),
            "load_error": f"{type(exc).__name__}: {exc}",
        })
        append_jsonl(run_path, metadata)
        raise
    metadata["dtype"] = dtype_name
    metadata["loaded_model_type"] = loaded_config.model_type
    metadata["device_map"] = {str(k): str(v) for k, v in getattr(model, "hf_device_map", {}).items()}
    metadata["gpu_max_memory_gib"] = os.environ.get("CCFA_GPU_MAX_MEMORY_GIB")
    if any(str(v) in {"cpu", "disk"} for v in getattr(model, "hf_device_map", {}).values()) and torch.cuda.is_available():
        raise RuntimeError("Insufficient GPU capacity: CPU/disk offload was not allowed for these experiments")
    run_started = time.perf_counter()
    written = 0
    for index, case in enumerate(cases, start=1):
        key = (case["case_id"], case["prompt_sha256"])
        if args.resume and key in completed:
            continue
        raw_text = ""
        prompt_tokens = None
        output_tokens = None
        parsed = None
        error = None
        started = time.perf_counter()
        feature_ids = set(case["feature_ids"])
        try:
            raw_text, prompt_tokens, output_tokens = generate_one(model, tokenizer, case)
            parsed, error = parse_prediction(raw_text, feature_ids, case.get("output_contract", "full"))
        except Exception as exc:
            error = f"inference failed: {type(exc).__name__}: {exc}"
        record = {
            "event": "annotation",
            "protocol_id": args.protocol_id,
            "run_id": run_id,
            "annotator_id": args.model,
            "model_id": spec["model_id"],
            "model_family": spec["family"],
            "case_id": case["case_id"],
            "split": case["split"],
            "prompt_sha256": case["prompt_sha256"],
            "timestamp_utc": utc_now(),
            "temperature": 0.0,
            "do_sample": False,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_new_tokens": int(case.get("max_new_tokens", MAX_NEW_TOKENS)),
            "output_contract": case.get("output_contract", "full"),
            "output_limit_reached": output_tokens == int(case.get("max_new_tokens", MAX_NEW_TOKENS)),
            "source_case_id": case.get("source_case_id", case["case_id"]),
            "condition": case.get("condition"),
            "generator_seed": case.get("generator_seed"),
            "display_order_seed": case.get("display_order_seed"),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "latency_seconds": round(time.perf_counter() - started, 4),
            "raw_output": raw_text,
            "parsed_prediction": parsed,
            "parse_or_inference_error": error,
        }
        append_jsonl(annotation_path, record)
        if error and "out of memory" in error.lower():
            raise RuntimeError("GPU out of memory; stopped the run to repair the resource allocation")
        completed.add(key)
        written += 1
        if index == 1 or index % 10 == 0 or index == len(cases):
            print(
                f"[{args.model}] {index}/{len(cases)} case={case['case_id']} "
                f"parse={'ok' if parsed is not None else 'failed'}",
                flush=True,
            )

    metadata.update({
        "event": "completed",
        "completed_at_utc": utc_now(),
        "elapsed_seconds": round(time.perf_counter() - run_started, 3),
        "annotation_records_written": written,
        "total_annotation_records_in_file": len(jsonl_read(annotation_path)),
    })
    append_jsonl(run_path, metadata)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[{args.model}] completed; new records={written}", flush=True)


if __name__ == "__main__":
    main()
