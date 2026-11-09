#!/usr/bin/env python3
"""Run a frozen local LLM protocol on the sampled circuit pilot queries."""

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

import torch
import transformers

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TRACK_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))
from annotate import ANNOTATORS, append_jsonl, jsonl_read, load_model, model_file_manifest, render_prompt, utc_now

PROTOCOL_ID = "harvard_circuit_zero_noise_output_specific_pilot_v1"
MAX_INPUT_TOKENS = 16384
MAX_NEW_TOKENS = 512


def parse_output(raw: str, allowed: set[str]):
    candidate = raw.replace("Ġ", " ").replace("Ċ", "\n")
    start = candidate.find("{")
    if start < 0:
        return None, "no JSON object found"
    try:
        value, end = json.JSONDecoder().raw_decode(candidate[start:])
    except Exception as exc:
        return None, f"JSON decode failed: {type(exc).__name__}: {exc}"
    if not isinstance(value, dict) or set(value) != {"explanations", "abstain"}:
        return None, "required top-level keys must be explanations and abstain"
    if not isinstance(value["abstain"], bool):
        return None, "abstain must be boolean"
    explanations = value["explanations"]
    if not isinstance(explanations, list) or len(explanations) > 4:
        return None, "explanations must be a list with at most four candidates"
    for ix, item in enumerate(explanations):
        if not isinstance(item, dict) or set(item) != {"features", "evidence", "confidence"}:
            return None, f"explanation {ix} has invalid keys"
        features = item["features"]
        evidence = item["evidence"]
        confidence = item["confidence"]
        if not isinstance(features, list) or any(not isinstance(f, str) for f in features):
            return None, f"explanation {ix} features must be strings"
        if len(features) != len(set(features)) or set(features) - allowed:
            return None, f"explanation {ix} has duplicate or non-input feature names"
        if not isinstance(evidence, str) or len(evidence.split()) > 20:
            return None, f"explanation {ix} evidence must be a string of at most 20 words"
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            return None, f"explanation {ix} confidence must be numeric in [0,1]"
    if value["abstain"] and explanations:
        return None, "abstain=true requires no explanations"
    return value, None


def generate_one(model, tokenizer, case):
    rendered = render_prompt(tokenizer, case["system_prompt"], case["user_prompt"])
    tokens = tokenizer(rendered, return_tensors="pt", truncation=False)
    prompt_length = int(tokens["input_ids"].shape[1])
    if prompt_length > MAX_INPUT_TOKENS:
        raise ValueError(f"input has {prompt_length} tokens; cap is {MAX_INPUT_TOKENS}; not truncated")
    input_ids = tokens["input_ids"].to(model.device)
    attention = tokens.get("attention_mask")
    if attention is not None:
        attention = attention.to(model.device)
    eos = tokenizer.eos_token_id
    if isinstance(eos, list):
        eos = eos[0]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": pad,
        "eos_token_id": eos,
    }
    if attention is not None:
        kwargs["attention_mask"] = attention
    with torch.inference_mode():
        generated_ids = model.generate(**kwargs)[0, input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True), prompt_length, int(generated_ids.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen35_9b", "deepseek_v2_lite_chat", "llama31_8b_instruct"], required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inputs-path", type=Path, default=TRACK_ROOT / "inputs_zero_noise_pilot.jsonl")
    parser.add_argument("--annotations-path", type=Path, default=None)
    parser.add_argument("--run-log-path", type=Path, default=None)
    parser.add_argument("--protocol-id", default=PROTOCOL_ID)
    args = parser.parse_args()
    spec = ANNOTATORS[args.model]
    model_path = Path(spec["model_path"])
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory unavailable: {model_path}")
    input_path = args.inputs_path
    cases = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    annotation_path = args.annotations_path or TRACK_ROOT / f"annotations_{args.model}.jsonl"
    run_path = args.run_log_path or TRACK_ROOT / "model_runs.jsonl"
    prior = jsonl_read(annotation_path)
    completed = {(r.get("case_id"), r.get("prompt_sha256")) for r in prior if r.get("event") == "annotation"}
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    run_id = f"{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    metadata = {
        "event": "started", "protocol_id": args.protocol_id, "run_id": run_id,
        "annotator_id": args.model, "model_id": spec["model_id"], "model_family": spec["family"],
        "model_path": str(model_path), "model_type": config.get("model_type"),
        "checkpoint_transformers_version": config.get("transformers_version"),
        "inference_transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "decoding": {"temperature": 0.0, "do_sample": False, "num_beams": 1,
                     "max_input_tokens": MAX_INPUT_TOKENS, "max_new_tokens": MAX_NEW_TOKENS,
                     "input_truncation": False},
        "n_queries": len(cases), "inputs_path": str(input_path.resolve()),
        "inputs_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "gpu": {"device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "visible_devices": visible},
        "model_files": model_file_manifest(model_path), "started_at_utc": utc_now(),
    }
    append_jsonl(run_path, metadata)
    print(f"[{args.model}] loading {model_path}; queries={len(cases)}; GPUs={visible}", flush=True)
    try:
        model, tokenizer, loaded_config, dtype_name = load_model(model_path)
    except KeyboardInterrupt:
        append_jsonl(run_path, {
            **metadata,
            "event": "cancelled",
            "completed_at_utc": utc_now(),
            "annotation_records_written": 0,
            "reason": "Interrupted during checkpoint loading; no inference query was run.",
        })
        raise
    metadata["dtype"] = dtype_name
    metadata["loaded_model_type"] = loaded_config.model_type
    started = time.perf_counter()
    n_written = 0
    for ix, case in enumerate(cases, 1):
        key = (case["case_id"], case["prompt_sha256"])
        if args.resume and key in completed:
            continue
        raw = ""
        prompt_tokens = output_tokens = None
        parsed = error = None
        case_started = time.perf_counter()
        try:
            raw, prompt_tokens, output_tokens = generate_one(model, tokenizer, case)
            parsed, error = parse_output(raw, set(case["eligible_feature_ids"]))
        except Exception as exc:
            error = f"inference failed: {type(exc).__name__}: {exc}"
        append_jsonl(annotation_path, {
        "event": "annotation", "protocol_id": args.protocol_id, "run_id": run_id,
            "annotator_id": args.model, "model_id": spec["model_id"], "case_id": case["case_id"],
            "prompt_sha256": case["prompt_sha256"], "timestamp_utc": utc_now(),
            "temperature": 0.0, "do_sample": False, "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens, "latency_seconds": round(time.perf_counter()-case_started, 4),
            "raw_output": raw, "parsed_prediction": parsed, "parse_or_inference_error": error,
        })
        completed.add(key)
        n_written += 1
        if ix == 1 or ix % 10 == 0 or ix == len(cases):
            print(f"[{args.model}] {ix}/{len(cases)} {case['case_id']} valid={parsed is not None}", flush=True)
    metadata.update({"event": "completed", "completed_at_utc": utc_now(),
                     "elapsed_seconds": round(time.perf_counter()-started, 3),
                     "annotation_records_written": n_written})
    append_jsonl(run_path, metadata)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[{args.model}] completed; new records={n_written}", flush=True)


if __name__ == "__main__":
    main()
