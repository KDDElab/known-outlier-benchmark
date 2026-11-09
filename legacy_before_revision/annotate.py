#!/usr/bin/env python3
"""Run one or more local Hugging Face annotators and preserve raw responses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
ANNOTATORS = {a["annotator_id"]: a for a in CONFIG["annotators"]}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def parse_annotation(raw_text: str, allowed_features: set[str]) -> tuple[dict | None, str | None]:
    normalization_notes = []
    if "Ġ" in raw_text or "Ċ" in raw_text:
        raw_text = raw_text.replace("Ġ", " ").replace("Ċ", "\n")
        normalization_notes.append(
            "tokenizer whitespace markers Ġ/Ċ normalized to spaces/newlines before JSON parsing"
        )
    decoder = json.JSONDecoder()
    start = raw_text.find("{")
    if start < 0:
        return None, "no JSON object found"
    try:
        parsed, _ = decoder.raw_decode(raw_text[start:])
    except Exception as exc:
        return None, f"JSON decode failed: {type(exc).__name__}: {exc}"
    if not isinstance(parsed, dict):
        return None, "top-level response must be a JSON object"

    features = parsed.get("feature_ids_ranked")
    if not isinstance(features, list):
        return None, "feature_ids_ranked must be a list"
    max_features = int(CONFIG.get("max_features", 3))
    if len(features) > max_features:
        return None, f"feature_ids_ranked exceeds the {max_features}-feature limit"
    if any(not isinstance(feature, str) for feature in features):
        return None, "all feature IDs must be strings"
    if len(features) != len(set(features)):
        return None, "feature_ids_ranked contains duplicates"
    unknown = sorted(set(features) - allowed_features)
    if unknown:
        return None, f"unknown feature IDs: {unknown}"

    abstain = parsed.get("abstain")
    if not isinstance(abstain, bool):
        return None, "abstain must be a boolean"
    if abstain and features:
        return None, "abstain=true requires an empty feature list"
    if not abstain and not features:
        return None, "non-abstaining response must include at least one feature"

    confidence = parsed.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None, "confidence must be numeric"
    if not 0.0 <= float(confidence) <= 1.0:
        return None, "confidence must be between zero and one"
    directions = parsed.get("deviation_direction", {})
    if not isinstance(directions, dict):
        return None, "deviation_direction must be an object"
    allowed_directions = {
        "above", "below", "rare_category", "unusual_combination", "unclear"
    }
    for feature_id, value in list(directions.items()):
        if isinstance(value, str) and "|" in value:
            options = [part.strip() for part in value.split("|")]
            if len(options) > 1 and all(option in allowed_directions for option in options):
                directions[feature_id] = "unclear"
                normalization_notes.append(
                    f"{feature_id}: ambiguous pipe-separated direction mapped to unclear"
                )
    if normalization_notes:
        parsed["deviation_direction"] = directions
        existing_notes = parsed.get("parse_normalization_notes", [])
        if not isinstance(existing_notes, list):
            existing_notes = []
        parsed["parse_normalization_notes"] = existing_notes + normalization_notes
    if any(
        not isinstance(value, str) or value not in allowed_directions
        for value in directions.values()
    ):
        return None, "deviation_direction contains an unsupported value"
    evidence = parsed.get("evidence", [])
    if not isinstance(evidence, list):
        return None, "evidence must be a list"
    return parsed, None


def model_file_manifest(model_path: Path) -> dict:
    config_file = model_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing model config: {config_file}")
    weight_files = sorted(
        p for p in model_path.iterdir()
        if p.is_file() and (
            p.name.endswith(".safetensors")
            or p.name.endswith(".bin")
        )
    )
    tokenizer_files = sorted(
        p for p in model_path.iterdir()
        if p.is_file() and p.name in {
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
        }
    )
    return {
        "config": {
            "name": config_file.name,
            "bytes": config_file.stat().st_size,
            "sha256": sha256_file(config_file),
        },
        "weights": [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in weight_files
        ],
        "tokenizer_files": [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in tokenizer_files
        ],
    }


def load_model(model_path: Path):
    config_hint = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    # The local DeepSeek-V2 custom code targets an older Transformers API;
    # this runtime provides the model natively, so load its native classes.
    use_remote_code = config_hint.get("model_type") != "deepseek_v2"
    model_config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=use_remote_code, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=use_remote_code, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model_class = (
        AutoModelForImageTextToText
        if model_config.model_type in {"qwen3_5", "gemma3"}
        else AutoModelForCausalLM
    )
    # The local Phi checkpoint ships a Transformers 4.45 custom modeling file
    # that is incompatible with the current Transformers 5.x runtime. Its phi3
    # architecture is built in, so use the native implementation for this family.
    trust_model_code = model_config.model_type not in {
        "phi3", "gemma3", "deepseek_v2"
    }
    model = model_class.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=trust_model_code,
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer, model_config, str(dtype)


def render_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"


def generate_one(model, tokenizer, case: dict) -> tuple[str, int]:
    rendered = render_prompt(tokenizer, case["system_prompt"], case["user_prompt"])
    maximum = int(CONFIG["decoding"]["max_input_tokens"])
    tokens = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=maximum,
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
    generation = {
        "input_ids": input_ids,
        "max_new_tokens": int(CONFIG["decoding"]["max_new_tokens"]),
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }
    if attention_mask is not None:
        generation["attention_mask"] = attention_mask
    with torch.inference_mode():
        output_ids = model.generate(**generation)
    generated = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), int(input_ids.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(ANNOTATORS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--annotation-output",
        default="annotations.jsonl",
        help="JSONL filename in runs/. Use a separate file when running models concurrently.",
    )
    args = parser.parse_args()
    if not args.all and not args.model:
        parser.error("choose --model annotator_id or --all")
    chosen_ids = sorted(ANNOTATORS) if args.all else [args.model]

    payload_path = ROOT / "runs" / "input_payloads.jsonl"
    if not payload_path.exists():
        raise FileNotFoundError("Run prepare_data.py before annotation.")
    payloads = jsonl_read(payload_path)
    annotation_name = Path(args.annotation_output)
    if annotation_name.name != str(annotation_name) or not annotation_name.name.startswith("annotations"):
        parser.error("--annotation-output must be a filename beginning with annotations")
    annotation_path = ROOT / "runs" / annotation_name
    run_path = ROOT / "runs" / "model_runs.jsonl"
    existing = jsonl_read(annotation_path)
    completed = {
        (row.get("annotator_id"), row.get("case_id"), row.get("prompt_sha256"))
        for row in existing
        if row.get("event") == "annotation"
        and not row.get("parse_or_inference_error")
        and isinstance(row.get("parsed_annotation"), dict)
    }

    for annotator_id in chosen_ids:
        spec = ANNOTATORS[annotator_id]
        model_path = Path(spec["model_path"])
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Model directory is unavailable for {annotator_id}: {model_path}"
            )
        run_id = f"{annotator_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        model_files = model_file_manifest(model_path)
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        gpu = None
        if torch.cuda.is_available():
            gpu = {
                "name": torch.cuda.get_device_name(torch.cuda.current_device()),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        metadata = {
            "event": "started",
            "run_id": run_id,
            "annotator_id": annotator_id,
            "model_id": spec["model_id"],
            "model_family": spec["family"],
            "model_path": str(model_path),
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "checkpoint_transformers_version": config.get("transformers_version"),
            "inference_transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "dtype": None,
            "decoding": CONFIG["decoding"],
            "gpu": gpu,
            "model_files": model_files,
            "started_at_utc": utc_now(),
            "protocol_id": CONFIG["protocol_id"],
        }
        append_jsonl(run_path, metadata)
        print(f"[{annotator_id}] loading {model_path}", flush=True)
        try:
            model, tokenizer, model_config, dtype_name = load_model(model_path)
        except Exception as exc:
            metadata["event"] = "load_failed"
            metadata["failed_at_utc"] = utc_now()
            metadata["load_error"] = f"{type(exc).__name__}: {exc}"
            append_jsonl(run_path, metadata)
            raise
        metadata["dtype"] = dtype_name
        metadata["loaded_model_type"] = model_config.model_type
        run_start = time.perf_counter()
        n_completed_this_run = 0
        for index, case in enumerate(payloads, start=1):
            key = (annotator_id, case["case_id"], case["prompt_sha256"])
            if args.resume and key in completed:
                continue
            allowed_features = set(
                json.loads(case["user_prompt"])["target_row"]["features"].keys()
            )
            started = time.perf_counter()
            raw_text = ""
            prompt_tokens = None
            error = None
            parsed = None
            try:
                raw_text, prompt_tokens = generate_one(model, tokenizer, case)
                parsed, error = parse_annotation(raw_text, allowed_features)
            except Exception as exc:
                error = f"inference failed: {type(exc).__name__}: {exc}"
            record = {
                "event": "annotation",
                "protocol_id": CONFIG["protocol_id"],
                "run_id": run_id,
                "annotator_id": annotator_id,
                "model_id": spec["model_id"],
                "model_family": spec["family"],
                "case_id": case["case_id"],
                "dataset_key": case["dataset_key"],
                "prompt_sha256": case["prompt_sha256"],
                "timestamp_utc": utc_now(),
                "temperature": 0.0,
                "do_sample": False,
                "max_new_tokens": int(CONFIG["decoding"]["max_new_tokens"]),
                "prompt_tokens": prompt_tokens,
                "latency_seconds": round(time.perf_counter() - started, 4),
                "raw_output": raw_text,
                "parsed_annotation": parsed,
                "parse_or_inference_error": error,
            }
            append_jsonl(annotation_path, record)
            completed.add(key)
            n_completed_this_run += 1
            if index == 1 or index % 10 == 0 or index == len(payloads):
                print(
                    f"[{annotator_id}] {index}/{len(payloads)} "
                    f"case={case['case_id']} parse={'ok' if parsed else 'failed'}",
                    flush=True,
                )
        metadata["event"] = "completed"
        metadata["completed_at_utc"] = utc_now()
        metadata["elapsed_seconds"] = round(time.perf_counter() - run_start, 3)
        metadata["annotation_records_written"] = n_completed_this_run
        append_jsonl(run_path, metadata)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[{annotator_id}] finished; wrote {n_completed_this_run} annotation records.",
            flush=True,
        )


if __name__ == "__main__":
    main()
