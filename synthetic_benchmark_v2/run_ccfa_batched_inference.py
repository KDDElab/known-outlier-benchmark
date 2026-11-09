#!/usr/bin/env python3
"""Resume the frozen queue with logged, fidelity-checked batched greedy inference."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import platform
import time
import uuid
from pathlib import Path
import torch
import transformers
from annotate_v2 import (ANNOTATORS, append_jsonl, find_model_manifest, load_model,
                         parse_prediction, render_prompt, utc_now, MAX_INPUT_TOKENS)
from prepare_ccfa_revision import OUT, HERE, PROTOCOL, read

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def signature(parsed):
    if parsed is None: return None
    return {"explanations": [x["features"] for x in parsed["explanations"]],
            "abstain": parsed["abstain"], "task_label": parsed.get("task_label"),
            "mechanism_feature_ids": parsed.get("mechanism_feature_ids")}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(ANNOTATORS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-token-budget", type=int, default=16384)
    args = parser.parse_args()
    model_id = args.model
    path = OUT / f"pending_{model_id}.jsonl"
    cases = read(path)
    annotations = OUT / f"new_{model_id}.jsonl"
    existing = read(annotations) if annotations.exists() else []
    completed = {r["case_id"]: r for r in existing}
    assert len(completed) == len(existing)
    run_path = OUT / f"model_runs_{model_id}.jsonl"
    run_id = f"{model_id}_batch_{uuid.uuid4().hex[:12]}"
    spec = ANNOTATORS[model_id]
    model_path = Path(spec["model_path"])
    metadata = {"event": "started", "run_id": run_id, "protocol_id": PROTOCOL,
                "annotator_id": model_id, "model_id": spec["model_id"],
                "model_path": str(model_path), "model_files": find_model_manifest(model_id, model_path),
                "inputs_sha256": sha(path), "runner_sha256": sha(__file__),
                "shared_loader_sha256": sha(HERE.parent / "annotate.py"),
                "parser_sha256": sha(HERE / "annotate_v2.py"),
                "inference_transformers_version": transformers.__version__, "torch_version": torch.__version__,
                "python_version": platform.python_version(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "batch_size_requested": args.batch_size, "batch_token_budget": args.batch_token_budget,
                "decoding": {"do_sample": False, "num_beams": 1, "max_input_tokens": MAX_INPUT_TOKENS,
                             "max_new_tokens": "512 or 1024 as frozen per case", "input_truncation": False},
                "started_at_utc": utc_now(), "existing_records": len(existing)}
    append_jsonl(run_path, metadata)
    print(f"Loading {model_id}; {len(cases)-len(existing)} pending; GPU={metadata['gpu']}", flush=True)
    model, tokenizer, config, dtype = load_model(model_path)
    tokenizer.padding_side = "left"
    device_map = {str(k): str(v) for k,v in getattr(model, "hf_device_map", {}).items()}
    if any(v in {"cpu", "disk"} for v in device_map.values()):
        raise RuntimeError("Offloaded model is outside the GPU execution contract")
    append_jsonl(run_path, {"event":"loaded", "run_id":run_id, "dtype":dtype, "device_map":device_map})
    eos = tokenizer.eos_token_id
    if isinstance(eos, list): eos = eos[0]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    encoded = {}
    for case in cases:
        rendered = render_prompt(tokenizer, case["system_prompt"], case["user_prompt"])
        encoded[case["case_id"]] = tokenizer(rendered, truncation=False)["input_ids"]

    def generate(batch):
        token_rows = [encoded[c["case_id"]] for c in batch]
        inputs = tokenizer.pad({"input_ids": token_rows}, padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k,v in inputs.items()}
        cap = batch[0].get("max_new_tokens", 512)
        assert all(c.get("max_new_tokens",512) == cap for c in batch)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=cap, do_sample=False, num_beams=1,
                                     use_cache=True, pad_token_id=pad, eos_token_id=eos)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        results = []
        for c, output in zip(batch, outputs):
            tail = output[inputs["input_ids"].shape[1]:].tolist()
            if eos in tail: tail = tail[:tail.index(eos)+1]
            raw = tokenizer.decode(tail, skip_special_tokens=True)
            parsed, error = parse_prediction(raw, set(c["feature_ids"]), c.get("output_contract","full"))
            results.append((raw, parsed, error, len(tail), elapsed))
        del inputs, outputs
        return results

    def batches(rows, size):
        current = []
        for c in sorted(rows, key=lambda x:(x.get("max_new_tokens",512),len(encoded[x["case_id"]]),x["case_id"])):
            if current and (len(current)>=size or c.get("max_new_tokens",512)!=current[0].get("max_new_tokens",512)
                            or (len(encoded[c["case_id"]])+c.get("max_new_tokens",512))*(len(current)+1)>args.batch_token_budget):
                yield current; current=[]
            current.append(c)
        if current: yield current

    # Check the exact localization outputs against serial inference on the same
    # loaded checkpoint. Keep diagnostics out of the publication predictions.
    eligible = [c for c in cases if len(encoded[c["case_id"]]) < args.batch_token_budget//3 and c.get("max_new_tokens",512)==512]
    probe = sorted(eligible,key=lambda c:c["case_id"])[:6]
    diagnostic = []
    serial_results = {}
    for c in probe:
        serial_results[c["case_id"]] = generate([c])[0]
    for batch in batches(probe, args.batch_size):
        for c, result in zip(batch,generate(batch)):
            serial = serial_results[c["case_id"]]
            diagnostic.append({"case_id": c["case_id"], "prompt_sha256": c["prompt_sha256"],
                "batch_size": len(batch), "serial_raw": serial[0], "batch_raw": result[0],
                "raw_exact": serial[0]==result[0], "localization_exact": signature(serial[1])==signature(result[1]),
                "serial_seconds":serial[4], "batch_seconds":result[4]})
    effective_batch = args.batch_size if diagnostic and all(d["localization_exact"] for d in diagnostic) else 1
    append_jsonl(run_path,{"event":"batch_fidelity_diagnostic","run_id":run_id,"n_cases":len(probe),
                          "effective_batch_size":effective_batch,"exact_localization_count":sum(d["localization_exact"] for d in diagnostic),
                          "note":"Finite diagnostic, not proof of identical outputs on all cases; no quality-based retry"})
    (OUT/f"batch_diagnostic_{model_id}.json").write_text(json.dumps(diagnostic,indent=2)+"\n")
    print(f"Fidelity diagnostic {sum(d['localization_exact'] for d in diagnostic)}/{len(diagnostic)}; batch={effective_batch}",flush=True)
    completed_ids = set(completed)
    pending = [c for c in cases if c["case_id"] not in completed_ids]
    written = 0
    def record(c, result, batch_size):
        nonlocal written
        raw, parsed, error, tokens, elapsed = result
        row={"event":"annotation","protocol_id":PROTOCOL,"run_id":run_id,"annotator_id":model_id,
             "model_id":spec["model_id"],"model_family":spec["family"],"case_id":c["case_id"],
             "source_case_id":c["source_case_id"],"condition":c["condition"],"generator_seed":c["generator_seed"],
             "display_order_seed":c.get("display_order_seed"),"split":c["split"],"prompt_sha256":c["prompt_sha256"],
             "timestamp_utc":utc_now(),"temperature":0.0,"do_sample":False,"max_input_tokens":MAX_INPUT_TOKENS,
             "max_new_tokens":c.get("max_new_tokens",512),"output_contract":c.get("output_contract","full"),
             "prompt_tokens":len(encoded[c["case_id"]]),"output_tokens":tokens,
             "output_limit_reached":tokens==c.get("max_new_tokens",512),
             "batch_size":batch_size,"batch_elapsed_seconds":elapsed,"latency_seconds":elapsed,
             "timing_scope":"entire padded batch; not an individual-query latency or additive GPU time",
             "raw_output":raw,"parsed_prediction":parsed,"parse_or_inference_error":error}
        append_jsonl(annotations,row);written+=1
    def execute(batch):
        try:
            results=generate(batch)
        except torch.cuda.OutOfMemoryError as exc:
            append_jsonl(run_path,{"event":"batch_oom","run_id":run_id,"case_ids":[c["case_id"] for c in batch],"batch_size":len(batch),"error":str(exc)})
            gc.collect();torch.cuda.empty_cache()
            if len(batch)>1:
                cut=len(batch)//2;execute(batch[:cut]);execute(batch[cut:]);return
            raise RuntimeError("Single-query GPU OOM: resource repair required; no model-output retry") from exc
        for c,result in zip(batch,results):record(c,result,len(batch))
    feasible=[]
    for c in pending:
        length=len(encoded[c["case_id"]])
        if length>MAX_INPUT_TOKENS:
            record(c,("",None,f"inference failed: ValueError: rendered input has {length} tokens, exceeding {MAX_INPUT_TOKENS}; input was not truncated",None,0.0),0)
        else:feasible.append(c)
    started=time.perf_counter()
    for batch_index,batch in enumerate(batches(feasible,effective_batch),1):
        execute(batch)
        if batch_index==1 or batch_index%10==0:
            print(f"{model_id}: {written}/{len(pending)} new records; batch {batch_index}",flush=True)
    append_jsonl(run_path,{"event":"completed","run_id":run_id,"protocol_id":PROTOCOL,
                          "annotator_id":model_id,"completed_at_utc":utc_now(),"new_records":written,
                          "elapsed_seconds":time.perf_counter()-started,"dtype":dtype,"device_map":device_map})
    print(f"Completed {model_id}: {written} new records",flush=True)

if __name__ == "__main__":main()
