#!/usr/bin/env python3
"""Execute the frozen CPU supplement queue; preserve commands and exit status."""
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from prepare_ccfa_revision import OUT, HERE, read, write

def main():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "2",
           "PYTHONPATH": str(OUT / "third_party" / "M2OE")}
    development = [r for r in read(OUT / "seed_20261106" / "inputs.jsonl") if r["split"] == "dev"]
    gold = {r["case_id"]: r for r in read(OUT / "seed_20261106" / "ground_truth.jsonl")}
    selected = {}
    for r in sorted(development, key=lambda x: x["case_id"]):
        selected.setdefault((gold[r["case_id"]]["template_id"], r["feature_count"]), r)
    assert len(selected) == 24
    dev_path = OUT / "inputs_dev_budget.jsonl"
    write(dev_path, list(selected.values()))
    write(OUT / "ground_truth_dev_budget.jsonl", [gold[r["case_id"]] for r in selected.values()])
    jobs = []
    def add(name, args, expected):
        jobs.append({"name": name, "command": [str(x) for x in args], "expected": str(expected)})
    for seed in (20261106, 20261107, 20261108):
        root = OUT / f"seed_{seed}"
        add(f"simple_{seed}", [sys.executable, HERE / "make_baselines_v2.py", "--inputs-path", root / "inputs.jsonl", "--ground-truth-path", root / "ground_truth.jsonl", "--outdir", root / "simple", "--replicates", "100"], root / "simple" / "manifest.json")
    add("sod_cop_legacy_panel", [sys.executable, HERE / "run_sod_cop_adaptations_v2.py", "--inputs-path", OUT / "inputs_panel.jsonl", "--outdir", OUT / "sod_cop_legacy_panel", "--encoding", "legacy_reference_clip"], OUT / "sod_cop_legacy_panel" / "manifest.json")
    add("m2oe_legacy_panel", ["/tmp/m2oe_tf_env/bin/python", HERE / "run_m2oe_baseline_v2.py", "--inputs-path", OUT / "inputs_panel.jsonl", "--output-path", OUT / "m2oe_legacy_panel" / "predictions_m2oe_choice.jsonl", "--encoding", "legacy_reference_clip"], OUT / "m2oe_legacy_panel" / "predictions_m2oe_choice.manifest.json")
    for steps in (256, 1024, 4096):
        for seed in (20261106, 20261117, 20261128):
            name = f"rlax_dev_{steps}_{seed}"
            add(name, [sys.executable, HERE / "run_rlax_adaptation_v2.py", "--inputs-path", dev_path, "--outdir", OUT / name, "--split", "dev", "--n-steps", str(steps), "--seed", str(seed)], OUT / name / "manifest.json")
    for epochs in (30, 90):
        name = f"m2oe_dev_epochs{epochs}"
        add(name, ["/tmp/m2oe_tf_env/bin/python", HERE / "run_m2oe_baseline_v2.py", "--inputs-path", dev_path, "--split", "dev", "--epochs", str(epochs), "--output-path", OUT / name / "predictions_m2oe_choice.jsonl"], OUT / name / "predictions_m2oe_choice.manifest.json")
    contract = {"dev_selection": "24 cases: lexicographically first case in each of 8 seed-20261106 dev templates x 3 field sizes; independent of performance",
                "development_isolation": "No new tuning on test outputs; dev sensitivity is conditional on seed-20261106 split; templates rotate across other splits",
                "jobs": jobs, "frozen_at_unix": time.time(), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest_path = OUT / "cpu_supplement_queue.json"
    if not manifest_path.exists(): manifest_path.write_text(json.dumps(contract, indent=2) + "\n")
    def run(job):
        if Path(job["expected"]).exists():
            return {"name": job["name"], "status": "existing_completed_artifact"}
        start = time.time()
        with (OUT / f"{job['name']}.log").open("w") as log:
            p = subprocess.run(job["command"], env=env, stdout=log, stderr=subprocess.STDOUT)
        result = {"name": job["name"], "returncode": p.returncode, "wall_seconds": time.time()-start,
                  "expected_exists": Path(job["expected"]).exists(), "command": job["command"]}
        (OUT / f"{job['name']}.execution.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, jobs))
    (OUT / "cpu_supplement_status.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(r.get("returncode", 0) for r in results): raise SystemExit(1)

if __name__ == "__main__": main()
