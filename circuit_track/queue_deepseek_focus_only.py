#!/usr/bin/env python3
"""Wait for the common-runtime DeepSeek run to release both GPUs, then run its matched circuit prompt."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ANNOTATIONS = HERE / "matched_information" / "annotations_deepseek_v2_lite_chat.jsonl"
INPUTS = HERE / "matched_information" / "inputs_focus_output_only.jsonl"
OUTPUT = HERE / "matched_information" / "annotations_deepseek_v2_lite_chat.jsonl"
RUN_LOG = HERE / "matched_information" / "model_runs.jsonl"
PROTOCOL = "harvard_circuit_focus_output_only_v1"
MAIN_PROTOCOL = "known_anomaly_subspace_v2_seed_20261106_rt522"
MAIN_ANNOTATIONS = ROOT / "synthetic_benchmark_v2" / "runs" / "multiseed" / "seed_20261106_rt522" / "annotations_deepseek_v2_lite_chat.jsonl"


def main_count() -> tuple[int, int]:
    rows = []
    if MAIN_ANNOTATIONS.exists():
        for line in MAIN_ANNOTATIONS.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "annotation" and row.get("protocol_id") == MAIN_PROTOCOL:
                rows.append(row)
    return len(rows), len({row.get("case_id") for row in rows})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True, help="PID of the in-flight DeepSeek common-runtime annotation")
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    while Path(f"/proc/{args.wait_pid}").exists():
        total, unique = main_count()
        print(f"{datetime.now(timezone.utc).isoformat()} waiting for DeepSeek PID {args.wait_pid} to exit; annotations={unique}/288", flush=True)
        time.sleep(args.poll_seconds)
    while True:
        total, unique = main_count()
        if total == unique == 288:
            break
        print(f"{datetime.now(timezone.utc).isoformat()} waiting for complete DeepSeek annotations={unique}/288", flush=True)
        time.sleep(args.poll_seconds)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    command = [
        sys.executable, str(HERE / "run_zero_noise_pilot.py"), "--model", "deepseek_v2_lite_chat",
        "--inputs-path", str(INPUTS), "--annotations-path", str(OUTPUT),
        "--run-log-path", str(RUN_LOG), "--protocol-id", PROTOCOL,
    ]
    print(f"{datetime.now(timezone.utc).isoformat()} starting DeepSeek focus-only circuit run after main process exit: {command}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    print(f"{datetime.now(timezone.utc).isoformat()} DeepSeek focus-only circuit run complete", flush=True)


if __name__ == "__main__":
    main()
