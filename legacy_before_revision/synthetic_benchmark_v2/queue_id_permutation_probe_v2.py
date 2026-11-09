#!/usr/bin/env python3
"""Wait for the queued Qwen circuit pilot, then run the frozen 96-case ID probe."""
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
PROBE = HERE / "runs" / "id_permutation_probe"
PROTOCOL = "known_anomaly_subspace_v2_opaque_feature_id_permutation_v1"
CIRCUIT_ANNOTATIONS = HERE.parent / "circuit_track" / "matched_information" / "annotations_qwen35_9b.jsonl"
CIRCUIT_PROTOCOL = "harvard_circuit_focus_output_only_v1"
OUTPUT = PROBE / "annotations_qwen35_9b_id_permutation.jsonl"
RUN_LOG = PROBE / "model_runs.jsonl"


def protocol_case_count(path: Path, protocol: str) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "annotation" and row.get("protocol_id") == protocol:
            rows.append(row)
    return len(rows), len({row.get("case_id") for row in rows})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True, help="PID of the current Qwen circuit-focus process")
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()

    while Path(f"/proc/{args.wait_pid}").exists():
        print(f"{datetime.now(timezone.utc).isoformat()} waiting for circuit PID {args.wait_pid}", flush=True)
        time.sleep(args.poll_seconds)
    while True:
        total, unique = protocol_case_count(CIRCUIT_ANNOTATIONS, CIRCUIT_PROTOCOL)
        if total == unique == 86:
            break
        print(f"{datetime.now(timezone.utc).isoformat()} circuit annotations {unique}/86 unique", flush=True)
        time.sleep(args.poll_seconds)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    command = [
        sys.executable, str(HERE / "annotate_v2.py"), "--model", "qwen35_9b", "--split", "test",
        "--inputs-path", str(PROBE / "inputs_opaque_id_permutation.jsonl"),
        "--annotations-path", str(OUTPUT), "--run-log-path", str(RUN_LOG),
        "--protocol-id", PROTOCOL,
    ]
    print(f"{datetime.now(timezone.utc).isoformat()} starting 96-case ID-permutation run: {command}", flush=True)
    subprocess.run(command, cwd=HERE.parent, env=env, check=True)
    subprocess.run([sys.executable, str(HERE / "analyze_id_permutation_probe_v2.py")], cwd=HERE.parent, check=True)
    subprocess.run([sys.executable, str(HERE / "combine_prompt_sensitivity_tests_v2.py")], cwd=HERE.parent, check=True)
    print(f"{datetime.now(timezone.utc).isoformat()} ID-permutation probe complete", flush=True)


if __name__ == "__main__":
    main()
