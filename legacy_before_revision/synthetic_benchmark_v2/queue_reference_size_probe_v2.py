#!/usr/bin/env python3
"""After the Qwen ID probe frees GPU 3, run the paired 8/16-reference conditions."""
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
PROBE = HERE / "runs" / "reference_size_probe"
ID_PROBE_SUMMARY = HERE / "runs" / "id_permutation_probe" / "id_permutation_summary.json"
RUN_LOG = PROBE / "model_runs.jsonl"


def unique_annotations(path: Path, protocol: str) -> int:
    if not path.exists():
        return 0
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "annotation" and row.get("protocol_id") == protocol:
            ids.add(row.get("case_id"))
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    while not ID_PROBE_SUMMARY.exists():
        path = HERE / "runs" / "id_permutation_probe" / "annotations_qwen35_9b_id_permutation.jsonl"
        print(f"{datetime.now(timezone.utc).isoformat()} waiting for ID probe summary; annotations={unique_annotations(path, 'known_anomaly_subspace_v2_opaque_feature_id_permutation_v1')}/96", flush=True)
        time.sleep(args.poll_seconds)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    for size in (8, 16):
        protocol = f"known_anomaly_subspace_v2_reference_size_{size}_v1"
        command = [
            sys.executable, str(HERE / "annotate_v2.py"), "--model", "qwen35_9b", "--split", "test",
            "--inputs-path", str(PROBE / f"inputs_reference_size_{size}.jsonl"),
            "--annotations-path", str(PROBE / f"annotations_qwen35_9b_reference_size_{size}.jsonl"),
            "--run-log-path", str(RUN_LOG), "--protocol-id", protocol,
        ]
        print(f"{datetime.now(timezone.utc).isoformat()} starting reference-size {size} run: {command}", flush=True)
        subprocess.run(command, cwd=HERE.parent, env=env, check=True)
        count = unique_annotations(PROBE / f"annotations_qwen35_9b_reference_size_{size}.jsonl", protocol)
        if count != 96:
            raise RuntimeError(f"Reference-size {size} run ended with {count}/96 unique annotations")
    subprocess.run([sys.executable, str(HERE / "analyze_reference_size_probe_v2.py")], cwd=HERE.parent, check=True)
    subprocess.run([sys.executable, str(HERE / "combine_prompt_sensitivity_tests_v2.py")], cwd=HERE.parent, check=True)
    subprocess.run([sys.executable, str(HERE / "combine_robustness_holm_v2.py")], cwd=HERE.parent, check=True)
    print(f"{datetime.now(timezone.utc).isoformat()} reference-size probe complete", flush=True)


if __name__ == "__main__":
    main()
