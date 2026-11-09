#!/usr/bin/env python3
"""Wait for the complete 78-test multiseed analysis, then export subgroup summaries."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIRED = HERE / "runs" / "multiseed_analysis" / "paired_tests.json"
METRICS = HERE / "runs" / "multiseed_analysis" / "multiseed_metrics.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    while True:
        try:
            data = json.loads(PAIRED.read_text(encoding="utf-8"))
            if data.get("completed_contrast_count") == data.get("planned_contrast_count") == 78 and METRICS.exists():
                break
            print(f"{datetime.now(timezone.utc).isoformat()} paired contrasts {data.get('completed_contrast_count')}/{data.get('planned_contrast_count')}; waiting", flush=True)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"{datetime.now(timezone.utc).isoformat()} waiting for complete paired-results file", flush=True)
        time.sleep(args.poll_seconds)
    command = [sys.executable, str(HERE / "analyze_multiseed_subgroups_v2.py")]
    print(f"{datetime.now(timezone.utc).isoformat()} running subgroup summaries: {command}", flush=True)
    subprocess.run(command, cwd=HERE, check=True)


if __name__ == "__main__":
    main()
