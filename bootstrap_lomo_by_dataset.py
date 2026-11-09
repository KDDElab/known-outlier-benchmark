#!/usr/bin/env python3
"""Dataset-cluster bootstrap for real-table LOMO weak-reference agreement."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "reports" / "leave_one_model_out_by_case.csv"
OUTPUT = ROOT / "reports" / "leave_one_model_out_dataset_cluster_bootstrap.csv"
N_BOOT = 10_000
SEED = 20261106


def main() -> None:
    data = pd.read_csv(INPUT)
    data["eligible"] = data["status"].eq("scored against peer strict-majority reference")
    data["valid_held_out"] = ~data["status"].isin([
        "held-out output missing or invalid", "held-out annotator abstained"
    ])
    data["peer_consensus_coverage"] = data["peer_consensus_coverage"].astype(str).str.lower().eq("true")
    data["jaccard"] = pd.to_numeric(data["jaccard"], errors="coerce")
    rng = np.random.default_rng(SEED)
    rows = []
    for model, frame in data.groupby("held_out_model_id", sort=True):
        clusters = sorted(frame["dataset_key"].unique())
        clustered = [frame.loc[frame["dataset_key"].eq(key)] for key in clusters]
        eligible_counts = np.array([int(group["eligible"].sum()) for group in clustered])
        valid_counts = np.array([int(group["valid_held_out"].sum()) for group in clustered])
        score_sums = np.array([group.loc[group["eligible"], "jaccard"].sum() for group in clustered], dtype=float)
        score_counts = np.array([group.loc[group["eligible"], "jaccard"].count() for group in clustered], dtype=float)
        draws = rng.integers(0, len(clusters), size=(N_BOOT, len(clusters)))
        sampled_eligible = eligible_counts[draws].sum(axis=1)
        sampled_valid = valid_counts[draws].sum(axis=1)
        coverage_draws = sampled_eligible / np.maximum(sampled_valid, 1)
        sampled_score_count = score_counts[draws].sum(axis=1)
        jaccard_draws = score_sums[draws].sum(axis=1) / np.maximum(sampled_score_count, 1)
        eligible = frame.loc[frame["eligible"]]
        valid = int(frame["valid_held_out"].sum())
        score = eligible["jaccard"].dropna().to_numpy(dtype=float)
        rows.append({
            "held_out_model_id": model,
            "n_queries": int(len(frame)),
            "n_dataset_clusters": len(clusters),
            "n_valid_held_out_outputs": valid,
            "n_lomo_eligible": int(len(eligible)),
            "peer_reference_coverage": len(eligible) / valid if valid else np.nan,
            "peer_reference_coverage_ci95_low": float(np.nanquantile(coverage_draws, .025)),
            "peer_reference_coverage_ci95_high": float(np.nanquantile(coverage_draws, .975)),
            "mean_jaccard_on_eligible": float(score.mean()) if len(score) else np.nan,
            "mean_jaccard_ci95_low": float(np.nanquantile(jaccard_draws, .025)),
            "mean_jaccard_ci95_high": float(np.nanquantile(jaccard_draws, .975)),
            "bootstrap_seed": SEED,
            "bootstrap_replicates": N_BOOT,
        })
    pd.DataFrame(rows).to_csv(OUTPUT, index=False, float_format="%.6f")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
