#!/usr/bin/env python3
"""Freeze repaired inputs and provenance-checked reuse, without inspecting scores."""
from __future__ import annotations
import copy
import hashlib
import json
import random
from pathlib import Path
from collections import Counter
from generate_benchmark import independently_ordered
from run_expanded_random_order_probe_v2 import prompt_payload
from run_m2oe_baseline_v2 import continuous_encoding

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "ccfa_revision"
OLD = HERE.parent / "artifact_release_staging" / "synthetic_benchmark_v2" / "runs" / "expanded_random_field_order_288"
MODELS = ("qwen35_9b", "deepseek_v2_lite_chat", "llama31_8b_instruct")
PROTOCOL = "known_outlier_ccfa_repaired_v1"

def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def finish(row, seed, condition, old_id):
    row = copy.deepcopy(row)
    row.update(source_case_id=old_id, generator_seed=seed, condition=condition,
               case_id=f"s{seed}_{condition}_{old_id}", protocol_id=PROTOCOL)
    row["prompt_sha256"] = hashlib.sha256((row["system_prompt"] + "\0" + row["user_prompt"]).encode()).hexdigest()
    return row

def replace_payload(row, payload, prefix=None):
    row = copy.deepcopy(row)
    if prefix is None:
        prefix, _ = prompt_payload(row["user_prompt"])
    row["user_prompt"] = prefix + json.dumps(payload, ensure_ascii=False, indent=2)
    return row

def main():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("Frozen protocol already exists; inspect it rather than overwriting running inputs")
    all_rows, all_gold, sources, panels = [], [], {}, {}
    by_seed = {}
    identity_checks = Counter()
    for seed, display_seed in ((20261106, 1001), (20261107, 2001), (20261108, 3001)):
        src = HERE if seed == 20261106 else HERE / "runs" / "multiseed" / f"seed_{seed}"
        original = read(src / "inputs.jsonl")
        truths = {r["case_id"]: r for r in read(src / "ground_truth.jsonl")}
        sources[str(seed)] = {str(p): sha(p) for p in (src / "inputs.jsonl", src / "ground_truth.jsonl")}
        by_seed[seed] = (original, truths)
        main_rows, main_gold = [], []
        for source in original:
            old_id = source["case_id"]
            reordered = independently_ordered(source, display_seed)
            _, before = prompt_payload(source["user_prompt"])
            _, after = prompt_payload(reordered["user_prompt"])
            assert before["target_row"] == after["target_row"]
            assert before["normal_reference_rows"] == after["normal_reference_rows"]
            assert {s["feature_id"]: s for s in before["feature_schema"]} == {s["feature_id"]: s for s in after["feature_schema"]}
            assert list(after["target_row"]) == reordered["feature_ids"]
            assert all(list(r) == reordered["feature_ids"] for r in after["normal_reference_rows"])
            row = finish(reordered, seed, "main", old_id)
            gold = {**truths[old_id], "case_id": row["case_id"], "source_case_id": old_id, "generator_seed": seed}
            main_rows.append(row); main_gold.append(gold)
            if source["split"] == "test":
                all_rows.append(row); all_gold.append(gold)
                identity_checks["main_value_schema_label_checks"] += 1
                # The injectivity check uses only visible values/schema.
                target, refs, groups = continuous_encoding(after)
                values = [after["target_row"], *after["normal_reference_rows"]]
                import numpy as np
                matrix = np.concatenate([target, refs])
                for item, dims in zip(after["feature_schema"], groups):
                    fid = item["feature_id"]
                    states = {json.dumps(v[fid], sort_keys=True) for v in values}
                    enc = {tuple(v[dims]) for v in matrix}
                    assert len(states) == len(enc), (seed, old_id, fid)
                    identity_checks["injective_observed_field_encodings"] += 1
        write(OUT / f"seed_{seed}" / "inputs.jsonl", main_rows)
        write(OUT / f"seed_{seed}" / "ground_truth.jsonl", main_gold)

    original, truths = by_seed[20261106]
    panel = [r for r in original if r["split"] == "test" and r["reference_count"] == 32]
    assert len(panel) == 96
    assert set(Counter(truths[r["case_id"]]["template_id"] for r in panel).values()) == {4}
    rng = random.Random(20261113)
    for source in panel:
        old_id = source["case_id"]
        base = independently_ordered(source, 1001)
        prefix, payload = prompt_payload(base["user_prompt"])
        positions = list(range(32)); rng.shuffle(positions)
        indices = {8: sorted(positions[:8]), 16: sorted(positions[:16]), 32: list(range(32))}
        panels[old_id] = {"template_id": truths[old_id]["template_id"], "reference_indices": indices}
        variants = [("front", source)]
        for order_seed in (1002, 1003, 1004):
            variants.append((f"order{order_seed}", independently_ordered(source, order_seed)))
        roles = set(truths[old_id]["rule_role_to_feature"].values())
        order = [x for x in base["feature_ids"] if x not in roles] + [x for x in base["feature_ids"] if x in roles]
        back = copy.deepcopy(payload)
        schema = {s["feature_id"]: s for s in back["feature_schema"]}
        back["feature_schema"] = [schema[x] for x in order]
        back["target_row"] = {x: back["target_row"][x] for x in order}
        back["normal_reference_rows"] = [{x: r[x] for x in order} for r in back["normal_reference_rows"]]
        rear = replace_payload(base, back); rear["feature_ids"] = order
        rear["display_order_seed"] = "role_back_diagnostic"
        variants.append(("back", rear))
        for size in (8, 16):
            p = copy.deepcopy(payload)
            p["normal_reference_rows"] = [p["normal_reference_rows"][i] for i in indices[size]]
            r = replace_payload(base, p); r["reference_count"] = size; r["reference_indices"] = indices[size]
            variants.append((f"refs{size}", r))
        full = copy.deepcopy(base); full["max_new_tokens"] = 1024
        variants.append(("full1024", full))
        minimal_prefix = (
            "The target row is already known to be anomalous. Return up to four ranked candidate "
            "inclusion-minimal feature subspaces that distinguish the target from the normal population. "
            "Multiple valid subspaces may exist. Use only feature IDs from the schema. "
            "Return one JSON object with exactly explanations and abstain. explanations must be an "
            "ordered array of objects containing only features (an array of feature IDs without duplicates). "
            "abstain must be a boolean. If uncertain, abstain. Do not include analysis, deliberation, or a preamble. "
            'Example shape: {"explanations":[{"features":["F001"]}],"abstain":false}.\n\n'
        )
        minimal = replace_payload(base, payload, minimal_prefix)
        minimal["output_contract"] = "minimal"
        variants.append(("minimal512", minimal))
        for condition, r in variants:
            r = finish(r, 20261106, condition, old_id)
            all_rows.append(r)
            all_gold.append({**truths[old_id], "case_id": r["case_id"], "source_case_id": old_id, "generator_seed": 20261106})
    assert len(all_rows) == len({r["case_id"] for r in all_rows}) == 1728
    write(OUT / "inputs_all_conditions.jsonl", all_rows)
    write(OUT / "ground_truth_all_conditions.jsonl", all_gold)

    reuse_manifest = {}
    for model in MODELS:
        cached = {}
        candidates = [OLD / f"annotations_{model}.jsonl", HERE / "runs" / "multiseed" / "seed_20261106_rt522" / f"annotations_{model}.jsonl"]
        for path in candidates:
            for record in read(path):
                if record.get("event") == "annotation":
                    key = (record["prompt_sha256"], record.get("max_new_tokens", 512), "full")
                    cached.setdefault(key, (record, path))
        reused, pending = [], []
        for row in all_rows:
            key = (row["prompt_sha256"], row.get("max_new_tokens", 512), row.get("output_contract", "full"))
            old = cached.get(key)
            if old is None:
                pending.append(row); continue
            rec, path = old
            assert rec["annotator_id"] == model
            assert rec.get("max_input_tokens") == 16384 and rec.get("do_sample") is False
            r = copy.deepcopy(rec)
            r.update(case_id=row["case_id"], protocol_id=PROTOCOL, condition=row["condition"],
                     source_case_id=row["source_case_id"], generator_seed=row["generator_seed"],
                     reused_from={"path": str(path), "case_id": rec["case_id"], "run_id": rec["run_id"], "file_sha256": sha(path)})
            reused.append(r)
        write(OUT / f"reused_{model}.jsonl", reused)
        write(OUT / f"pending_{model}.jsonl", pending)
        reuse_manifest[model] = {"reused": len(reused), "pending": len(pending),
                                 "reuse_conditions": dict(Counter(r["condition"] for r in reused))}
    protocol = {
        "protocol_id": PROTOCOL,
        "revision_scope": "Version reevaluation after inspection of old results; not preregistered or a new unseen-template test",
        "sources": sources, "reuse": reuse_manifest,
        "main_generator_seeds": [20261106, 20261107, 20261108],
        "display_seeds": [1001, 2001, 3001], "independent_orders": [1001, 1002, 1003, 1004],
        "panel_selection": "All 96 seed-20261106 test cases with 32 references: four per test template; no model performance used",
        "panel": panels, "checks": dict(identity_checks),
        "display_policy": "Independent seeded shuffle after all values and gold; identity permutations allowed. Existing outputs reused only if prompt hashes match this policy.",
        "encoding": "schema_domain_v1", "candidate_budget": 4, "M2OE_unordered_budget": 3,
        "scoring": "All-query denominator; explicit abstain, empty, invalid and missing count as misses. Size conditional on answered; top1 separately from list mean.",
        "primary_endpoints": ["support_top1_minimal", "support_any_minimal_hit"],
        "estimator": "Within seed/template case mean, then mean over available seeds per template, then equal mean over 38 templates",
        "main_test_family": "18 two-sided template sign-flips: 3 LLM x 2 Gower plus 3 LLM pairs, each Exact@1 and Hit@4; Holm across all 18",
        "bootstrap_repetitions": 2000, "sign_flip_repetitions": 20000, "analysis_seed": 20261107,
        "RLAX": "Original 256 interactions / 4 PPO epochs main; separate 256/1024/4096 interaction dev sensitivity with three optimization seeds. No cross-seed template blindness claim.",
        "optional_not_executed": "New expert study, new natural-domain gold, 64-reference panel and additional models are outside this core revision",
        "input_sha256": sha(OUT / "inputs_all_conditions.jsonl"), "gold_sha256": sha(OUT / "ground_truth_all_conditions.jsonl"),
        "code_sha256": sha(__file__),
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"n_conditions": len(all_rows), "reuse": reuse_manifest, "checks": dict(identity_checks)}, indent=2))

if __name__ == "__main__":
    main()
