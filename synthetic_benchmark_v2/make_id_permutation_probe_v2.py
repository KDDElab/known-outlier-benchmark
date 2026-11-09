#!/usr/bin/env python3
"""Create a paired opaque-feature-ID relabeling probe for the fixed 96 cases."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SOURCE = RUNS / "order_probe_original_inputs.jsonl"
OUT = RUNS / "id_permutation_probe"
PROTOCOL = "known_anomaly_subspace_v2_opaque_feature_id_permutation_v1"
SEED = 20261109


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 96 or len({row["case_id"] for row in rows}) != 96:
        raise RuntimeError("The frozen order-probe source must contain exactly 96 unique cases")

    rng = random.Random(SEED)
    variants: list[dict] = []
    mappings: dict[str, dict[str, str]] = {}
    for source in rows:
        row = dict(source)
        old_ids = sorted(set(source["feature_ids"]))
        new_ids = list(old_ids)
        rng.shuffle(new_ids)
        if len(old_ids) > 1 and new_ids == old_ids:
            new_ids = new_ids[1:] + new_ids[:1]
        old_to_new = dict(zip(old_ids, new_ids, strict=True))
        new_to_old = {value: key for key, value in old_to_new.items()}
        if len(new_to_old) != len(old_ids):
            raise RuntimeError(f"Non-bijective ID mapping for {source['case_id']}")

        token_pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(old_ids, key=len, reverse=True))) + r")\b")
        for key in ("system_prompt", "user_prompt"):
            row[key] = token_pattern.sub(lambda match: old_to_new[match.group(0)], source[key])
        row["feature_ids"] = [old_to_new[item] for item in source["feature_ids"]]
        row["protocol_id"] = PROTOCOL
        row["prompt_sha256"] = hashlib.sha256((row["system_prompt"] + "\0" + row["user_prompt"]).encode()).hexdigest()

        # Check that all machine-readable feature occurrences in the embedded payload were renamed.
        original_offset = source["user_prompt"].find('\n{\n  "feature_schema"')
        marker = '\n{\n  "feature_schema"'
        offset = row["user_prompt"].find(marker)
        if offset < 0 or original_offset < 0:
            raise RuntimeError(f"Embedded payload marker missing for {source['case_id']}")
        original_payload = json.loads(source["user_prompt"][original_offset + 1 :])
        payload = json.loads(row["user_prompt"][offset + 1 :])
        schema_ids = {item["feature_id"] for item in payload["feature_schema"]}
        if schema_ids != set(row["feature_ids"]):
            raise RuntimeError(f"Renamed schema and ID list disagree for {source['case_id']}")
        if row["feature_ids"] != [old_to_new[item] for item in source["feature_ids"]]:
            raise RuntimeError(f"Feature order changed for {source['case_id']}")
        original_schema = {item["feature_id"]: item for item in original_payload["feature_schema"]}
        renamed_schema = {item["feature_id"]: item for item in payload["feature_schema"]}
        for old_id, new_id in old_to_new.items():
            old_item = {key: value for key, value in original_schema[old_id].items() if key != "feature_id"}
            new_item = {key: value for key, value in renamed_schema[new_id].items() if key != "feature_id"}
            if old_item != new_item:
                raise RuntimeError(f"Schema values changed while renaming {source['case_id']}/{old_id}")
        for old_row, new_row in zip(
            [original_payload["target_row"], *original_payload["normal_reference_rows"]],
            [payload["target_row"], *payload["normal_reference_rows"]],
            strict=True,
        ):
            for old_id, new_id in old_to_new.items():
                if old_row[old_id] != new_row[new_id]:
                    raise RuntimeError(f"Table values/order changed while renaming {source['case_id']}/{old_id}")
        for item in [payload["target_row"], *payload["normal_reference_rows"]]:
            if set(item) != schema_ids:
                raise RuntimeError(f"Renamed row keys and schema disagree for {source['case_id']}")

        variants.append(row)
        mappings[source["case_id"]] = {"old_to_new": old_to_new, "new_to_old": new_to_old}

    variant_path = OUT / "inputs_opaque_id_permutation.jsonl"
    write_jsonl(variant_path, variants)
    case_templates = {
        row["case_id"]: row.get("template_id")
        for row in (json.loads(line) for line in (HERE / "case_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    }
    template_count = len({case_templates.get(row["case_id"]) for row in variants} - {None})
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    manifest = {
        "protocol_id": PROTOCOL,
        "n_cases": len(variants),
        "n_templates": template_count,
        "selection_source": str(SOURCE.relative_to(HERE)),
        "selection_source_sha256": source_hash,
        "permutation_seed": SEED,
        "design": "Within each case, apply a deterministic bijection over opaque F### feature IDs. Values, feature order, reference-row order, task wording, and case membership are unchanged. Model predictions are mapped back to original IDs before scoring.",
        "does_not_test": "Semantic feature names, prompt wording, row order, or field order.",
        "primary_metrics": ["support_any_minimal_hit", "support_best_jaccard"],
        "inference_plan": "Two-sided template-level sign flips and template-cluster bootstrap; Holm across two primary metrics; top-one set stability descriptive.",
        "status": "Post-hoc sensitivity probe; condition and metrics were fixed before this condition's inference, not preregistered.",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "case_mappings": mappings,
        "variant_input_sha256": hashlib.sha256(variant_path.read_bytes()).hexdigest(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"n_cases": len(variants), "n_templates": manifest["n_templates"], "out": str(variant_path)}, indent=2))


if __name__ == "__main__":
    main()
