#!/usr/bin/env python3
"""Generate v2 known-anomaly subspace benchmark with template-group splits."""

from __future__ import annotations

import hashlib
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SEED = 20261106
CASES_PER_TEMPLATE = 12
DIMENSIONS = (6, 12, 24)
REFERENCE_SIZES = (8, 16, 32)
NOISE_PROFILES = (
    "independent",
    "high_cardinality_noise",
    "noisy_proxy",
    "deterministic_proxy_preserved",
    "deterministic_proxy_broken",
    "missingness_noise",
)
NUMERIC = [0.0, 0.25, 0.5, 0.75, 1.0]
ORDINAL = [0, 1, 2, 3, 4]
BINARY = [0, 1]


def field(kind: str, values: list) -> dict:
    return {"type": kind, "values": values}


def atom(op: str, role: str, value=None, other=None) -> dict:
    result = {"op": op, "role": role}
    if value is not None:
        result["value"] = value
    if other is not None:
        result["other"] = other
    return result


def combine(op: str, *children: dict, k: int | None = None) -> dict:
    result = {"op": op, "children": list(children)}
    if k is not None:
        result["k"] = k
    return result


def templates() -> list[dict]:
    cat4 = ["c0", "c1", "c2", "c3"]
    cat5 = ["c0", "c1", "c2", "c3", "c4"]
    cat6 = ["c0", "c1", "c2", "c3", "c4", "c5"]
    out = []

    def add(family, template_id, roles, rule, description, decoy=False):
        out.append({
            "task_label": family,
            "template_id": template_id,
            "roles": roles,
            "rule": rule,
            "rule_description": description,
            "decoy_role": "decoy" if decoy else None,
        })

    # Univariate numeric mechanisms.
    add("numeric_marginal", "num_ge_075", {"signal": field("numeric", NUMERIC)}, atom("ge", "signal", 0.75), "one numeric value is at or above a threshold")
    add("numeric_marginal", "num_ge_050", {"signal": field("numeric", NUMERIC)}, atom("ge", "signal", 0.5), "one numeric value is at or above a threshold")
    add("numeric_marginal", "num_le_025", {"signal": field("numeric", NUMERIC)}, atom("le", "signal", 0.25), "one numeric value is at or below a threshold")
    add("numeric_marginal", "num_le_050", {"signal": field("numeric", NUMERIC)}, atom("le", "signal", 0.5), "one numeric value is at or below a threshold")
    add("numeric_marginal", "num_outside_mid", {"signal": field("ordinal", ORDINAL)}, combine("or", atom("lt", "signal", 1), atom("gt", "signal", 3)), "one ordinal value is outside an interval")

    # Univariate categorical mechanisms with varying domain cardinality.
    add("categorical_rare", "cat_eq_c3_d4", {"signal": field("categorical", cat4)}, atom("eq", "signal", "c3"), "one category value is anomalous")
    add("categorical_rare", "cat_eq_c4_d5", {"signal": field("categorical", cat5)}, atom("eq", "signal", "c4"), "one category value is anomalous")
    add("categorical_rare", "cat_eq_c5_d6", {"signal": field("categorical", cat6)}, atom("eq", "signal", "c5"), "one category value is anomalous")
    add("categorical_rare", "cat_in_rare_pair", {"signal": field("categorical", cat5)}, atom("in", "signal", ["c3", "c4"]), "one of a pair of category values is anomalous")
    add("categorical_rare", "cat_not_common", {"signal": field("categorical", cat5)}, atom("not_in", "signal", ["c0", "c1", "c2"]), "a category outside the common subset is anomalous")

    # Two-field conjunctions, including mixed-type conditions.
    add("pairwise_interaction", "pair_binary_and", {"left": field("binary", BINARY), "right": field("binary", BINARY)}, combine("and", atom("eq", "left", 1), atom("eq", "right", 1)), "two binary conditions must hold together")
    add("pairwise_interaction", "pair_num_edges", {"left": field("numeric", NUMERIC), "right": field("numeric", NUMERIC)}, combine("and", atom("ge", "left", 0.75), atom("le", "right", 0.25)), "two numeric conditions must hold together")
    add("pairwise_interaction", "pair_cat_num", {"left": field("categorical", cat4), "right": field("numeric", NUMERIC)}, combine("and", atom("eq", "left", "c2"), atom("ge", "right", 0.75)), "a category and a numeric condition must hold together")
    add("pairwise_interaction", "pair_cat_binary", {"left": field("categorical", cat5), "right": field("binary", BINARY)}, combine("and", atom("eq", "left", "c4"), atom("eq", "right", 0)), "a category and a binary condition must hold together")
    add("pairwise_interaction", "pair_num_cat_set", {"left": field("numeric", NUMERIC), "right": field("categorical", cat6)}, combine("and", atom("gt", "left", 0.5), atom("in", "right", ["c4", "c5"])), "numeric and categorical conditions must hold together")

    # Context-conditioned group mechanisms.
    add("conditional_group", "cond_binary_num", {"context": field("binary", BINARY), "signal": field("numeric", NUMERIC)}, combine("and", atom("eq", "context", 1), atom("ge", "signal", 0.75)), "a numeric value is anomalous in one binary context")
    add("conditional_group", "cond_cat_zero", {"context": field("categorical", cat4), "signal": field("binary", BINARY)}, combine("and", atom("eq", "context", "c2"), atom("eq", "signal", 0)), "a zero signal is anomalous in one category group")
    add("conditional_group", "cond_num_cat", {"context": field("numeric", NUMERIC), "signal": field("categorical", cat5)}, combine("and", atom("le", "context", 0.25), atom("eq", "signal", "c4")), "a category is anomalous in one numeric context")
    add("conditional_group", "cond_cat_num_low", {"context": field("categorical", cat6), "signal": field("numeric", NUMERIC)}, combine("and", atom("eq", "context", "c5"), atom("le", "signal", 0.25)), "a low numeric value is anomalous in one category group")
    add("conditional_group", "cond_num_num", {"context": field("numeric", NUMERIC), "signal": field("numeric", NUMERIC)}, combine("and", atom("ge", "context", 0.75), atom("eq", "signal", 0.5)), "a numeric value is anomalous under a numeric context")

    # Alternative sufficient reasons; each branch is independently sufficient.
    add("multiple_valid", "or_binary", {"left": field("binary", BINARY), "right": field("binary", BINARY)}, combine("or", atom("eq", "left", 1), atom("eq", "right", 1)), "either of two fields independently signals anomaly")
    add("multiple_valid", "or_num_cat", {"left": field("numeric", NUMERIC), "right": field("categorical", cat5)}, combine("or", atom("ge", "left", 0.75), atom("eq", "right", "c4")), "either numeric or categorical evidence is sufficient")
    add("multiple_valid", "or_cat_num_low", {"left": field("categorical", cat4), "right": field("numeric", NUMERIC)}, combine("or", atom("eq", "left", "c3"), atom("le", "right", 0.25)), "either categorical or numeric evidence is sufficient")
    add("multiple_valid", "or_two_categories", {"left": field("categorical", cat5), "right": field("categorical", cat5)}, combine("or", atom("eq", "left", "c4"), atom("eq", "right", "c4")), "either of two category values is sufficient")
    add("multiple_valid", "or_binary_num", {"left": field("binary", BINARY), "right": field("numeric", NUMERIC)}, combine("or", atom("eq", "left", 0), atom("ge", "right", 0.75)), "either binary or numeric evidence is sufficient")

    # A deliberately changed but rule-irrelevant decoy accompanies the signal.
    add("modified_vs_mechanism", "decoy_binary", {"signal": field("binary", BINARY), "decoy": field("binary", BINARY)}, atom("eq", "signal", 1), "one field governs the rule while an incidental field also changes", True)
    add("modified_vs_mechanism", "decoy_num_cat", {"signal": field("numeric", NUMERIC), "decoy": field("categorical", cat5)}, atom("ge", "signal", 0.75), "one field governs the rule while an incidental category also changes", True)
    add("modified_vs_mechanism", "decoy_cat_num", {"signal": field("categorical", cat5), "decoy": field("numeric", NUMERIC)}, atom("eq", "signal", "c4"), "one category governs the rule while an incidental number also changes", True)
    add("modified_vs_mechanism", "decoy_binary_zero", {"signal": field("binary", BINARY), "decoy": field("categorical", cat4)}, atom("eq", "signal", 0), "one field governs the rule while an incidental category also changes", True)
    add("modified_vs_mechanism", "decoy_num_low", {"signal": field("numeric", NUMERIC), "decoy": field("binary", BINARY)}, atom("le", "signal", 0.25), "one numeric field governs the rule while an incidental flag also changes", True)

    # Relational violations.
    add("relational", "rel_binary_neq", {"left": field("binary", BINARY), "right": field("binary", BINARY)}, atom("neq_roles", "left", other="right"), "two field values disagree")
    add("relational", "rel_cat_neq", {"left": field("categorical", cat4), "right": field("categorical", cat4)}, atom("neq_roles", "left", other="right"), "two category values disagree")
    add("relational", "rel_num_gt", {"left": field("ordinal", ORDINAL), "right": field("ordinal", ORDINAL)}, atom("gt_roles", "left", other="right"), "one ordinal field exceeds another")
    add("relational", "rel_num_gap", {"left": field("numeric", NUMERIC), "right": field("numeric", NUMERIC)}, atom("abs_diff_gt", "left", 0.5, other="right"), "two numeric fields differ by more than a threshold")
    add("relational", "rel_sum", {"left": field("numeric", NUMERIC), "right": field("numeric", NUMERIC)}, atom("sum_ge", "left", 1.25, other="right"), "a relation over two numeric fields exceeds a threshold")

    # Three-way interactions with binary and mixed-type variants.
    add("higher_order", "three_parity", {"a": field("binary", BINARY), "b": field("binary", BINARY), "c": field("binary", BINARY)}, combine("parity", atom("eq", "a", 1), atom("eq", "b", 1), atom("eq", "c", 1)), "odd parity across three binary fields is anomalous")
    add("higher_order", "three_and", {"a": field("binary", BINARY), "b": field("binary", BINARY), "c": field("binary", BINARY)}, combine("and", atom("eq", "a", 1), atom("eq", "b", 1), atom("eq", "c", 1)), "three binary conditions must all hold")
    add("higher_order", "three_at_least_two", {"a": field("binary", BINARY), "b": field("binary", BINARY), "c": field("binary", BINARY)}, combine("at_least", atom("eq", "a", 1), atom("eq", "b", 1), atom("eq", "c", 1), k=2), "at least two of three binary conditions must hold")
    add("higher_order", "three_mixed", {"a": field("categorical", cat4), "b": field("binary", BINARY), "c": field("ordinal", ORDINAL)}, combine("and", atom("eq", "a", "c3"), atom("eq", "b", 1), atom("ge", "c", 3)), "a category, a binary flag, and an ordinal condition interact")
    add("higher_order", "three_xor_mixed", {"a": field("binary", BINARY), "b": field("categorical", cat5), "c": field("numeric", NUMERIC)}, combine("parity", atom("eq", "a", 1), atom("eq", "b", "c4"), atom("ge", "c", 0.75)), "odd parity across mixed-type conditions is anomalous")
    return out


TASK_DESCRIPTIONS = {
    "numeric_marginal": "single-field numeric condition",
    "categorical_rare": "single-field categorical condition",
    "pairwise_interaction": "two-field interaction",
    "conditional_group": "context- or group-conditioned pattern",
    "multiple_valid": "multiple independently sufficient alternatives",
    "modified_vs_mechanism": "changed field that may be irrelevant to the rule",
    "relational": "cross-field relational violation",
    "higher_order": "three-way or higher-order interaction",
}


def eval_expr(expr: dict, assignment: dict) -> bool:
    op = expr["op"]
    if op in {"and", "or", "parity", "at_least"}:
        values = [eval_expr(child, assignment) for child in expr["children"]]
        if op == "and":
            return all(values)
        if op == "or":
            return any(values)
        if op == "parity":
            return sum(values) % 2 == 1
        return sum(values) >= expr["k"]
    left = assignment[expr["role"]]
    if op == "eq":
        return left == expr["value"]
    if op == "neq":
        return left != expr["value"]
    if op == "in":
        return left in expr["value"]
    if op == "not_in":
        return left not in expr["value"]
    if op == "lt":
        return left < expr["value"]
    if op == "le":
        return left <= expr["value"]
    if op == "gt":
        return left > expr["value"]
    if op == "ge":
        return left >= expr["value"]
    right = assignment[expr["other"]]
    if op == "neq_roles":
        return left != right
    if op == "gt_roles":
        return left > right
    if op == "abs_diff_gt":
        return abs(left - right) > expr["value"]
    if op == "sum_ge":
        return left + right >= expr["value"]
    raise ValueError(f"Unsupported expression: {op}")


def bind_expr(expr: dict, role_to_feature: dict[str, str]) -> dict:
    bound = dict(expr)
    if "role" in bound:
        bound["role"] = role_to_feature[bound["role"]]
    if "other" in bound and bound["other"] in role_to_feature:
        bound["other"] = role_to_feature[bound["other"]]
    if "children" in expr:
        bound["children"] = [bind_expr(child, role_to_feature) for child in expr["children"]]
    return bound


def all_role_assignments(role_specs: dict[str, dict]) -> list[dict]:
    roles = list(role_specs)
    domains = [role_specs[role]["values"] for role in roles]
    return [dict(zip(roles, values)) for values in itertools.product(*domains)]


def rule_labels(template: dict) -> tuple[list[dict], list[dict], list[str]]:
    assignments = all_role_assignments(template["roles"])
    clean = [row for row in assignments if not eval_expr(template["rule"], row)]
    anomalous = [row for row in assignments if eval_expr(template["rule"], row)]
    role_names = list(template["roles"])
    mechanism = []
    for role in role_names:
        for row in assignments:
            if any(eval_expr(template["rule"], {**row, role: v}) != eval_expr(template["rule"], row)
                   for v in template["roles"][role]["values"] if v != row[role]):
                mechanism.append(role)
                break
    return clean, anomalous, mechanism


def rule_roles(expr: dict) -> set[str]:
    roles = set()
    if "role" in expr:
        roles.add(expr["role"])
    if "other" in expr and isinstance(expr["other"], str):
        roles.add(expr["other"])
    for child in expr.get("children", []):
        roles.update(rule_roles(child))
    return roles


def make_feature_id(index: int) -> str:
    return f"F{index:03d}"


NEUTRAL_NAMES = [
    "record_code", "measure_alpha", "channel_beta", "reading_gamma", "group_delta",
    "status_epsilon", "signal_zeta", "value_eta", "class_theta", "metric_iota",
    "field_kappa", "series_lambda", "attribute_mu", "column_nu", "index_xi",
    "observation_omicron", "factor_pi", "dimension_rho", "feature_sigma", "entry_tau",
    "marker_upsilon", "sample_phi", "component_chi", "variable_psi",
]


def sample_value(values: list, rng: np.random.Generator):
    return values[int(rng.integers(0, len(values)))]


def complete_reference_labels(
    template: dict,
    role_to_feature: dict[str, str],
    target: dict,
    clean_role_assignments: list[dict],
    proxy_meta: dict | None,
) -> tuple[list[list[str]], list[list[str]], list[list[str]], list[list[str]]]:
    active_ids = [role_to_feature[role] for role in template["roles"]]
    rule_valid = []
    for subset_size in range(len(active_ids) + 1):
        for subset in itertools.combinations(active_ids, subset_size):
            selected_roles = [role for role, fid in role_to_feature.items() if fid in subset]
            completions = [row for row in all_role_assignments(template["roles"])
                           if all(row[role] == target[role_to_feature[role]] for role in selected_roles)]
            if completions and all(eval_expr(template["rule"], row) for row in completions):
                rule_valid.append(list(subset))
    rule_min = [s for s in rule_valid if not any(set(o) < set(s) for o in rule_valid)]

    candidate_ids = active_ids[:]
    if proxy_meta and proxy_meta.get("kind") == "deterministic_copy":
        candidate_ids.append(proxy_meta["feature_id"])
    support_valid = []
    for subset_size in range(len(candidate_ids) + 1):
        for subset in itertools.combinations(candidate_ids, subset_size):
            has_clean_completion = False
            for assignment in clean_role_assignments:
                if any(target[fid] != assignment[next(role for role, role_fid in role_to_feature.items() if role_fid == fid)]
                       for fid in subset if fid in active_ids):
                    continue
                if proxy_meta and proxy_meta.get("kind") == "deterministic_copy" and proxy_meta["feature_id"] in subset:
                    source_role = next(role for role, fid in role_to_feature.items() if fid == proxy_meta["source_feature_id"])
                    if target[proxy_meta["feature_id"]] != assignment[source_role]:
                        continue
                has_clean_completion = True
                break
            if not has_clean_completion:
                support_valid.append(list(subset))
    support_min = [s for s in support_valid if not any(set(o) < set(s) for o in support_valid)]
    return rule_valid, rule_min, support_valid, support_min


def make_case(
    template: dict,
    template_index: int,
    case_index: int,
    split: str,
    rng: np.random.Generator,
    case_id: str,
    factors: dict,
) -> tuple[dict, dict, dict]:
    dim = factors["feature_count"]
    n_ref = factors["reference_count"]
    profile = factors["nuisance_profile"]
    name_mode = "opaque" if case_index % 2 == 0 else "neutral_names"

    feature_ids = [make_feature_id(i) for i in range(1, dim + 1)]
    permuted_ids = list(rng.permutation(feature_ids))
    role_names = list(template["roles"])
    role_to_feature = {role: permuted_ids[i] for i, role in enumerate(role_names)}
    active_ids = set(role_to_feature.values())
    noise_ids = [fid for fid in permuted_ids if fid not in active_ids]

    clean, anomalous, mechanism_roles = rule_labels(template)
    before_roles = sample_value(clean, rng)
    target_roles = sample_value([row for row in anomalous if any(row[r] != before_roles[r] for r in role_names)], rng)
    before_feature_values = {role_to_feature[r]: before_roles[r] for r in role_names}
    target_feature_values = {role_to_feature[r]: target_roles[r] for r in role_names}

    schema_by_id = {}
    for role, fid in role_to_feature.items():
        spec = template["roles"][role]
        schema_by_id[fid] = {"feature_id": fid, "value_type": spec["type"], "allowed_values": list(spec["values"])}

    proxy_meta = None
    special_noise_id = noise_ids[0] if noise_ids else None
    for i, fid in enumerate(noise_ids):
        if profile == "high_cardinality_noise" and i == 0:
            values = [f"level_{j:02d}" for j in range(64)]
            schema_by_id[fid] = {"feature_id": fid, "value_type": "categorical", "allowed_values": values}
        elif profile in {"noisy_proxy", "deterministic_proxy_preserved", "deterministic_proxy_broken"} and i == 0:
            possible_sources = sorted(rule_roles(template["rule"]))
            source_role = possible_sources[case_index % len(possible_sources)]
            source_id = role_to_feature[source_role]
            spec = template["roles"][source_role]
            schema_by_id[fid] = {"feature_id": fid, "value_type": spec["type"], "allowed_values": list(spec["values"])}
            if profile == "noisy_proxy":
                flip_probability = 0.2 if case_index % 2 == 0 else 0.4
                proxy_meta = {"kind": "noisy_copy", "feature_id": fid, "source_feature_id": source_id, "flip_probability": flip_probability}
            else:
                proxy_meta = {"kind": "deterministic_copy", "feature_id": fid, "source_feature_id": source_id}
        elif profile == "missingness_noise" and i == 0:
            values = [f"token_{j:02d}" for j in range(8)]
            schema_by_id[fid] = {
                "feature_id": fid,
                "value_type": "categorical",
                "allowed_values": values + [None],
                "missing_rate": 0.15,
                "nullable": True,
            }
        else:
            kind = ("numeric", "categorical", "binary", "ordinal", "datetime", "identifier")[(i + case_index) % 6]
            if kind == "numeric":
                values = list(NUMERIC)
            elif kind == "binary":
                values = list(BINARY)
            elif kind == "ordinal":
                values = list(ORDINAL)
            elif kind == "datetime":
                values = [f"2026-01-{day:02d}" for day in range(1, 13)]
            elif kind == "identifier":
                values = [f"id_{j:03d}" for j in range(64)]
            else:
                values = ["u0", "u1", "u2", "u3"]
            schema_by_id[fid] = {"feature_id": fid, "value_type": kind, "allowed_values": values}

    def sample_noise(fid: str, active_values: dict[str, object], target_row: bool, before_active: dict[str, object]):
        if proxy_meta and fid == proxy_meta["feature_id"]:
            source = proxy_meta["source_feature_id"]
            if proxy_meta["kind"] == "deterministic_copy":
                if target_row and profile == "deterministic_proxy_broken":
                    return before_row[fid]
                return active_values[source]
            if target_row:
                return before_row[fid]
            if rng.random() >= proxy_meta["flip_probability"]:
                return active_values[source]
        if profile == "missingness_noise" and fid == special_noise_id:
            if rng.random() < schema_by_id[fid]["missing_rate"]:
                return None
        values = schema_by_id[fid]["allowed_values"]
        if profile == "missingness_noise" and fid == special_noise_id:
            values = [value for value in values if value is not None]
        return sample_value(values, rng)

    def assemble_row(active_values: dict[str, object], target_row: bool = False) -> dict:
        row = {fid: sample_noise(fid, active_values, target_row, before_feature_values) for fid in noise_ids}
        row.update(active_values)
        if template["decoy_role"]:
            decoy_id = role_to_feature[template["decoy_role"]]
            base = row[decoy_id]
            vals = schema_by_id[decoy_id]["allowed_values"]
            if target_row and base == before_row[decoy_id]:
                alternatives = [v for v in vals if v != base]
                if alternatives:
                    row[decoy_id] = sample_value(alternatives, rng)
        return {fid: row[fid] for fid in permuted_ids}

    before_row = assemble_row(before_feature_values, False)
    # The target inherits all non-intervened nuisance values from the baseline.
    target_row = dict(before_row)
    target_row.update(target_feature_values)
    if proxy_meta and proxy_meta.get("kind") == "deterministic_copy":
        if profile == "deterministic_proxy_preserved":
            target_row[proxy_meta["feature_id"]] = target_feature_values[proxy_meta["source_feature_id"]]
        elif profile == "deterministic_proxy_broken":
            target_row[proxy_meta["feature_id"]] = before_row[proxy_meta["feature_id"]]
    if template["decoy_role"]:
        decoy_id = role_to_feature[template["decoy_role"]]
        if target_row[decoy_id] == before_row[decoy_id]:
            alternatives = [v for v in schema_by_id[decoy_id]["allowed_values"] if v != target_row[decoy_id]]
            target_row[decoy_id] = sample_value(alternatives, rng)

    clean_rows = []
    for _ in range(n_ref):
        clean_assignment = sample_value(clean, rng)
        active_values = {role_to_feature[role]: clean_assignment[role] for role in role_names}
        clean_rows.append(assemble_row(active_values, False))
    rng.shuffle(clean_rows)

    # Hide generator roles; use neutral display names only as an experimental factor.
    schema = []
    name_order = list(rng.permutation(NEUTRAL_NAMES[:dim]))
    for position, fid in enumerate(permuted_ids):
        item = dict(schema_by_id[fid])
        if name_mode == "neutral_names":
            item["field_name"] = name_order[position]
        schema.append(item)

    clean_role_assignments = clean
    rule_valid_sets, rule_min_sets, support_valid_sets, support_min_sets = complete_reference_labels(
        template, role_to_feature, target_row, clean_role_assignments, proxy_meta
    )
    directly_intervened = [role_to_feature[role] for role in role_names if before_roles[role] != target_roles[role]]
    if template["decoy_role"]:
        directly_intervened.append(role_to_feature[template["decoy_role"]])
    changed_ids = sorted(set(directly_intervened))
    value_changed_ids = [fid for fid in permuted_ids if before_row[fid] != target_row[fid]]
    mechanism_ids = [role_to_feature[role] for role in mechanism_roles]
    task_options = "\n".join(f"- {key}: {value}" for key, value in TASK_DESCRIPTIONS.items())
    system_prompt = (
        "You localize explanations for an already-known anomalous table row. "
        "Use only the schema, normal reference rows, target row, and supplied task-label options. "
        "Do not claim real-world causality. Return one JSON object only."
    )
    user_prompt = (
        "The target row is already known to be anomalous. "
        "Infer the broad task label, identify the fields that appear essential to the anomalous "
        "pattern, and return up to four ranked candidate "
        "inclusion-minimal feature subspaces that distinguish the target from the normal population. "
        "A field may be a useful descriptive explanation without being a rule mechanism, and a "
        "field changed by a hidden generator may be irrelevant to the mechanism. Multiple valid "
        "subspaces may exist. Directly modified fields cannot be inferred from this unpaired input "
        "and are retained only as hidden provenance labels for a separate paired-input track. "
        "Use only feature IDs from the schema. Return JSON with exactly these fields: "
        "task_label, mechanism_feature_ids, explanations, abstain. "
        "task_label must be one of the supplied keys; mechanism_feature_ids must be an array; "
        "explanations must be an ordered array of objects with features, evidence, confidence "
        "(number from 0 to 1); abstain must be a boolean. If uncertain, lower confidence or abstain. "
        "An explanation features array must contain no duplicates. Do not include analysis, "
        "deliberation, or a preamble. Keep each evidence value to one factual sentence of at most "
        "20 words. Example shape: "
        "{\"task_label\":\"pairwise_interaction\",\"mechanism_feature_ids\":[],"
        "\"explanations\":[{\"features\":[\"F001\"],"
        "\"evidence\":\"contrast with normal references\",\"confidence\":0.6}],\"abstain\":false}.\n\n"
        "Allowed task labels:\n" + task_options + "\n\n" +
        json.dumps({
            "feature_schema": schema,
            "normal_reference_rows": clean_rows,
            "target_row": target_row,
        }, ensure_ascii=False, indent=2)
    )
    prompt_hash = hashlib.sha256((system_prompt + "\0" + user_prompt).encode("utf-8")).hexdigest()

    input_record = {
        "case_id": case_id,
        "split": split,
        "feature_count": dim,
        "feature_ids": list(permuted_ids),
        "reference_count": n_ref,
        "nuisance_profile": profile,
        "field_name_mode": name_mode,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt_sha256": prompt_hash,
    }
    ground_truth = {
        "case_id": case_id,
        "task_label": template["task_label"],
        "template_id": template["template_id"],
        "template_group_id": template["template_id"],
        "split": split,
        "rule_ast": bind_expr(template["rule"], role_to_feature),
        "rule_role_to_feature": role_to_feature,
        "modified_feature_ids": changed_ids,
        "changed_values": [{"feature_id": fid, "before": before_row[fid], "after": target_row[fid]} for fid in changed_ids],
        "value_different_from_baseline_feature_ids": value_changed_ids,
        "mechanism_feature_ids": mechanism_ids,
        "rule_sufficient_subspaces": [sorted(s) for s in rule_valid_sets],
        "rule_minimal_subspaces": [sorted(s) for s in rule_min_sets],
        "reference_support_sufficient_subspaces": [sorted(s) for s in support_valid_sets],
        "reference_support_sufficient_minimal_subspaces": [sorted(s) for s in support_min_sets],
        "nuisance_profile": profile,
        "proxy_metadata": proxy_meta,
        "feature_count": dim,
        "reference_count": n_ref,
        "field_name_mode": name_mode,
        "before_row_hidden": before_row,
        "target_row": target_row,
        "normal_reference_rows": clean_rows,
        "label_semantics": {
            "mechanism": "rule-essential features under the procedural predicate",
            "rule_sufficient": "subset whose target-matching full-domain completions all satisfy the anomaly rule",
            "rule_minimal": "inclusion-minimal rule-sufficient subset",
            "reference_support_sufficient": "subset whose target projection has no completion in the complete clean population support",
            "reference_support_minimal": "inclusion-minimal subset sufficient to exclude the complete clean population support",
            "modified": "directly changed fields between the hidden baseline and injected target; not inferable from unpaired main-task inputs",
            "value_different_from_baseline": "all fields whose values differ between hidden baseline and target, including propagated proxy changes",
        },
    }
    hidden_manifest = {
        "case_id": case_id,
        "split": split,
        "task_label": template["task_label"],
        "template_id": template["template_id"],
        "template_group_id": template["template_id"],
        "feature_count": dim,
        "reference_count": n_ref,
        "nuisance_profile": profile,
        "field_name_mode": name_mode,
    }
    return input_record, ground_truth, hidden_manifest


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--outdir", type=Path, default=ROOT)
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_templates = templates()
    assert len(all_templates) == 40
    assert all(sum(t["task_label"] == family for t in all_templates) == 5 for family in TASK_DESCRIPTIONS)

    inputs, gold, case_manifest = [], [], []
    family_positions: dict[str, int] = {}
    for t_index, template in enumerate(all_templates):
        family = template["task_label"]
        family_position = family_positions.get(family, 0)
        family_positions[family] = family_position + 1
        if args.seed == SEED:
            # Preserve the original v2 split exactly for the registered seed.
            split = ("train", "dev", "test", "test", "test")[family_position]
        else:
            # Keep whole templates grouped while rotating the held-out templates
            # across independent benchmark replicates.
            family_index = list(TASK_DESCRIPTIONS).index(family)
            rotation = (args.seed - SEED) % 5
            train_position = (family_index + rotation) % 5
            dev_position = (train_position + 1) % 5
            split = "train" if family_position == train_position else (
                "dev" if family_position == dev_position else "test"
            )
        dims = list(DIMENSIONS) * (CASES_PER_TEMPLATE // len(DIMENSIONS))
        refs = list(REFERENCE_SIZES) * (CASES_PER_TEMPLATE // len(REFERENCE_SIZES))
        profiles = list(NOISE_PROFILES) * (CASES_PER_TEMPLATE // len(NOISE_PROFILES))
        profiles += list(NOISE_PROFILES[:CASES_PER_TEMPLATE - len(profiles)])
        rng.shuffle(dims)
        rng.shuffle(refs)
        rng.shuffle(profiles)
        for case_index in range(CASES_PER_TEMPLATE):
            factors = {
                "feature_count": dims[case_index],
                "reference_count": refs[case_index],
                "nuisance_profile": profiles[case_index],
            }
            inputs.append((template, t_index, case_index, split, factors))
    rng.shuffle(inputs)

    materialized = []
    for global_index, (template, t_index, case_index, split, factors) in enumerate(inputs, start=1):
        case_id = f"V2_{global_index:04d}"
        materialized.append(make_case(template, t_index, case_index, split, rng, case_id, factors))
    inputs_out = [item[0] for item in materialized]
    gold_out = [item[1] for item in materialized]
    manifest_out = [item[2] for item in materialized]

    write_jsonl(outdir / "inputs.jsonl", inputs_out)
    write_jsonl(outdir / "ground_truth.jsonl", gold_out)
    write_jsonl(outdir / "case_manifest.jsonl", manifest_out)
    (outdir / "template_catalog.json").write_text(
        json.dumps(all_templates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    counts = {split: sum(row["split"] == split for row in manifest_out) for split in ("train", "dev", "test")}
    template_counts = {split: sorted({row["template_id"] for row in manifest_out if row["split"] == split})
                       for split in ("train", "dev", "test")}
    manifest = {
        "benchmark_id": "known_anomaly_subspace_benchmark_v2",
        "replicate_seed": args.seed,
        "prompt_revision": "compact_json_v1",
        "prompt_output_constraints": "JSON only; each evidence field at most 20 words; no analysis or preamble",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generator_runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
        "seed": args.seed,
        "cases_per_template": CASES_PER_TEMPLATE,
        "n_templates": len(all_templates),
        "templates_per_family": 5,
        "n_families": len(TASK_DESCRIPTIONS),
        "split_case_counts": counts,
        "split_template_counts": {k: len(v) for k, v in template_counts.items()},
        "split_template_ids": template_counts,
        "feature_counts": list(DIMENSIONS),
        "feature_types": ["binary", "categorical", "datetime", "identifier", "numeric", "ordinal"],
        "reference_sizes": list(REFERENCE_SIZES),
        "nuisance_profiles": list(NOISE_PROFILES),
        "nuisance_profile_definitions": {
            "independent": "independent low-cardinality noise fields",
            "high_cardinality_noise": "independent 64-level categorical distractor",
            "noisy_proxy": "one proxy field correlated with a mechanism field, with 0.2 or 0.4 flip probability",
            "deterministic_proxy_preserved": "a deterministic copy of a mechanism field preserved in the target",
            "deterministic_proxy_broken": "normal rows contain a deterministic copy, but the target proxy is not updated",
            "missingness_noise": "one independent categorical distractor with 15% missingness",
        },
        "main_input_excludes_before_row": True,
        "main_task_does_not_request_modified_feature_prediction": True,
        "main_task_output_fields": ["task_label", "mechanism_feature_ids", "explanations", "abstain"],
        "split_unit": "rule template group; no template id crosses splits",
        "primary_explanation_gold": "minimal subspaces absent from complete clean population support",
        "separate_procedural_gold": "rule-sufficient and minimal subspaces; mechanism and modified fields (modified fields are hidden provenance only for the unpaired primary task)",
        "artifacts_sha256": {},
    }
    for name in ("inputs.jsonl", "ground_truth.jsonl", "case_manifest.jsonl", "template_catalog.json"):
        manifest["artifacts_sha256"][name] = hashlib.sha256((outdir / name).read_bytes()).hexdigest()
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split_case_counts": counts, "split_template_counts": manifest["split_template_counts"],
                      "templates_per_family": 5, "cases_per_template": CASES_PER_TEMPLATE}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
