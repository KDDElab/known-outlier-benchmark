#!/usr/bin/env python3
"""Paper-guided RLAX adaptation for the single-target mixed-type benchmark.

This is an independent reproduction/adaptation, not the authors' implementation.
It follows the published subspace-margin reward, PPO action policy, MDR/policy
feature rankings, and Gain-based prefix selection, with the query's normal rows
split into disjoint reference-X and negative-example sets as in the paper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
PAPER = "https://doi.org/10.1007/s10994-026-07182-z"
OFFICIAL_REPO = "https://github.com/AIDALab-DIMES/RLAX"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def payload(case: dict) -> dict:
    prompt = case["user_prompt"]
    start = prompt.find("\n{")
    if start < 0:
        raise ValueError(f"Cannot locate JSON input payload for {case['case_id']}")
    return json.loads(prompt[start + 1 :])


def normalized_matrix(schema: list[dict], target: dict, refs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode each original field as one scalar, with reference-fitted scaling."""
    rows = [target, *refs]
    columns = []
    for spec in schema:
        fid, kind = spec["feature_id"], spec["value_type"]
        values = [row.get(fid) for row in rows]
        if kind in {"numeric", "ordinal", "date", "datetime"}:
            numeric = np.asarray([
                np.nan if v is None else float(np.datetime64(str(v), "D").astype("int64"))
                if kind in {"date", "datetime"} else float(v) for v in values
            ], dtype=float)
            finite_refs = numeric[1:][np.isfinite(numeric[1:])]
            lo = float(finite_refs.min()) if finite_refs.size else 0.0
            hi = float(finite_refs.max()) if finite_refs.size else lo
            span = hi - lo
            if not np.isfinite(span) or span == 0.0:
                span = 1.0
            col = np.where(np.isfinite(numeric), (numeric - lo) / span, np.nan)
        else:
            col = np.asarray([np.nan if v is None else float(v) if isinstance(v, bool) else v for v in values], dtype=object)
        columns.append(col)
    matrix = np.zeros((len(rows), len(schema)), dtype=np.float32)
    missing = np.zeros((len(rows), len(schema)), dtype=bool)
    for j, spec in enumerate(schema):
        col = columns[j]
        if spec["value_type"] in {"numeric", "ordinal", "date", "datetime"}:
            vals = np.asarray(col, dtype=float)
            valid = np.isfinite(vals)
            missing[:, j] = ~valid
            fill = float(np.nanmedian(vals[1:])) if np.any(np.isfinite(vals[1:])) else 0.0
            matrix[:, j] = np.where(valid, vals, fill).astype(np.float32)
        else:
            vals = list(col)
            ref_categories = sorted({str(v) for v in vals[1:] if v is not None and not (isinstance(v, float) and np.isnan(v))})
            lookup = {v: i for i, v in enumerate(ref_categories)}
            encoded = []
            for value in vals:
                is_missing = value is None or (isinstance(value, float) and np.isnan(value))
                encoded.append(-1 if is_missing else lookup.get(str(value), len(lookup)))
            matrix[:, j] = np.asarray(encoded, dtype=np.float32)
    numeric_columns = np.asarray([item["value_type"] in {"numeric", "ordinal", "date", "datetime"} for item in schema], dtype=bool)
    return matrix, numeric_columns, missing


class PolicyValue(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(n_features + 2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.policy = nn.Linear(64, 2 * n_features + 1)
        self.value = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(state)
        return self.policy(hidden), self.value(hidden).squeeze(-1)


class MarginEnv:
    def __init__(self, matrix: np.ndarray, numeric_columns: np.ndarray, missing: np.ndarray, feature_ids: list[str], seed: int, max_steps: int):
        self.x = matrix
        self.feature_ids = feature_ids
        self.d = len(feature_ids)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(np.arange(1, len(matrix)))
        cut = max(1, len(indices) // 2)
        self.x_ref = indices[:cut]
        self.x_neg = indices[cut:]
        if not len(self.x_neg):
            self.x_neg = self.x_ref[-1:]
            self.x_ref = self.x_ref[:-1]
        self.k = max(1, min(3, len(self.x_ref)))
        self.max_adds = min(self.d, max(int(math.ceil(math.sqrt(self.d))), 30))
        self.max_steps = max_steps
        raw_diff = matrix[:, None, :] - matrix[None, :, :]
        self.sqdiff = np.square(raw_diff, dtype=np.float32)
        for j, is_numeric in enumerate(numeric_columns):
            if is_numeric:
                self.sqdiff[:, :, j] += (missing[:, None, j] != missing[None, :, j]).astype(np.float32)
            else:
                self.sqdiff[:, :, j] = (matrix[:, None, j] != matrix[None, :, j]).astype(np.float32)
        self.margin_cache: dict[tuple[int, ...], float] = {(): 0.0}
        self.gain_cache: dict[tuple[int, ...], float] = {}
        self.reset()

    def margin(self, selected: tuple[int, ...]) -> float:
        key = tuple(sorted(selected))
        if not key:
            return 0.0
        if key not in self.margin_cache:
            dist = np.sqrt(np.maximum(0.0, self.sqdiff[:, :, key].sum(axis=2)))
            def score(query_index: int) -> float:
                vals = np.sort(dist[query_index, self.x_ref])[: self.k]
                return float(vals.mean())
            pos = score(0)
            neg = float(np.mean([score(int(index)) for index in self.x_neg]))
            self.margin_cache[key] = (pos - neg) / math.sqrt(len(key))
        return self.margin_cache[key]

    def gain(self, selected: tuple[int, ...], alpha: float = 0.5) -> float:
        key = tuple(sorted(selected))
        if key not in self.gain_cache:
            if not key:
                self.gain_cache[key] = 0.0
            else:
                dist = np.sqrt(np.maximum(0.0, self.sqdiff[:, :, key].sum(axis=2)))
                def score(index: int) -> float:
                    return float(np.sort(dist[index, self.x_ref])[: self.k].mean())
                values = [score(0)] + [score(int(i)) for i in self.x_neg]
                labels = [1] + [0] * len(self.x_neg)
                sep = float(roc_auc_score(labels, values))
                self.gain_cache[key] = sep - alpha * len(key) / self.d
        return self.gain_cache[key]

    def reset(self) -> np.ndarray:
        self.selected: set[int] = set()
        self.adds = 0
        self.negative_streak = 0
        self.steps = 0
        self.previous_margin = 0.0
        return self.observation()

    def observation(self) -> np.ndarray:
        state = np.zeros(self.d + 2, dtype=np.float32)
        if self.selected:
            state[list(self.selected)] = 1.0
        state[-2] = self.adds / max(1, self.max_adds)
        state[-1] = self.negative_streak / 3.0
        return state

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(2 * self.d + 1, dtype=bool)
        for j in range(self.d):
            mask[j] = j not in self.selected and self.adds < self.max_adds
            mask[self.d + j] = j in self.selected
        mask[-1] = True  # stop
        return mask

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        self.steps += 1
        if action == 2 * self.d:
            return self.observation(), 0.0, True
        if action < self.d:
            self.selected.add(action)
            self.adds += 1
        else:
            self.selected.remove(action - self.d)
        current = self.margin(tuple(self.selected))
        reward = current - self.previous_margin
        self.previous_margin = current
        self.negative_streak = self.negative_streak + 1 if reward < 0 else 0
        done = self.adds >= self.max_adds or self.negative_streak >= 3 or self.steps >= self.max_steps
        return self.observation(), float(reward), bool(done)


def train_policy(env: MarginEnv, seed: int, n_steps: int, ppo_epochs: int) -> tuple[PolicyValue, dict[int, float]]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    model = PolicyValue(env.d)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    states, actions, masks, rewards, dones, old_log_probs, old_values = [], [], [], [], [], [], []
    mdr_sums = np.zeros(env.d, dtype=np.float64)
    mdr_counts = np.zeros(env.d, dtype=np.float64)
    transitions = 0
    episode_id = 0
    while transitions < n_steps:
        state = env.reset()
        episode_rewards = []
        episode_actions = []
        episode_states = []
        episode_masks = []
        episode_logp = []
        episode_values = []
        done = False
        while not done and transitions < n_steps:
            legal = env.action_mask()
            with torch.no_grad():
                logits, value = model(torch.from_numpy(state).unsqueeze(0))
                logits = logits.squeeze(0).masked_fill(torch.from_numpy(~legal), -1e9)
                distribution = Categorical(logits=logits)
                action_tensor = distribution.sample()
                action = int(action_tensor.item())
                log_prob = float(distribution.log_prob(action_tensor).item())
                value_scalar = float(value.item())
            next_state, reward, done = env.step(action)
            episode_states.append(state.copy()); episode_actions.append(action); episode_masks.append(legal)
            episode_logp.append(log_prob); episode_values.append(value_scalar); episode_rewards.append(reward)
            if action < env.d:
                mdr_sums[action] += (0.95 ** (len(episode_rewards) - 1)) * reward
                mdr_counts[action] += 1
            state = next_state
            transitions += 1
        with torch.no_grad():
            last_value = 0.0 if done else float(model(torch.from_numpy(state).unsqueeze(0))[1].item())
        adv = np.zeros(len(episode_rewards), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(episode_rewards))):
            next_value = last_value if t == len(episode_rewards) - 1 else episode_values[t + 1]
            nonterminal = 0.0 if (t == len(episode_rewards) - 1 and done) else 1.0
            delta = episode_rewards[t] + 0.95 * next_value * nonterminal - episode_values[t]
            gae = delta + 0.95 * 0.95 * nonterminal * gae
            adv[t] = gae
        ret = adv + np.asarray(episode_values, dtype=np.float32)
        states.extend(episode_states); actions.extend(episode_actions); masks.extend(episode_masks)
        rewards.extend(ret.tolist()); dones.extend(adv.tolist()); old_log_probs.extend(episode_logp); old_values.extend(episode_values)
        episode_id += 1
    states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64)
    masks_t = torch.as_tensor(np.asarray(masks), dtype=torch.bool)
    returns_t = torch.as_tensor(rewards, dtype=torch.float32)
    advantages_t = torch.as_tensor(dones, dtype=torch.float32)
    old_logp_t = torch.as_tensor(old_log_probs, dtype=torch.float32)
    old_value_t = torch.as_tensor(old_values, dtype=torch.float32)
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)
    batch_size = len(states)
    batch_rng = np.random.default_rng(seed + 17)
    for _ in range(ppo_epochs):
        order = batch_rng.permutation(batch_size)
        for start in range(0, batch_size, 64):
            ids = torch.as_tensor(order[start : start + 64], dtype=torch.int64)
            logits, values = model(states_t[ids])
            logits = logits.masked_fill(~masks_t[ids], -1e9)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(actions_t[ids])
            ratio = torch.exp(logp - old_logp_t[ids])
            unclipped = ratio * advantages_t[ids]
            clipped = torch.clamp(ratio, 0.8, 1.2) * advantages_t[ids]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * torch.square(values - returns_t[ids]).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + value_loss - 0.0 * entropy
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5); optimizer.step()
    mdr = {i: float(mdr_sums[i] / mdr_counts[i]) if mdr_counts[i] else 0.0 for i in range(env.d)}
    return model, mdr


def policy_ranking(model: PolicyValue, env: MarginEnv) -> list[int]:
    state = env.reset(); ranking: list[int] = []; visited: set[int] = set()
    while len(ranking) < env.d and env.steps < env.max_steps:
        legal = env.action_mask()
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(state).unsqueeze(0))
            logits = logits.squeeze(0).masked_fill(torch.from_numpy(~legal), -1e9)
            probs = torch.softmax(logits, dim=-1).numpy()
        action = int(np.argmax(probs))
        if action == 2 * env.d:
            break
        if action < env.d and action not in visited:
            ranking.append(action); visited.add(action)
        state, _, done = env.step(action)
        if done:
            break
    while len(ranking) < env.d:
        remaining = [i for i in range(env.d) if i not in visited]
        if not remaining:
            break
        ranking.extend(remaining)
    return ranking


def explain(case: dict, seed: int, n_steps: int, ppo_epochs: int) -> dict:
    obj = payload(case)
    schema = obj["feature_schema"]
    feature_ids = [item["feature_id"] for item in schema]
    matrix, numeric_columns, missing = normalized_matrix(schema, obj["target_row"], obj["normal_reference_rows"])
    case_seed = int(hashlib.sha256(f"{seed}:{case.get('source_case_id', case['case_id'])}".encode()).hexdigest()[:8], 16)
    env = MarginEnv(matrix, numeric_columns, missing, feature_ids, case_seed, max_steps=max(2 * len(feature_ids) + 4, 32))
    model, mdr = train_policy(env, case_seed, n_steps=n_steps, ppo_epochs=ppo_epochs)
    mdr_order = sorted(range(env.d), key=lambda i: (-mdr[i], i))
    policy_order = policy_ranking(model, env)
    candidates: dict[tuple[int, ...], dict] = {}
    for ranking_name, ranking in (("MDR", mdr_order), ("policy", policy_order)):
        prefix: list[int] = []
        for idx in ranking:
            prefix.append(idx)
            key = tuple(sorted(prefix))
            if key not in candidates:
                candidates[key] = {"features": [feature_ids[i] for i in prefix], "gain": env.gain(key), "ranking": ranking_name}
    ordered = sorted(candidates.values(), key=lambda row: (-row["gain"], len(row["features"]), row["features"]))[:4]
    explanations = [{**item, "confidence": 0.5, "evidence": "Reference-guided PPO subspace-margin adaptation; candidate is ranked by held-out Gain."} for item in ordered]
    return {"case_id": case["case_id"], "task_label": "unknown", "mechanism_feature_ids": [],
            "mechanism_metric_applicable": False, "explanations": explanations, "abstain": False,
            "rlax_mdr_order": [feature_ids[i] for i in mdr_order], "rlax_policy_order": [feature_ids[i] for i in policy_order]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-path", type=Path, default=HERE / "inputs.jsonl")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    args = parser.parse_args()
    rows = [row for row in read_jsonl(args.inputs_path) if row["split"] == args.split]
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "predictions_rlax_reference_adaptation.jsonl"
    started = time.perf_counter(); results = []
    for i, case in enumerate(rows, start=1):
        results.append(explain(case, args.seed, args.n_steps, args.ppo_epochs))
        if i % 24 == 0 or i == len(rows):
            out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
            print(f"RLAX adaptation {i}/{len(rows)} case={case['case_id']}", flush=True)
    manifest = {
        "n_cases": len(rows), "split": args.split,
        "method": "Independent paper-guided RLAX mixed-type single-positive adaptation; per-case PPO with subspace-margin-delta reward; MDR and policy feature orderings; top four distinct prefix subsets ranked by AUC minus alpha*|S|/d.",
        "reference_protocol": "Each case's normal reference sample is deterministically split into disjoint X for kNN density scores and E_neg inlier examples; target is E_pos; k=min(3, |X|).",
        "encoding": "numeric/ordinal/date reference min-max with squared numeric differences and an in-field missingness mismatch penalty; categorical/binary/identifier dimensions use 0/1 mismatch, all under per-field Euclidean aggregation; no test labels are used.",
        "paper_url": PAPER, "official_repository_url": OFFICIAL_REPO,
        "official_repository_snapshot": "Checked-out main commit a6fcceca50ca64d8558fec31ef2dff5c34474e39 contains README.md and LICENSE only; this is an independent adaptation, not canonical implementation.",
        "scorer_schema_adapter": "task_label=unknown, mechanism_feature_ids=[], mechanism_metric_applicable=false, abstain=false; confidence=0.5 and fixed evidence are required-format placeholders only and are not interpreted as calibrated method outputs.",
        "hyperparameters": {"seed": args.seed, "n_steps_per_case": args.n_steps, "ppo_epochs": args.ppo_epochs, "policy_architecture": "2x64 Tanh actor-critic", "learning_rate": 3e-4, "gamma": 0.95, "gae_lambda": 0.95, "clip_range": 0.2, "batch_size": 64, "max_gradient_norm": 0.5, "negative_reward_patience": 3, "alpha_gain": 0.5, "distance": "Euclidean over encoded original-field dimensions", "k_neighbors": "min(3, |X|)"},
        "elapsed_seconds": time.perf_counter() - started,
        "optimization_accounting": {"interactions_per_query": args.n_steps,
                                    "rollout_batches_per_query": 1,
                                    "minibatch_updates_per_query": args.ppo_epochs * math.ceil(args.n_steps / 64),
                                    "initial_policy": "fresh per query", "datetime_encoding": "days, reference scaled without clipping"},
        "inputs_sha256": hashlib.sha256(args.inputs_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
