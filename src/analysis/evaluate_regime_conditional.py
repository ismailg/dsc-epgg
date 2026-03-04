from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

# Allow direct execution: `python src/analysis/evaluate_regime_conditional.py ...`
if __package__ is None or __package__ == "":
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from src.algos.PPO import PPOAgentV2
from src.environments import pgg_parallel_v0
from src.wrappers import ObservationWrapper


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def _condition_seed_from_path(path: str) -> Tuple[str, int]:
    name = os.path.basename(path)
    m = re.search(r"(cond[0-9]+)_seed([0-9]+)\.pt$", name)
    if not m:
        return "unknown", -1
    return m.group(1), int(m.group(2))


def _regime_label(f_val: float, n_agents: int) -> str:
    if f_val <= 1.0:
        return "competitive"
    if f_val <= float(n_agents):
        return "mixed"
    return "cooperative"


def _env_cfg_from_train_cfg(cfg: Dict, greedy: bool = False) -> Dict:
    return dict(
        n_agents=int(cfg["n_agents"]),
        num_game_iterations=int(cfg["T"]),
        mult_fact=[float(v) for v in cfg["F"]],
        F=[float(v) for v in cfg["F"]],
        uncertainties=[float(v) for v in cfg["sigmas"]],
        fraction=False,
        rho=float(cfg.get("rho", 0.05)),
        # Greedy eval measures policy preference, so disable tremble noise.
        epsilon_tremble=0.0 if greedy else float(cfg.get("epsilon_tremble", 0.05)),
        endowment=float(cfg.get("endowment", 4.0)),
    )


def _build_eval_objects(payload: Dict, greedy: bool = False):
    cfg = payload["config"]
    n_agents = int(cfg["n_agents"])
    agent_ids = [f"agent_{i}" for i in range(n_agents)]
    comm_enabled = bool(cfg.get("comm_enabled", False))
    n_senders = int(cfg.get("n_senders", 0))
    if comm_enabled and n_senders == 0:
        n_senders = n_agents
    sender_ids = agent_ids[:n_senders]
    vocab_size = int(cfg.get("vocab_size", 2))

    wrapper = ObservationWrapper(
        n_agents=n_agents,
        comm_enabled=comm_enabled,
        n_senders=n_senders,
        sender_ids=sender_ids,
        vocab_size=vocab_size,
        # Keep greedy mode deterministic by disabling message dropout.
        msg_dropout=0.0 if greedy else float(cfg.get("msg_dropout", 0.1)),
        default_endowment=float(cfg.get("endowment", 4.0)),
    )
    value_time_feature = bool(cfg.get("value_time_feature", False))
    value_obs_dim = wrapper.obs_dim + (1 if value_time_feature else 0)

    agents = {}
    for agent_id in agent_ids:
        can_send = comm_enabled and (agent_id in sender_ids)
        agent = PPOAgentV2(
            obs_dim=wrapper.obs_dim,
            action_size=2,
            can_send=can_send,
            vocab_size=vocab_size,
            hidden_size=int(cfg.get("hidden_size", 64)),
            value_obs_dim=value_obs_dim,
            lr=float(cfg.get("lr", 3e-4)),
        )
        state = payload["agents"][agent_id]
        agent.action_actor.load_state_dict(state["action_actor"])
        agent.value_net.load_state_dict(state["value_net"])
        if can_send and agent.message_actor is not None and state.get("message_actor") is not None:
            agent.message_actor.load_state_dict(state["message_actor"])
        agent.action_actor.eval()
        agent.value_net.eval()
        if agent.message_actor is not None:
            agent.message_actor.eval()
        agents[agent_id] = agent

    env = pgg_parallel_v0.parallel_env(_env_cfg_from_train_cfg(cfg, greedy=greedy))
    return cfg, env, wrapper, agents, agent_ids, sender_ids, value_time_feature


def _eval_checkpoint(
    checkpoint_path: str,
    n_eval_episodes: int,
    eval_seed: int,
    greedy: bool = False,
) -> List[Dict]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    cfg, env, wrapper, agents, agent_ids, sender_ids, value_time_feature = _build_eval_objects(
        payload, greedy=greedy
    )
    _seed_everything(eval_seed)

    n_agents = int(cfg["n_agents"])
    T = int(cfg["T"])
    comm_enabled = bool(cfg.get("comm_enabled", False))
    condition, train_seed = _condition_seed_from_path(checkpoint_path)

    regime_acc = defaultdict(lambda: {"n_rounds": 0, "coop_sum": 0.0, "reward_sum": 0.0})
    f_acc = defaultdict(lambda: {"n_rounds": 0, "coop_sum": 0.0, "reward_sum": 0.0})

    with torch.no_grad():
        for _ in range(n_eval_episodes):
            raw_obs = env.reset()
            wrapper.reset(agent_ids)
            current_messages = None
            done = False
            steps = 0

            while (not done) and (steps < T):
                aug_obs = {
                    agent_id: wrapper.build_obs(agent_id, raw_obs[agent_id], current_messages)
                    for agent_id in agent_ids
                }

                if comm_enabled and len(sender_ids) > 0:
                    proposed = {}
                    for sender_id in sender_ids:
                        if greedy:
                            obs_t = torch.tensor(
                                aug_obs[sender_id],
                                dtype=torch.float32,
                                device=agents[sender_id].action_actor.net[0].weight.device,
                            )
                            logits = agents[sender_id].message_actor(obs_t)
                            msg = int(torch.argmax(logits).item())
                        else:
                            msg, _lp, _ent, _probs = agents[sender_id].sample_message(
                                aug_obs[sender_id]
                            )
                        proposed[sender_id] = int(msg)
                    dropped = wrapper.apply_msg_dropout(proposed)
                    for sender_id, msg in proposed.items():
                        wrapper.update_msg_marginals(sender_id, msg)
                    current_messages = dropped
                    aug_obs = {
                        agent_id: wrapper.build_obs(agent_id, raw_obs[agent_id], current_messages)
                        for agent_id in agent_ids
                    }

                intended_actions = {}
                t_frac = float(steps) / float(max(1, T - 1))
                for agent_id in agent_ids:
                    value_obs = (
                        np.concatenate(
                            [aug_obs[agent_id], np.array([t_frac], dtype=np.float32)], axis=0
                        ).astype(np.float32)
                        if value_time_feature
                        else aug_obs[agent_id]
                    )
                    if greedy:
                        obs_t = torch.tensor(
                            aug_obs[agent_id],
                            dtype=torch.float32,
                            device=agents[agent_id].action_actor.net[0].weight.device,
                        )
                        logits = agents[agent_id].action_actor(obs_t)
                        action = int(torch.argmax(logits).item())
                    else:
                        action, _lp, _value, _ent, _probs = agents[agent_id].sample_action(
                            aug_obs[agent_id], value_obs=value_obs
                        )
                    intended_actions[agent_id] = int(action)

                raw_next, rewards, done, infos = env.step(intended_actions)
                executed_actions = infos.get("executed_actions", intended_actions)
                true_f = float(infos.get("true_f", _as_float(env.current_multiplier)))

                wrapper.update(executed_actions)

                coop_step = float(np.mean([int(v) for v in executed_actions.values()]))
                reward_step = float(np.mean([_as_float(v) for v in rewards.values()]))
                regime = _regime_label(true_f, n_agents=n_agents)
                f_key = f"{true_f:.3f}"

                regime_acc[regime]["n_rounds"] += 1
                regime_acc[regime]["coop_sum"] += coop_step
                regime_acc[regime]["reward_sum"] += reward_step

                f_acc[f_key]["n_rounds"] += 1
                f_acc[f_key]["coop_sum"] += coop_step
                f_acc[f_key]["reward_sum"] += reward_step

                raw_obs = raw_next
                steps += 1

    rows = []
    for regime, acc in sorted(regime_acc.items()):
        n = max(1, int(acc["n_rounds"]))
        rows.append(
            {
                "checkpoint": checkpoint_path,
                "condition": condition,
                "train_seed": train_seed,
                "comm_enabled": int(comm_enabled),
                "eval_seed": eval_seed,
                "eval_policy": "greedy" if greedy else "sample",
                "scope": "regime",
                "key": regime,
                "n_rounds": int(acc["n_rounds"]),
                "coop_rate": float(acc["coop_sum"] / n),
                "avg_reward": float(acc["reward_sum"] / n),
            }
        )

    for f_key, acc in sorted(f_acc.items(), key=lambda x: float(x[0])):
        n = max(1, int(acc["n_rounds"]))
        rows.append(
            {
                "checkpoint": checkpoint_path,
                "condition": condition,
                "train_seed": train_seed,
                "comm_enabled": int(comm_enabled),
                "eval_seed": eval_seed,
                "eval_policy": "greedy" if greedy else "sample",
                "scope": "f_value",
                "key": f_key,
                "n_rounds": int(acc["n_rounds"]),
                "coop_rate": float(acc["coop_sum"] / n),
                "avg_reward": float(acc["reward_sum"] / n),
            }
        )
    return rows


def _write_csv(path: str, rows: List[Dict]):
    if len(rows) == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "checkpoint",
        "condition",
        "train_seed",
        "comm_enabled",
        "eval_seed",
        "eval_policy",
        "scope",
        "key",
        "n_rounds",
        "coop_rate",
        "avg_reward",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _condition_summary(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(lambda: {"n_rounds": 0, "coop_weighted": 0.0, "reward_weighted": 0.0})
    for row in rows:
        if row["scope"] != "regime":
            continue
        key = (row["condition"], row["key"])
        n = int(row["n_rounds"])
        grouped[key]["n_rounds"] += n
        grouped[key]["coop_weighted"] += float(row["coop_rate"]) * n
        grouped[key]["reward_weighted"] += float(row["avg_reward"]) * n

    out = []
    for (condition, regime), acc in sorted(grouped.items()):
        n = max(1, int(acc["n_rounds"]))
        out.append(
            {
                "condition": condition,
                "regime": regime,
                "n_rounds": int(acc["n_rounds"]),
                "coop_rate": float(acc["coop_weighted"] / n),
                "avg_reward": float(acc["reward_weighted"] / n),
            }
        )
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints_glob", type=str, default="outputs/train/grid/*.pt")
    p.add_argument("--n_eval_episodes", type=int, default=300)
    p.add_argument("--eval_seed", type=int, default=9001)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--out_csv", type=str, default="outputs/train/grid/regime_eval.csv")
    p.add_argument(
        "--out_condition_csv",
        type=str,
        default="outputs/train/grid/regime_eval_condition_summary.csv",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ckpts = sorted(glob.glob(args.checkpoints_glob))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"no checkpoints matched: {args.checkpoints_glob}")

    all_rows: List[Dict] = []
    for idx, ckpt in enumerate(ckpts):
        run_seed = int(args.eval_seed + idx)
        print(f"[eval] {idx + 1}/{len(ckpts)} -> {ckpt} (seed={run_seed})")
        rows = _eval_checkpoint(
            checkpoint_path=ckpt,
            n_eval_episodes=args.n_eval_episodes,
            eval_seed=run_seed,
            greedy=bool(args.greedy),
        )
        all_rows.extend(rows)

    _write_csv(args.out_csv, all_rows)
    cond_rows = _condition_summary(all_rows)
    os.makedirs(os.path.dirname(args.out_condition_csv), exist_ok=True)
    with open(args.out_condition_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["condition", "regime", "n_rounds", "coop_rate", "avg_reward"]
        )
        writer.writeheader()
        for row in cond_rows:
            writer.writerow(row)

    print(f"[eval] rows={len(all_rows)} out={args.out_csv}")
    print(f"[eval] condition_summary_rows={len(cond_rows)} out={args.out_condition_csv}")
    for row in cond_rows:
        print(
            "[summary] "
            f"{row['condition']} {row['regime']} "
            f"coop={row['coop_rate']:.3f} reward={row['avg_reward']:.3f} "
            f"n_rounds={row['n_rounds']}"
        )


if __name__ == "__main__":
    main()
