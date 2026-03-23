from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np
import torch

# Allow direct execution: `python src/experiments_pgg_v0/train_ppo.py ...`
if __package__ is None or __package__ == "":
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from src.algos.PPO import PPOAgentV2, PPOTrainer
from src.algos.trajectory_buffer import TrajectoryBuffer
from src.analysis.regime_audit import regime_audit
from src.environments import pgg_parallel_v0
from src.logging import SessionLogger
from src.wrappers import ObservationWrapper


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    n_agents: int = 4
    T: int = 100
    n_episodes: int = 100
    endowment: float = 4.0

    F: tuple = (0.5, 1.5, 2.5, 3.5, 5.0)
    sigmas: tuple = (0.5, 0.5, 0.5, 0.5)
    rho: float = 0.05
    epsilon_tremble: float = 0.05

    comm_enabled: bool = False
    n_senders: int = 0
    vocab_size: int = 2
    msg_dropout: float = 0.1
    msg_training_intervention: str = "none"
    msg_training_history_len: int = 4096
    episode_offset: int = 0
    schedule_total_episodes: int = 0

    hidden_size: int = 64
    value_time_feature: bool = True
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    entropy_schedule: str = "none"  # one of: none, linear, cosine
    entropy_coeff_final: float = 0.001
    msg_entropy_coeff: Optional[float] = None
    msg_entropy_coeff_final: Optional[float] = None
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 32
    sign_lambda: float = 0.1
    list_lambda: float = 0.1

    seed: int = 42
    log_interval: int = 10
    save_path: str = "outputs/ppo_agents.pt"
    init_ckpt: str = ""
    reward_scale: float = 20.0
    lr_schedule: str = "none"  # one of: none, linear, cosine
    min_lr: float = 1e-5
    early_stop_patience: int = 0
    early_stop_min_delta: float = 1e-6
    enable_comm_fallback: bool = True
    max_comm_debug_cycles: int = 2
    log_sessions: bool = False
    session_log_dir: str = "outputs/sessions"
    condition_name: str = "default"
    consolidate_sessions: bool = False
    run_regime_audit: bool = False
    audit_sessions: int = 100
    use_wandb: bool = False
    wandb_project: str = "dsc-epgg"
    wandb_mode: str = "offline"
    regime_log_interval: int = 500
    metrics_jsonl_path: str = ""
    checkpoint_interval: int = 0
    mi_null_perms: int = 200
    mi_alpha: float = 0.05
    log_trainer_responsiveness: bool = True


def minimal_test_config(**overrides):
    cfg = TrainConfig(
        n_agents=4,
        T=10,
        n_episodes=10,
        comm_enabled=False,
        n_senders=0,
        sigmas=(0.5, 0.5, 0.5, 0.5),
        save_path="outputs/test_agents.pt",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _agent_ids(n_agents: int) -> List[str]:
    return [f"agent_{i}" for i in range(n_agents)]


def _sender_ids(cfg: TrainConfig) -> List[str]:
    return _agent_ids(cfg.n_agents)[: int(cfg.n_senders)]


def _to_float_rewards(rewards: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k, v in rewards.items():
        if isinstance(v, torch.Tensor):
            out[k] = float(v.detach().cpu().item())
        else:
            out[k] = float(v)
    return out


def _safe_is_finite(metrics: Dict[str, float]) -> bool:
    for val in metrics.values():
        if not math.isfinite(float(val)):
            return False
    return True


def _regime_label(true_f: float, n_agents: int) -> str:
    if true_f <= 1.0:
        return "competitive"
    if true_f <= float(n_agents):
        return "mixed"
    return "cooperative"


def _bucket() -> Dict[str, float]:
    return {"n_rounds": 0.0, "coop_sum": 0.0, "reward_sum": 0.0}


def _update_bucket(acc: Dict[str, Dict[str, float]], key: str, coop: float, reward: float):
    if key not in acc:
        acc[key] = _bucket()
    acc[key]["n_rounds"] += 1.0
    acc[key]["coop_sum"] += float(coop)
    acc[key]["reward_sum"] += float(reward)


def _readout_bucket(acc: Dict[str, Dict[str, float]], key: str) -> Dict[str, float]:
    bucket = acc.get(key, _bucket())
    n = int(bucket["n_rounds"])
    denom = max(1, n)
    return {
        "n_rounds": n,
        "coop_rate": float(bucket["coop_sum"] / float(denom)),
        "avg_reward": float(bucket["reward_sum"] / float(denom)),
    }


def _append_jsonl(path: str, row: Dict):
    if str(path or "").strip() == "":
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _mutual_information_from_counts(counts: np.ndarray) -> float:
    arr = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(arr))
    if total <= 0.0:
        return 0.0
    p_xy = arr / total
    p_x = np.sum(p_xy, axis=1, keepdims=True)
    p_y = np.sum(p_xy, axis=0, keepdims=True)
    mask = p_xy > 0.0
    denom = np.maximum(p_x * p_y, 1e-12)
    return float(np.sum(p_xy[mask] * np.log2(p_xy[mask] / denom[mask])))


def _entropy_from_counts_1d(counts: np.ndarray) -> float:
    arr = np.asarray(counts, dtype=np.float64).reshape(-1)
    total = float(np.sum(arr))
    if total <= 0.0:
        return 0.0
    p = arr / total
    mask = p > 0.0
    return float(-np.sum(p[mask] * np.log2(p[mask])))


def _mi_null_independence_stats(
    counts: np.ndarray,
    n_perms: int,
    alpha: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    arr = np.asarray(counts, dtype=np.float64)
    observed_mi = _mutual_information_from_counts(arr)
    total = int(round(float(np.sum(arr))))
    n_perms = int(max(1, n_perms))
    alpha = float(alpha)
    if total <= 0:
        return {
            "mi_perm_p95": 0.0,
            "mi_p_value": 1.0,
            "mi_significant": False,
            "mi_observed": float(observed_mi),
        }

    p_x = np.sum(arr, axis=1)
    p_y = np.sum(arr, axis=0)
    sx = float(np.sum(p_x))
    sy = float(np.sum(p_y))
    if sx <= 0.0 or sy <= 0.0:
        return {
            "mi_perm_p95": 0.0,
            "mi_p_value": 1.0,
            "mi_significant": False,
            "mi_observed": float(observed_mi),
        }
    p_x = p_x / sx
    p_y = p_y / sy
    p_ind = np.outer(p_x, p_y).reshape(-1)
    p_ind = p_ind / np.maximum(np.sum(p_ind), 1e-12)

    null_mi = np.zeros((n_perms,), dtype=np.float64)
    for i in range(n_perms):
        sampled = rng.multinomial(total, p_ind).reshape(arr.shape)
        null_mi[i] = _mutual_information_from_counts(sampled)

    p95 = float(np.percentile(null_mi, 95.0))
    p_value = float((1.0 + np.sum(null_mi >= observed_mi)) / float(n_perms + 1))
    return {
        "mi_perm_p95": p95,
        "mi_p_value": p_value,
        "mi_significant": bool(p_value < alpha and observed_mi > 1e-8),
        "mi_observed": float(observed_mi),
    }


def _checkpoint_with_episode(base_path: str, episode_num: int) -> str:
    root, ext = os.path.splitext(base_path)
    if ext == "":
        ext = ".pt"
    return f"{root}_ep{int(episode_num)}{ext}"


def _apply_training_message_intervention(
    intervention: str,
    delivered: Dict[str, int],
    vocab_size: int,
    sender_history: Optional[Dict[str, Deque[int]]] = None,
) -> Dict[str, int]:
    mode = str(intervention or "none").strip().lower()
    out = {sender_id: int(msg) for sender_id, msg in delivered.items()}
    if mode == "none":
        return out
    if mode == "uniform":
        return {
            sender_id: int(np.random.randint(0, vocab_size))
            for sender_id in out.keys()
        }
    if mode == "public_random":
        shared = int(np.random.randint(0, vocab_size))
        return {sender_id: shared for sender_id in out.keys()}
    if mode == "fixed0":
        return {sender_id: 0 for sender_id in out.keys()}
    if mode == "fixed1":
        if int(vocab_size) < 2:
            raise ValueError("msg_training_intervention=fixed1 requires vocab_size >= 2")
        return {sender_id: 1 for sender_id in out.keys()}
    if mode == "sender_shuffle":
        if sender_history is None:
            raise ValueError(
                "msg_training_intervention=sender_shuffle requires sender_history"
            )
        shuffled = {}
        for sender_id, msg in out.items():
            history = sender_history.get(sender_id)
            if history is None or len(history) == 0:
                shuffled[sender_id] = int(msg)
                continue
            idx = int(np.random.randint(0, len(history)))
            shuffled[sender_id] = int(history[idx])
        return shuffled
    raise ValueError(
        "unknown msg_training_intervention="
        f"{intervention!r}; expected one of: none,uniform,public_random,fixed0,fixed1,sender_shuffle"
    )


def _update_training_message_history(
    sender_history: Dict[str, Deque[int]],
    natural_messages: Dict[str, int],
) -> None:
    for sender_id, msg in natural_messages.items():
        sender_history.setdefault(str(sender_id), deque()).append(int(msg))


def _set_agents_lr(agents: Dict[str, PPOAgentV2], lr_value: float):
    for agent in agents.values():
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = float(lr_value)


def _scheduled_value(
    initial: float,
    final: float,
    schedule: str,
    progress: float,
) -> float:
    mode = str(schedule or "none").strip().lower()
    if mode == "none":
        return float(initial)
    progress = min(1.0, max(0.0, float(progress)))
    if mode == "linear":
        mix = progress
    elif mode == "cosine":
        mix = 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        raise ValueError(f"unknown schedule mode: {schedule}")
    return float(initial + (final - initial) * mix)


def _build_env(cfg: TrainConfig):
    env_cfg = dict(
        n_agents=cfg.n_agents,
        num_game_iterations=cfg.T,
        mult_fact=list(cfg.F),
        F=list(cfg.F),
        uncertainties=list(cfg.sigmas),
        fraction=False,
        rho=cfg.rho,
        epsilon_tremble=cfg.epsilon_tremble,
        endowment=cfg.endowment,
    )
    return pgg_parallel_v0.parallel_env(env_cfg)


def _build_agents(cfg: TrainConfig, obs_dim: int, sender_ids: List[str]):
    value_obs_dim = obs_dim + (1 if cfg.value_time_feature else 0)
    agents = {}
    for agent_id in _agent_ids(cfg.n_agents):
        agents[agent_id] = PPOAgentV2(
            obs_dim=obs_dim,
            action_size=2,
            can_send=(cfg.comm_enabled and agent_id in sender_ids),
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            value_obs_dim=value_obs_dim,
            lr=cfg.lr,
        )
    return agents


def _safe_load_state_dict(module: torch.nn.Module, loaded: Dict):
    current = module.state_dict()
    compatible = {}
    for key, tensor in loaded.items():
        if key in current and tuple(current[key].shape) == tuple(tensor.shape):
            compatible[key] = tensor
    module.load_state_dict(compatible, strict=False)


def _maybe_load_agents(agents: Dict[str, PPOAgentV2], init_ckpt: str):
    ckpt = str(init_ckpt or "").strip()
    if ckpt == "":
        return
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"init_ckpt not found: {ckpt}")
    payload = torch.load(ckpt, map_location="cpu")
    saved_agents = payload.get("agents", {})
    for agent_id, agent in agents.items():
        if agent_id not in saved_agents:
            continue
        saved = saved_agents[agent_id]
        _safe_load_state_dict(agent.action_actor, saved["action_actor"])
        _safe_load_state_dict(agent.value_net, saved["value_net"])
        if agent.message_actor is not None and saved.get("message_actor") is not None:
            _safe_load_state_dict(agent.message_actor, saved["message_actor"])


def _save_agents(path: str, agents: Dict[str, PPOAgentV2], cfg: TrainConfig):
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    payload = {
        "config": cfg.__dict__,
        "agents": {},
    }
    for agent_id, agent in agents.items():
        payload["agents"][agent_id] = {
            "action_actor": agent.action_actor.state_dict(),
            "value_net": agent.value_net.state_dict(),
            "message_actor": (
                agent.message_actor.state_dict() if agent.message_actor is not None else None
            ),
        }
    torch.save(payload, path)
    _write_run_manifest(path=path, cfg=cfg)


def _git_commit_hash() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _write_run_manifest(path: str, cfg: TrainConfig):
    manifest_path = os.path.splitext(path)[0] + ".run.json"
    payload = {
        "timestamp_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "git_commit": _git_commit_hash(),
        "config": cfg.__dict__,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _single_run(cfg: TrainConfig):
    _seed_everything(cfg.seed)
    agent_ids = _agent_ids(cfg.n_agents)
    sender_ids = _sender_ids(cfg)
    value_obs_dim = (5 + len(sender_ids) * cfg.vocab_size if cfg.comm_enabled else 5) + (
        1 if cfg.value_time_feature else 0
    )
    if str(cfg.msg_training_intervention).strip().lower() != "none":
        print(
            "[msg-training-intervention] "
            f"mode={cfg.msg_training_intervention} sign_lambda={cfg.sign_lambda} "
            f"list_lambda={cfg.list_lambda}"
        )

    env = _build_env(cfg)
    wrapper = ObservationWrapper(
        n_agents=cfg.n_agents,
        comm_enabled=cfg.comm_enabled,
        n_senders=len(sender_ids),
        sender_ids=sender_ids,
        vocab_size=cfg.vocab_size,
        msg_dropout=cfg.msg_dropout,
        default_endowment=cfg.endowment,
    )
    agents = _build_agents(cfg, obs_dim=wrapper.obs_dim, sender_ids=sender_ids)
    _maybe_load_agents(agents=agents, init_ckpt=cfg.init_ckpt)
    sender_history: Dict[str, Deque[int]] = {}
    if cfg.comm_enabled and str(cfg.msg_training_intervention).strip().lower() == "sender_shuffle":
        sender_history = {
            sender_id: deque(maxlen=max(1, int(cfg.msg_training_history_len)))
            for sender_id in sender_ids
        }
    if cfg.lr_schedule == "linear":
        _set_agents_lr(agents, cfg.lr)
    ppo = PPOTrainer(
        agents=agents,
        clip_ratio=cfg.clip_ratio,
        value_coeff=cfg.value_coeff,
        entropy_coeff=cfg.entropy_coeff,
        msg_entropy_coeff=cfg.msg_entropy_coeff,
        max_grad_norm=cfg.max_grad_norm,
        ppo_epochs=cfg.ppo_epochs,
        mini_batch_size=cfg.mini_batch_size,
        sign_lambda=cfg.sign_lambda,
        list_lambda=cfg.list_lambda,
    )
    session_logger = (
        SessionLogger(
            save_dir=cfg.session_log_dir,
            condition_name=cfg.condition_name,
            seed=cfg.seed,
        )
        if cfg.log_sessions
        else None
    )
    wandb_run = None
    if cfg.use_wandb:
        try:
            import wandb  # local import to keep dependency optional at runtime

            wandb_run = wandb.init(
                project=cfg.wandb_project,
                mode=cfg.wandb_mode,
                config=cfg.__dict__,
            )
        except Exception as exc:
            print(f"[wandb] disabled due to init error: {exc}")
            wandb_run = None

    metrics_over_time = []
    cumulative_regime_acc = {label: _bucket() for label in ("competitive", "mixed", "cooperative")}
    window_regime_acc = {label: _bucket() for label in ("competitive", "mixed", "cooperative")}
    cumulative_f_acc: Dict[str, Dict[str, float]] = {}
    window_f_acc: Dict[str, Dict[str, float]] = {}
    sender_agent_idx = {sender_id: agent_ids.index(sender_id) for sender_id in sender_ids}
    f_keys_sorted = sorted({f"{float(v):.3f}" for v in cfg.F}, key=float)
    f_key_to_idx = {key: idx for idx, key in enumerate(f_keys_sorted)}

    def _new_comm_window_counts():
        return {
            sender_id: {
                "msg_f": np.zeros((cfg.vocab_size, len(f_keys_sorted)), dtype=np.float64),
                "msg_action": np.zeros((cfg.vocab_size, 2), dtype=np.float64),
            }
            for sender_id in sender_ids
        }

    window_comm_counts = _new_comm_window_counts() if (cfg.comm_enabled and len(sender_ids) > 0) else {}
    window_responsiveness = (
        {agent_id: [] for agent_id in agent_ids}
        if (cfg.comm_enabled and len(sender_ids) > 0 and cfg.log_trainer_responsiveness)
        else {}
    )
    mi_rng = np.random.default_rng(int(cfg.seed) + 1729)
    best_metric = -float("inf")
    stale_epochs = 0
    for episode in range(cfg.n_episodes):
        abs_episode0 = int(cfg.episode_offset) + int(episode)
        abs_episode1 = int(abs_episode0) + 1
        total_schedule_episodes = int(cfg.schedule_total_episodes)
        if total_schedule_episodes <= 0:
            total_schedule_episodes = int(cfg.episode_offset) + int(cfg.n_episodes)
        progress = float(abs_episode0) / float(max(1, total_schedule_episodes - 1))
        if cfg.lr_schedule != "none":
            episode_lr = _scheduled_value(
                initial=float(cfg.lr),
                final=float(cfg.min_lr),
                schedule=str(cfg.lr_schedule),
                progress=progress,
            )
            _set_agents_lr(agents, episode_lr)
        else:
            episode_lr = float(cfg.lr)

        ppo.entropy_coeff = _scheduled_value(
            initial=float(cfg.entropy_coeff),
            final=float(cfg.entropy_coeff_final),
            schedule=str(cfg.entropy_schedule),
            progress=progress,
        )
        msg_entropy_initial = (
            float(cfg.msg_entropy_coeff)
            if cfg.msg_entropy_coeff is not None
            else float(cfg.entropy_coeff)
        )
        if cfg.msg_entropy_coeff_final is not None:
            msg_entropy_final = float(cfg.msg_entropy_coeff_final)
        elif cfg.msg_entropy_coeff is None:
            msg_entropy_final = float(cfg.entropy_coeff_final)
        else:
            msg_entropy_final = msg_entropy_initial
        ppo.msg_entropy_coeff = _scheduled_value(
            initial=msg_entropy_initial,
            final=msg_entropy_final,
            schedule=str(cfg.entropy_schedule),
            progress=progress,
        )

        buffer = TrajectoryBuffer(
            agent_ids=agent_ids,
            T=cfg.T,
            obs_dim=wrapper.obs_dim,
            value_obs_dim=value_obs_dim,
            comm_enabled=cfg.comm_enabled,
            vocab_size=cfg.vocab_size,
            sender_ids=sender_ids,
        )

        raw_obs = env.reset()
        wrapper.reset(agent_ids)
        current_messages = None
        done = False
        steps = 0

        while not done and steps < cfg.T:
            aug_obs = {
                agent_id: wrapper.build_obs(agent_id, raw_obs[agent_id], current_messages)
                for agent_id in agent_ids
            }

            message_actions = {}
            message_log_probs = {}
            if cfg.comm_enabled and len(sender_ids) > 0:
                proposed = {}
                for sender_id in sender_ids:
                    msg, msg_lp, _msg_ent, _msg_probs = agents[sender_id].sample_message(
                        aug_obs[sender_id]
                    )
                    proposed[sender_id] = msg
                    message_actions[sender_id] = msg
                    message_log_probs[sender_id] = msg_lp

                dropped = wrapper.apply_msg_dropout(proposed)
                if str(cfg.msg_training_intervention).strip().lower() == "none":
                    for sender_id, msg in proposed.items():
                        wrapper.update_msg_marginals(sender_id, msg)
                    current_messages = dropped
                else:
                    if str(cfg.msg_training_intervention).strip().lower() == "sender_shuffle":
                        delivered = _apply_training_message_intervention(
                            intervention=cfg.msg_training_intervention,
                            delivered=dropped,
                            vocab_size=cfg.vocab_size,
                            sender_history=sender_history,
                        )
                        _update_training_message_history(sender_history, dropped)
                    else:
                        delivered = _apply_training_message_intervention(
                            intervention=cfg.msg_training_intervention,
                            delivered=dropped,
                            vocab_size=cfg.vocab_size,
                        )
                    for sender_id, msg in delivered.items():
                        wrapper.update_msg_marginals(sender_id, msg)
                    current_messages = delivered
                aug_obs = {
                    agent_id: wrapper.build_obs(agent_id, raw_obs[agent_id], current_messages)
                    for agent_id in agent_ids
                }

            intended_actions = {}
            action_log_probs = {}
            values = {}
            listening_bonus = {agent_id: 0.0 for agent_id in agent_ids}
            t_frac = float(steps) / float(max(1, cfg.T - 1))
            value_aug_obs = {}
            for agent_id in agent_ids:
                if cfg.value_time_feature:
                    value_aug_obs[agent_id] = np.concatenate(
                        [aug_obs[agent_id], np.array([t_frac], dtype=np.float32)], axis=0
                    ).astype(np.float32)
                else:
                    value_aug_obs[agent_id] = aug_obs[agent_id]
            for agent_id in agent_ids:
                action, action_lp, value, _ent, _ = agents[agent_id].sample_action(
                    aug_obs[agent_id], value_obs=value_aug_obs[agent_id]
                )
                intended_actions[agent_id] = int(action)
                action_log_probs[agent_id] = float(action_lp)
                values[agent_id] = float(value)

                if cfg.comm_enabled and len(sender_ids) > 0:
                    obs_t = torch.tensor(
                        aug_obs[agent_id], dtype=torch.float32, device=agents[agent_id].action_actor.net[0].weight.device
                    ).unsqueeze(0)
                    probs_with = agents[agent_id].action_distribution(obs_t).squeeze(0)
                    obs_no_msg = obs_t.clone()
                    msg_start = wrapper.message_start_idx
                    obs_no_msg[:, msg_start:] = 0.0
                    probs_without = agents[agent_id].action_distribution(obs_no_msg).squeeze(0)
                    eps = 1e-8
                    # KL(pi(a|with_msg) || pi(a|without_msg)): larger means messages
                    # shift action predictions more strongly.
                    kl = torch.sum(
                        probs_with * (torch.log(probs_with + eps) - torch.log(probs_without + eps))
                    )
                    listening_bonus[agent_id] = -float(
                        kl.detach().cpu().item()
                    )

                    if cfg.log_trainer_responsiveness:
                        # Diagnostics-only responsiveness proxy:
                        # KL(pi(a|actual_msg) || pi(a|marginal_msg)), where marginal_msg
                        # uses per-sender empirical message marginals from wrapper state.
                        obs_marginal = obs_t.clone()
                        msg_start = int(wrapper.message_start_idx)
                        for s_idx, sender_id in enumerate(sender_ids):
                            start = msg_start + s_idx * int(cfg.vocab_size)
                            end = start + int(cfg.vocab_size)
                            probs_msg = wrapper.msg_marginals.get(sender_id)
                            if probs_msg is None:
                                probs_msg = (
                                    np.ones((cfg.vocab_size,), dtype=np.float32) / float(cfg.vocab_size)
                                )
                            obs_marginal[:, start:end] = torch.tensor(
                                probs_msg,
                                dtype=torch.float32,
                                device=obs_t.device,
                            ).unsqueeze(0)
                        probs_marginal = agents[agent_id].action_distribution(obs_marginal).squeeze(0)
                        kl_diag = torch.sum(
                            probs_with * (torch.log(probs_with + eps) - torch.log(probs_marginal + eps))
                        )
                        window_responsiveness[agent_id].append(float(kl_diag.detach().cpu().item()))

            raw_obs_next, rewards, done, infos = env.step(intended_actions)
            rewards_raw = _to_float_rewards(rewards)
            rewards_train = {
                agent_id: (rewards_raw[agent_id] / float(cfg.reward_scale))
                for agent_id in agent_ids
            }
            executed_actions = infos.get("executed_actions", intended_actions)
            flips = infos.get("flips", {agent_id: False for agent_id in agent_ids})
            true_f = float(infos.get("true_f", env.current_multiplier.item()))

            wrapper.update(executed_actions)

            buffer.store(
                obs=aug_obs,
                actions=intended_actions,
                rewards=rewards_train,
                raw_rewards=rewards_raw,
                values=values,
                log_probs=action_log_probs,
                done=bool(done),
                executed_actions=executed_actions,
                flips=flips,
                true_f=true_f,
                f_hats=raw_obs,
                messages=current_messages,
                value_obs=value_aug_obs,
                message_actions=message_actions if len(message_actions) > 0 else None,
                message_log_probs=message_log_probs if len(message_log_probs) > 0 else None,
                listening_bonus=listening_bonus,
            )
            raw_obs = raw_obs_next
            steps += 1

        if done:
            last_values = np.zeros((cfg.n_agents,), dtype=np.float32)
        else:
            final_aug_obs = {
                agent_id: wrapper.build_obs(agent_id, raw_obs[agent_id], current_messages)
                for agent_id in agent_ids
            }
            t_frac = float(steps) / float(max(1, cfg.T - 1))
            final_value_obs = {
                agent_id: (
                    np.concatenate(
                        [final_aug_obs[agent_id], np.array([t_frac], dtype=np.float32)], axis=0
                    ).astype(np.float32)
                    if cfg.value_time_feature
                    else final_aug_obs[agent_id]
                )
                for agent_id in agent_ids
            }
            last_values = np.array(
                [
                    agents[agent_id]
                    .value(
                        torch.tensor(
                            final_value_obs[agent_id], dtype=torch.float32, device=agents[agent_id].action_actor.net[0].weight.device
                        ).unsqueeze(0)
                    )
                    .detach()
                    .cpu()
                    .item()
                    for agent_id in agent_ids
                ],
                dtype=np.float32,
            )

        advantages, returns = buffer.compute_gae(
            last_values=last_values, gamma=cfg.gamma, lam=cfg.lam
        )
        train_metrics = ppo.update(buffer, advantages, returns)
        if not _safe_is_finite(train_metrics):
            raise FloatingPointError(f"non-finite PPO metrics at episode {episode}: {train_metrics}")

        if session_logger is not None:
            session_logger.log_session(buffer)

        coop_rate = float(np.mean(buffer.executed_actions[: buffer.t])) if buffer.t > 0 else 0.0
        avg_reward = float(np.mean(buffer.agent_rewards[: buffer.t])) if buffer.t > 0 else 0.0
        episode_regime_acc = {
            "competitive": _bucket(),
            "mixed": _bucket(),
            "cooperative": _bucket(),
        }
        if buffer.t > 0:
            coop_per_step = np.mean(buffer.executed_actions[: buffer.t], axis=1)
            reward_per_step = np.mean(buffer.agent_rewards[: buffer.t], axis=1)
            for t in range(buffer.t):
                f_val = float(buffer.true_f[t])
                regime = _regime_label(f_val, n_agents=cfg.n_agents)
                coop_t = float(coop_per_step[t])
                rew_t = float(reward_per_step[t])

                _update_bucket(episode_regime_acc, regime, coop_t, rew_t)
                _update_bucket(cumulative_regime_acc, regime, coop_t, rew_t)
                _update_bucket(window_regime_acc, regime, coop_t, rew_t)

                f_key = f"{f_val:.3f}"
                _update_bucket(cumulative_f_acc, f_key, coop_t, rew_t)
                _update_bucket(window_f_acc, f_key, coop_t, rew_t)

                if cfg.comm_enabled and len(sender_ids) > 0 and buffer.message_actions is not None:
                    f_idx = f_key_to_idx.get(f_key)
                    if f_idx is None:
                        continue
                    for sender_id in sender_ids:
                        a_idx = sender_agent_idx[sender_id]
                        # Use intended message (pre-dropout) to measure sender signaling capacity.
                        msg = int(buffer.message_actions[t, a_idx])
                        act = int(buffer.executed_actions[t, a_idx])
                        if msg < 0 or msg >= cfg.vocab_size:
                            continue
                        if act < 0 or act > 1:
                            continue
                        window_comm_counts[sender_id]["msg_f"][msg, f_idx] += 1.0
                        window_comm_counts[sender_id]["msg_action"][msg, act] += 1.0
        episode_metrics = {
            "episode": int(abs_episode1),
            "episode_local": int(episode + 1),
            "steps": buffer.t,
            "coop_rate": coop_rate,
            "avg_reward": avg_reward,
            "lr_current": float(episode_lr),
            "entropy_coeff_current": float(ppo.entropy_coeff),
            "msg_entropy_coeff_current": float(ppo.msg_entropy_coeff),
            **train_metrics,
        }
        for regime in ("competitive", "mixed", "cooperative"):
            view = _readout_bucket(episode_regime_acc, regime)
            episode_metrics[f"regime_{regime}_rounds"] = int(view["n_rounds"])
            episode_metrics[f"regime_{regime}_coop"] = float(view["coop_rate"])
            episode_metrics[f"regime_{regime}_reward"] = float(view["avg_reward"])
        metrics_over_time.append(episode_metrics)
        if wandb_run is not None:
            wandb_run.log(episode_metrics, step=abs_episode1)

        if (episode + 1) % max(1, cfg.log_interval) == 0:
            print(
                f"[episode {abs_episode1:04d}] "
                f"coop={coop_rate:.3f} avg_reward={avg_reward:.3f} "
                f"loss={train_metrics['loss_total']:.4f}"
            )

        if (episode + 1) % max(1, cfg.regime_log_interval) == 0:
            summary_chunks = []
            for regime in ("competitive", "mixed", "cooperative"):
                win = _readout_bucket(window_regime_acc, regime)
                summary_chunks.append(
                    f"{regime[:4]}={win['coop_rate']:.3f}(n={int(win['n_rounds'])})"
                )
                _append_jsonl(
                    cfg.metrics_jsonl_path,
                    {
                        "episode": int(abs_episode1),
                        "seed": int(cfg.seed),
                        "condition": str(cfg.condition_name),
                        "scope": "regime",
                        "key": regime,
                        "window": "window",
                        "n_rounds": int(win["n_rounds"]),
                        "coop_rate": float(win["coop_rate"]),
                        "avg_reward": float(win["avg_reward"]),
                    },
                )
                cum = _readout_bucket(cumulative_regime_acc, regime)
                _append_jsonl(
                    cfg.metrics_jsonl_path,
                    {
                        "episode": int(abs_episode1),
                        "seed": int(cfg.seed),
                        "condition": str(cfg.condition_name),
                        "scope": "regime",
                        "key": regime,
                        "window": "cumulative",
                        "n_rounds": int(cum["n_rounds"]),
                        "coop_rate": float(cum["coop_rate"]),
                        "avg_reward": float(cum["avg_reward"]),
                    },
                )

            for f_key in sorted(window_f_acc.keys(), key=float):
                win = _readout_bucket(window_f_acc, f_key)
                _append_jsonl(
                    cfg.metrics_jsonl_path,
                    {
                        "episode": int(abs_episode1),
                        "seed": int(cfg.seed),
                        "condition": str(cfg.condition_name),
                        "scope": "f_value",
                        "key": str(f_key),
                        "window": "window",
                        "n_rounds": int(win["n_rounds"]),
                        "coop_rate": float(win["coop_rate"]),
                        "avg_reward": float(win["avg_reward"]),
                    },
                )
            for f_key in sorted(cumulative_f_acc.keys(), key=float):
                cum = _readout_bucket(cumulative_f_acc, f_key)
                _append_jsonl(
                    cfg.metrics_jsonl_path,
                    {
                        "episode": int(abs_episode1),
                        "seed": int(cfg.seed),
                        "condition": str(cfg.condition_name),
                        "scope": "f_value",
                        "key": str(f_key),
                        "window": "cumulative",
                        "n_rounds": int(cum["n_rounds"]),
                        "coop_rate": float(cum["coop_rate"]),
                        "avg_reward": float(cum["avg_reward"]),
                    },
                )

            if cfg.comm_enabled and len(sender_ids) > 0:
                all_msg_f = np.zeros((cfg.vocab_size, len(f_keys_sorted)), dtype=np.float64)
                all_msg_action = np.zeros((cfg.vocab_size, 2), dtype=np.float64)
                for sender_id in sender_ids:
                    msg_f_counts = window_comm_counts[sender_id]["msg_f"]
                    msg_action_counts = window_comm_counts[sender_id]["msg_action"]
                    all_msg_f += msg_f_counts
                    all_msg_action += msg_action_counts

                    mi_stats_f = _mi_null_independence_stats(
                        msg_f_counts,
                        n_perms=cfg.mi_null_perms,
                        alpha=cfg.mi_alpha,
                        rng=mi_rng,
                    )
                    mi_stats_action = _mi_null_independence_stats(
                        msg_action_counts,
                        n_perms=cfg.mi_null_perms,
                        alpha=cfg.mi_alpha,
                        rng=mi_rng,
                    )
                    msg_entropy = _entropy_from_counts_1d(np.sum(msg_f_counts, axis=1))
                    msg_entropy_max = float(np.log2(max(1, int(cfg.vocab_size))))
                    _append_jsonl(
                        cfg.metrics_jsonl_path,
                        {
                            "episode": int(abs_episode1),
                            "seed": int(cfg.seed),
                            "condition": str(cfg.condition_name),
                            "scope": "comm",
                            "key": sender_id,
                            "window": "window",
                            "metric": "mi_message_f",
                            "mi": float(mi_stats_f["mi_observed"]),
                            "mi_unit": "bits",
                            "mi_perm_p95": float(mi_stats_f["mi_perm_p95"]),
                            "mi_p_value": float(mi_stats_f["mi_p_value"]),
                            "mi_significant": bool(mi_stats_f["mi_significant"]),
                            "mi_null_method": "independence_multinomial",
                            "mi_null_perms": int(cfg.mi_null_perms),
                            "h_message": float(msg_entropy),
                            "h_message_max": float(msg_entropy_max),
                            "n_pairs": int(np.sum(msg_f_counts)),
                        },
                    )
                    _append_jsonl(
                        cfg.metrics_jsonl_path,
                        {
                            "episode": int(abs_episode1),
                            "seed": int(cfg.seed),
                            "condition": str(cfg.condition_name),
                            "scope": "comm",
                            "key": sender_id,
                            "window": "window",
                            "metric": "mi_message_action",
                            "mi": float(mi_stats_action["mi_observed"]),
                            "mi_unit": "bits",
                            "mi_perm_p95": float(mi_stats_action["mi_perm_p95"]),
                            "mi_p_value": float(mi_stats_action["mi_p_value"]),
                            "mi_significant": bool(mi_stats_action["mi_significant"]),
                            "mi_null_method": "independence_multinomial",
                            "mi_null_perms": int(cfg.mi_null_perms),
                            "h_message": float(msg_entropy),
                            "h_message_max": float(msg_entropy_max),
                            "n_pairs": int(np.sum(msg_action_counts)),
                        },
                    )

                mi_all_stats_f = _mi_null_independence_stats(
                    all_msg_f,
                    n_perms=cfg.mi_null_perms,
                    alpha=cfg.mi_alpha,
                    rng=mi_rng,
                )
                mi_all_stats_action = _mi_null_independence_stats(
                    all_msg_action,
                    n_perms=cfg.mi_null_perms,
                    alpha=cfg.mi_alpha,
                    rng=mi_rng,
                )
                msg_all_entropy = _entropy_from_counts_1d(np.sum(all_msg_f, axis=1))
                msg_entropy_max = float(np.log2(max(1, int(cfg.vocab_size))))
                mi_all_msg_f = float(mi_all_stats_f["mi_observed"])
                mi_all_msg_action = float(mi_all_stats_action["mi_observed"])
                summary_chunks.append(f"mi(m;f)={mi_all_msg_f:.3f}")
                summary_chunks.append(f"mi(m;a)={mi_all_msg_action:.3f}")
                all_resp = np.array([], dtype=np.float64)
                if cfg.log_trainer_responsiveness:
                    all_resp = np.array(
                        [x for values in window_responsiveness.values() for x in values],
                        dtype=np.float64,
                    )
                if all_resp.size > 0:
                    summary_chunks.append(f"resp={float(np.mean(all_resp)):.3f}")
                _append_jsonl(
                    cfg.metrics_jsonl_path,
                    {
                        "episode": int(abs_episode1),
                        "seed": int(cfg.seed),
                        "condition": str(cfg.condition_name),
                        "scope": "comm",
                        "key": "all_senders",
                        "window": "window",
                        "metric": "mi_message_f",
                        "mi": float(mi_all_msg_f),
                        "mi_unit": "bits",
                        "mi_perm_p95": float(mi_all_stats_f["mi_perm_p95"]),
                        "mi_p_value": float(mi_all_stats_f["mi_p_value"]),
                        "mi_significant": bool(mi_all_stats_f["mi_significant"]),
                        "mi_null_method": "independence_multinomial",
                        "mi_null_perms": int(cfg.mi_null_perms),
                        "h_message": float(msg_all_entropy),
                        "h_message_max": float(msg_entropy_max),
                        "n_pairs": int(np.sum(all_msg_f)),
                    },
                )
                _append_jsonl(
                    cfg.metrics_jsonl_path,
                    {
                        "episode": int(abs_episode1),
                        "seed": int(cfg.seed),
                        "condition": str(cfg.condition_name),
                        "scope": "comm",
                        "key": "all_senders",
                        "window": "window",
                        "metric": "mi_message_action",
                        "mi": float(mi_all_msg_action),
                        "mi_unit": "bits",
                        "mi_perm_p95": float(mi_all_stats_action["mi_perm_p95"]),
                        "mi_p_value": float(mi_all_stats_action["mi_p_value"]),
                        "mi_significant": bool(mi_all_stats_action["mi_significant"]),
                        "mi_null_method": "independence_multinomial",
                        "mi_null_perms": int(cfg.mi_null_perms),
                        "h_message": float(msg_all_entropy),
                        "h_message_max": float(msg_entropy_max),
                        "n_pairs": int(np.sum(all_msg_action)),
                    },
                )
                if cfg.log_trainer_responsiveness:
                    for agent_id in agent_ids:
                        agent_resp = np.asarray(window_responsiveness.get(agent_id, []), dtype=np.float64)
                        if agent_resp.size == 0:
                            continue
                        _append_jsonl(
                            cfg.metrics_jsonl_path,
                            {
                                "episode": int(abs_episode1),
                                "seed": int(cfg.seed),
                                "condition": str(cfg.condition_name),
                                "scope": "comm",
                                "key": str(agent_id),
                                "window": "window",
                                "metric": "responsiveness_kl",
                                "value": float(np.mean(agent_resp)),
                                "value_std": float(np.std(agent_resp)),
                                "n_pairs": int(agent_resp.size),
                            },
                        )
                    if all_resp.size > 0:
                        _append_jsonl(
                            cfg.metrics_jsonl_path,
                            {
                                "episode": int(abs_episode1),
                                "seed": int(cfg.seed),
                                "condition": str(cfg.condition_name),
                                "scope": "comm",
                                "key": "all_agents",
                                "window": "window",
                                "metric": "responsiveness_kl",
                                "value": float(np.mean(all_resp)),
                                "value_std": float(np.std(all_resp)),
                                "n_pairs": int(all_resp.size),
                            },
                        )

            print(
                f"[regime @ episode {abs_episode1:04d}] " + " ".join(summary_chunks)
            )
            window_regime_acc = {
                "competitive": _bucket(),
                "mixed": _bucket(),
                "cooperative": _bucket(),
            }
            window_f_acc = {}
            if cfg.comm_enabled and len(sender_ids) > 0:
                window_comm_counts = _new_comm_window_counts()
                if cfg.log_trainer_responsiveness:
                    window_responsiveness = {agent_id: [] for agent_id in agent_ids}

        if (
            cfg.checkpoint_interval > 0
            and abs_episode1 % int(cfg.checkpoint_interval) == 0
            and (episode + 1) < cfg.n_episodes
        ):
            ckpt_path = _checkpoint_with_episode(cfg.save_path, abs_episode1)
            ckpt_cfg = TrainConfig(**cfg.__dict__)
            ckpt_cfg.save_path = ckpt_path
            _save_agents(ckpt_path, agents, ckpt_cfg)
            print(f"[checkpoint] saved intermediate checkpoint: {ckpt_path}")

        # Early stopping on avg_reward if enabled.
        if cfg.early_stop_patience > 0:
            current = avg_reward
            if current > best_metric + cfg.early_stop_min_delta:
                best_metric = current
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= cfg.early_stop_patience:
                    print(
                        f"[early-stop] stopping at episode {abs_episode1} "
                        f"(best_avg_reward={best_metric:.4f})"
                    )
                    break

    _save_agents(cfg.save_path, agents, cfg)
    if wandb_run is not None:
        wandb_run.finish()

    if session_logger is not None and cfg.consolidate_sessions:
        consolidated = session_logger.consolidate(delete_parts=False)
        print(f"[session-logger] consolidated sessions -> {consolidated}")

    if cfg.run_regime_audit:
        env_cfg = dict(
            n_agents=cfg.n_agents,
            num_game_iterations=cfg.T,
            mult_fact=list(cfg.F),
            F=list(cfg.F),
            uncertainties=list(cfg.sigmas),
            fraction=False,
            rho=cfg.rho,
            epsilon_tremble=cfg.epsilon_tremble,
            endowment=cfg.endowment,
        )
        audit = regime_audit(env_cfg, n_sessions=cfg.audit_sessions)
        print("[regime-audit]", json.dumps(audit))

    return metrics_over_time


def train(config: TrainConfig):
    if config.comm_enabled and config.enable_comm_fallback:
        for attempt in range(config.max_comm_debug_cycles):
            try:
                return _single_run(config)
            except FloatingPointError as exc:
                print(
                    f"[comm-debug] attempt {attempt + 1}/{config.max_comm_debug_cycles} "
                    f"failed: {exc}"
                )
                if attempt == config.max_comm_debug_cycles - 1:
                    fallback_cfg = TrainConfig(**config.__dict__)
                    fallback_cfg.comm_enabled = False
                    fallback_cfg.n_senders = 0
                    print("[comm-debug] falling back to no-communication PPO baseline.")
                    metrics = _single_run(fallback_cfg)
                    for row in metrics:
                        row["fallback_no_comm"] = True
                    return metrics
    return _single_run(config)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", type=int, default=4)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--endowment", type=float, default=4.0)
    parser.add_argument("--F", nargs="*", type=float, default=[0.5, 1.5, 2.5, 3.5, 5.0])
    parser.add_argument("--sigmas", nargs="*", type=float, default=[0.5, 0.5, 0.5, 0.5])
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--epsilon_tremble", type=float, default=0.05)
    parser.add_argument("--comm_enabled", action="store_true")
    parser.add_argument("--n_senders", type=int, default=0)
    parser.add_argument("--vocab_size", type=int, default=2)
    parser.add_argument("--msg_dropout", type=float, default=0.1)
    parser.add_argument(
        "--msg_training_intervention",
        type=str,
        default="none",
        choices=["none", "uniform", "public_random", "fixed0", "fixed1", "sender_shuffle"],
    )
    parser.add_argument("--msg_training_history_len", type=int, default=4096)
    parser.add_argument("--episode_offset", type=int, default=0)
    parser.add_argument("--schedule_total_episodes", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--disable_value_time_feature", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--value_coeff", type=float, default=0.5)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument(
        "--entropy_schedule",
        type=str,
        choices=["none", "linear", "cosine"],
        default="none",
    )
    parser.add_argument("--entropy_coeff_final", type=float, default=0.001)
    parser.add_argument(
        "--msg_entropy_coeff",
        type=float,
        default=None,
        help="Entropy coeff for message head. If omitted, uses --entropy_coeff for both.",
    )
    parser.add_argument(
        "--msg_entropy_coeff_final",
        type=float,
        default=None,
        help="Optional final entropy coeff for message head schedule.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=32)
    parser.add_argument("--sign_lambda", type=float, default=0.1)
    parser.add_argument("--list_lambda", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="outputs/ppo_agents.pt")
    parser.add_argument("--init_ckpt", type=str, default="")
    parser.add_argument("--reward_scale", type=float, default=20.0)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        choices=["none", "linear", "cosine"],
        default="none",
    )
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-6)
    parser.add_argument("--disable_comm_fallback", action="store_true")
    parser.add_argument("--max_comm_debug_cycles", type=int, default=2)
    parser.add_argument("--log_sessions", action="store_true")
    parser.add_argument("--session_log_dir", type=str, default="outputs/sessions")
    parser.add_argument("--condition_name", type=str, default="default")
    parser.add_argument("--consolidate_sessions", action="store_true")
    parser.add_argument("--run_regime_audit", action="store_true")
    parser.add_argument("--audit_sessions", type=int, default=100)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dsc-epgg")
    parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--regime_log_interval", type=int, default=500)
    parser.add_argument("--metrics_jsonl_path", type=str, default="")
    parser.add_argument("--checkpoint_interval", type=int, default=0)
    parser.add_argument("--mi_null_perms", type=int, default=200)
    parser.add_argument("--mi_alpha", type=float, default=0.05)
    parser.add_argument("--disable_trainer_responsiveness_logging", action="store_true")
    return parser.parse_args()


def args_to_config(args) -> TrainConfig:
    resolved_n_senders = int(args.n_senders)
    if bool(args.comm_enabled) and resolved_n_senders == 0:
        # v3.2 default: in comm-on conditions, all agents can send.
        resolved_n_senders = int(args.n_agents)

    cfg = TrainConfig(
        n_agents=args.n_agents,
        T=args.T,
        n_episodes=args.n_episodes,
        endowment=args.endowment,
        F=tuple(args.F),
        sigmas=tuple(args.sigmas),
        rho=args.rho,
        epsilon_tremble=args.epsilon_tremble,
        comm_enabled=bool(args.comm_enabled),
        n_senders=resolved_n_senders,
        vocab_size=args.vocab_size,
        msg_dropout=args.msg_dropout,
        msg_training_intervention=args.msg_training_intervention,
        msg_training_history_len=args.msg_training_history_len,
        episode_offset=args.episode_offset,
        schedule_total_episodes=args.schedule_total_episodes,
        hidden_size=args.hidden_size,
        value_time_feature=not bool(args.disable_value_time_feature),
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        clip_ratio=args.clip_ratio,
        value_coeff=args.value_coeff,
        entropy_coeff=args.entropy_coeff,
        entropy_schedule=args.entropy_schedule,
        entropy_coeff_final=args.entropy_coeff_final,
        msg_entropy_coeff=args.msg_entropy_coeff,
        msg_entropy_coeff_final=args.msg_entropy_coeff_final,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        sign_lambda=args.sign_lambda,
        list_lambda=args.list_lambda,
        seed=args.seed,
        log_interval=args.log_interval,
        save_path=args.save_path,
        init_ckpt=args.init_ckpt,
        reward_scale=args.reward_scale,
        lr_schedule=args.lr_schedule,
        min_lr=args.min_lr,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        enable_comm_fallback=not bool(args.disable_comm_fallback),
        max_comm_debug_cycles=args.max_comm_debug_cycles,
        log_sessions=bool(args.log_sessions),
        session_log_dir=args.session_log_dir,
        condition_name=args.condition_name,
        consolidate_sessions=bool(args.consolidate_sessions),
        run_regime_audit=bool(args.run_regime_audit),
        audit_sessions=args.audit_sessions,
        use_wandb=bool(args.use_wandb),
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        regime_log_interval=args.regime_log_interval,
        metrics_jsonl_path=args.metrics_jsonl_path,
        checkpoint_interval=args.checkpoint_interval,
        mi_null_perms=args.mi_null_perms,
        mi_alpha=args.mi_alpha,
        log_trainer_responsiveness=not bool(args.disable_trainer_responsiveness_logging),
    )
    if len(cfg.sigmas) != cfg.n_agents:
        raise ValueError(f"len(sigmas) must equal n_agents ({cfg.n_agents})")
    if cfg.reward_scale <= 0:
        raise ValueError("reward_scale must be > 0")
    if float(cfg.entropy_coeff_final) < 0.0:
        raise ValueError("entropy_coeff_final must be >= 0")
    if cfg.msg_entropy_coeff is not None and float(cfg.msg_entropy_coeff) < 0.0:
        raise ValueError("msg_entropy_coeff must be >= 0")
    if cfg.msg_entropy_coeff_final is not None and float(cfg.msg_entropy_coeff_final) < 0.0:
        raise ValueError("msg_entropy_coeff_final must be >= 0")
    if cfg.n_senders < 0 or cfg.n_senders > cfg.n_agents:
        raise ValueError("n_senders must be in [0, n_agents]")
    if cfg.comm_enabled and cfg.n_senders <= 0:
        raise ValueError("comm_enabled requires n_senders > 0")
    if (not cfg.comm_enabled) and str(cfg.msg_training_intervention) != "none":
        raise ValueError("msg_training_intervention requires comm_enabled")
    if str(cfg.msg_training_intervention) != "none" and (
        float(cfg.sign_lambda) != 0.0 or float(cfg.list_lambda) != 0.0
    ):
        raise ValueError(
            "msg_training_intervention requires sign_lambda=0 and list_lambda=0"
        )
    if str(cfg.msg_training_intervention) == "fixed1" and int(cfg.vocab_size) < 2:
        raise ValueError("msg_training_intervention=fixed1 requires vocab_size >= 2")
    if int(cfg.msg_training_history_len) <= 0:
        raise ValueError("msg_training_history_len must be > 0")
    if int(cfg.episode_offset) < 0:
        raise ValueError("episode_offset must be >= 0")
    if int(cfg.schedule_total_episodes) < 0:
        raise ValueError("schedule_total_episodes must be >= 0")
    if int(cfg.schedule_total_episodes) > 0 and int(cfg.schedule_total_episodes) < (
        int(cfg.episode_offset) + int(cfg.n_episodes)
    ):
        raise ValueError(
            "schedule_total_episodes must be >= episode_offset + n_episodes"
        )
    if cfg.regime_log_interval <= 0:
        raise ValueError("regime_log_interval must be > 0")
    if cfg.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be >= 0")
    if cfg.mi_null_perms <= 0:
        raise ValueError("mi_null_perms must be > 0")
    if not (0.0 < float(cfg.mi_alpha) < 1.0):
        raise ValueError("mi_alpha must be in (0, 1)")
    return cfg


if __name__ == "__main__":
    args = parse_args()
    cfg = args_to_config(args)
    metrics = train(cfg)
    print(json.dumps({"episodes": len(metrics), "last": metrics[-1] if metrics else {}}, indent=2))
