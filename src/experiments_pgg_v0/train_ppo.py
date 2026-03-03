from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List

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

    hidden_size: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 32
    sign_lambda: float = 0.1
    list_lambda: float = 0.1

    seed: int = 42
    log_interval: int = 10
    save_path: str = "outputs/ppo_agents.pt"
    lr_schedule: str = "none"  # one of: none, linear
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


def _set_agents_lr(agents: Dict[str, PPOAgentV2], lr_value: float):
    for agent in agents.values():
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = float(lr_value)


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
    agents = {}
    for agent_id in _agent_ids(cfg.n_agents):
        agents[agent_id] = PPOAgentV2(
            obs_dim=obs_dim,
            action_size=2,
            can_send=(cfg.comm_enabled and agent_id in sender_ids),
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            lr=cfg.lr,
        )
    return agents


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
    if cfg.lr_schedule == "linear":
        _set_agents_lr(agents, cfg.lr)
    ppo = PPOTrainer(
        agents=agents,
        clip_ratio=cfg.clip_ratio,
        value_coeff=cfg.value_coeff,
        entropy_coeff=cfg.entropy_coeff,
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
    best_metric = -float("inf")
    stale_epochs = 0
    for episode in range(cfg.n_episodes):
        if cfg.lr_schedule == "linear":
            progress = float(episode) / float(max(1, cfg.n_episodes - 1))
            episode_lr = cfg.lr - (cfg.lr - cfg.min_lr) * progress
            episode_lr = max(float(cfg.min_lr), float(episode_lr))
            _set_agents_lr(agents, episode_lr)

        buffer = TrajectoryBuffer(
            agent_ids=agent_ids,
            T=cfg.T,
            obs_dim=wrapper.obs_dim,
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
                for sender_id, msg in proposed.items():
                    wrapper.update_msg_marginals(sender_id, msg)
                current_messages = dropped
                aug_obs = {
                    agent_id: wrapper.build_obs(agent_id, raw_obs[agent_id], current_messages)
                    for agent_id in agent_ids
                }

            intended_actions = {}
            action_log_probs = {}
            values = {}
            listening_bonus = {agent_id: 0.0 for agent_id in agent_ids}
            for agent_id in agent_ids:
                action, action_lp, value, _ent, _ = agents[agent_id].sample_action(
                    aug_obs[agent_id]
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

            raw_obs_next, rewards, done, infos = env.step(intended_actions)
            rewards = _to_float_rewards(rewards)
            executed_actions = infos.get("executed_actions", intended_actions)
            flips = infos.get("flips", {agent_id: False for agent_id in agent_ids})
            true_f = float(infos.get("true_f", env.current_multiplier.item()))

            wrapper.update(executed_actions)

            buffer.store(
                obs=aug_obs,
                actions=intended_actions,
                rewards=rewards,
                values=values,
                log_probs=action_log_probs,
                done=bool(done),
                executed_actions=executed_actions,
                flips=flips,
                true_f=true_f,
                f_hats=raw_obs,
                messages=current_messages,
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
            last_values = np.array(
                [
                    agents[agent_id]
                    .value(
                        torch.tensor(
                            final_aug_obs[agent_id], dtype=torch.float32, device=agents[agent_id].action_actor.net[0].weight.device
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
        episode_metrics = {
            "episode": episode,
            "steps": buffer.t,
            "coop_rate": coop_rate,
            "avg_reward": avg_reward,
            **train_metrics,
        }
        metrics_over_time.append(episode_metrics)
        if wandb_run is not None:
            wandb_run.log(episode_metrics, step=episode)

        if (episode + 1) % max(1, cfg.log_interval) == 0:
            print(
                f"[episode {episode + 1:04d}] "
                f"coop={coop_rate:.3f} avg_reward={avg_reward:.3f} "
                f"loss={train_metrics['loss_total']:.4f}"
            )

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
                        f"[early-stop] stopping at episode {episode + 1} "
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
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--value_coeff", type=float, default=0.5)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=32)
    parser.add_argument("--sign_lambda", type=float, default=0.1)
    parser.add_argument("--list_lambda", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="outputs/ppo_agents.pt")
    parser.add_argument("--lr_schedule", type=str, choices=["none", "linear"], default="none")
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
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        clip_ratio=args.clip_ratio,
        value_coeff=args.value_coeff,
        entropy_coeff=args.entropy_coeff,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        sign_lambda=args.sign_lambda,
        list_lambda=args.list_lambda,
        seed=args.seed,
        log_interval=args.log_interval,
        save_path=args.save_path,
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
    )
    if len(cfg.sigmas) != cfg.n_agents:
        raise ValueError(f"len(sigmas) must equal n_agents ({cfg.n_agents})")
    if cfg.n_senders < 0 or cfg.n_senders > cfg.n_agents:
        raise ValueError("n_senders must be in [0, n_agents]")
    if cfg.comm_enabled and cfg.n_senders <= 0:
        raise ValueError("comm_enabled requires n_senders > 0")
    return cfg


if __name__ == "__main__":
    args = parse_args()
    cfg = args_to_config(args)
    metrics = train(cfg)
    print(json.dumps({"episodes": len(metrics), "last": metrics[-1] if metrics else {}}, indent=2))
