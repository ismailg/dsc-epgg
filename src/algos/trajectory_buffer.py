from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


class TrajectoryBuffer:
    """
    Fixed-length trajectory storage for PPO/GAE in multi-agent sessions.
    """

    def __init__(
        self,
        agent_ids: Iterable[str],
        T: int,
        obs_dim: int,
        comm_enabled: bool = False,
        vocab_size: int = 0,
        sender_ids: Optional[Iterable[str]] = None,
    ) -> None:
        self.agent_ids = list(agent_ids)
        self.agent_to_idx = {agent_id: i for i, agent_id in enumerate(self.agent_ids)}
        self.n_agents = len(self.agent_ids)
        self.T = int(T)
        self.obs_dim = int(obs_dim)

        self.observations = np.zeros((self.T, self.n_agents, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.T, self.n_agents), dtype=np.int32)
        self.rewards = np.zeros((self.T, self.n_agents), dtype=np.float32)
        self.values = np.zeros((self.T, self.n_agents), dtype=np.float32)
        self.log_probs = np.zeros((self.T, self.n_agents), dtype=np.float32)
        self.dones = np.zeros((self.T,), dtype=bool)

        # Set-C / diagnostics.
        self.executed_actions = np.zeros((self.T, self.n_agents), dtype=np.int32)
        self.flips = np.zeros((self.T, self.n_agents), dtype=bool)
        self.true_f = np.zeros((self.T,), dtype=np.float32)
        self.f_hats = np.zeros((self.T, self.n_agents), dtype=np.float32)
        self.agent_rewards = np.zeros((self.T, self.n_agents), dtype=np.float32)

        # Optional communication traces.
        self.comm_enabled = bool(comm_enabled)
        self.sender_ids = list(sender_ids) if sender_ids is not None else []
        self.sender_to_idx = {sender_id: i for i, sender_id in enumerate(self.sender_ids)}
        if self.comm_enabled and len(self.sender_ids) > 0:
            n_senders = len(self.sender_ids)
            self.messages = np.zeros((self.T, n_senders), dtype=np.int32)
            self.message_actions = np.full((self.T, self.n_agents), -1, dtype=np.int32)
            self.message_log_probs = np.zeros((self.T, self.n_agents), dtype=np.float32)
        else:
            self.messages = None
            self.message_actions = None
            self.message_log_probs = None

        # Listening auxiliary signal from rollout-time distribution shift.
        self.listening_bonus = np.zeros((self.T, self.n_agents), dtype=np.float32)

        self.t = 0

    def _agent_index(self, agent_id: str) -> int:
        return self.agent_to_idx[agent_id]

    def store(
        self,
        obs: Dict[str, np.ndarray],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        values: Dict[str, float],
        log_probs: Dict[str, float],
        done: bool,
        executed_actions: Dict[str, int],
        flips: Dict[str, bool],
        true_f: float,
        f_hats: Dict[str, np.ndarray],
        messages: Optional[Dict[str, int]] = None,
        message_actions: Optional[Dict[str, int]] = None,
        message_log_probs: Optional[Dict[str, float]] = None,
        listening_bonus: Optional[Dict[str, float]] = None,
    ) -> None:
        if self.t >= self.T:
            raise IndexError("trajectory buffer overflow")

        for agent_id in self.agent_ids:
            i = self._agent_index(agent_id)
            self.observations[self.t, i] = np.asarray(obs[agent_id], dtype=np.float32)
            self.actions[self.t, i] = int(actions[agent_id])
            self.rewards[self.t, i] = float(rewards[agent_id])
            self.agent_rewards[self.t, i] = float(rewards[agent_id])
            self.values[self.t, i] = float(values[agent_id])
            self.log_probs[self.t, i] = float(log_probs[agent_id])
            self.executed_actions[self.t, i] = int(executed_actions[agent_id])
            self.flips[self.t, i] = bool(flips[agent_id])

            raw = np.asarray(f_hats[agent_id]).reshape(-1)
            self.f_hats[self.t, i] = float(raw[0]) if raw.size > 0 else 0.0

            if listening_bonus is not None:
                self.listening_bonus[self.t, i] = float(listening_bonus.get(agent_id, 0.0))

        self.true_f[self.t] = float(true_f)
        self.dones[self.t] = bool(done)

        if self.messages is not None and messages is not None:
            for sender_id, msg in messages.items():
                s_idx = self.sender_to_idx.get(sender_id)
                if s_idx is not None:
                    self.messages[self.t, s_idx] = int(msg)

        if self.message_actions is not None and message_actions is not None:
            for agent_id, action in message_actions.items():
                i = self._agent_index(agent_id)
                self.message_actions[self.t, i] = int(action)

        if self.message_log_probs is not None and message_log_probs is not None:
            for agent_id, lp in message_log_probs.items():
                i = self._agent_index(agent_id)
                self.message_log_probs[self.t, i] = float(lp)

        self.t += 1

    def compute_gae(
        self,
        last_values: np.ndarray,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        if self.t == 0:
            return np.zeros((0, self.n_agents), dtype=np.float32), np.zeros(
                (0, self.n_agents), dtype=np.float32
            )

        last_values = np.asarray(last_values, dtype=np.float32).reshape(self.n_agents)
        rewards = self.rewards[: self.t]
        values = self.values[: self.t]
        dones = self.dones[: self.t]

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = np.zeros((self.n_agents,), dtype=np.float32)

        for step in reversed(range(self.t)):
            if step == self.t - 1:
                next_values = last_values
                next_non_terminal = 1.0 - float(dones[step])
            else:
                next_values = values[step + 1]
                next_non_terminal = 1.0 - float(dones[step])

            delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[step] = last_gae

        returns = advantages + values
        return advantages.astype(np.float32), returns.astype(np.float32)

    def reset(self) -> None:
        self.t = 0

