from pathlib import Path

import torch

from src.experiments_pgg_v0 import train_ppo
from src.experiments_pgg_v0.train_ppo import minimal_test_config, train
from src.wrappers.observation_wrapper import ObservationWrapper


def test_uniform_training_intervention_updates_marginals_with_delivered_tokens(
    tmp_path: Path, monkeypatch
):
    ckpt = tmp_path / "cond1_seed777.pt"
    recorded = []
    orig_update = ObservationWrapper.update_msg_marginals

    def fake_apply(intervention, delivered, vocab_size):
        return {sender_id: 1 for sender_id in delivered.keys()}

    def record_update(self, sender_id, message):
        recorded.append(int(message))
        return orig_update(self, sender_id, message)

    monkeypatch.setattr(train_ppo, "_apply_training_message_intervention", fake_apply)
    monkeypatch.setattr(ObservationWrapper, "update_msg_marginals", record_update)

    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=2,
        T=4,
        comm_enabled=True,
        n_senders=4,
        seed=777,
        save_path=str(ckpt),
        condition_name="cond1",
        sign_lambda=0.0,
        list_lambda=0.0,
        msg_training_intervention="uniform",
    )
    train(cfg)

    assert len(recorded) > 0
    assert set(recorded) == {1}
    payload = torch.load(ckpt, map_location="cpu")
    assert payload["config"]["msg_training_intervention"] == "uniform"


def test_public_random_training_intervention_shares_one_token_across_senders():
    delivered = {"agent_0": 0, "agent_1": 1, "agent_2": 0, "agent_3": 1}
    out = train_ppo._apply_training_message_intervention(
        intervention="public_random",
        delivered=delivered,
        vocab_size=2,
    )
    assert set(out.keys()) == set(delivered.keys())
    assert len(set(out.values())) == 1


def test_episode_offset_drives_absolute_checkpoint_numbering(tmp_path: Path):
    ckpt = tmp_path / "cond2_seed888.pt"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=3,
        T=4,
        comm_enabled=False,
        n_senders=0,
        seed=888,
        save_path=str(ckpt),
        condition_name="cond2",
        checkpoint_interval=1,
        episode_offset=10,
        schedule_total_episodes=13,
    )
    train(cfg)

    assert (tmp_path / "cond2_seed888_ep11.pt").exists()
    assert (tmp_path / "cond2_seed888_ep12.pt").exists()
    payload = torch.load(ckpt, map_location="cpu")
    assert payload["config"]["episode_offset"] == 10
    assert payload["config"]["schedule_total_episodes"] == 13
