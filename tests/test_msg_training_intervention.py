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
