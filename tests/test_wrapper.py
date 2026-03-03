import numpy as np

from src.wrappers import ObservationWrapper


def test_obs_dimension_consistency():
    wrapper = ObservationWrapper(
        n_agents=4,
        comm_enabled=True,
        n_senders=3,
        sender_ids=["agent_0", "agent_1", "agent_2"],
        vocab_size=2,
    )
    wrapper.reset(["agent_0", "agent_1", "agent_2", "agent_3"])
    messages = {"agent_0": 0, "agent_1": 1, "agent_2": 0}
    for _ in range(20):
        obs = wrapper.build_obs("agent_0", np.array([1.5, 4.0], dtype=np.float32), messages)
        assert obs.shape == (wrapper.obs_dim,)
        assert obs.dtype == np.float32
        wrapper.update({"agent_0": 1, "agent_1": 0, "agent_2": 1, "agent_3": 0})


def test_ewma_computation():
    wrapper = ObservationWrapper(n_agents=4, ewma_decay=0.9)
    wrapper.reset(["agent_0", "agent_1", "agent_2", "agent_3"])
    wrapper.update({"agent_0": 1, "agent_1": 1, "agent_2": 1, "agent_3": 1})
    assert abs(wrapper.ewma_coop - 0.1) < 1e-6
    wrapper.update({"agent_0": 1, "agent_1": 1, "agent_2": 1, "agent_3": 1})
    assert abs(wrapper.ewma_coop - 0.19) < 1e-6


def test_message_dropout_rate():
    wrapper = ObservationWrapper(
        n_agents=4,
        comm_enabled=True,
        n_senders=3,
        sender_ids=["agent_0", "agent_1", "agent_2"],
        vocab_size=2,
        msg_dropout=0.5,
    )
    wrapper.reset(["agent_0", "agent_1", "agent_2", "agent_3"])
    original = {"agent_0": 0, "agent_1": 0, "agent_2": 0}
    changed = 0
    total = 0
    for _ in range(8000):
        dropped = wrapper.apply_msg_dropout(original)
        for sender_id in original:
            total += 1
            if dropped[sender_id] != original[sender_id]:
                changed += 1
    rate = changed / float(total)
    assert 0.20 <= rate <= 0.35


def test_history_features_lag():
    wrapper = ObservationWrapper(n_agents=4)
    wrapper.reset(["agent_0", "agent_1", "agent_2", "agent_3"])
    obs0 = wrapper.build_obs("agent_0", np.array([2.0, 4.0], dtype=np.float32))
    assert obs0[2] == 0.0

    wrapper.update({"agent_0": 1, "agent_1": 1, "agent_2": 0, "agent_3": 0})
    obs1 = wrapper.build_obs("agent_0", np.array([2.0, 4.0], dtype=np.float32))
    assert abs(float(obs1[2]) - 0.5) < 1e-6
    assert int(obs1[3]) == 1

