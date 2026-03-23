from collections import Counter
from pathlib import Path

from src.analysis.evaluate_regime_conditional import (
    _condition_seed_from_path,
    _condition_summary,
    _eval_checkpoint,
)
from src.experiments_pgg_v0.train_ppo import minimal_test_config, train


def _make_checkpoint(
    tmp_path: Path,
    *,
    condition: str,
    seed: int,
    comm_enabled: bool,
) -> Path:
    ckpt = tmp_path / f"{condition}_seed{seed}.pt"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=2,
        T=6,
        comm_enabled=comm_enabled,
        n_senders=4 if comm_enabled else 0,
        seed=seed,
        save_path=str(ckpt),
        condition_name=condition,
    )
    train(cfg)
    assert ckpt.exists()
    return ckpt


def _make_comm_checkpoint(tmp_path: Path) -> Path:
    return _make_checkpoint(
        tmp_path,
        condition="cond1",
        seed=999,
        comm_enabled=True,
    )


def test_eval_outputs_ablation_and_comm_rows(tmp_path):
    ckpt = _make_comm_checkpoint(tmp_path)
    (
        rows_none,
        comm_none,
        posterior_none,
        trace_none,
        sender_none,
        receiver_none,
        _sender_causal_none,
    ) = _eval_checkpoint(
        checkpoint_path=str(ckpt),
        n_eval_episodes=2,
        eval_seed=1234,
        greedy=True,
        msg_intervention="none",
        mi_null_perms=40,
        mi_alpha=0.05,
        collect_semantics=True,
    )
    (
        rows_ablate,
        comm_ablate,
        _posterior_ablate,
        trace_ablate,
        sender_ablate,
        receiver_ablate,
        _sender_causal_ablate,
    ) = _eval_checkpoint(
        checkpoint_path=str(ckpt),
        n_eval_episodes=2,
        eval_seed=1234,
        greedy=True,
        msg_intervention="marginal",
        mi_null_perms=40,
        mi_alpha=0.05,
        collect_semantics=True,
    )
    assert len(rows_none) > 0
    assert len(rows_ablate) > 0
    assert all(r["ablation"] == "none" for r in rows_none)
    assert all(r["ablation"] == "marginal" for r in rows_ablate)
    assert all(r["history_intervention"] == "none" for r in rows_none)
    assert all(r["history_intervention"] == "none" for r in comm_none)
    assert any(r.get("metric") == "mi_message_f" for r in comm_none)
    assert any(r.get("metric") == "responsiveness_kl" for r in comm_ablate)
    assert len(posterior_none) == 0
    assert len(trace_none) > 0
    assert len(trace_ablate) > 0
    assert all(r["history_intervention"] == "none" for r in trace_none)
    assert any("recv_pattern" in r for r in trace_none)
    assert any(r.get("summary") == "p_msg1_given_fhat" for r in sender_none)
    assert any(r.get("summary") == "p_coop_given_any_token_fhat" for r in receiver_ablate)
    assert any(r.get("summary") == "p_msg1_given_action" for r in sender_ablate)


def test_eval_supports_permute_slots_ablation(tmp_path):
    ckpt = _make_comm_checkpoint(tmp_path)
    rows, comm_rows, _posterior, trace_rows, _sender_rows, receiver_rows, _sender_causal = _eval_checkpoint(
        checkpoint_path=str(ckpt),
        n_eval_episodes=2,
        eval_seed=4321,
        greedy=True,
        msg_intervention="permute_slots",
        mi_null_perms=20,
        mi_alpha=0.05,
        collect_semantics=True,
    )
    assert len(rows) > 0
    assert len(comm_rows) > 0
    assert len(trace_rows) > 0
    assert all(r["ablation"] == "permute_slots" for r in rows)
    assert any(r.get("summary") == "p_coop_given_sender_token_fhat" for r in receiver_rows)


def test_eval_supports_public_random_ablation(tmp_path):
    ckpt = _make_comm_checkpoint(tmp_path)
    rows, _comm_rows, _posterior, trace_rows, _sender_rows, _receiver_rows, _sender_causal = _eval_checkpoint(
        checkpoint_path=str(ckpt),
        n_eval_episodes=2,
        eval_seed=2468,
        greedy=True,
        msg_intervention="public_random",
        mi_null_perms=20,
        mi_alpha=0.05,
        collect_semantics=True,
    )
    assert len(rows) > 0
    assert len(trace_rows) > 0
    assert all(r["ablation"] == "public_random" for r in rows)
    delivered_cols = sorted(
        key for key in trace_rows[0].keys() if key.startswith("delivered_msg_")
    )
    assert len(delivered_cols) > 1
    for row in trace_rows:
        delivered = {int(row[col]) for col in delivered_cols if row[col] != ""}
        assert len(delivered) == 1


def test_sender_shuffle_preserves_sender_marginals(tmp_path):
    ckpt = _make_comm_checkpoint(tmp_path)
    (
        _rows_none,
        _comm_none,
        _posterior_none,
        trace_none,
        _sender_none,
        _receiver_none,
        _sender_causal_none,
    ) = _eval_checkpoint(
        checkpoint_path=str(ckpt),
        n_eval_episodes=3,
        eval_seed=1357,
        greedy=True,
        msg_intervention="none",
        mi_null_perms=20,
        mi_alpha=0.05,
        collect_semantics=True,
    )
    (
        rows_shuffle,
        _comm_shuffle,
        _posterior_shuffle,
        trace_shuffle,
        _sender_shuffle,
        _receiver_shuffle,
        _sender_causal_shuffle,
    ) = _eval_checkpoint(
        checkpoint_path=str(ckpt),
        n_eval_episodes=3,
        eval_seed=1357,
        greedy=True,
        msg_intervention="sender_shuffle",
        mi_null_perms=20,
        mi_alpha=0.05,
        collect_semantics=True,
    )
    assert len(rows_shuffle) > 0
    assert len(trace_none) == len(trace_shuffle)
    delivered_cols = sorted(
        key for key in trace_none[0].keys() if key.startswith("delivered_msg_")
    )
    for col in delivered_cols:
        natural_counts = Counter(
            int(row[col]) for row in trace_none if row["agent_id"] == "agent_0" and row[col] != ""
        )
        shuffled_counts = Counter(
            int(row[col]) for row in trace_shuffle if row["agent_id"] == "agent_0" and row[col] != ""
        )
        assert natural_counts == shuffled_counts
    assert all(r["ablation"] == "sender_shuffle" for r in rows_shuffle)


def test_eval_history_intervention_clamps_temporal_features_in_no_comm(tmp_path):
    ckpt = _make_checkpoint(
        tmp_path,
        condition="cond2",
        seed=123,
        comm_enabled=False,
    )
    rows, comm_rows, posterior_rows, trace_rows, sender_rows, receiver_rows, sender_causal_rows = _eval_checkpoint(
        checkpoint_path=str(ckpt),
        n_eval_episodes=2,
        eval_seed=2026,
        greedy=True,
        history_intervention="clamp_temporal_high",
        mi_null_perms=20,
        mi_alpha=0.05,
        collect_semantics=True,
    )
    assert len(rows) > 0
    assert len(trace_rows) > 0
    assert len(comm_rows) == 0
    assert len(posterior_rows) == 0
    assert len(sender_rows) == 0
    assert len(receiver_rows) == 0
    assert len(sender_causal_rows) == 0
    assert all(r["history_intervention"] == "clamp_temporal_high" for r in rows)
    assert all(r["history_intervention"] == "clamp_temporal_high" for r in trace_rows)
    assert {row["obs_last_coop_fraction"] for row in trace_rows} == {1.0}
    assert {row["obs_own_last_action"] for row in trace_rows} == {1.0}
    assert {row["obs_ewma_coop"] for row in trace_rows} == {1.0}


def test_condition_summary_separates_ablation():
    rows = [
        {
            "scope": "regime",
            "condition": "cond1",
            "key": "cooperative",
            "eval_policy": "greedy",
            "ablation": "none",
            "history_intervention": "none",
            "n_rounds": 10,
            "coop_rate": 0.8,
            "avg_reward": 1.0,
        },
        {
            "scope": "regime",
            "condition": "cond1",
            "key": "cooperative",
            "eval_policy": "greedy",
            "ablation": "none",
            "history_intervention": "zero_temporal",
            "n_rounds": 10,
            "coop_rate": 0.3,
            "avg_reward": 1.0,
        },
    ]
    out = _condition_summary(rows)
    assert len(out) == 2
    keys = {(r["eval_policy"], r["ablation"], r["history_intervention"]) for r in out}
    assert ("greedy", "none", "none") in keys
    assert ("greedy", "none", "zero_temporal") in keys


def test_condition_seed_parser_accepts_control_suffixes():
    assert _condition_seed_from_path("cond1_seed101_fixed0_ep25000.pt") == ("cond1", 101)
    assert _condition_seed_from_path("cond1_seed202_public_random.pt") == ("cond1", 202)
    assert _condition_seed_from_path("cond2_seed303_uniform_ep50000.pt") == ("cond2", 303)
