from pathlib import Path

from src.analysis.evaluate_regime_conditional import (
    _condition_seed_from_path,
    _condition_summary,
    _eval_checkpoint,
)
from src.experiments_pgg_v0.train_ppo import minimal_test_config, train


def _make_comm_checkpoint(tmp_path: Path) -> Path:
    ckpt = tmp_path / "cond1_seed999.pt"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=2,
        T=6,
        comm_enabled=True,
        n_senders=4,
        seed=999,
        save_path=str(ckpt),
        condition_name="cond1",
    )
    train(cfg)
    assert ckpt.exists()
    return ckpt


def test_eval_outputs_ablation_and_comm_rows(tmp_path):
    ckpt = _make_comm_checkpoint(tmp_path)
    rows_none, comm_none, posterior_none, trace_none, sender_none, receiver_none = _eval_checkpoint(
        checkpoint_path=str(ckpt),
        n_eval_episodes=2,
        eval_seed=1234,
        greedy=True,
        msg_intervention="none",
        mi_null_perms=40,
        mi_alpha=0.05,
        collect_semantics=True,
    )
    rows_ablate, comm_ablate, _posterior_ablate, trace_ablate, sender_ablate, receiver_ablate = _eval_checkpoint(
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
    assert any(r.get("metric") == "mi_message_f" for r in comm_none)
    assert any(r.get("metric") == "responsiveness_kl" for r in comm_ablate)
    assert len(posterior_none) == 0
    assert len(trace_none) > 0
    assert len(trace_ablate) > 0
    assert any("recv_pattern" in r for r in trace_none)
    assert any(r.get("summary") == "p_msg1_given_fhat" for r in sender_none)
    assert any(r.get("summary") == "p_coop_given_any_token_fhat" for r in receiver_ablate)


def test_eval_supports_permute_slots_ablation(tmp_path):
    ckpt = _make_comm_checkpoint(tmp_path)
    rows, comm_rows, _posterior, trace_rows, _sender_rows, receiver_rows = _eval_checkpoint(
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


def test_condition_summary_separates_ablation():
    rows = [
        {
            "scope": "regime",
            "condition": "cond1",
            "key": "cooperative",
            "eval_policy": "greedy",
            "ablation": "none",
            "n_rounds": 10,
            "coop_rate": 0.8,
            "avg_reward": 1.0,
        },
        {
            "scope": "regime",
            "condition": "cond1",
            "key": "cooperative",
            "eval_policy": "greedy",
            "ablation": "marginal",
            "n_rounds": 10,
            "coop_rate": 0.3,
            "avg_reward": 1.0,
        },
    ]
    out = _condition_summary(rows)
    assert len(out) == 2
    keys = {(r["eval_policy"], r["ablation"]) for r in out}
    assert ("greedy", "none") in keys
    assert ("greedy", "marginal") in keys


def test_condition_seed_parser_accepts_control_suffixes():
    assert _condition_seed_from_path("cond1_seed101_fixed0_ep25000.pt") == ("cond1", 101)
    assert _condition_seed_from_path("cond1_seed202_public_random.pt") == ("cond1", 202)
    assert _condition_seed_from_path("cond2_seed303_uniform_ep50000.pt") == ("cond2", 303)
