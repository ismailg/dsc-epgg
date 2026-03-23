from src.analysis.plot_phase3_online_vs_checkpoint import _mean_sem


def test_mean_sem_singleton_and_pair():
    mean1, sem1 = _mean_sem([0.4])
    assert mean1 == 0.4
    assert sem1 == 0.0
    mean2, sem2 = _mean_sem([0.2, 0.4])
    assert abs(mean2 - 0.3) < 1e-9
    assert sem2 > 0.0
