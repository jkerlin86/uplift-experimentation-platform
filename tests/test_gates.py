import numpy as np
from scipy import stats

from uplift_platform.learners import crossfit_uplift
from uplift_platform.metrics import qini_coefficient
from uplift_platform.synthetic import simulate_rct, true_tau

SMALL_PARAMS = {
    "max_iter": 40,
    "learning_rate": 0.08,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
}


def test_synthetic_dgp_shape():
    sim = simulate_rct(n=50_000, seed=1)
    assert sim["tau"].min() > 0
    assert sim["tau"].max() < 0.05
    assert 0.80 < sim["t"].mean() < 0.90
    # The effect is heterogeneous: the top decile of true tau carries a
    # much larger mean effect than the population.
    k = len(sim["tau"]) // 10
    assert np.sort(sim["tau"])[-k:].mean() > 1.5 * sim["tau"].mean()


def test_synthetic_recovery_small():
    sim = simulate_rct(n=60_000, seed=2)
    oof = crossfit_uplift(
        sim["x"], sim["y"].astype(float), sim["t"], propensity=0.85,
        n_splits=2, seed=2, params=SMALL_PARAMS,
    )
    true_ate = sim["tau"].mean()
    for key in ("tau_t", "tau_x"):
        assert abs(oof[key].mean() - true_ate) < 0.004
        assert stats.spearmanr(oof[key], sim["tau"]).statistic > 0.3


def test_placebo_collapses_on_synthetic_null():
    rng = np.random.default_rng(4)
    n = 60_000
    x = rng.normal(size=(n, 8))
    t = (rng.random(n) < 0.85).astype(int)
    y = rng.binomial(1, 0.02, n).astype(float)  # no effect at all
    oof = crossfit_uplift(
        x, y, t, propensity=0.85, n_splits=2, seed=4, params=SMALL_PARAMS
    )
    assert abs(qini_coefficient(y, t, oof["tau_t"])) < 0.35
    assert abs(true_tau(x).mean() - simulate_rct(n=1000, seed=0)["tau"].mean()) < 0.01
