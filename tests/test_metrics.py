import numpy as np
import pytest

from uplift_platform.metrics import (
    dr_policy_value,
    policy_table,
    qini_coefficient,
    qini_curve,
    uplift_calibration,
)

# Hand-checked example. Rows in score order:
#   t=1,y=1 ; t=1,y=1 ; t=0,y=0 ; t=1,y=0 ; t=0,y=1 ; t=0,y=0
# Totals: N_t = 3, N_c = 3, Y_t = 2, Y_c = 1.
# Randomisation-corrected total: Q(N) = 2 - 1 * (3 / 3) = 1.
# Top 2 rows: Y_t = 2, Y_c = 0, so Q = 2 regardless of the arm ratio term.
Y = np.array([1, 1, 0, 0, 1, 0], dtype=float)
T = np.array([1, 1, 0, 1, 0, 0])
SCORE = np.array([0.95, 0.90, 0.80, 0.70, 0.60, 0.50])


def test_qini_total_matches_hand_computation():
    curve = qini_curve(Y, T, SCORE, n_points=6)
    assert curve["total_incremental_conversions"] == pytest.approx(1.0)
    assert curve["qini"][-1] == pytest.approx(1.0)
    assert curve["random"][-1] == pytest.approx(1.0)


def test_qini_top_slice_matches_hand_computation():
    curve = qini_curve(Y, T, SCORE, n_points=6)
    idx = curve["phi"].index(pytest.approx(2 / 6))
    assert curve["qini"][idx] == pytest.approx(2.0)


def test_qini_perfect_ranking_scores_one():
    oracle_score = np.array([3, 3, 1, 2, 0, 1], dtype=float)
    assert qini_coefficient(Y, T, oracle_score) == pytest.approx(1.0)


def test_qini_coefficient_between_random_and_perfect():
    coef = qini_coefficient(Y, T, SCORE)
    assert 0.0 < coef < 1.0


def test_qini_constant_score_is_random():
    rng = np.random.default_rng(5)
    n = 4000
    t = (rng.random(n) < 0.85).astype(int)
    y = rng.binomial(1, 0.01 + 0.005 * t).astype(float)
    coef = qini_coefficient(y, t, np.zeros(n))
    assert abs(coef) < 0.35


def test_dr_policy_value_hand_computation():
    y = np.array([1, 0, 1, 0], dtype=float)
    t = np.array([1, 1, 0, 0])
    mu = np.zeros(4)
    all_treated = dr_policy_value(y, t, mu, mu, np.ones(4, dtype=bool), propensity=0.5)
    none_treated = dr_policy_value(y, t, mu, mu, np.zeros(4, dtype=bool), propensity=0.5)
    # Contributions: treated rows contribute y / 0.5 under treat-all, control
    # rows contribute y / 0.5 under treat-none; both means are 0.5.
    assert all_treated["value"] == pytest.approx(0.5)
    assert none_treated["value"] == pytest.approx(0.5)


def test_policy_table_keys_and_shares():
    rng = np.random.default_rng(9)
    n = 2000
    t = (rng.random(n) < 0.85).astype(int)
    y = rng.binomial(1, 0.02, n).astype(float)
    mu0 = np.full(n, 0.02)
    mu1 = np.full(n, 0.025)
    score = rng.random(n)
    table = policy_table(y, t, mu0, mu1, score, [0.1, 0.5], propensity=0.85)
    assert table["top_10pct"]["treated_share"] == pytest.approx(0.1)
    assert table["treat_all"]["treated_share"] == pytest.approx(1.0)
    assert table["treat_none"]["incremental_vs_treat_none"] == pytest.approx(0.0)


def test_calibration_bins_cover_sample():
    rng = np.random.default_rng(13)
    n = 5000
    t = (rng.random(n) < 0.85).astype(int)
    y = rng.binomial(1, 0.02, n).astype(float)
    cal = uplift_calibration(y, t, rng.random(n), n_bins=10)
    assert sum(c["n"] for c in cal) == n
    assert [c["decile"] for c in cal] == list(range(1, 11))
