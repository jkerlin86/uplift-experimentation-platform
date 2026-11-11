import numpy as np
import pandas as pd
import pytest

from uplift_platform.harness import (
    balance_check,
    exposure_readout,
    itt_estimate,
    srm_check,
    standard_readout,
)


def test_itt_known_values():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 1], dtype=float)
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    est = itt_estimate(y, t)
    assert est["rate_treated"] == pytest.approx(0.5)
    assert est["rate_control"] == pytest.approx(0.25)
    assert est["ate"] == pytest.approx(0.25)
    assert est["ci_low"] < est["ate"] < est["ci_high"]


def test_srm_exact_split_passes():
    out = srm_check(8500, 1500, expected_treated_share=0.85)
    assert out["chi2"] == pytest.approx(0.0)
    assert out["p_value"] == pytest.approx(1.0)
    assert out["verdict"] == "no_srm_detected"


def test_srm_wrong_split_flagged():
    out = srm_check(5000, 5000, expected_treated_share=0.85)
    assert out["p_value"] < 0.001
    assert out["verdict"] == "srm_flagged"


def test_balance_randomised_data_is_balanced():
    rng = np.random.default_rng(7)
    n = 40_000
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=["f0", "f1", "f2", "f3"])
    df["treatment"] = (rng.random(n) < 0.85).astype(int)
    out = balance_check(df, ["f0", "f1", "f2", "f3"], seed=7)
    assert out["max_abs_smd"] < 0.05
    assert out["joint"]["df"] == 4
    assert out["joint"]["p_value"] > 0.001


def test_balance_detects_imbalance():
    rng = np.random.default_rng(7)
    n = 20_000
    t = (rng.random(n) < 0.5).astype(int)
    df = pd.DataFrame({"f0": rng.normal(size=n) + 0.8 * t, "treatment": t})
    out = balance_check(df, ["f0"], seed=7)
    assert out["max_abs_smd"] > 0.5
    assert out["joint"]["p_value"] < 0.001


def _exposure_frame():
    rng = np.random.default_rng(3)
    n_t, n_c = 8500, 1500
    exposure = np.concatenate([rng.binomial(1, 0.04, n_t), np.zeros(n_c, dtype=int)])
    t = np.concatenate([np.ones(n_t, dtype=int), np.zeros(n_c, dtype=int)])
    y = rng.binomial(1, 0.003, n_t + n_c)
    return pd.DataFrame({"treatment": t, "exposure": exposure, "conversion": y})


def test_exposure_structural_zero_warns():
    out = exposure_readout(_exposure_frame(), "conversion")
    assert out["available"] is True
    assert out["label"] == "descriptive_only_not_causal"
    assert out["structural_zero_in_control"] is True
    assert "structurally zero" in out["warning"]
    assert out["control_exposure_share"] == 0.0


def test_exposure_missing_column():
    df = pd.DataFrame({"treatment": [1, 0], "conversion": [1, 0]})
    out = exposure_readout(df, "conversion")
    assert out["available"] is False


def test_standard_readout_structure():
    rng = np.random.default_rng(11)
    n = 10_000
    df = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    df["treatment"] = (rng.random(n) < 0.85).astype(int)
    df["conversion"] = rng.binomial(1, 0.003 + 0.001 * df["treatment"])
    df["exposure"] = 0
    out = standard_readout(df, "conversion", ["f0", "f1"])
    for key in ("n_rows", "itt", "srm", "balance", "exposure"):
        assert key in out
