"""Synthetic RCT with a known, nonlinear heterogeneous treatment effect.

Used by the recovery gate: the true effect function tau(x) is known for
every row, so the gate can check that the pipeline recovers the ATE,
ranks rows by true effect, and captures the true top decile. The design
mirrors the real data: binary outcome, constant propensity 0.85, and a
rare outcome.
"""

from __future__ import annotations

import numpy as np


def true_tau(x: np.ndarray) -> np.ndarray:
    """Known heterogeneous effect: smooth, nonlinear, in a few features."""
    z = 1.3 * x[:, 0] - 1.0 * x[:, 1] + 0.8 * x[:, 0] * x[:, 2]
    gate = 1.0 / (1.0 + np.exp(-z))
    step = np.where(x[:, 3] > -0.5, 1.0, 0.35)
    return 0.002 + 0.045 * gate * step


def baseline_risk(x: np.ndarray) -> np.ndarray:
    z = -4.4 + 0.4 * x[:, 4] - 0.3 * x[:, 5] + 0.2 * x[:, 1]
    return 1.0 / (1.0 + np.exp(-z))


def simulate_rct(n: int = 300_000, seed: int = 20261113, propensity: float = 0.85) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8)).astype(np.float64)
    tau = true_tau(x)
    mu0 = baseline_risk(x)
    t = (rng.random(n) < propensity).astype(int)
    p = mu0 + t * tau
    y = (rng.random(n) < p).astype(int)
    return {"x": x, "y": y, "t": t, "tau": tau, "mu0": mu0, "propensity": propensity}
