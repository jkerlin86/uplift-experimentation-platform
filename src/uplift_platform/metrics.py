"""Uplift evaluation metrics: Qini curves and coefficients, decile
calibration, and doubly robust policy values.

Qini uses the randomisation-corrected form for unequal arm sizes
(Radcliffe): for the top-ranked set S,

    Q(S) = Y_t(S) - Y_c(S) * (N_t(S) / N_c(S)),

where Y and N are conversion counts and row counts inside S, split by
arm. The random-targeting reference is the straight line from 0 to
Q(full sample). The perfect reference orders rows by their realised
contribution (treated converters first, control converters last). The
Qini coefficient reported is the area between the model curve and the
random line divided by the area between the perfect curve and the
random line, so 0 is random targeting and 1 is the realised oracle.
"""

from __future__ import annotations

import numpy as np

Z_95 = 1.959963984540054


def _qini_values(y: np.ndarray, t: np.ndarray, order: np.ndarray, ks: np.ndarray) -> np.ndarray:
    ys, ts = y[order], t[order]
    cum_y_t = np.cumsum(ys * ts)
    cum_y_c = np.cumsum(ys * (1 - ts))
    cum_n_t = np.cumsum(ts)
    cum_n_c = np.cumsum(1 - ts)
    idx = ks - 1
    y_t, y_c = cum_y_t[idx], cum_y_c[idx]
    n_t, n_c = cum_n_t[idx], cum_n_c[idx]
    ratio = np.where(n_c > 0, n_t / np.maximum(n_c, 1), 0.0)
    return y_t - y_c * ratio


def qini_curve(
    y: np.ndarray, t: np.ndarray, score: np.ndarray, n_points: int = 51
) -> dict:
    """Qini curve for a score, with random and perfect references."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    score = np.asarray(score, dtype=float)
    n = len(y)
    if n_points >= n:
        ks = np.arange(1, n + 1)
    else:
        ks = np.unique(np.linspace(1, n, n_points).astype(int))
    order = np.argsort(-score, kind="stable")
    q = _qini_values(y, t, order, ks)
    total = float(q[-1])
    phi = ks / n
    random = phi * total
    oracle_key = np.where(t == 1, np.where(y == 1, 3.0, 2.0), np.where(y == 1, 0.0, 1.0))
    oracle_order = np.argsort(-oracle_key, kind="stable")
    perfect = _qini_values(y, t, oracle_order, ks)
    phi_full = np.concatenate([[0.0], phi])
    area_model = float(np.trapezoid(np.concatenate([[0.0], q]), phi_full))
    area_random = float(np.trapezoid(np.concatenate([[0.0], random]), phi_full))
    area_perfect = float(np.trapezoid(np.concatenate([[0.0], perfect]), phi_full))
    denom = area_perfect - area_random
    coefficient = float((area_model - area_random) / denom) if denom != 0 else 0.0
    return {
        "phi": phi.tolist(),
        "qini": q.tolist(),
        "random": random.tolist(),
        "perfect": perfect.tolist(),
        "total_incremental_conversions": total,
        "qini_coefficient": coefficient,
        "area_model": area_model,
        "area_random": area_random,
        "area_perfect": area_perfect,
        "normalisation": (
            "coefficient = (area under model curve - area under random line) / "
            "(area under perfect curve - area under random line); curve values are "
            "randomisation-corrected incremental conversions (Radcliffe form)"
        ),
    }


def qini_coefficient(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    return qini_curve(y, t, score)["qini_coefficient"]


def uplift_calibration(
    y: np.ndarray, t: np.ndarray, score: np.ndarray, n_bins: int = 10
) -> list[dict]:
    """Observed versus predicted uplift by predicted-uplift decile.

    Deciles are equal-count bins of the predicted uplift score, highest
    first. The observed uplift in a bin is the treated-minus-control
    conversion rate inside the bin; the scaled column multiplies that
    rate by the bin size to express it as incremental conversions.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="stable")
    bins = np.array_split(order, n_bins)
    out = []
    for rank, idx in enumerate(bins, start=1):
        yb, tb = y[idx], t[idx]
        rate_t = float(yb[tb == 1].mean()) if (tb == 1).any() else 0.0
        rate_c = float(yb[tb == 0].mean()) if (tb == 0).any() else 0.0
        observed = rate_t - rate_c
        out.append(
            {
                "decile": rank,
                "n": len(idx),
                "n_treated": int((tb == 1).sum()),
                "n_control": int((tb == 0).sum()),
                "rate_treated": rate_t,
                "rate_control": rate_c,
                "observed_uplift": float(observed),
                "mean_predicted_uplift": float(score[idx].mean()),
                "incremental_conversions_in_bin": float(observed * len(idx)),
            }
        )
    return out


def dr_policy_value(
    y: np.ndarray,
    t: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    treat_mask: np.ndarray,
    propensity: float = 0.85,
) -> dict:
    """Doubly robust value of a deterministic treatment policy.

    The policy treats exactly the rows where ``treat_mask`` is true.
    With cross-fitted outcome models mu0/mu1 and the known constant
    propensity, the per-row doubly robust contribution is

        mu_pi(x) + 1{t = pi(x)} / P(t = pi(x)) * (y - mu_t(x)),

    whose mean is an unbiased estimate of the policy's mean outcome.
    The standard error is the influence-function standard error of that
    mean. Units are conversions per person.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    mu0 = np.asarray(mu0, dtype=float)
    mu1 = np.asarray(mu1, dtype=float)
    pi = np.asarray(treat_mask, dtype=bool)
    mu_pi = np.where(pi, mu1, mu0)
    mu_t = np.where(t == 1, mu1, mu0)
    p_assign = np.where(t == 1, propensity, 1.0 - propensity)
    matches = (t == 1) == pi
    contrib = mu_pi + np.where(matches, (y - mu_t) / p_assign, 0.0)
    value = float(contrib.mean())
    se = float(contrib.std(ddof=1) / np.sqrt(len(y)))
    return {
        "value": value,
        "se": se,
        "ci_low": float(value - Z_95 * se),
        "ci_high": float(value + Z_95 * se),
        "treated_share": float(pi.mean()),
    }


def policy_table(
    y: np.ndarray,
    t: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    score: np.ndarray,
    quantiles: list[float],
    propensity: float = 0.85,
) -> dict:
    """Policy values for treat-top-q policies plus the two benchmarks."""
    n = len(y)
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    out: dict = {}
    for q in quantiles:
        k = round(q * n)
        mask = np.zeros(n, dtype=bool)
        mask[order[:k]] = True
        out[f"top_{round(100 * q)}pct"] = dr_policy_value(
            y, t, mu0, mu1, mask, propensity
        )
    out["treat_all"] = dr_policy_value(
        y, t, mu0, mu1, np.ones(n, dtype=bool), propensity
    )
    out["treat_none"] = dr_policy_value(
        y, t, mu0, mu1, np.zeros(n, dtype=bool), propensity
    )
    base = out["treat_none"]["value"]
    for v in out.values():
        v["incremental_vs_treat_none"] = float(v["value"] - base)
    return out
