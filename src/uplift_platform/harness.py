"""Reusable experiment-readout harness.

Functions here take a dataframe with a treatment column, an outcome
column, optional feature columns, and an optional exposure column, and
return plain dictionaries that can be serialised into a results file.
They are written for randomised experiments with a known, constant
assignment propensity (the Criteo v2 design uses 0.85), but the
propensity is a parameter throughout, never hard-coded inside an
estimator.

Exposure policy: assignment is the primary contrast (intention to
treat). When an exposure column is supplied, the harness additionally
returns a clearly labelled per-exposure descriptive estimate. Exposure
in the Criteo v2 data is structurally zero for every control row, so a
per-exposure contrast can never be a randomised comparison there; the
harness detects that case and attaches an explicit warning instead of
letting the number pass as causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

Z_95 = 1.959963984540054


def itt_estimate(y: np.ndarray, t: np.ndarray) -> dict:
    """Difference-in-means (intention-to-treat) estimate with SE and CI."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    y1, y0 = y[t == 1], y[t == 0]
    rate_t, rate_c = float(y1.mean()), float(y0.mean())
    diff = rate_t - rate_c
    se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    return {
        "n_treated": len(y1),
        "n_control": len(y0),
        "rate_treated": rate_t,
        "rate_control": rate_c,
        "ate": float(diff),
        "se": se,
        "ci_low": float(diff - Z_95 * se),
        "ci_high": float(diff + Z_95 * se),
        "z": float(diff / se) if se > 0 else 0.0,
        "p_value": float(2 * stats.norm.sf(abs(diff / se))) if se > 0 else 1.0,
    }


def standardised_mean_differences(
    x: np.ndarray, t: np.ndarray, feature_names: list[str]
) -> list[dict]:
    """Per-feature standardised mean differences between arms."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=int)
    out = []
    for j, name in enumerate(feature_names):
        a, b = x[t == 1, j], x[t == 0, j]
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd = float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0
        out.append(
            {
                "feature": name,
                "mean_treated": float(a.mean()),
                "mean_control": float(b.mean()),
                "sd_pooled": float(pooled),
                "smd": smd,
                "abs_smd": abs(smd),
            }
        )
    return out


def _joint_balance_test(x: np.ndarray, t: np.ndarray, seed: int, max_rows: int) -> dict:
    """Joint balance check: likelihood-ratio test of a logistic model of
    treatment on all features against an intercept-only model. Under
    randomisation the features jointly carry no signal about assignment,
    so the statistic is compared with a chi-square distribution with
    ``n_features`` degrees of freedom. Large samples are subsampled with
    a fixed seed so the test stays a diagnostic, not a significance
    machine driven by n."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    n = len(t)
    if n > max_rows:
        idx = np.sort(rng.choice(n, size=max_rows, replace=False))
        xs, ts = x[idx], t[idx]
    else:
        xs, ts = x, t
    mu, sd = xs.mean(axis=0), xs.std(axis=0)
    sd[sd == 0] = 1.0
    z = (xs - mu) / sd
    model = LogisticRegression(C=1000.0, max_iter=1000, solver="lbfgs")
    model.fit(z, ts)
    p = np.clip(model.predict_proba(z)[:, 1], 1e-12, 1 - 1e-12)
    ll_full = float(np.sum(ts * np.log(p) + (1 - ts) * np.log(1 - p)))
    p0 = float(ts.mean())
    ll_null = float(np.sum(ts * np.log(p0) + (1 - ts) * np.log(1 - p0)))
    lr = max(0.0, 2 * (ll_full - ll_null))
    df = int(x.shape[1])
    return {
        "method": "logistic likelihood-ratio test of assignment on all features",
        "n_used": len(ts),
        "lr_statistic": float(lr),
        "df": df,
        "p_value": float(stats.chi2.sf(lr, df)),
    }


def balance_check(
    df: pd.DataFrame,
    feature_columns: list[str],
    treatment_column: str = "treatment",
    seed: int = 20261111,
    joint_max_rows: int = 200_000,
) -> dict:
    """Randomisation balance: per-feature SMDs plus a joint check."""
    x = df[feature_columns].to_numpy(dtype=float)
    t = df[treatment_column].to_numpy(dtype=int)
    per_feature = standardised_mean_differences(x, t, feature_columns)
    abs_smds = np.array([r["abs_smd"] for r in per_feature])
    return {
        "per_feature": per_feature,
        "max_abs_smd": float(abs_smds.max()),
        "mean_abs_smd": float(abs_smds.mean()),
        "n_features_over_0_1": int((abs_smds > 0.1).sum()),
        "joint": _joint_balance_test(x, t, seed=seed, max_rows=joint_max_rows),
    }


def srm_check(
    n_treated: int,
    n_control: int,
    expected_treated_share: float = 0.85,
    alpha: float = 0.001,
) -> dict:
    """Sample ratio mismatch check against the designed split.

    Chi-square goodness of fit of the observed arm counts against the
    designed assignment share. The conventional SRM flag threshold is
    p < 0.001; the verdict field uses that threshold.
    """
    n = n_treated + n_control
    exp_t = n * expected_treated_share
    exp_c = n * (1 - expected_treated_share)
    chi2 = float((n_treated - exp_t) ** 2 / exp_t + (n_control - exp_c) ** 2 / exp_c)
    p_value = float(stats.chi2.sf(chi2, 1))
    return {
        "n_treated": int(n_treated),
        "n_control": int(n_control),
        "observed_treated_share": float(n_treated / n),
        "expected_treated_share": float(expected_treated_share),
        "chi2": chi2,
        "p_value": p_value,
        "alpha": alpha,
        "verdict": "srm_flagged" if p_value < alpha else "no_srm_detected",
    }


def exposure_readout(
    df: pd.DataFrame,
    outcome_column: str,
    treatment_column: str = "treatment",
    exposure_column: str = "exposure",
) -> dict:
    """Per-exposure descriptive estimate, with the structural warning.

    This is never a causal estimate: exposure is post-assignment. When
    exposure is structurally zero in the control arm (as in Criteo v2),
    no like-for-like exposed control group exists, and the returned
    dictionary says so in the ``warning`` field. The contrast reported
    is exposed-treated versus all control, labelled descriptive.
    """
    if exposure_column not in df.columns:
        return {"available": False, "reason": f"no '{exposure_column}' column in dataframe"}
    t = df[treatment_column].to_numpy(dtype=int)
    e = df[exposure_column].to_numpy(dtype=int)
    y = df[outcome_column].to_numpy(dtype=float)
    control_exposure_share = float(e[t == 0].mean()) if (t == 0).any() else 0.0
    treated_exposure_share = float(e[t == 1].mean()) if (t == 1).any() else 0.0
    structural_zero = bool(control_exposure_share == 0.0 and treated_exposure_share > 0.0)
    exposed = (t == 1) & (e == 1)
    control = t == 0
    if exposed.sum() == 0 or control.sum() == 0:
        return {
            "available": False,
            "reason": "no exposed treated rows or no control rows",
            "treated_exposure_share": treated_exposure_share,
            "control_exposure_share": control_exposure_share,
        }
    diff = float(y[exposed].mean() - y[control].mean())
    se = float(
        np.sqrt(
            y[exposed].var(ddof=1) / exposed.sum() + y[control].var(ddof=1) / control.sum()
        )
    )
    warning = (
        "Exposure is structurally zero in the control arm: exposed users exist "
        "only among the treated, so no randomised exposed-versus-unexposed "
        "contrast exists. This figure is descriptive only and must not be "
        "read as the effect of exposure."
        if structural_zero
        else "Exposure is post-assignment; this contrast is descriptive only "
        "and must not be read as the effect of exposure."
    )
    return {
        "available": True,
        "label": "descriptive_only_not_causal",
        "treated_exposure_share": treated_exposure_share,
        "control_exposure_share": control_exposure_share,
        "structural_zero_in_control": structural_zero,
        "warning": warning,
        "n_exposed_treated": int(exposed.sum()),
        "rate_exposed_treated": float(y[exposed].mean()),
        "rate_control": float(y[control].mean()),
        "descriptive_difference": diff,
        "se": se,
        "ci_low": float(diff - Z_95 * se),
        "ci_high": float(diff + Z_95 * se),
    }


def standard_readout(
    df: pd.DataFrame,
    outcome_column: str,
    feature_columns: list[str] | None = None,
    treatment_column: str = "treatment",
    exposure_column: str = "exposure",
    expected_treated_share: float = 0.85,
    seed: int = 20261111,
) -> dict:
    """One structured readout: ITT, SRM, balance (if features), exposure."""
    t = df[treatment_column].to_numpy(dtype=int)
    y = df[outcome_column].to_numpy(dtype=float)
    out: dict = {
        "n_rows": len(df),
        "outcome": outcome_column,
        "itt": itt_estimate(y, t),
        "srm": srm_check(int((t == 1).sum()), int((t == 0).sum()), expected_treated_share),
    }
    if feature_columns:
        out["balance"] = balance_check(df, feature_columns, treatment_column, seed=seed)
    out["exposure"] = exposure_readout(df, outcome_column, treatment_column, exposure_column)
    return out
