"""Cross-fitted uplift learners: T-learner and X-learner.

One cross-fitting pass fits the shared stage-1 outcome models (one per
arm) on each training fold and produces, for every held-out row,
out-of-fold predictions mu0, mu1, the T-learner uplift (mu1 - mu0), and
the X-learner uplift. The X-learner's second stage regresses
imputed treatment effects on features within each arm of the training
fold and combines the two arm-specific effect models with the known
constant propensity, following Kuenzel et al. (2019): with
g = P(treatment), tau_x(x) = g * tau_0(x) + (1 - g) * tau_1(x), where
tau_1 is fitted on treated rows and tau_0 on control rows. All
weighting uses the known design propensity (0.85 for Criteo v2), not a
fitted propensity model.

The S-learner is deliberately not implemented; see
docs/coding_decisions.md.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

DEFAULT_PARAMS = {
    "max_iter": 150,
    "learning_rate": 0.06,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 100,
    "l2_regularization": 1.0,
}


def _clf(params: dict, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **params)


def _reg(params: dict, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(early_stopping=False, random_state=seed, **params)


def crossfit_uplift(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    propensity: float = 0.85,
    n_splits: int = 5,
    seed: int = 20261111,
    params: dict | None = None,
) -> dict:
    """Out-of-fold mu0, mu1, T-learner and X-learner uplift scores."""
    params = dict(DEFAULT_PARAMS if params is None else params)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    n = len(y)
    mu0 = np.full(n, np.nan)
    mu1 = np.full(n, np.nan)
    tau_x = np.full(n, np.nan)
    fold_id = np.zeros(n, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(skf.split(x, t)):
        fold_id[test] = fold
        tr_t = train[t[train] == 1]
        tr_c = train[t[train] == 0]
        m1 = _clf(params, seed + fold).fit(x[tr_t], y[tr_t])
        m0 = _clf(params, seed + 100 + fold).fit(x[tr_c], y[tr_c])
        mu1[test] = m1.predict_proba(x[test])[:, 1]
        mu0[test] = m0.predict_proba(x[test])[:, 1]
        # X-learner stage 2 on the training fold, using this fold's
        # stage-1 models for the imputed effects.
        d1 = y[tr_t] - m0.predict_proba(x[tr_t])[:, 1]
        d0 = m1.predict_proba(x[tr_c])[:, 1] - y[tr_c]
        tau1_model = _reg(params, seed + 200 + fold).fit(x[tr_t], d1)
        tau0_model = _reg(params, seed + 300 + fold).fit(x[tr_c], d0)
        tau_x[test] = propensity * tau0_model.predict(x[test]) + (
            1.0 - propensity
        ) * tau1_model.predict(x[test])
    return {
        "mu0": mu0,
        "mu1": mu1,
        "tau_t": mu1 - mu0,
        "tau_x": tau_x,
        "fold": fold_id,
    }
