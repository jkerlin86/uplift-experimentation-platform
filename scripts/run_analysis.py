"""Run the full E4 analysis and write reports/results.json.

Pipeline: chunked CSV-to-parquet conversion; full-sample streaming
statistics (arm counts, outcome rates, exposure shares, per-feature
balance moments); a seeded analysis sample; one cross-fitting pass
producing out-of-fold mu0/mu1 and T-learner and X-learner uplift scores
for every sampled row; Qini, calibration, and doubly robust policy
values per estimator; a placebo gate (shuffled assignment); and a
synthetic recovery gate (known heterogeneous effect). Every number the
README and executive summary cite is written to reports/results.json
here and nowhere else.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uplift_platform.dataio import (
    convert_csv_to_parquet,
    draw_analysis_sample,
    iter_batches,
)
from uplift_platform.harness import (
    balance_check,
    exposure_readout,
    itt_estimate,
    srm_check,
)
from uplift_platform.learners import crossfit_uplift
from uplift_platform.metrics import (
    policy_table,
    qini_coefficient,
    qini_curve,
    uplift_calibration,
)
from uplift_platform.synthetic import simulate_rct


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def full_sample_stats(parquet: Path, features: list[str]) -> dict:
    """Stream the parquet once, accumulating everything the full-sample
    readouts need: arm counts, outcome sums, exposure sums, and per-arm
    feature sums and squared sums for standardised mean differences."""
    nf = len(features)
    counts = {0: 0, 1: 0}
    conv = {0: 0, 1: 0}
    visit = {0: 0, 1: 0}
    expo = {0: 0, 1: 0}
    fsum = {0: np.zeros(nf), 1: np.zeros(nf)}
    fsq = {0: np.zeros(nf), 1: np.zeros(nf)}
    for batch in iter_batches(parquet):
        frame = batch.to_pandas()
        t = frame["treatment"].to_numpy(dtype=int)
        for arm in (0, 1):
            sel = t == arm
            counts[arm] += int(sel.sum())
            conv[arm] += int(frame.loc[sel, "conversion"].sum())
            visit[arm] += int(frame.loc[sel, "visit"].sum())
            expo[arm] += int(frame.loc[sel, "exposure"].sum())
            x = frame.loc[sel, features].to_numpy(dtype=np.float64)
            fsum[arm] += x.sum(axis=0)
            fsq[arm] += (x**2).sum(axis=0)
    per_feature = []
    for j, name in enumerate(features):
        stats_arm = {}
        for arm in (0, 1):
            n = counts[arm]
            mean = fsum[arm][j] / n
            var = fsq[arm][j] / n - mean**2
            var = var * n / (n - 1)
            stats_arm[arm] = (mean, var)
        pooled = np.sqrt((stats_arm[1][1] + stats_arm[0][1]) / 2)
        smd = float((stats_arm[1][0] - stats_arm[0][0]) / pooled) if pooled > 0 else 0.0
        per_feature.append(
            {
                "feature": name,
                "mean_treated": float(stats_arm[1][0]),
                "mean_control": float(stats_arm[0][0]),
                "smd": smd,
                "abs_smd": abs(smd),
            }
        )
    n_total = counts[0] + counts[1]
    return {
        "n_rows": n_total,
        "n_treated": counts[1],
        "n_control": counts[0],
        "treated_share": counts[1] / n_total,
        "conversion": {
            "rate_treated": conv[1] / counts[1],
            "rate_control": conv[0] / counts[0],
        },
        "visit": {"rate_treated": visit[1] / counts[1], "rate_control": visit[0] / counts[0]},
        "exposure": {
            "treated_share_exposed": expo[1] / counts[1],
            "control_share_exposed": expo[0] / counts[0] if counts[0] else 0.0,
        },
        "per_feature_balance": per_feature,
    }


def main() -> None:
    started = time.time()
    config = json.loads((ROOT / "configs/default.json").read_text())
    dcfg, design, scfg = config["data"], config["design"], config["sample"]
    lcfg, ecfg = config["learners"], config["evaluation"]
    features = dcfg["feature_columns"]
    propensity = float(design["propensity"])
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    parquet = processed / dcfg["processed_file"]

    print("[1/7] converting CSV to parquet (chunked) ...", flush=True)
    n_rows = convert_csv_to_parquet(
        ROOT / "data/raw" / dcfg["raw_file"], parquet, int(dcfg["csv_chunk_rows"])
    )
    print(f"      parquet rows: {n_rows}", flush=True)

    print("[2/7] full-sample streaming statistics ...", flush=True)
    full = full_sample_stats(parquet, features)
    # Full-sample difference-in-means with the standard two-sample SE,
    # computed from the streamed rates and counts.
    def dim(outcome: str) -> dict:
        r = full[outcome]
        n1, n0 = full["n_treated"], full["n_control"]
        p1, p0 = r["rate_treated"], r["rate_control"]
        se = float(np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0))
        diff = float(p1 - p0)
        return {
            "rate_treated": float(p1),
            "rate_control": float(p0),
            "ate": diff,
            "se": se,
            "ci_low": diff - 1.959963984540054 * se,
            "ci_high": diff + 1.959963984540054 * se,
            "p_value": float(2 * stats.norm.sf(abs(diff / se))),
            "n_treated": n1,
            "n_control": n0,
        }

    srm = srm_check(full["n_treated"], full["n_control"], propensity, design["srm_alpha"])
    abs_smds = np.array([r["abs_smd"] for r in full["per_feature_balance"]])

    print("[3/7] drawing the seeded analysis sample ...", flush=True)
    sample = draw_analysis_sample(
        parquet,
        target_n=int(scfg["analysis_n"]),
        probability=float(scfg["sampling_probability"]),
        seed=int(scfg["seed"]),
        feature_columns=features,
    )
    sample.to_parquet(processed / "analysis_sample.parquet", index=False)
    print(f"      analysis sample rows: {len(sample)}", flush=True)

    joint = balance_check(sample, features, seed=int(scfg["seed"]))["joint"]
    exposure = exposure_readout(sample, dcfg["primary_outcome"])

    x = sample[features].to_numpy(dtype=np.float64)
    y = sample[dcfg["primary_outcome"]].to_numpy(dtype=float)
    t = sample[dcfg["treatment_column"]].to_numpy(dtype=int)

    print("[4/7] cross-fitting T-learner and X-learner ...", flush=True)
    params = {k: lcfg[k] for k in ("max_iter", "learning_rate", "max_leaf_nodes",
                                   "min_samples_leaf", "l2_regularization")}
    oof = crossfit_uplift(
        x, y, t, propensity=propensity, n_splits=int(lcfg["n_splits"]),
        seed=int(scfg["seed"]), params=params,
    )
    pd.DataFrame(
        {
            "fold": oof["fold"],
            "treatment": t,
            "conversion": y.astype(int),
            "visit": sample[dcfg["secondary_outcome"]].to_numpy(dtype=int),
            "mu0": oof["mu0"],
            "mu1": oof["mu1"],
            "tau_t": oof["tau_t"],
            "tau_x": oof["tau_x"],
        }
    ).to_parquet(processed / "oof_scores.parquet", index=False)

    estimators: dict = {}
    for key, score_key in (("t_learner", "tau_t"), ("x_learner", "tau_x")):
        score = oof[score_key]
        estimators[key] = {
            "qini": qini_curve(y, t, score, n_points=int(ecfg["qini_points"])),
            "calibration": uplift_calibration(
                y, t, score, n_bins=int(ecfg["calibration_bins"])
            ),
            "policy_values": policy_table(
                y, t, oof["mu0"], oof["mu1"], score,
                [float(q) for q in ecfg["policy_quantiles"]], propensity,
            ),
            "mean_predicted_uplift": float(score.mean()),
        }

    print("[5/7] placebo gate: shuffled assignment ...", flush=True)
    pcfg = config["placebo"]
    rng = np.random.default_rng(int(pcfg["seed"]))
    t_shuffled = rng.permutation(t)
    oof_placebo = crossfit_uplift(
        x, y, t_shuffled, propensity=propensity, n_splits=int(pcfg["n_splits"]),
        seed=int(pcfg["seed"]), params=params,
    )
    placebo = {
        "seed": int(pcfg["seed"]),
        "n_splits": int(pcfg["n_splits"]),
        "t_learner_qini_coefficient": qini_coefficient(y, t_shuffled, oof_placebo["tau_t"]),
        "x_learner_qini_coefficient": qini_coefficient(y, t_shuffled, oof_placebo["tau_x"]),
        "max_abs_qini_threshold": float(pcfg["max_abs_qini"]),
    }
    placebo["passed"] = bool(
        max(
            abs(placebo["t_learner_qini_coefficient"]),
            abs(placebo["x_learner_qini_coefficient"]),
        )
        < float(pcfg["max_abs_qini"])
    )

    print("[6/7] synthetic recovery gate ...", flush=True)
    gcfg = config["synthetic"]
    sim = simulate_rct(n=int(gcfg["n"]), seed=int(gcfg["seed"]), propensity=propensity)
    oof_syn = crossfit_uplift(
        sim["x"], sim["y"].astype(float), sim["t"], propensity=propensity,
        n_splits=int(gcfg["n_splits"]), seed=int(gcfg["seed"]), params=params,
    )
    true_ate = float(sim["tau"].mean())
    k_top = round(0.10 * len(sim["tau"]))
    true_top_mean = float(np.sort(sim["tau"])[-k_top:].mean())
    syn: dict = {"true_ate": true_ate, "n": int(gcfg["n"]), "estimators": {}}
    for key, score in (("t_learner", oof_syn["tau_t"]), ("x_learner", oof_syn["tau_x"])):
        top_idx = np.argsort(-score, kind="stable")[:k_top]
        capture = float(sim["tau"][top_idx].mean() / true_top_mean)
        spearman = float(stats.spearmanr(score, sim["tau"]).statistic)
        syn["estimators"][key] = {
            "ate_estimate": float(score.mean()),
            "ate_abs_error": float(abs(score.mean() - true_ate)),
            "top_decile_true_tau_capture_ratio": capture,
            "spearman_vs_true_tau": spearman,
        }
    syn["difference_in_means"] = itt_estimate(sim["y"].astype(float), sim["t"])
    tol = gcfg["tolerances"]
    syn["tolerances"] = tol
    syn["passed"] = bool(
        all(
            e["ate_abs_error"] <= tol["ate_abs_error_max"]
            and e["top_decile_true_tau_capture_ratio"] >= tol["top_decile_capture_ratio_min"]
            and e["spearman_vs_true_tau"] >= tol["spearman_min"]
            for e in syn["estimators"].values()
        )
        and abs(syn["difference_in_means"]["ate"] - true_ate) <= tol["ate_abs_error_max"]
    )

    print("[7/7] writing reports/results.json ...", flush=True)
    results = {
        "project": "E4 uplift experimentation platform (portfolio Project 7)",
        "dataset": {
            "name": "Criteo Uplift Prediction Dataset v2 (file version 2.1)",
            "licence": dcfg["licence"],
            "sha256": dcfg["sha256"],
            "bytes": dcfg["bytes"],
            "limitation": (
                "The released data are a non-uniform subsample, so estimated levels "
                "are NOT Criteo's true incrementality; all effects are within-sample "
                "estimates."
            ),
            **full,
        },
        "design": {
            "propensity": propensity,
            "propensity_source": design["propensity_source"],
            "primary_outcome": dcfg["primary_outcome"],
            "secondary_outcome": dcfg["secondary_outcome"],
        },
        "ate": {
            "conversion": dim("conversion"),
            "visit": dim("visit"),
        },
        "balance": {
            "per_feature": full["per_feature_balance"],
            "max_abs_smd": float(abs_smds.max()),
            "mean_abs_smd": float(abs_smds.mean()),
            "basis": "per-feature SMDs on all rows (streamed moments); joint test on the analysis sample",
            "joint": joint,
        },
        "srm": srm,
        "exposure": exposure,
        "analysis_sample": {
            "n": len(sample),
            "seed": int(scfg["seed"]),
            "sampling_probability": float(scfg["sampling_probability"]),
            "n_splits_crossfit": int(lcfg["n_splits"]),
            "base_learner": "HistGradientBoosting (classifier for outcome models, regressor for X-learner effect models)",
            "learner_params": params,
        },
        "estimators": estimators,
        "placebo_gate": placebo,
        "synthetic_gate": syn,
        "runtime_seconds": time.time() - started,
    }
    out = ROOT / "reports/results.json"
    out.write_text(json.dumps(_clean(results), indent=2))
    print(f"      wrote {out} in {results['runtime_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
