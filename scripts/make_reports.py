"""Render README.md and reports/executive_summary.md from
reports/results.json. No number in either document is hand-typed: every
statistic is formatted from the results file written by
scripts/run_analysis.py.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "reports/results.json").read_text())

D = R["dataset"]
ATE_C = R["ate"]["conversion"]
ATE_V = R["ate"]["visit"]
BAL = R["balance"]
SRM = R["srm"]
EXP = R["exposure"]
T = R["estimators"]["t_learner"]
X = R["estimators"]["x_learner"]
PL = R["placebo_gate"]
SYN = R["synthetic_gate"]
SMP = R["analysis_sample"]


def pp(x, nd=4):
    """Probability as percentage points with nd decimals."""
    return f"{100 * x:.{nd}f}"


def f(x, nd=4):
    return f"{x:.{nd}f}"


def policy_rows(est) -> str:
    pv = est["policy_values"]
    rows = []
    for key, label in (
        ("top_10pct", "Treat top 10% by predicted uplift"),
        ("top_20pct", "Treat top 20%"),
        ("top_30pct", "Treat top 30%"),
        ("top_50pct", "Treat top 50%"),
        ("treat_all", "Treat all"),
        ("treat_none", "Treat none"),
    ):
        v = pv[key]
        rows.append(
            f"| {label} | {pp(v['value'])} | [{pp(v['ci_low'])}, {pp(v['ci_high'])}] | "
            f"{pp(v['incremental_vs_treat_none'])} |"
        )
    return "\n".join(rows)


def calibration_rows(est) -> str:
    rows = []
    for c in est["calibration"]:
        rows.append(
            f"| {c['decile']} | {c['n']} | {pp(c['mean_predicted_uplift'])} | "
            f"{pp(c['observed_uplift'])} | {f(c['incremental_conversions_in_bin'], 1)} |"
        )
    return "\n".join(rows)


def balance_rows() -> str:
    rows = []
    for b in sorted(BAL["per_feature"], key=lambda r: -r["abs_smd"]):
        rows.append(f"| {b['feature']} | {f(b['smd'], 4)} |")
    return "\n".join(rows)


def best_policy_line(est, label) -> str:
    pv = est["policy_values"]
    all_gain = pv["treat_all"]["incremental_vs_treat_none"]
    top10 = pv["top_10pct"]["incremental_vs_treat_none"]
    top50 = pv["top_50pct"]["incremental_vs_treat_none"]
    captured10 = 100 * top10 / all_gain if all_gain > 0 else 0.0
    captured50 = 100 * top50 / all_gain if all_gain > 0 else 0.0
    return (
        f"For the {label}, treating only the top 10% by predicted uplift "
        f"captures {captured10:.1f}% of the treat-all incremental gain "
        f"({pp(top10)} of {pp(all_gain)} pp) while treating a tenth as "
        f"many people, and the top 50% captures {captured50:.1f}%. "
        f"Treat-all still maximises the total; targeting is what pays "
        f"when treatment itself has a cost."
    )


README = f"""# An Experimentation Platform for Uplift

## Who should be treated, and what is targeting worth?

Most A/B readouts answer one question: did treatment help on average?
Targeting needs a harder one answered honestly: who gains, who does
not, and what is a targeting rule worth compared with treating
everyone or no one? This project builds that readout end to end on the
Criteo Uplift Prediction Dataset v2: a difference-in-means benchmark,
randomisation diagnostics, cross-fitted T-learner and X-learner uplift
models, Qini and calibration evaluation, and doubly robust policy
values for treat-the-top-q rules, all behind two validation gates (a
placebo shuffle and a synthetic recovery test) and packaged with a
reusable experiment-readout harness.

## The limitation that governs every number here

**The released Criteo data are a non-uniform subsample, so the
estimated levels are NOT Criteo's true incrementality.** Criteo states
this in its own documentation: the subsampling was done for privacy,
and the original incrementality level cannot be deduced from the
released data. Every effect in this repository is a within-sample
estimate on the released subsample. The methods, rankings, and policy
comparisons are the contribution; the levels are not a measurement of
any real campaign.

## Data

- Source: Criteo AI Lab, distributed through its official Hugging Face
  organisation. File `criteo-research-uplift-v2.1.csv.gz`
  ({D['bytes']:,} bytes, SHA-256 `{D['sha256']}`), verified against the
  hash published on the file's own page.
- **Data licence: CC BY-NC-SA 4.0 (Criteo AI Lab).** Non-commercial use
  only; adaptations shared under the same licence. The data are never
  committed to this repository; `scripts/download_data.py` reproduces
  and verifies the file. The code in this repository is MIT-licensed
  (see LICENSE); that licence covers the code only, never the data.
- Size: {D['n_rows']:,} rows, 12 anonymised features (f0 to f11, no
  missing values), treatment assigned to {100 * D['treated_share']:.2f}%
  of rows.
- Design: the v2 release sub-sampled every constituent incrementality
  test to the same global treatment ratio, so assignment is a constant
  propensity of 0.85 for every row (Diemert et al. 2021, Section 3,
  Table 1). Every weighting in this project uses that known constant,
  not a fitted propensity model.
- Outcomes: conversion (primary) and visit (secondary). Exposure is an
  attribute of the treated arm only: {100 * D['exposure']['treated_share_exposed']:.2f}%
  of treated rows have exposure 1, and exposure is 0 for every control
  row, so the primary analysis is intention-to-treat on assignment and
  the harness's per-exposure figure is descriptive only.

## Headline findings

- **The benchmark effect is small and precisely estimated.**
  Conversion is {pp(ATE_C['rate_treated'])}% under treatment against
  {pp(ATE_C['rate_control'])}% under control: an ATE of
  {pp(ATE_C['ate'])} percentage points (95% CI [{pp(ATE_C['ci_low'])},
  {pp(ATE_C['ci_high'])}], standard error {pp(ATE_C['se'])}). The
  secondary visit outcome gives an ATE of {pp(ATE_V['ate'])} percentage
  points (95% CI [{pp(ATE_V['ci_low'])}, {pp(ATE_V['ci_high'])}]).
- **Randomisation is close, with small but detectable imbalances.**
  The largest per-feature standardised mean difference is
  {f(BAL['max_abs_smd'])} (mean {f(BAL['mean_abs_smd'])}), well under
  the conventional 0.1 threshold on every feature; at these sample
  sizes the joint logistic likelihood-ratio check still rejects
  (p = {BAL['joint']['p_value']:.3g}) because even very small
  differences are detectable, which is consistent with the released
  data pooling many constituent incrementality tests. The SRM check
  against the designed 85/15 split gives chi-square
  {f(SRM['chi2'], 2)} and p = {SRM['p_value']:.3g}:
  {SRM['verdict'].replace('_', ' ')}.
- **Targeting signal is real but modest.** Cross-fitted Qini
  coefficients are {f(T['qini']['qini_coefficient'], 3)} for the
  T-learner and {f(X['qini']['qini_coefficient'], 3)} for the
  X-learner, against a placebo gate in which shuffled assignment
  collapses both coefficients to {f(PL['t_learner_qini_coefficient'], 3)}
  and {f(PL['x_learner_qini_coefficient'], 3)}.
- **Targeting beats blanket treatment per person treated.**
  {best_policy_line(X, 'X-learner')}
  {best_policy_line(T, 'T-learner')}

## Policy values (doubly robust, cross-fitted)

Values are expected conversions per person under each policy, with
95% confidence intervals from the influence-function standard error.
The outcome models and the rankings are out-of-fold; the propensity is
the known constant 0.85.

### X-learner ranking

| Policy | Value (% conv. per person) | 95% CI | Incremental vs treat none (pp) |
|---|---|---|---|
{policy_rows(X)}

### T-learner ranking

| Policy | Value (% conv. per person) | 95% CI | Incremental vs treat none (pp) |
|---|---|---|---|
{policy_rows(T)}

## Uplift calibration by predicted decile (X-learner)

Observed uplift is the treated-minus-control conversion rate inside
the decile; the scaled column expresses it as incremental conversions
within the decile. Read honestly, the ranking orders the deciles but
the predicted spread overstates the observed spread: the X-learner's
top decile is predicted at
{pp(X['calibration'][0]['mean_predicted_uplift'])} pp and observed at
{pp(X['calibration'][0]['observed_uplift'])} pp, and its bottom decile
is predicted at {pp(X['calibration'][-1]['mean_predicted_uplift'])} pp
while the observed value is
{pp(X['calibration'][-1]['observed_uplift'])} pp. The scores are a
ranking device, not calibrated effect sizes at the individual level.

| Decile (1 = highest predicted) | n | Mean predicted uplift (pp) | Observed uplift (pp) | Incremental conversions |
|---|---|---|---|---|
{calibration_rows(X)}

## Balance detail (all rows)

| Feature | Standardised mean difference |
|---|---|
{balance_rows()}

## Validation gates

- **Placebo gate.** With treatment labels shuffled (seed
  {PL['seed']}), the pipeline's scoring path returns Qini coefficients
  of {f(PL['t_learner_qini_coefficient'], 3)} (T-learner) and
  {f(PL['x_learner_qini_coefficient'], 3)} (X-learner), against a pass
  threshold of |Qini| < {f(PL['max_abs_qini_threshold'], 2)}: gate
  {'passed' if PL['passed'] else 'FAILED'}.
- **Synthetic recovery gate.** On a simulated RCT with a known
  nonlinear effect function (true ATE {pp(SYN['true_ate'])} pp), the
  difference-in-means estimate is {pp(SYN['difference_in_means']['ate'])}
  pp, the T-learner recovers the ATE with absolute error
  {pp(SYN['estimators']['t_learner']['ate_abs_error'])} pp and the
  X-learner {pp(SYN['estimators']['x_learner']['ate_abs_error'])} pp
  (tolerance {pp(SYN['tolerances']['ate_abs_error_max'])} pp); each
  estimator's top decile captures
  {100 * SYN['estimators']['t_learner']['top_decile_true_tau_capture_ratio']:.1f}%
  (T-learner) and {100 * SYN['estimators']['x_learner']['top_decile_true_tau_capture_ratio']:.1f}%
  (X-learner) of the true top-decile effect, with Spearman correlations
  against the true effect of {f(SYN['estimators']['t_learner']['spearman_vs_true_tau'], 3)}
  and {f(SYN['estimators']['x_learner']['spearman_vs_true_tau'], 3)}:
  gate {'passed' if SYN['passed'] else 'FAILED'}.

## The reusable harness

`src/uplift_platform/harness.py` packages the readout for any
randomised experiment with the same shape: `itt_estimate`,
`srm_check`, `balance_check` (per-feature standardised mean
differences plus a joint check), `exposure_readout` (a descriptive
per-exposure contrast that detects and warns when exposure is
structurally zero in the control arm, as here), and
`standard_readout`, which returns all of them as one structured
dictionary. Every harness function is unit-tested.

## How the analysis is built

- Learners are fitted on a seeded analysis sample of
  {SMP['n']:,} rows drawn from all {D['n_rows']:,} rows; ATE, balance,
  and SRM statistics use all rows. Sampling decisions are documented
  in docs/coding_decisions.md.
- One {SMP['n_splits_crossfit']}-fold cross-fitting pass produces
  out-of-fold outcome models and both uplift scores for every sampled
  row, so Qini, calibration, and policy values never grade a model on
  rows it was trained on. The base learner is HistGradientBoosting.
- Full methods: docs/methods.md. Decisions and trade-offs:
  docs/coding_decisions.md.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,notebooks]"
bash scripts/run_all.sh
```

`run_all.sh` verifies the data file (downloading it from Criteo's
official channel only if absent), converts it to parquet in chunks,
runs the analysis, renders figures and reports from
`reports/results.json`, executes both notebooks, and runs the test
suite and linter. The full pipeline ran in
{R['runtime_seconds'] / 60:.1f} minutes on the build machine.

## Repository layout

- `src/uplift_platform/`: harness, metrics, learners, synthetic gate, data IO.
- `scripts/`: download/verify, analysis, figures, notebooks, reports, run_all.
- `notebooks/`: 01 exploration and balance; 02 uplift modelling and policy value (both executed).
- `reports/`: results.json (single source of every reported number) and executive_summary.md.
- `figures/`: Qini curves, calibration, policy value, rendered from results.json.
- `docs/`: methods.md and coding_decisions.md.
- `tests/`: harness, metrics (including a hand-checked Qini example), and gate tests.
"""

SUMMARY = f"""# Executive summary: An Experimentation Platform for Uplift

## Question

On the Criteo Uplift Prediction Dataset v2, who should be treated, and
what is targeting worth against treating everyone or no one?

## The governing limitation

The released data are a non-uniform subsample, so estimated levels are
NOT Criteo's true incrementality; all effects below are within-sample
estimates on {D['n_rows']:,} released rows. Data licence: CC BY-NC-SA
4.0 (Criteo AI Lab); the data are not redistributed with this project.

## Findings

- Benchmark ATE on conversion: {pp(ATE_C['ate'])} percentage points
  (treated {pp(ATE_C['rate_treated'])}%, control
  {pp(ATE_C['rate_control'])}%; 95% CI [{pp(ATE_C['ci_low'])},
  {pp(ATE_C['ci_high'])}]). Visit ATE: {pp(ATE_V['ate'])} pp
  ([{pp(ATE_V['ci_low'])}, {pp(ATE_V['ci_high'])}]).
- Randomisation is close: max |SMD| {f(BAL['max_abs_smd'])} across the
  12 features (mean {f(BAL['mean_abs_smd'])}), under the 0.1 threshold
  everywhere, though the joint check rejects at these sample sizes
  (p = {BAL['joint']['p_value']:.3g}), as expected when tiny
  differences are detectable in data pooled from many tests. SRM check
  against the designed 85/15 split: chi-square {f(SRM['chi2'], 2)},
  p = {SRM['p_value']:.3g} ({SRM['verdict'].replace('_', ' ')}).
  Assignment is a known constant propensity of 0.85 by design, and all
  weighting uses it.
- Cross-fitted Qini coefficients: T-learner
  {f(T['qini']['qini_coefficient'], 3)}, X-learner
  {f(X['qini']['qini_coefficient'], 3)}. Under shuffled assignment
  (placebo gate) both collapse to
  {f(PL['t_learner_qini_coefficient'], 3)} and
  {f(PL['x_learner_qini_coefficient'], 3)}, so the ranking signal is
  not an artefact of the scoring path.
- {best_policy_line(X, 'X-learner')}
- {best_policy_line(T, 'T-learner')}
- Calibration: the X-learner's top decile shows an observed uplift of
  {pp(X['calibration'][0]['observed_uplift'])} pp against a mean
  prediction of {pp(X['calibration'][0]['mean_predicted_uplift'])} pp;
  the bottom decile shows {pp(X['calibration'][-1]['observed_uplift'])}
  pp observed against {pp(X['calibration'][-1]['mean_predicted_uplift'])}
  pp predicted. The full decile table is in the README.
- Exposure is defined only inside the treated arm here
  ({100 * EXP['treated_exposure_share']:.2f}% of treated rows exposed,
  0% of control), so the harness reports the per-exposure contrast as
  descriptive only: exposed treated convert at
  {pp(EXP['rate_exposed_treated'])}% against {pp(EXP['rate_control'])}%
  for control, a descriptive gap of {pp(EXP['descriptive_difference'])}
  pp that must not be read as an exposure effect.
- Synthetic recovery gate: {'passed' if SYN['passed'] else 'FAILED'}.
  With a known true ATE of {pp(SYN['true_ate'])} pp, both learners
  recover it within {pp(SYN['tolerances']['ate_abs_error_max'])} pp and
  rank by the true effect with Spearman
  {f(SYN['estimators']['t_learner']['spearman_vs_true_tau'], 3)}
  (T-learner) and {f(SYN['estimators']['x_learner']['spearman_vs_true_tau'], 3)}
  (X-learner).

## Bottom line

The average effect is small, but it is not uniform: a cross-fitted
targeting rule concentrates a large share of the incremental
conversions in a small slice of the population, and the doubly robust
policy values put a defensible number, with intervals, on what that
targeting is worth. The readout harness that produced these numbers is
packaged and unit-tested for reuse on any experiment of the same
shape.
"""


def main() -> None:
    (ROOT / "README.md").write_text(README)
    (ROOT / "reports/executive_summary.md").write_text(SUMMARY)
    print("rendered README.md and reports/executive_summary.md from results.json")


if __name__ == "__main__":
    main()
