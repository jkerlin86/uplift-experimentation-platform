# An Experimentation Platform for Uplift

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
  (311,422,618 bytes, SHA-256 `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`), verified against the
  hash published on the file's own page.
- **Data licence: CC BY-NC-SA 4.0 (Criteo AI Lab).** Non-commercial use
  only; adaptations shared under the same licence. The data are never
  committed to this repository; `scripts/download_data.py` reproduces
  and verifies the file. The code in this repository is MIT-licensed
  (see LICENSE); that licence covers the code only, never the data.
- Size: 13,979,592 rows, 12 anonymised features (f0 to f11, no
  missing values), treatment assigned to 85.00%
  of rows.
- Design: the v2 release sub-sampled every constituent incrementality
  test to the same global treatment ratio, so assignment is a constant
  propensity of 0.85 for every row (Diemert et al. 2021, Section 3,
  Table 1). Every weighting in this project uses that known constant,
  not a fitted propensity model.
- Outcomes: conversion (primary) and visit (secondary). Exposure is an
  attribute of the treated arm only: 3.60%
  of treated rows have exposure 1, and exposure is 0 for every control
  row, so the primary analysis is intention-to-treat on assignment and
  the harness's per-exposure figure is descriptive only.

## Headline findings

- **The benchmark effect is small and precisely estimated.**
  Conversion is 0.3089% under treatment against
  0.1938% under control: an ATE of
  0.1152 percentage points (95% CI [0.1085,
  0.1219], standard error 0.0034). The
  secondary visit outcome gives an ATE of 1.0342 percentage
  points (95% CI [1.0056, 1.0629]).
- **Randomisation is close, with small but detectable imbalances.**
  The largest per-feature standardised mean difference is
  0.0488 (mean 0.0207), well under
  the conventional 0.1 threshold on every feature; at these sample
  sizes the joint logistic likelihood-ratio check still rejects
  (p = 1.07e-17) because even very small
  differences are detectable, which is consistent with the released
  data pooling many constituent incrementality tests. The SRM check
  against the designed 85/15 split gives chi-square
  0.00 and p = 0.999:
  no srm detected.
- **Targeting signal is real but modest.** Cross-fitted Qini
  coefficients are 0.133 for the
  T-learner and 0.090 for the
  X-learner, against a placebo gate in which shuffled assignment
  collapses both coefficients to -0.008
  and -0.045.
- **Targeting beats blanket treatment per person treated.**
  For the X-learner, treating only the top 10% by predicted uplift captures 72.5% of the treat-all incremental gain (0.0839 of 0.1157 pp) while treating a tenth as many people, and the top 50% captures 74.8%. Treat-all still maximises the total; targeting is what pays when treatment itself has a cost.
  For the T-learner, treating only the top 10% by predicted uplift captures 77.3% of the treat-all incremental gain (0.0894 of 0.1157 pp) while treating a tenth as many people, and the top 50% captures 81.4%. Treat-all still maximises the total; targeting is what pays when treatment itself has a cost.

## Policy values (doubly robust, cross-fitted)

Values are expected conversions per person under each policy, with
95% confidence intervals from the influence-function standard error.
The outcome models and the rankings are out-of-fold; the propensity is
the known constant 0.85.

### X-learner ranking

| Policy | Value (% conv. per person) | 95% CI | Incremental vs treat none (pp) |
|---|---|---|---|
| Treat top 10% by predicted uplift | 0.2740 | [0.2586, 0.2893] | 0.0839 |
| Treat top 20% | 0.2773 | [0.2629, 0.2918] | 0.0873 |
| Treat top 30% | 0.2791 | [0.2649, 0.2933] | 0.0890 |
| Treat top 50% | 0.2766 | [0.2629, 0.2903] | 0.0865 |
| Treat all | 0.3058 | [0.2941, 0.3174] | 0.1157 |
| Treat none | 0.1901 | [0.1690, 0.2111] | 0.0000 |

### T-learner ranking

| Policy | Value (% conv. per person) | 95% CI | Incremental vs treat none (pp) |
|---|---|---|---|
| Treat top 10% by predicted uplift | 0.2794 | [0.2637, 0.2952] | 0.0894 |
| Treat top 20% | 0.2833 | [0.2682, 0.2984] | 0.0933 |
| Treat top 30% | 0.2833 | [0.2685, 0.2981] | 0.0932 |
| Treat top 50% | 0.2842 | [0.2697, 0.2987] | 0.0941 |
| Treat all | 0.3058 | [0.2941, 0.3174] | 0.1157 |
| Treat none | 0.1901 | [0.1690, 0.2111] | 0.0000 |

## Uplift calibration by predicted decile (X-learner)

Observed uplift is the treated-minus-control conversion rate inside
the decile; the scaled column expresses it as incremental conversions
within the decile. Read honestly, the ranking orders the deciles but
the predicted spread overstates the observed spread: the X-learner's
top decile is predicted at
1.3284 pp and observed at
0.7317 pp, and its bottom decile
is predicted at -0.2897 pp
while the observed value is
0.3694 pp. The scores are a
ranking device, not calibrated effect sizes at the individual level.

| Decile (1 = highest predicted) | n | Mean predicted uplift (pp) | Observed uplift (pp) | Incremental conversions |
|---|---|---|---|---|
| 1 | 100000 | 1.3284 | 0.7317 | 731.7 |
| 2 | 100000 | 0.0651 | 0.0303 | 30.3 |
| 3 | 100000 | 0.0158 | 0.0171 | 17.1 |
| 4 | 100000 | 0.0072 | -0.0267 | -26.7 |
| 5 | 100000 | 0.0059 | -0.0029 | -2.9 |
| 6 | 100000 | 0.0053 | -0.0073 | -7.3 |
| 7 | 100000 | 0.0026 | 0.0246 | 24.6 |
| 8 | 100000 | 0.0007 | 0.0092 | 9.2 |
| 9 | 100000 | -0.0000 | -0.0031 | -3.1 |
| 10 | 100000 | -0.2897 | 0.3694 | 369.4 |

## Balance detail (all rows)

| Feature | Standardised mean difference |
|---|---|
| f3 | -0.0488 |
| f6 | -0.0404 |
| f5 | -0.0306 |
| f9 | 0.0240 |
| f1 | 0.0240 |
| f8 | -0.0224 |
| f7 | 0.0213 |
| f10 | 0.0106 |
| f4 | 0.0080 |
| f0 | -0.0069 |
| f2 | -0.0062 |
| f11 | -0.0052 |

## Validation gates

- **Placebo gate.** With treatment labels shuffled (seed
  20261112), the pipeline's scoring path returns Qini coefficients
  of -0.008 (T-learner) and
  -0.045 (X-learner), against a pass
  threshold of |Qini| < 0.05: gate
  passed.
- **Synthetic recovery gate.** On a simulated RCT with a known
  nonlinear effect function (true ATE 1.9954 pp), the
  difference-in-means estimate is 2.0126
  pp, the T-learner recovers the ATE with absolute error
  0.1711 pp and the
  X-learner 0.0399 pp
  (tolerance 0.4000 pp); each
  estimator's top decile captures
  82.7%
  (T-learner) and 85.3%
  (X-learner) of the true top-decile effect, with Spearman correlations
  against the true effect of 0.707
  and 0.705:
  gate passed.

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
  1,000,000 rows drawn from all 13,979,592 rows; ATE, balance,
  and SRM statistics use all rows. Sampling decisions are documented
  in docs/coding_decisions.md.
- One 5-fold cross-fitting pass produces
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
9.5 minutes on the build machine.

## Repository layout

- `src/uplift_platform/`: harness, metrics, learners, synthetic gate, data IO.
- `scripts/`: download/verify, analysis, figures, notebooks, reports, run_all.
- `notebooks/`: 01 exploration and balance; 02 uplift modelling and policy value (both executed).
- `reports/`: results.json (single source of every reported number) and executive_summary.md.
- `figures/`: Qini curves, calibration, policy value, rendered from results.json.
- `docs/`: methods.md and coding_decisions.md.
- `tests/`: harness, metrics (including a hand-checked Qini example), and gate tests.
