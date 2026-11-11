# Methods

## Data and design

The data are the Criteo Uplift Prediction Dataset v2 (file version
2.1): 13,979,592 rows, 12 anonymised features, and treatment,
conversion, visit, and exposure columns, with no missing values.
Conversion is the primary outcome and visit the secondary outcome.
Treatment is analysed as assignment (intention to treat).

One design fact drives the whole analysis. In the v2 release, Criteo
sub-sampled every constituent incrementality test to the same global
treatment ratio (Diemert et al. 2021, Section 3, Table 1), so the
assignment propensity is a known constant, e = 0.85, for every row.
All weighting in this project (X-learner combination, doubly robust
policy values) uses that constant. A fitted propensity model is used
only inside the joint balance diagnostic, never as an estimator input.

The released data are a non-uniform subsample. Criteo's documentation
states that the original incrementality level cannot be deduced from
the released data, so every level reported here is a within-sample
estimate, not Criteo's true incrementality.

## Benchmark: difference in means

The benchmark estimand is the intention-to-treat average treatment
effect, estimated by the difference in arm means on all rows, with the
standard two-sample standard error and a normal 95% interval. Under
randomisation this is unbiased without any modelling, and it is the
yardstick the uplift machinery has to beat in usefulness, not in
headline size.

## Randomisation diagnostics

Balance is assessed per feature with standardised mean differences
(arm mean difference divided by the pooled standard deviation),
computed on all rows from streamed moments. The joint check is a
likelihood-ratio test of a logistic model of assignment on all 12
features against an intercept-only model, run on a seeded subsample of
the analysis sample (200,000 rows): with millions of rows a joint test
becomes a significance machine, so it is used as a diagnostic at a
fixed, disclosed size. The sample ratio mismatch check is a chi-square
goodness-of-fit test of the observed arm counts against the designed
85/15 split, flagged at the conventional p < 0.001 threshold.

## Uplift estimators

Two meta-learners are fitted, both with HistGradientBoosting base
learners, in one 5-fold cross-fitting pass over the analysis sample,
so every row's scores are out of fold:

- Stage 1 (shared): one classifier per arm, giving mu1(x) and mu0(x),
  the treated and control outcome models.
- T-learner: tau_t(x) = mu1(x) - mu0(x).
- X-learner (Kuenzel et al. 2019): within each training fold, imputed
  effects D1 = y - mu0(x) on treated rows and D0 = mu1(x) - y on
  control rows are regressed on features to give tau_1(x) and
  tau_0(x); the held-out score is tau_x(x) = e * tau_0(x) +
  (1 - e) * tau_1(x) with the known propensity e = 0.85.

The S-learner is omitted on purpose; see docs/coding_decisions.md.

## Qini curves and coefficients

Scores are evaluated with the randomisation-corrected Qini curve for
unequal arm sizes (Radcliffe form): for the top-ranked set S,

    Q(S) = Y_t(S) - Y_c(S) * (N_t(S) / N_c(S)),

in units of incremental conversions. The random-targeting reference
is the straight line from 0 to Q(full sample); the perfect reference
orders rows by realised contribution (treated converters first,
control converters last). The reported Qini coefficient is the area
between the model curve and the random line divided by the area
between the perfect curve and the random line: 0 is random targeting,
1 is the realised oracle. The normalisation is stated wherever the
coefficient is used, because Qini coefficients are not comparable
across normalisations.

## Calibration

Rows are split into equal-count deciles of predicted uplift, highest
first. Within each decile the observed uplift is the treated-minus-
control conversion rate; it is compared with the mean predicted
uplift in the decile and scaled by the decile size into incremental
conversions. Calibration here is descriptive: with a 0.15 control
share, each decile's observed rate difference carries noticeable
sampling noise, which the report does not hide.

## Policy value

For a policy that treats a set S (the top q percent by predicted
uplift, for q in 10, 20, 30, 50, plus treat-all and treat-none), the
value is estimated with the doubly robust (augmented inverse
propensity) estimator built from the cross-fitted outcome models and
the known propensity:

    V = mean over rows of [ mu_pi(x) + 1{t = pi(x)} / P(t = pi(x)) * (y - mu_t(x)) ].

Because the scores that define S and the outcome models are both out
of fold, the value estimate does not grade a policy on data it was
fitted on. Intervals are influence-function intervals (standard error
of the per-row doubly robust contributions). Units are expected
conversions per person; reports also give the increment over the
treat-none value.

## Exposure

Exposure is an attribute of the treated arm only in this data: it is 1
for a minority of treated rows and 0 for every control row, so no
randomised exposed-versus-unexposed contrast exists. The harness
returns a clearly labelled descriptive contrast (exposed treated
versus all control) with an explicit warning; the primary analysis
never conditions on exposure.

## Validation gates

- Placebo gate: treatment labels are shuffled with a fixed seed and
  the full scoring path (cross-fitting at 3 folds) is rerun. Qini
  coefficients for both estimators must collapse to near zero
  (|Qini| < 0.05 in the config) for the gate to pass. This detects
  leakage or a scoring path that manufactures ranking signal.
- Synthetic recovery gate: a simulated RCT with a known nonlinear
  effect function (see src/uplift_platform/synthetic.py) is run
  through the same estimators and policy machinery. The gate checks
  the recovered ATE against the truth, the share of the true
  top-decile effect captured by each estimator's estimated top
  decile, and the Spearman correlation between estimated and true
  effects, against the tolerances stated in configs/default.json.

## Reproducibility

All randomness is seeded from configs/default.json. ATE, balance, and
SRM statistics use all 13,979,592 rows; learner fitting uses the
seeded analysis sample described in docs/coding_decisions.md.
reports/results.json is the single source of every number in the
README and the executive summary, which are rendered from it by
scripts/make_reports.py.
