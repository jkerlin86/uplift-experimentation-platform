# Executive summary: An Experimentation Platform for Uplift

## Question

On the Criteo Uplift Prediction Dataset v2, who should be treated, and
what is targeting worth against treating everyone or no one?

## The governing limitation

The released data are a non-uniform subsample, so estimated levels are
NOT Criteo's true incrementality; all effects below are within-sample
estimates on 13,979,592 released rows. Data licence: CC BY-NC-SA
4.0 (Criteo AI Lab); the data are not redistributed with this project.

## Findings

- Benchmark ATE on conversion: 0.1152 percentage points
  (treated 0.3089%, control
  0.1938%; 95% CI [0.1085,
  0.1219]). Visit ATE: 1.0342 pp
  ([1.0056, 1.0629]).
- Randomisation is close: max |SMD| 0.0488 across the
  12 features (mean 0.0207), under the 0.1 threshold
  everywhere, though the joint check rejects at these sample sizes
  (p = 1.07e-17), as expected when tiny
  differences are detectable in data pooled from many tests. SRM check
  against the designed 85/15 split: chi-square 0.00,
  p = 0.999 (no srm detected).
  Assignment is a known constant propensity of 0.85 by design, and all
  weighting uses it.
- Cross-fitted Qini coefficients: T-learner
  0.133, X-learner
  0.090. Under shuffled assignment
  (placebo gate) both collapse to
  -0.008 and
  -0.045, so the ranking signal is
  not an artefact of the scoring path.
- For the X-learner, treating only the top 10% by predicted uplift captures 72.5% of the treat-all incremental gain (0.0839 of 0.1157 pp) while treating a tenth as many people, and the top 50% captures 74.8%. Treat-all still maximises the total; targeting is what pays when treatment itself has a cost.
- For the T-learner, treating only the top 10% by predicted uplift captures 77.3% of the treat-all incremental gain (0.0894 of 0.1157 pp) while treating a tenth as many people, and the top 50% captures 81.4%. Treat-all still maximises the total; targeting is what pays when treatment itself has a cost.
- Calibration: the X-learner's top decile shows an observed uplift of
  0.7317 pp against a mean
  prediction of 1.3284 pp;
  the bottom decile shows 0.3694
  pp observed against -0.2897
  pp predicted. The full decile table is in the README.
- Exposure is defined only inside the treated arm here
  (3.63% of treated rows exposed,
  0% of control), so the harness reports the per-exposure contrast as
  descriptive only: exposed treated convert at
  5.4422% against 0.1838%
  for control, a descriptive gap of 5.2584
  pp that must not be read as an exposure effect.
- Synthetic recovery gate: passed.
  With a known true ATE of 1.9954 pp, both learners
  recover it within 0.4000 pp and
  rank by the true effect with Spearman
  0.707
  (T-learner) and 0.705
  (X-learner).

## Bottom line

The average effect is small, but it is not uniform: a cross-fitted
targeting rule concentrates a large share of the incremental
conversions in a small slice of the population, and the doubly robust
policy values put a defensible number, with intervals, on what that
targeting is worth. The readout harness that produced these numbers is
packaged and unit-tested for reuse on any experiment of the same
shape.
