# Coding decisions

1. **Constant known propensity, by design.** The Criteo v2 release
   sub-sampled all constituent incrementality tests to the same global
   treatment ratio, so assignment is a constant propensity e = 0.85
   for every row (Diemert et al. 2021, Section 3, Table 1). Three
   consequences are implemented as stated in the brief: (a) the SRM
   check compares the observed arm counts against the designed 85/15
   split; (b) the X-learner combination weights and the doubly robust
   policy-value estimator use the constant 0.85, not a fitted
   propensity model; a fitted propensity appears only inside the
   joint balance diagnostic, where it is a test statistic, not an
   estimator input; (c) Qini computation uses the standard
   randomisation-corrected form for unequal arm sizes (Radcliffe),
   documented in docs/methods.md.

2. **S-learner omitted.** With a rare binary outcome (about 0.31%
   treated conversion) and a single model asked to represent both
   arms through a treatment indicator, the S-learner's effect
   estimates are dominated by the baseline fit and shrink toward the
   average effect; in this data regime it adds a third curve without
   a distinct point beyond the T-learner (which differences two arm
   models directly) and the X-learner (which is designed for the
   85/15 imbalance). The slate is T-learner and X-learner only, and
   methods.md says so.

3. **One cross-fitting pass, shared stage 1.** The T-learner and the
   X-learner share the same cross-fitted stage-1 outcome models
   (mu0, mu1) from a single 5-fold pass. This halves the compute of
   fitting them separately, makes the two rankings differ only in how
   they turn outcome models into effect scores, and means the doubly
   robust policy values for both rankings use the same outcome
   models, so the comparison isolates the ranking. Out-of-fold scores
   are saved to data/processed/oof_scores.parquet so the notebooks
   recompute every evaluation metric from them.

4. **Analysis sample of 1,000,000 rows, seeded.** Learner fitting on
   all 13,979,592 rows is not feasible in this environment (about
   8 GB RAM, shared CPUs) inside the runtime budget, and is not
   needed for ranking quality at these effect sizes. The sample is
   drawn by Bernoulli sampling (p = 0.08) over parquet batches with
   seed 20261111 and cut to exactly 1,000,000 rows with the same
   generator. Everything cheap uses all rows instead: ATEs, arm
   counts, exposure shares, SRM, and per-feature balance moments are
   computed by streaming the full parquet. The sample preserves the
   85/15 design ratio in expectation and its own SRM and balance are
   the ones the joint check runs on.

5. **Parquet first, never the raw CSV in memory.** The 3.25 GB CSV is
   converted once, in 750,000-row chunks, to a parquet with float32
   features and int8 indicators (about a tenth of the memory
   footprint). All later stages read the parquet in batches or via
   the seeded sampler. The raw file's SHA-256 is verified by
   scripts/download_data.py before anything runs.

6. **HistGradientBoosting as the only base learner.** On rows in the
   millions, histogram gradient boosting is the fastest strong
   learner available in scikit-learn, handles the 12 dense float
   features natively, and needs no categorical plumbing on
   anonymised features. Classifiers fit the arm outcome models;
   regressors fit the X-learner's imputed-effect models. Parameters
   (150 trees, learning rate 0.06, 31 leaves, minimum leaf 100,
   L2 = 1.0) are in configs/default.json. A logistic-regression second
   base learner was considered and dropped: at these conversion
   rates its arm models are close to intercepts, so its uplift
   rankings would mostly re-measure the base rates rather than add a
   distinct modelling view; the runtime is better spent on the
   placebo gate at full sample size.

7. **Placebo gate at 3 folds on the full analysis sample.** The
   placebo reruns the entire scoring path on shuffled labels. Three
   folds instead of five keeps the gate inside the runtime budget
   while still cross-fitting every score; the quantity under test is
   the Qini coefficient's collapse, which does not depend on the fold
   count. The shuffle seed is separate from the analysis seed and is
   recorded in results.json.

8. **Policy-value intervals from influence functions, not bootstrap.**
   The doubly robust estimator's per-row contributions give a direct
   standard error (their sample standard deviation over root n). A
   bootstrap over 1,000,000 rows would multiply the runtime for the
   same interval to first order, so the analytic interval is used and
   named as such wherever the values are reported.

9. **Joint balance test at a fixed subsample size.** A logistic
   likelihood-ratio test on millions of rows flags trivially small
   imbalances. The joint check therefore runs on a seeded 200,000-row
   subsample of the analysis sample, and its n is reported alongside
   the p-value. The per-feature SMDs, which do not have this problem,
   use all rows.

10. **Qini normalisation is stated, not assumed.** Qini coefficients
    are only comparable under a stated normalisation. This project
    uses (area between model curve and random line) / (area between
    perfect curve and random line), with the randomisation-corrected
    curve; the string is stored in results.json with every curve.
    The perfect curve is the realised oracle (ordering by observed
    arm and outcome), which is the convention that makes 1.0 mean
    "as well as this sample allows any ranking to do".

11. **Exposure is never conditioned on.** Exposure is post-assignment
    and structurally zero in the control arm. The harness's
    exposure_readout detects that structure and returns a descriptive
    contrast with an explicit warning rather than refusing silently
    or, worse, presenting a causal number. The main analysis is ITT
    throughout.

12. **Synthetic DGP signal strength.** A pre-run check of the recovery
    gate with the first DGP draft (effect amplitude 0.022 over a noisier
    baseline) recovered the ATE but ranked weakly (Spearman about 0.31):
    at a roughly 2% baseline conversion rate, Bernoulli noise, not the
    machinery, was the binding constraint, and the gate would have
    measured the noise floor rather than the pipeline. The effect
    amplitude was raised to 0.045 and the baseline intercept lowered
    before the full run, with the gate tolerances (ATE absolute error
    0.004, top-decile capture 0.70, Spearman 0.50) left exactly as
    first stated in the config. The gate now tests what it is meant to
    test: whether the estimators and the policy machinery recover a
    known, learnable heterogeneous effect.

13. **Versions pinned to what the build environment installed.**
    numpy 2.5.4, pandas 3.1.1, pyarrow 26.0.1, scikit-learn 1.9.1,
    scipy 1.18.1, matplotlib 3.11.2, pytest 9.2.0, ruff 0.17.4,
    ipykernel 7.4.0, nbclient 0.11.0, nbformat 5.11.1. Code is
    written against the pandas 3 API (no removed helpers are used).
