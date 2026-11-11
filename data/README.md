# Data

## Licence (read first)

**The Criteo Uplift Prediction Dataset is released by Criteo AI Lab
under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International licence (CC BY-NC-SA 4.0).** Non-commercial use only;
adaptations must be shared under the same licence. This licence
covers the data only. The code in this repository is MIT-licensed
(see the repository LICENSE), and the code licence must not be read
as covering the data.

## Disclosed limitation

Criteo's own documentation states that for privacy reasons the
released data have been sub-sampled non-uniformly, so that the
original incrementality level cannot be deduced from the dataset.
Every effect size this project reports is a within-sample estimate on
the released subsample. It is never Criteo's true incrementality.

## What is here

`data/raw/` holds the dataset exactly as downloaded (gitignored;
recreate and verify with `scripts/download_data.py`).
`data/processed/` holds the derived artifacts written by
`scripts/run_analysis.py`: the typed parquet conversion of the full
file, the seeded analysis sample, and the cross-fitted out-of-fold
scores. Processed data are also never committed.

## Source

The dataset was obtained from Criteo AI Lab's own distribution
channel: Criteo AI Lab's dataset index links its Hugging Face
organisation as an official channel, and the file was downloaded from
the `criteo/criteo-uplift` repository there. No Kaggle copy and no
third-party mirror was used.

| File | Source | Bytes | SHA-256 | Coverage |
|---|---|---|---|---|
| criteo-research-uplift-v2.1.csv.gz | Criteo AI Lab via its official Hugging Face organisation (criteo/criteo-uplift) | 311,422,618 | 2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc | 13,979,592 rows; 16 columns (f0 to f11, treatment, conversion, visit, exposure); treatment share 85.00% |

The SHA-256 above matches the hash published on the file's own page
in the Criteo repository, and `scripts/download_data.py` re-verifies
it on every run. Retrieved 2026-11-11. Full provenance, including the
access routes that did not work, is in the project gate note
(02_Dataset_Research/e4-data-gate-2026-11-11.md in the portfolio
workspace).

## Version note

The file is v2 (file version 2.1). Version 1 was decommissioned by
Criteo after early users found a flaw; v2 rebalances the treatment
effect across subpopulations and sub-samples every constituent test
to the same global treatment ratio of 0.85 (Diemert et al. 2021,
arXiv:2111.10106).
