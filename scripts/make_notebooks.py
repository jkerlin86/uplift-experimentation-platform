"""Build the two analysis notebooks (unexecuted). Execute with
scripts/execute_notebooks.py. The notebooks read the processed parquet
files written by scripts/run_analysis.py and recompute every number
they show with the project package; nothing is copied from a report.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = """import sys
sys.path.insert(0, "../src")
import json
import pandas as pd
from uplift_platform.harness import balance_check, exposure_readout, itt_estimate, srm_check

pd.set_option("display.width", 140)
config = json.loads(open("../configs/default.json").read())
FEATURES = config["data"]["feature_columns"]
PROPENSITY = config["design"]["propensity"]
"""


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def nb1():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "name": "python3"}
    nb.cells = [
        md(
            "# Exploration, balance, and the experiment readout\n\n"
            "The Criteo Uplift Prediction Dataset v2 (CC BY-NC-SA 4.0) is a "
            "collection of incrementality tests rebalanced to a constant "
            "assignment propensity of 0.85. The released data are a "
            "non-uniform subsample, so every level estimated here is a "
            "within-sample estimate, not Criteo's true incrementality. This "
            "notebook loads the seeded analysis sample saved by "
            "scripts/run_analysis.py and runs the packaged readout harness: "
            "ITT estimates, the SRM check, randomisation balance, and the "
            "exposure readout."
        ),
        code(SETUP),
        md("## The analysis sample"),
        code(
            "sample = pd.read_parquet('../data/processed/analysis_sample.parquet')\n"
            "print('rows:', len(sample))\n"
            "sample[['treatment', 'conversion', 'visit', 'exposure']].mean()"
        ),
        md("## Intention-to-treat estimates (difference in means)"),
        code(
            "for outcome in ('conversion', 'visit'):\n"
            "    est = itt_estimate(sample[outcome].to_numpy(float), "
            "sample['treatment'].to_numpy(int))\n"
            "    print(outcome, {k: est[k] for k in "
            "('rate_treated', 'rate_control', 'ate', 'se', 'ci_low', 'ci_high')})"
        ),
        md("## Sample ratio mismatch check against the designed 85/15 split"),
        code(
            "t = sample['treatment']\n"
            "srm_check(int((t == 1).sum()), int((t == 0).sum()), PROPENSITY)"
        ),
        md("## Randomisation balance: standardised mean differences"),
        code(
            "bal = balance_check(sample, FEATURES)\n"
            "pd.DataFrame(bal['per_feature'])[['feature', 'smd', 'abs_smd']]"
        ),
        code(
            "print('max |SMD|:', bal['max_abs_smd'])\n"
            "print('joint check:', bal['joint'])"
        ),
        md(
            "## Exposure readout\n\n"
            "Exposure is an attribute of the treated arm only in this data: "
            "it is 1 for a minority of treated rows and 0 for every control "
            "row. The harness therefore returns a descriptive contrast with "
            "an explicit warning, never a causal per-exposure effect."
        ),
        code("exposure_readout(sample, 'conversion')"),
    ]
    return nb


def nb2():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "name": "python3"}
    nb.cells = [
        md(
            "# Uplift modelling, Qini, calibration, and policy value\n\n"
            "This notebook loads the cross-fitted out-of-fold scores written "
            "by scripts/run_analysis.py (one row per analysis-sample user, "
            "with mu0, mu1, and T-learner and X-learner uplift scores, all "
            "produced out of fold) and recomputes the evaluation slate: Qini "
            "coefficients, decile calibration, and doubly robust policy "
            "values at the known constant propensity 0.85. It closes by "
            "comparing the recomputed Qini coefficients with the values in "
            "reports/results.json."
        ),
        code(
            "import sys\n"
            "sys.path.insert(0, '../src')\n"
            "import json\n"
            "import pandas as pd\n"
            "from uplift_platform.metrics import policy_table, qini_coefficient, "
            "uplift_calibration\n\n"
            "pd.set_option('display.width', 140)\n"
            "oof = pd.read_parquet('../data/processed/oof_scores.parquet')\n"
            "y = oof['conversion'].to_numpy(float)\n"
            "t = oof['treatment'].to_numpy(int)\n"
            "print('scored rows:', len(oof))"
        ),
        md("## Qini coefficients per estimator"),
        code(
            "qini = {\n"
            "    't_learner': qini_coefficient(y, t, oof['tau_t'].to_numpy()),\n"
            "    'x_learner': qini_coefficient(y, t, oof['tau_x'].to_numpy()),\n"
            "}\n"
            "qini"
        ),
        md("## Calibration by predicted decile (X-learner)"),
        code(
            "pd.DataFrame(uplift_calibration(y, t, oof['tau_x'].to_numpy()))"
        ),
        md("## Doubly robust policy values"),
        code(
            "for key in ('tau_t', 'tau_x'):\n"
            "    table = policy_table(y, t, oof['mu0'].to_numpy(), "
            "oof['mu1'].to_numpy(), oof[key].to_numpy(), [0.1, 0.2, 0.3, 0.5], 0.85)\n"
            "    print(key)\n"
            "    print(pd.DataFrame(table).T[['value', 'se', 'ci_low', 'ci_high', "
            "'incremental_vs_treat_none']])"
        ),
        md("## Agreement with reports/results.json"),
        code(
            "results = json.loads(open('../reports/results.json').read())\n"
            "for key in ('t_learner', 'x_learner'):\n"
            "    stored = results['estimators'][key]['qini']['qini_coefficient']\n"
            "    print(key, 'recomputed:', round(qini[key], 6), 'stored:', round(stored, 6))"
        ),
    ]
    return nb


def main() -> None:
    NB.mkdir(exist_ok=True)
    nbf.write(nb1(), NB / "01_exploration_and_balance.ipynb")
    nbf.write(nb2(), NB / "02_uplift_modelling_and_policy_value.ipynb")
    print("notebooks written (unexecuted)")


if __name__ == "__main__":
    main()
