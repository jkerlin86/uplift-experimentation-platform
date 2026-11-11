#!/usr/bin/env bash
# Full pipeline: verify data, analysis, figures, notebooks, reports, tests.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/download_data.py
python scripts/run_analysis.py
python scripts/make_figures.py
python scripts/make_notebooks.py
python scripts/execute_notebooks.py
python scripts/make_reports.py
python -m pytest tests -q
python -m ruff check src scripts tests
