"""Execute both notebooks in place with the project environment."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["01_exploration_and_balance.ipynb", "02_uplift_modelling_and_policy_value.ipynb"]


def main() -> None:
    for name in NAMES:
        path = ROOT / "notebooks" / name
        nb = nbf.read(path, as_version=4)
        client = NotebookClient(
            nb, timeout=1200, kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        )
        client.execute()
        nbf.write(nb, path)
        errors = [
            out
            for cell in nb.cells
            for out in cell.get("outputs", [])
            if out.get("output_type") == "error"
        ]
        print(name, "executed; error outputs:", len(errors))
        assert not errors, f"{name} has error outputs"


if __name__ == "__main__":
    main()
