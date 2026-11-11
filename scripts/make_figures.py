"""Render the report figures from reports/results.json only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "reports/results.json").read_text())
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

T = R["estimators"]["t_learner"]
X = R["estimators"]["x_learner"]


def qini_curves() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for est, label, color in ((T, "T-learner", "tab:blue"), (X, "X-learner", "tab:orange")):
        q = est["qini"]
        ax.plot(q["phi"], q["qini"], label=f"{label} (Qini {q['qini_coefficient']:.3f})",
                color=color)
    q = T["qini"]
    ax.plot(q["phi"], q["random"], "--", color="grey", label="Random targeting")
    ax.plot(q["phi"], q["perfect"], ":", color="green", label="Perfect (realised oracle)")
    ax.set_xlabel("Share of population targeted (ranked by predicted uplift)")
    ax.set_ylabel("Incremental conversions (randomisation-corrected)")
    ax.set_title("Qini curves, conversion outcome, cross-fitted scores")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_qini_curves.png", dpi=150)
    plt.close(fig)


def calibration() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, est, label in zip(axes, (T, X), ("T-learner", "X-learner"), strict=True):
        cal = est["calibration"]
        deciles = [c["decile"] for c in cal]
        ax.plot(deciles, [100 * c["mean_predicted_uplift"] for c in cal], "o-",
                label="Mean predicted uplift")
        ax.plot(deciles, [100 * c["observed_uplift"] for c in cal], "s-",
                label="Observed uplift (treated minus control)")
        ax.set_xlabel("Predicted uplift decile (1 = highest)")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Uplift (percentage points)")
    fig.suptitle("Uplift calibration by predicted decile")
    fig.tight_layout()
    fig.savefig(FIG / "fig_uplift_calibration.png", dpi=150)
    plt.close(fig)


def policy_values() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    keys = ["top_10pct", "top_20pct", "top_30pct", "top_50pct", "treat_all"]
    labels = ["Top 10%", "Top 20%", "Top 30%", "Top 50%", "Treat all"]
    for est, label, color in ((T, "T-learner", "tab:blue"), (X, "X-learner", "tab:orange")):
        pv = est["policy_values"]
        vals = [100 * pv[k]["incremental_vs_treat_none"] for k in keys]
        errs = [100 * 1.96 * pv[k]["se"] for k in keys]
        ax.errorbar(labels, vals, yerr=errs, marker="o", capsize=3, label=label, color=color)
    none_val = T["policy_values"]["treat_none"]["value"]
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1)
    ax.set_ylabel("Incremental conversions per 100 people vs treat none (pp)")
    ax.set_title(
        "Doubly robust policy value (cross-fitted)\n"
        f"treat-none baseline: {100 * none_val:.3f} conversions per 100 people"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_policy_value.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    qini_curves()
    calibration()
    policy_values()
    print("figures written to", FIG)
