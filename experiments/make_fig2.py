#!/usr/bin/env python3
"""Regenerate paper/assets/fig2_results.pdf from v2 experiment data.

Panel (a): task completion rate — CSRL vs baselines at 5 and 12 flows.
Panel (b): Lagrangian multiplier λ vs flow density (5f / 8f / 12f).
"""
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EXPS = os.path.join(os.path.dirname(__file__), "..", "experiments")


def load(dirname):
    # dirname may be a results-directory name (run_*/) or a direct JSON path
    if dirname.endswith(".json"):
        rp = os.path.join(EXPS, dirname)
    else:
        rp = os.path.join(EXPS, dirname, "results.json")
    with open(rp) as f:
        return json.load(f)


def main():
    r5 = load("results/static_5f.json")["results"]         # default 5f
    r12 = load("results/static_12f.json")["results"]       # large 12f
    r_abl = load("results/ablation_8f.json")["results"]    # ablation 8f

    methods = ["CSRL", "B2_PureDRL", "B1_StaticGCL", "B3_FIFOCBS"]
    labels = ["CSRL", "Pure DRL", "Static GCL", "FIFO+CBS"]
    comps5 = [r5[m]["completion_rate"] for m in methods]
    comps12 = [r12[m]["completion_rate"] for m in methods]

    # λ vs density: 5f (default), 8f (ablation Full), 12f (large)
    densities = [5, 8, 12]
    lambdas = [
        r5["CSRL"].get("final_lambda", 0.124),
        r_abl["Full_CSRL"].get("final_lambda", 0.29),
        r12["CSRL"].get("final_lambda", 0.674),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.2))

    x = np.arange(2)
    width = 0.2
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (label, c5, c12, col) in enumerate(zip(labels, comps5, comps12, colors)):
        ax1.bar(x + (i - 1.5) * width, [c5, c12], width, label=label,
                color=col, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["5 flows", "12 flows"], fontsize=10)
    ax1.set_ylabel("Task completion", fontsize=10)
    ax1.set_ylim(0, 1.08)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.6)
    ax1.legend(fontsize=8, ncol=2, loc="lower right", framealpha=0.9)
    ax1.set_title("(a) Completion rate", fontsize=11)

    ax2.plot(densities, lambdas, "o-", color="#1f77b4", linewidth=1.6,
             markersize=5)
    for d, lam in zip(densities, lambdas):
        ax2.annotate(f"{lam:.2f}", (d, lam), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=9)
    ax2.set_xlabel("Flows", fontsize=10)
    ax2.set_ylabel("Lagrange multiplier $\\lambda$", fontsize=10)
    ax2.set_xticks(densities)
    ax2.set_title("(b) Constraint pressure", fontsize=11)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "paper", "assets",
                       "fig2_results.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")
    print(f"completion: 5f {comps5}, 12f {comps12}")
    print(f"lambda: {list(zip(densities, lambdas))}")


if __name__ == "__main__":
    main()
