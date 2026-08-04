#!/usr/bin/env python3
"""Generate summary.png from all experiment result directories.

Usage:
  python3 experiments/viz.py --dirs run_A run_B run_C
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_results(dirpath: str):
    rp = os.path.join(dirpath, "results.json")
    if not os.path.exists(rp):
        return None, ""
    with open(rp) as f:
        data = json.load(f)
    results = data.get("results", {})
    mode = data.get("mode", "")
    config = data.get("config", {})
    return results, mode, config


def plot_summary(dirs: list[str], outdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect per-directory data
    all_data = []
    for d in dirs:
        res, mode, cfg = load_results(d)
        if res:
            all_data.append((os.path.basename(d), res, mode, cfg))

    # ─────────────────────────────────────────────────────
    # Build structured data
    # ─────────────────────────────────────────────────────

    # flow_data: {method: {n_flows: completion_rate}}
    flow_data: dict[str, dict[int, float]] = {}
    p99_flow_data: dict[str, dict[int, float]] = {}

    # Dynamic arrival results
    dyn_results: dict[str, dict] = {}

    # Ablation results
    abl_results: dict[str, dict] = {}

    # P99 consolidated across all experiments
    p99_consolidated: dict[str, list[tuple[str, float]]] = {}

    method_labels = {
        "B1_StaticGCL": "Static GCL",
        "B3_FIFOCBS": "FIFO+CBS",
        "B2_PureDRL": "Pure DRL",
        "CSRL": "CSRL",
    }

    for dname, res, mode, cfg in all_data:
        nf = cfg.get("flows", None)
        for method, metrics in res.items():
            label = method_labels.get(method, method.replace("_", " "))

            # Completion rate by flows
            if nf and "completion_rate" in metrics:
                cr = metrics["completion_rate"]
                if label not in flow_data:
                    flow_data[label] = {}
                    p99_flow_data[label] = {}
                flow_data[label][nf] = max(flow_data[label].get(nf, 0), cr)
                p99_flow_data[label][nf] = max(p99_flow_data[label].get(nf, 0),
                                                metrics.get("p99_us", 0))

            # Dynamic arrival
            if mode == "dynamic":
                dyn_results[label] = metrics

            # Ablation
            if mode == "ablation":
                abl_results[label] = metrics

            # P99 consolidated: (label, nf, p99)
            if nf and "p99_us" in metrics:
                if label not in p99_consolidated:
                    p99_consolidated[label] = []
                p99_consolidated[label].append((str(nf), metrics["p99_us"]))

    # ── Create figure ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    colors = {"CSRL": "tab:blue", "Pure DRL": "tab:orange",
              "Static GCL": "tab:green", "FIFO+CBS": "tab:red"}
    markers = {"CSRL": "o", "Pure DRL": "s", "Static GCL": "^", "FIFO+CBS": "D"}

    # ── Subplot 1: Completion rate vs flows ──
    for method, data_map in flow_data.items():
        xs = sorted(data_map.keys())
        ys = [data_map[x] for x in xs]
        ax1.plot(xs, ys, marker=markers.get(method, "x"),
                 color=colors.get(method, "gray"), linewidth=2, markersize=8, label=method)
        if len(xs) >= 2:
            for x, y in zip(xs, ys):
                ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                             xytext=(0, -12), ha="center", fontsize=7, color=colors.get(method, "gray"))

    ax1.set_xlabel("Number of Flows", fontsize=11)
    ax1.set_ylabel("Completion Rate", fontsize=11)
    ax1.set_title("Completion Rate vs. Number of Flows", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower left")
    ax1.set_ylim(0, 1.10)
    ax1.grid(True, alpha=0.3)

    # ── Subplot 2: Dynamic arrival ──
    if dyn_results:
        methods_d = ["Static GCL", "FIFO+CBS", "Pure DRL", "CSRL"]
        methods_d = [m for m in methods_d if m in dyn_results]
        x = np.arange(len(methods_d))
        w = 0.25
        base_crs = [dyn_results[m].get("completion_rate_base", 0) for m in methods_d]
        new_crs = [dyn_results[m].get("completion_rate_new", 0) for m in methods_d]
        accept_rates = [dyn_results[m].get("arrival_acceptance_rate", 0) for m in methods_d]

        ax2.bar(x - w, base_crs, w, label="Base Flows (5)", color="steelblue")
        ax2.bar(x, new_crs, w, label="New Arrivals (3)", color="darkorange")
        ax2.bar(x + w, accept_rates, w, label="Acceptance Rate", color="seagreen")

        ax2.set_xticks(x)
        ax2.set_xticklabels(methods_d, fontsize=9)
        ax2.set_ylabel("Rate", fontsize=11)
        ax2.set_title("Dynamic Arrival: 5 Base + 3 Arriving Flows", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=8, loc="lower left")
        ax2.set_ylim(0, 1.15)
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "No dynamic arrival data", ha="center", va="center", transform=ax2.transAxes)

    # ── Subplot 3: Ablation bar chart ──
    if abl_results:
        labels_a = ["Full CSRL", "No Safety Shield", "No Semantic", "No NC Constraint"]
        labels_a = [l for l in labels_a if l in abl_results]
        crs = [abl_results[m].get("completion_rate", 0) for m in labels_a]
        p99s = [abl_results[m].get("p99_us", 0) for m in labels_a]
        dv = [abl_results[m].get("deadline_violations", 0) for m in labels_a]
        colors_a = ["#2c7bb6", "#fdae61", "#abd9e9", "#d7191c"]

        x = np.arange(len(labels_a))
        w = 0.35
        ax3.bar(x - w / 2, crs, w, label="Completion Rate", color=colors_a)
        for xi, c in zip(x, crs):
            ax3.text(xi - w / 2, c + 0.02, f"{c:.3f}", ha="center", fontsize=8)

        ax3.set_ylabel("Completion Rate", fontsize=11)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels_a, fontsize=8, rotation=10)
        ax3.set_title("Ablation Study: 8 Flows", fontsize=12, fontweight="bold")
        ax3.set_ylim(0, 1.15)
        ax3.grid(True, alpha=0.3, axis="y")

        ax3b = ax3.twinx()
        ax3b.bar(x + w / 2, [min(p, 300000) for p in p99s], w,
                 label="P99 (μs)", color="lightcoral", alpha=0.6)
        ax3b.set_ylabel("P99 Delay (μs)", fontsize=9)
        for xi, p in zip(x, p99s):
            ax3b.text(xi + w / 2, min(p, 300000) + 5000, f"{p:.0f}",
                      ha="center", fontsize=7, rotation=90, va="bottom")

        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    else:
        ax3.text(0.5, 0.5, "No ablation data", ha="center", va="center", transform=ax3.transAxes)

    # ── Subplot 4: P99 delay across flow counts ──
    if p99_flow_data:
        for method in ["CSRL", "Pure DRL", "Static GCL", "FIFO+CBS"]:
            if method in p99_flow_data and p99_flow_data[method]:
                data_map = p99_flow_data[method]
                xs = sorted(data_map.keys())
                ys = [data_map[x] / 1000.0 for x in xs]
                ax4.plot(xs, ys, marker=markers.get(method, "x"),
                         color=colors.get(method, "gray"), linewidth=2,
                         markersize=8, label=method)
                for x, y in zip(xs, ys):
                    ax4.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                                 xytext=(0, 8), ha="center", fontsize=7,
                                 color=colors.get(method, "gray"))

        ax4.set_xlabel("Number of Flows", fontsize=11)
        ax4.set_ylabel("P99 Delay (ms)", fontsize=11)
        ax4.set_title("P99 Delay Across Number of Flows", fontsize=12, fontweight="bold")
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

    fig.suptitle("TSN Semantic-Aware Scheduling — Comprehensive Experiment Results",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()

    path = os.path.join(outdir, "summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", required=True, help="Experiment dirs")
    parser.add_argument("--outdir", default=None, help="Output dir (default: first of --dirs)")
    args = parser.parse_args()
    outdir = args.outdir or args.dirs[0]
    plot_summary(args.dirs, outdir)
