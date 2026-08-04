#!/usr/bin/env python3
"""Multi-seed replication + λ stress test for the TSN experiment suite.

Usage:
  python3 experiments/run_matrix.py                     # 多种子 default + large
  python3 experiments/run_matrix.py --modes large --flows 16   # λ 压力测试

Runs each mode × seed sequentially; results land in run_YYYYMMDD_HHMMSS/
directories (one per (mode, seed) pair) with a summary JSON.
"""
from __future__ import annotations
import json, os, sys, time, datetime, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments import run_full


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modes", nargs="+", default=["default", "large"],
                   choices=["default", "large", "dynamic", "ablation"])
    p.add_argument("--flows", type=int, default=0, help="override flow count")
    p.add_argument("--seeds", type=str, default="123,456,789")
    p.add_argument("--train", type=int, default=60000)
    p.add_argument("--eval", type=int, default=10)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    summary = {}

    for mode in args.modes:
        for seed in seeds:
            print(f"\n{'#'*70}\n# {mode} | seed={seed} | flows={args.flows or 'default'}\n{'#'*70}",
                  flush=True)
            # fresh args namespace per run
            class A: pass
            a = A()
            a.mode = mode
            a.flows = args.flows or (12 if mode == "large" else 5 if mode == "default" else 8)
            a.switches = 3
            a.rate = 1000.0
            a.hyperperiod = 10000.0
            a.train = args.train
            a.eval = args.eval
            a.seed = seed

            mode_fn = {
                "default": run_full.run,
                "large": run_full.run_large_scale,
                "dynamic": run_full.run_dynamic_arrival,
                "ablation": run_full.run_ablation,
            }[mode]

            t0 = time.time()
            outdir, results = mode_fn(a)
            summary[f"{mode}_s{seed}"] = {
                "outdir": outdir,
                "elapsed_s": round(time.time() - t0, 1),
                "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_flow_completion"}
                            for k, v in results.items()},
            }
            print(f"[{mode} s{seed}] done in {time.time()-t0:.0f}s", flush=True)

    out = os.path.join(os.path.dirname(__file__), f"matrix_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary written to {out}")


if __name__ == "__main__":
    main()
