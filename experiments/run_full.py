#!/usr/bin/env python3
"""Full comparative experiment: CSRL vs 3 baselines on TSN scheduling.

Modes:
  default  — 5 flows, 4 methods (original behavior)
  large    — 12 flows, 4 methods (static benchmark)
  dynamic  — 8 flows, staggered arrival (5 base + 3 arriving)
  ablation — 8 flows, 4 configs (Full / NoShield / NoSemantic / NoNC)

Usage:
  python3 experiments/run_full.py --mode default --flows 5 --train 20000 --eval 10
  python3 experiments/run_full.py --mode large --flows 12 --train 30000 --eval 15
  python3 experiments/run_full.py --mode dynamic --flows 8 --train 30000 --eval 15
  python3 experiments/run_full.py --mode ablation --flows 8 --train 20000 --eval 15
"""
from __future__ import annotations
import json, os, sys, time, datetime
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.csrl.train import generate_random_flows
from src.csrl.tsn_env import CSRLConfig, TSNEnv
from src.csrl.csrl_agent import CSRLAgent, ConstraintManager
from src.csrl.safety_shield import SafetyShield
from src.nc_engine.topology import make_line_topology
from src.experiments.baselines import StaticGCLScheduler, FIFOCBSScheduler


def _flow_expected_packets(sf, steps_per_ep: int, hyperperiod_us: float) -> int:
    """Packets a flow *should* have transmitted over the evaluation.

    Within one hyperperiod an accepted flow releases HP/period frames;
    a rejected flow releases none but still counts toward the task
    completion denominator (rejected = not completed).
    """
    n_pkt = max(1, int(round(hyperperiod_us / max(sf.period_us, 1.0))))
    return steps_per_ep * n_pkt


def evaluate(env, scheduler, n_episodes: int, steps_per_ep: int = 100) -> dict:
    """Run eval episodes and aggregate per-flow metrics across episodes.

    Unlike the previous implementation, statistics are collected from
    every episode (not only the last one): per-episode snapshots are
    exported before each ``reset()`` clears the flow state.

    Completion is measured against the *expected* packet count, so a flow
    rejected by the scheduler counts as not completed — the metric cannot
    be inflated by refusing to schedule hard flows.
    """
    hp = env.config.hyperperiod_us

    # per-flow aggregates across all episodes
    total_expected: dict[str, int] = {}
    total_completed: dict[str, int] = {}
    total_delays: dict[str, list[float]] = {}
    total_sent: dict[str, int] = {}
    admitted_eps: dict[str, int] = {}

    for _ in range(n_episodes):
        obs, _ = env.reset()

        for _ in range(steps_per_ep):
            act, _ = scheduler.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(act)
            if terminated or truncated:
                break

        # export snapshot before the next reset wipes flow state
        for sf in env.sim_flows:
            fid = sf.flow_id
            n_expected = _flow_expected_packets(sf, steps_per_ep, hp)
            completed = sum(1 for d in sf.e2e_delays if d <= sf.deadline_us)
            total_expected[fid] = total_expected.get(fid, 0) + n_expected
            total_completed[fid] = total_completed.get(fid, 0) + completed
            total_delays.setdefault(fid, []).extend(list(sf.e2e_delays))
            total_sent[fid] = total_sent.get(fid, 0) + sf.packets_sent
            admitted_eps[fid] = admitted_eps.get(fid, 0) + (1 if sf.accepted else 0)

    # per-flow completion rate
    flow_cr: dict[str, float] = {}
    pw_map: dict[str, float] = {}
    for sf in env.sim_flows:
        fid = sf.flow_id
        n_ok = total_completed.get(fid, 0)
        n_exp = total_expected.get(fid, 1)
        flow_cr[fid] = n_ok / max(n_exp, 1)
        pw_map[fid] = sf.priority_weight

    all_delays = [d for ds in total_delays.values() for d in ds]

    # semantic-weighted completion (paper eq. reward definition)
    w_sum = sum(pw_map.values())
    semantic_cr = sum(pw_map[f] * flow_cr[f] for f in pw_map) / max(w_sum, 1e-9)

    # critical (L0-L2, pw >= 0.3) vs best-effort breakdown
    crit = [cr for fid, cr in flow_cr.items() if pw_map[fid] >= 0.30]
    be = [cr for fid, cr in flow_cr.items() if pw_map[fid] < 0.30]
    critical_cr = float(np.mean(crit)) if crit else 0.0
    be_cr = float(np.mean(be)) if be else 0.0

    arr = np.array(all_delays) if all_delays else np.array([0.0])
    return {
        "completion_rate": float(np.mean(list(flow_cr.values()))),   # equal-weight task completion
        "completion_rate_semantic": float(semantic_cr),              # semantic-weighted
        "completion_rate_critical": float(critical_cr),              # L0-L2 only
        "completion_rate_be": float(be_cr),                          # L3 only
        "admission_rate": float(np.mean([1.0 if admitted_eps.get(fid, 0) > 0 else 0.0 for fid in pw_map])),
        "p50_us": float(np.percentile(arr, 50)), "p99_us": float(np.percentile(arr, 99)),
        "p999_us": float(np.percentile(arr, 99.9)), "mean_us": float(np.mean(arr)),
        "max_us": float(np.max(arr)), "min_us": float(np.min(arr)),
        "n_e2e_samples": len(all_delays),
        "packets_sent": int(sum(total_sent.values())),
        "n_episodes": n_episodes,
        "per_flow_completion": flow_cr,
    }


def train_agent(env, shield, cm, steps, seed=42, nc_interval=1000):
    """Train CSRLAgent with the safety shield active and λ in the loop.

    Uses ``agent.train()`` so that (a) the shield filters actions during
    training, and (b) periodic schedulability validation drives the
    Lagrangian dual ascent, with the updated λ pushed into the reward.
    """
    agent = CSRLAgent(env=env, constraint_manager=cm, safety_shield=shield, seed=seed)
    t0 = time.time()
    agent.train(
        total_timesteps=steps,
        nc_validation_interval=nc_interval,
        log_interval=max(steps // 5, 1),
    )
    return agent, time.time() - t0


class PW:
    def __init__(self, a): self._a = a
    def predict(self, o, deterministic=True): return self._a.model.predict(o, deterministic=deterministic)


# ─────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────

def _make_base_config(args):
    return dict(n_switches=args.switches, max_active_flows=args.flows,
                hyperperiod_us=args.hyperperiod, link_rate_mbps=args.rate,
                frame_size_bytes=256.0, seed=args.seed)


def _make_shield(args, topo):
    return SafetyShield(topo, args.rate, enabled=True,
                        hyperperiod_us=args.hyperperiod, frame_size_bytes=256.0, guard_band_us=5.0)


def _print_flows(flows):
    for f in flows:
        print(f"  {f.flow_id:>12s}  pw={f.priority_weight:.2f}  "
              f"deadline={f.delayable_boundary_us:>8.0f}us  class={f.stream_class.name}")


def _print_summary(results):
    hdr = f"{'Method':<20s} {'Comp':>7s} {'Sem':>7s} {'Crit':>7s} {'BE':>6s} {'Adm':>6s} {'Time':>8s}"
    print(hdr); print("-" * len(hdr))
    for nm, r in results.items():
        t = r.get("train_time_s", r.get("prep_time_s", 0))
        print(f"{nm:<20s} {r['completion_rate']:>7.3f} {r['completion_rate_semantic']:>7.3f} "
              f"{r['completion_rate_critical']:>7.3f} {r['completion_rate_be']:>6.3f} "
              f"{r['admission_rate']:>6.3f} {t:>8.1f}")


def _p(r, name):
    print(f"  comp={r['completion_rate']:.3f}  sem={r['completion_rate_semantic']:.3f}  "
          f"crit={r['completion_rate_critical']:.3f}  be={r['completion_rate_be']:.3f}  "
          f"adm={r['admission_rate']:.3f}  p99={r['p99_us']:.0f}us  "
          f"samples={r['n_e2e_samples']}")


# ─────────────────────────────────────────────────────────
# Mode: default — original 5-flow comparison
# ─────────────────────────────────────────────────────────

def run(args):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(os.path.dirname(__file__), f"run_{ts}")
    os.makedirs(outdir, exist_ok=True)

    flows = generate_random_flows(n_flows=args.flows, seed=args.seed)
    base = _make_base_config(args)
    cfg = CSRLConfig(**base)
    topo = make_line_topology(args.switches, args.rate)

    print(f"\n{'='*70}")
    print(f"Mode: default | {args.flows} flows | {args.switches} switches | {args.rate}Mbps")
    print(f"Training: {args.train} steps | Eval: {args.eval} episodes × 100 steps")
    print(f"Output: {outdir}\n")
    _print_flows(flows)

    results = {}
    mk = lambda c=None: TSNEnv(config=c or cfg, topology=topo, flows=flows)

    # B1: Static GCL
    print("\n── B1: Static GCL ──")
    t0 = time.time()
    b1 = StaticGCLScheduler(flows=flows, config=cfg, topology=topo)
    r = evaluate(mk(), b1, args.eval)
    r["prep_time_s"] = time.time() - t0; results["B1_StaticGCL"] = r
    _p(r, "B1_StaticGCL")

    # B3: FIFO+CBS
    print("── B3: FIFO+CBS ──")
    b3 = FIFOCBSScheduler(flows=flows, config=cfg, topology=topo)
    r = evaluate(mk(), b3, args.eval); results["B3_FIFOCBS"] = r
    _p(r, "B3_FIFOCBS")

    # B2: Pure DRL
    print(f"── B2: Pure DRL ({args.train} steps) ──")
    cfg_b2 = CSRLConfig(**base, use_safety_shield=False, reward_semantic_scale=0.0)
    cm0 = ConstraintManager(initial_lambda=0.0, lr_lambda=0.0, max_lambda=0.0)
    so = SafetyShield(topo, args.rate, enabled=False)
    a2, t2 = train_agent(mk(cfg_b2), so, cm0, args.train, seed=args.seed)
    r = evaluate(mk(cfg_b2), PW(a2), args.eval); r["train_time_s"] = t2
    results["B2_PureDRL"] = r; _p(r, "B2_PureDRL")

    # CSRL
    print(f"── CSRL ({args.train} steps) ──")
    cm = ConstraintManager(); s1 = _make_shield(args, topo)
    a1, t1 = train_agent(mk(), s1, cm, args.train, seed=args.seed)
    r = evaluate(mk(), PW(a1), args.eval)
    r["train_time_s"] = t1; r["final_lambda"] = cm.value; results["CSRL"] = r
    _p(r, "CSRL")

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump({"config": vars(args), "mode": "default", "results": results}, f, indent=2, default=str)

    print(f"\nResults: {outdir}/results.json\n")
    _print_summary(results)
    return outdir, results


# ─────────────────────────────────────────────────────────
# Mode: large — 12 flows static benchmark
# ─────────────────────────────────────────────────────────

def run_large_scale(args):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(os.path.dirname(__file__), f"run_{ts}")
    os.makedirs(outdir, exist_ok=True)

    flows = generate_random_flows(n_flows=args.flows, seed=args.seed)
    base = _make_base_config(args)
    cfg = CSRLConfig(**base)
    topo = make_line_topology(args.switches, args.rate)

    print(f"\n{'='*70}")
    print(f"Mode: large-scale | {args.flows} flows | {args.switches} switches | {args.rate}Mbps")
    print(f"Training: {args.train} steps | Eval: {args.eval} episodes × 100 steps")
    print(f"Output: {outdir}\n")
    _print_flows(flows)

    results = {}
    mk = lambda c=None: TSNEnv(config=c or cfg, topology=topo, flows=flows)

    # B1: Static GCL
    print("\n── B1: Static GCL ──")
    t0 = time.time()
    b1 = StaticGCLScheduler(flows=flows, config=cfg, topology=topo)
    r = evaluate(mk(), b1, args.eval)
    r["prep_time_s"] = time.time() - t0; results["B1_StaticGCL"] = r
    _p(r, "B1_StaticGCL")

    # B3: FIFO+CBS
    print("── B3: FIFO+CBS ──")
    b3 = FIFOCBSScheduler(flows=flows, config=cfg, topology=topo)
    r = evaluate(mk(), b3, args.eval); results["B3_FIFOCBS"] = r
    _p(r, "B3_FIFOCBS")

    # B2: Pure DRL
    print(f"── B2: Pure DRL ({args.train} steps) ──")
    cfg_b2 = CSRLConfig(**base, use_safety_shield=False, reward_semantic_scale=0.0)
    cm0 = ConstraintManager(initial_lambda=0.0, lr_lambda=0.0, max_lambda=0.0)
    so = SafetyShield(topo, args.rate, enabled=False)
    a2, t2 = train_agent(mk(cfg_b2), so, cm0, args.train, seed=args.seed)
    r = evaluate(mk(cfg_b2), PW(a2), args.eval); r["train_time_s"] = t2
    results["B2_PureDRL"] = r; _p(r, "B2_PureDRL")

    # CSRL
    print(f"── CSRL ({args.train} steps) ──")
    cm = ConstraintManager(); s1 = _make_shield(args, topo)
    a1, t1 = train_agent(mk(), s1, cm, args.train, seed=args.seed)
    r = evaluate(mk(), PW(a1), args.eval)
    r["train_time_s"] = t1; r["final_lambda"] = cm.value; results["CSRL"] = r
    _p(r, "CSRL")

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump({"config": vars(args), "mode": "large", "results": results}, f, indent=2, default=str)

    print(f"\nResults: {outdir}/results.json\n")
    _print_summary(results)
    return outdir, results


# ─────────────────────────────────────────────────────────
# Mode: dynamic — staggered flow arrival
# ─────────────────────────────────────────────────────────

def evaluate_dynamic(env, scheduler, n_episodes: int, steps_per_ep: int,
                     arrival_interval: int, n_base: int) -> dict:
    """Dynamic arrival: start with n_base flows, add remaining every arrival_interval steps.

    Metrics are split by base flows and arriving flows.  Completion is
    measured against the *expected* packet count per flow (HP/period per
    accepted step): a flow that the scheduler keeps rejecting contributes
    zero completions to the denominator — the "new flow 100%" artifact of
    the previous implementation (zero samples forced to 1.0) is gone.
    """
    hp = env.config.hyperperiod_us
    n_total = env.config.max_active_flows
    arriving_ids = {sf.flow_id for sf in env.sim_flows[n_base:n_total]}

    # per-flow aggregates across episodes
    total_expected: dict[str, int] = {}
    total_completed: dict[str, int] = {}
    total_delays: dict[str, list[float]] = {}
    arrival_events: int = 0

    for ep_i in range(n_episodes):
        obs, _ = env.reset()

        # mark base flows accepted, arriving flows rejected initially
        for i, sf in enumerate(env.sim_flows):
            sf.accepted = (i < n_base)

        next_arrival_idx = n_base
        ep_events = 0

        for step in range(steps_per_ep):
            # trigger arrival if time
            if next_arrival_idx < n_total and step > 0 and step % arrival_interval == 0:
                env.sim_flows[next_arrival_idx].accepted = True
                arrival_events += 1
                ep_events += 1
                next_arrival_idx += 1

            act, _ = scheduler.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(act)
            if terminated or truncated:
                break

        # export snapshot (expected packets account for rejection)
        for sf in env.sim_flows:
            fid = sf.flow_id
            n_pkt = max(1, int(round(hp / max(sf.period_us, 1.0))))
            # a flow only produces expected packets while accepted
            expected = steps_per_ep * n_pkt if sf.accepted else 0
            completed = sum(1 for d in sf.e2e_delays if d <= sf.deadline_us)
            total_expected[fid] = total_expected.get(fid, 0) + expected
            total_completed[fid] = total_completed.get(fid, 0) + completed
            total_delays.setdefault(fid, []).extend(list(sf.e2e_delays))

    flow_cr: dict[str, float] = {}
    for sf in env.sim_flows:
        fid = sf.flow_id
        flow_cr[fid] = total_completed.get(fid, 0) / max(total_expected.get(fid, 0), 1)

    base_cr = np.mean([cr for fid, cr in flow_cr.items() if fid not in arriving_ids]) if n_base > 0 else 0.0
    new_cr = np.mean([cr for fid, cr in flow_cr.items() if fid in arriving_ids]) if len(arriving_ids) > 0 else 0.0
    overall_cr = float(np.mean(list(flow_cr.values())))

    all_delays = [d for ds in total_delays.values() for d in ds]
    arr_all = np.array(all_delays) if all_delays else np.array([0.0])
    n_arriving = n_total - n_base

    return {
        "completion_rate": overall_cr,
        "completion_rate_base": float(base_cr),
        "completion_rate_new": float(new_cr),
        "p50_us": float(np.percentile(arr_all, 50)),
        "p99_us": float(np.percentile(arr_all, 99)),
        "n_e2e_samples": len(all_delays),
        "arrival_events": arrival_events,
        "arrival_acceptance_rate": float(new_cr),  # accepted-and-completed ≈ admission effectiveness
        "n_episodes": n_episodes,
        "n_base": n_base,
        "n_arriving": n_arriving,
        "per_flow_completion": flow_cr,
    }


def run_dynamic_arrival(args):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(os.path.dirname(__file__), f"run_{ts}")
    os.makedirs(outdir, exist_ok=True)

    flows = generate_random_flows(n_flows=args.flows, seed=args.seed)
    n_base = 5
    arrival_interval = 20  # steps between arrivals

    base = _make_base_config(args)
    # training environment mirrors the dynamic evaluation distribution:
    # the policy must learn to admit mid-episode arrivals instead of
    # learning a static schedule on the full flow set.
    dyn_base = dict(base, dynamic_arrival=True, n_base_flows=n_base,
                    arrival_interval=arrival_interval)
    cfg = CSRLConfig(**dyn_base)
    topo = make_line_topology(args.switches, args.rate)

    print(f"\n{'='*70}")
    print(f"Mode: dynamic-arrival | {n_base} base + {args.flows - n_base} arriving | {args.switches} switches")
    print(f"Training: {args.train} steps | Eval: {args.eval} episodes × 100 steps | Arrival every {arrival_interval} steps")
    print(f"Output: {outdir}\n")
    _print_flows(flows)

    results = {}
    mk = lambda c=None: TSNEnv(config=c or cfg, topology=topo, flows=flows)

    # B1: Static GCL (always-accept) — the static scheduler only sees the
    # base flows at t=0; arriving flows get no reserved window
    print("\n── B1: Static GCL (base-flows only) ──")
    t0 = time.time()
    b1 = StaticGCLScheduler(flows=flows, config=cfg, topology=topo,
                            static_flows=flows[:n_base])
    r = evaluate_dynamic(mk(), b1, args.eval, steps_per_ep=100,
                         arrival_interval=arrival_interval, n_base=n_base)
    r["prep_time_s"] = time.time() - t0; results["B1_StaticGCL"] = r
    print(f"  comp={r['completion_rate']:.3f}  comp_base={r['completion_rate_base']:.3f}  "
          f"comp_new={r['completion_rate_new']:.3f}  accept_rate={r['arrival_acceptance_rate']:.3f}")

    # B3: FIFO+CBS (always accepts)
    print("── B3: FIFO+CBS (always-accept) ──")
    b3 = FIFOCBSScheduler(flows=flows, config=cfg, topology=topo)
    r = evaluate_dynamic(mk(), b3, args.eval, steps_per_ep=100,
                         arrival_interval=arrival_interval, n_base=n_base)
    results["B3_FIFOCBS"] = r
    print(f"  comp={r['completion_rate']:.3f}  comp_base={r['completion_rate_base']:.3f}  "
          f"comp_new={r['completion_rate_new']:.3f}  accept_rate={r['arrival_acceptance_rate']:.3f}")

    # B2: Pure DRL
    print(f"── B2: Pure DRL ({args.train} steps) ──")
    cfg_b2 = CSRLConfig(**base, use_safety_shield=False, reward_semantic_scale=0.0)
    cm0 = ConstraintManager(initial_lambda=0.0, lr_lambda=0.0, max_lambda=0.0)
    so = SafetyShield(topo, args.rate, enabled=False)
    a2, t2 = train_agent(mk(cfg_b2), so, cm0, args.train, seed=args.seed)
    r = evaluate_dynamic(mk(cfg_b2), PW(a2), args.eval, steps_per_ep=100,
                         arrival_interval=arrival_interval, n_base=n_base)
    r["train_time_s"] = t2; results["B2_PureDRL"] = r
    print(f"  comp={r['completion_rate']:.3f}  comp_base={r['completion_rate_base']:.3f}  "
          f"comp_new={r['completion_rate_new']:.3f}  accept_rate={r['arrival_acceptance_rate']:.3f}")

    # CSRL
    print(f"── CSRL ({args.train} steps) ──")
    cm = ConstraintManager(); s1 = _make_shield(args, topo)
    a1, t1 = train_agent(mk(), s1, cm, args.train, seed=args.seed)
    r = evaluate_dynamic(mk(), PW(a1), args.eval, steps_per_ep=100,
                         arrival_interval=arrival_interval, n_base=n_base)
    r["train_time_s"] = t1; r["final_lambda"] = cm.value; results["CSRL"] = r
    print(f"  comp={r['completion_rate']:.3f}  comp_base={r['completion_rate_base']:.3f}  "
          f"comp_new={r['completion_rate_new']:.3f}  accept_rate={r['arrival_acceptance_rate']:.3f}")

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump({"config": vars(args), "mode": "dynamic", "n_base": n_base,
                    "arrival_interval": arrival_interval, "results": results}, f, indent=2, default=str)

    print(f"\nResults: {outdir}/results.json\n")
    print(f"{'Method':<20s} {'Comp':>7s} {'Base':>7s} {'New':>7s} {'Accept':>7s} {'P99':>8s} {'Time':>8s}")
    print("-" * 78)
    for nm, r in results.items():
        t = r.get("train_time_s", r.get("prep_time_s", 0))
        print(f"{nm:<20s} {r['completion_rate']:>7.3f} {r['completion_rate_base']:>7.3f} "
              f"{r['completion_rate_new']:>7.3f} {r['arrival_acceptance_rate']:>7.3f} "
              f"{r['p99_us']:>8.0f} {t:>8.1f}")

    return outdir, results


# ─────────────────────────────────────────────────────────
# Mode: ablation — 4 configs on 8 flows
# ─────────────────────────────────────────────────────────

def run_ablation(args):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(os.path.dirname(__file__), f"run_{ts}")
    os.makedirs(outdir, exist_ok=True)

    flows = generate_random_flows(n_flows=args.flows, seed=args.seed)
    base = _make_base_config(args)
    topo = make_line_topology(args.switches, args.rate)

    print(f"\n{'='*70}")
    print(f"Mode: ablation | {args.flows} flows | {args.switches} switches | {args.rate}Mbps")
    print(f"Training: {args.train} steps each | Eval: {args.eval} episodes × 100 steps")
    print(f"Output: {outdir}\n")
    _print_flows(flows)

    results = {}

    def _train_eval(name, config_kwargs, cm_kwargs, shield_enabled):
        print(f"── {name} ({args.train} steps) ──")
        cfg = CSRLConfig(**{**base, **config_kwargs})
        shield = _make_shield(args, topo) if shield_enabled else SafetyShield(topo, args.rate, enabled=False)
        cm = ConstraintManager(**cm_kwargs)
        env = TSNEnv(config=cfg, topology=topo, flows=flows)
        agent, t = train_agent(env, shield, cm, args.train, seed=args.seed)
        eval_env = TSNEnv(config=cfg, topology=topo, flows=flows)
        r = evaluate(eval_env, PW(agent), args.eval)
        r["train_time_s"] = t
        r["final_lambda"] = cm.value
        return r

    # 1: Full CSRL
    r = _train_eval(
        "Full_CSRL",
        dict(use_safety_shield=True, reward_semantic_scale=1.0),
        dict(initial_lambda=0.1, lr_lambda=0.002, max_lambda=1.0),
        shield_enabled=True,
    )
    results["Full_CSRL"] = r; _p(r, "Full_CSRL")

    # 2: No Safety Shield
    r = _train_eval(
        "No_Safety_Shield",
        dict(use_safety_shield=False, reward_semantic_scale=1.0),
        dict(initial_lambda=0.1, lr_lambda=0.002, max_lambda=1.0),
        shield_enabled=False,
    )
    results["No_Safety_Shield"] = r; _p(r, "No_Safety_Shield")

    # 3: No Semantic
    r = _train_eval(
        "No_Semantic",
        dict(use_safety_shield=True, reward_semantic_scale=0.0),
        dict(initial_lambda=0.1, lr_lambda=0.002, max_lambda=1.0),
        shield_enabled=True,
    )
    results["No_Semantic"] = r; _p(r, "No_Semantic")

    # 4: No NC Constraint
    r = _train_eval(
        "No_NC_Constraint",
        dict(use_safety_shield=True, reward_semantic_scale=1.0),
        dict(initial_lambda=0.0, lr_lambda=0.0, max_lambda=0.0),
        shield_enabled=True,
    )
    results["No_NC_Constraint"] = r; _p(r, "No_NC_Constraint")

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump({"config": vars(args), "mode": "ablation", "results": results}, f, indent=2, default=str)

    print(f"\nResults: {outdir}/results.json\n")
    _print_summary(results)
    return outdir, results


def run_scarcity(args):
    """Scarce-resource mode: 10 same-period ST flows on one switch port,
    tx=12μs, deadline=100μs — at most 8 of 10 can meet their deadline, so
    who fails is a forced choice that exposes semantic weighting."""
    from src.csrl.train import generate_scarcity_flows

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(os.path.dirname(__file__), f"run_{ts}")
    os.makedirs(outdir, exist_ok=True)

    flows = generate_scarcity_flows(n_be=2, seed=args.seed)
    base = dict(n_switches=3, max_active_flows=12, hyperperiod_us=args.hyperperiod,
                link_rate_mbps=args.rate, frame_size_bytes=1500.0, seed=args.seed,
                deadline_multiplier=0.2, shared_st_queue=True, st_window_us=100.0,
                single_switch=True)
    topo = make_line_topology(args.switches, args.rate)

    print(f"\n{'='*70}")
    print(f"Mode: scarcity | 10 ST same-period + 2 BE | 1 switch port | tx=12μs | deadline=100μs")
    print(f"Training: {args.train} steps | Eval: {args.eval} episodes\n")
    _print_flows(flows)

    results = {}

    for name, cls, kw in [("B1_StaticGCL", StaticGCLScheduler, {}),
                          ("B3_FIFOCBS", FIFOCBSScheduler, {})]:
        print(f"\n── {name} ──")
        env = TSNEnv(config=CSRLConfig(**base), topology=topo, flows=flows)
        sched = cls(flows=flows, config=CSRLConfig(**base), topology=topo, **kw)
        r = evaluate(env, sched, args.eval)
        results[name] = r
        _p(r, name)

    for name, kw, cm_kw, shield_on in [
        ("B2_PureDRL", dict(reward_semantic_scale=0.0),
         dict(initial_lambda=0.0, lr_lambda=0.0, max_lambda=0.0), False),
        ("CSRL", dict(reward_semantic_scale=1.0),
         dict(initial_lambda=0.1, lr_lambda=0.002, max_lambda=1.0), True),
    ]:
        print(f"\n── {name} ({args.train} steps) ──")
        c = CSRLConfig(**{**base, **kw})
        env = TSNEnv(config=c, topology=topo, flows=flows)
        shield = SafetyShield(topo, args.rate, enabled=shield_on,
                              hyperperiod_us=args.hyperperiod, frame_size_bytes=1500.0)
        cm = ConstraintManager(**cm_kw)
        agent, t = train_agent(env, shield, cm, args.train, seed=args.seed)
        r = evaluate(TSNEnv(config=c, topology=topo, flows=flows), PW(agent), args.eval)
        r["train_time_s"] = t
        r["final_lambda"] = cm.value
        results[name] = r
        _p(r, name)

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump({"config": vars(args), "mode": "scarcity", "results": results},
                  f, indent=2, default=str)
    print(f"\nResults: {outdir}/results.json")
    _print_summary(results)
    return outdir, results


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["default", "large", "dynamic", "ablation", "scarcity"], default="default")
    p.add_argument("--flows", type=int, default=5)
    p.add_argument("--switches", type=int, default=3)
    p.add_argument("--rate", type=float, default=1000.0)
    p.add_argument("--hyperperiod", type=float, default=10000.0)
    p.add_argument("--train", type=int, default=60000)
    p.add_argument("--eval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    mode_fn = {
        "default": run,
        "large": run_large_scale,
        "dynamic": run_dynamic_arrival,
        "ablation": run_ablation,
        "scarcity": run_scarcity,
    }
    mode_fn[args.mode](args)
