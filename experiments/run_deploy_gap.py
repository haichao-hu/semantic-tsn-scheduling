#!/usr/bin/env python3
"""deadline<period deployment-gap experiment: training with mutual-exclusion
penalty (overlap_penalty_scale) then deployment evaluation with the Safety
Shield active.  Tests whether semantic weighting learns to protect critical
flows from window overlap (which the shield vetoes at deployment)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.csrl.train import generate_scarcity_flows
from src.csrl.tsn_env import CSRLConfig, TSNEnv
from src.csrl.csrl_agent import CSRLAgent, ConstraintManager
from src.csrl.safety_shield import SafetyShield
from src.nc_engine.topology import make_line_topology
from src.nc_engine.schedulability import SchedEntry
from src.intent_ontology.types import StreamClass


def make_entries(env):
    ents = {}
    for sf in env.sim_flows:
        ents[sf.flow_id] = SchedEntry(
            flow_id=sf.flow_id, queue=sf.queue,
            gate_start_us=sf.gate_start_us, gate_size_us=sf.gate_size_us,
            period_us=sf.period_us, deadline_us=sf.deadline_us,
            path=list(sf.path), priority_weight=sf.priority_weight,
            dispatch_phase_us=sf.phase_offset_us)
    return ents


def deploy_eval(agent, st_flows, base, topo, n_steps=100):
    env = TSNEnv(config=CSRLConfig(**base), topology=topo, flows=st_flows)
    shield = SafetyShield(topo, 1000.0, enabled=True, hyperperiod_us=10000.0)
    obs, _ = env.reset()
    for _ in range(n_steps):
        act, _ = agent.predict(obs, deterministic=True)
        decoded = env._decode_action(act)
        safe = shield.filter_action(decoded, st_flows, None, None, make_entries(env))
        for sf in env.sim_flows:
            d = safe.get(sf.flow_id)
            if d is None:
                continue
            sf.accepted = d["accept"]
            sf.phase_offset_us = d["dispatch_offset_us"]
            sf.gate_start_us = d["gate_start_us"]
            sf.gate_size_us = d["gate_size_us"]
        env._simulate_hyperperiod()
        obs = env._get_obs()
    total = crit = 0
    flow_cr = {}
    for sf in env.sim_flows:
        n_pkt = max(1, round(10000.0 / max(sf.period_us, 1)))
        exp = n_steps * n_pkt
        ok = sum(1 for d in sf.e2e_delays if d <= sf.deadline_us)
        cr = ok / exp
        flow_cr[sf.flow_id] = round(cr, 2)
        total += cr
        if sf.priority_weight >= 0.3:
            crit += cr
    return total / 6, crit / 5, flow_cr, len(shield.warnings)


def main():
    for seed in [42, 123, 456]:
        flows = generate_scarcity_flows(n_be=0, seed=seed)
        st_flows = [f for f in flows if f.stream_class == StreamClass.SCHEDULED_TRAFFIC][:6]
        base = dict(n_switches=3, max_active_flows=6, hyperperiod_us=10000.0,
                    link_rate_mbps=1000.0, frame_size_bytes=256.0, seed=42,
                    deadline_multiplier=0.2, shared_st_queue=True, st_window_us=100.0,
                    overlap_penalty_scale=0.5)
        topo = make_line_topology(3, 1000.0)

        for name, kw, cm_kw, shield_on in [
            ("CSRL", dict(reward_semantic_scale=1.0),
             dict(initial_lambda=0.1, lr_lambda=0.002, max_lambda=1.0), True),
            ("PureDRL", dict(reward_semantic_scale=0.0),
             dict(initial_lambda=0.0, lr_lambda=0.0, max_lambda=0.0), False),
        ]:
            c = CSRLConfig(**{**base, **kw})
            env = TSNEnv(config=c, topology=topo, flows=st_flows)
            shield = SafetyShield(topo, 1000.0, enabled=shield_on, hyperperiod_us=10000.0)
            cm = ConstraintManager(**cm_kw)
            agent = CSRLAgent(env=env, constraint_manager=cm, safety_shield=shield, seed=42,
                              ppo_kwargs=dict(ent_coef=0.03, n_steps=512))
            t0 = time.time()
            agent.train(total_timesteps=120000, nc_validation_interval=12000,
                        log_interval=120000, warmup_ratio=0.4)
            comp, crit, fcr, nw = deploy_eval(agent, st_flows, base, topo)
            print(f"seed={seed} {name}: comp={comp:.3f} crit={crit:.3f} 拦截={nw} "
                  f"λ={cm.value:.2f} t={time.time()-t0:.0f}s", flush=True)
            print(f"   {fcr}", flush=True)


if __name__ == "__main__":
    main()
