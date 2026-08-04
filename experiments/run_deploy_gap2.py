#!/usr/bin/env python3
"""overlap penalty scale=20: penalty exceeds completion reward, forcing the
policy to learn mutual exclusion during training.

Run:  python3 experiments/run_deploy_gap2.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.csrl.train import generate_scarcity_flows
from src.csrl.tsn_env import CSRLConfig, TSNEnv
from src.csrl.csrl_agent import CSRLAgent, ConstraintManager
from src.csrl.safety_shield import SafetyShield
from src.nc_engine.topology import make_line_topology
from src.intent_ontology.types import StreamClass
from experiments.run_deploy_gap import make_entries, deploy_eval


def main():
    for seed in [42, 123]:
        flows = generate_scarcity_flows(n_be=0, seed=seed)
        st_flows = [f for f in flows if f.stream_class == StreamClass.SCHEDULED_TRAFFIC][:6]
        base = dict(n_switches=3, max_active_flows=6, hyperperiod_us=10000.0,
                    link_rate_mbps=1000.0, frame_size_bytes=256.0, seed=42,
                    deadline_multiplier=0.2, shared_st_queue=True, st_window_us=100.0,
                    overlap_penalty_scale=20.0)
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
            print(f"seed={seed} {name} scale20: comp={comp:.3f} crit={crit:.3f} 拦截={nw} "
                  f"λ={cm.value:.2f} t={time.time()-t0:.0f}s", flush=True)
            print(f"   {fcr}", flush=True)


if __name__ == "__main__":
    main()
