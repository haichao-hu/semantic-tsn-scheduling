from __future__ import annotations

from typing import Any

from src.intent_ontology.types import FlowSemantics
from src.nc_engine.topology import TSNTopology, make_line_topology
from src.csrl.tsn_env import CSRLConfig, TSNEnv
from src.csrl.csrl_agent import CSRLAgent, ConstraintManager
from src.csrl.safety_shield import SafetyShield
from src.csrl.train import load_scenarios_from_ontology

from .runner import _compute_percentiles, _nc_validate_current_schedule


def _evaluate_agent(
    agent: CSRLAgent,
    env: TSNEnv,
    flows: list[FlowSemantics],
    topology: TSNTopology | None,
    link_rate_mbps: float,
    n_episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Run evaluation episodes and return metrics dict."""
    all_e2e: list[float] = []
    all_jitter: list[float] = []
    total_wcd_violations = 0
    deadline_checks = 0
    deadline_violations = 0

    for ep in range(n_episodes):
        try:
            obs, _ = env.reset(seed=seed + ep)
        except Exception:
            continue

        for _ in range(50):
            try:
                action, _ = agent.predict(obs, deterministic=True)
            except Exception:
                break
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception:
                break
            if terminated or truncated:
                break

        for sf in env.sim_flows:
            all_e2e.extend(sf.e2e_delays)
            all_jitter.extend(sf.jitter_samples)
            for d in sf.e2e_delays:
                deadline_checks += 1
                if d > sf.deadline_us:
                    deadline_violations += 1

        wcd_viols = _nc_validate_current_schedule(env, flows, topology, link_rate_mbps)
        total_wcd_violations += max(0, wcd_viols)

    completion_rate = (
        1.0 - deadline_violations / max(deadline_checks, 1)
        if deadline_checks > 0
        else 0.0
    )
    p50, p99, p999 = _compute_percentiles(all_e2e)
    avg_jitter = float(sum(all_jitter) / max(len(all_jitter), 1)) if all_jitter else 0.0

    return {
        "completion_rate": round(completion_rate, 6),
        "p50_delay_us": p50,
        "p99_delay_us": p99,
        "p999_delay_us": p999,
        "wcd_violations": total_wcd_violations,
        "avg_jitter_us": avg_jitter,
        "deadline_violations": deadline_violations,
        "n_e2e_samples": len(all_e2e),
        "total_flows": len(flows),
    }


def run_ablation(
    scenario: str = "agv_fleet",
    n_episodes: int = 50,
    n_switches: int = 3,
    link_rate_mbps: float = 1000.0,
    seed: int = 42,
) -> dict[str, dict]:
    """Run ablation study: Full CSRL vs. components removed.

    Compares four configurations:
      1. **Full CSRL** — semantic + safety shield + NC constraint (λ>0)
      2. **No Safety Shield** — ``use_safety_shield=False``
      3. **No Semantic** — ``reward_semantic_scale=0``, uniform reward
      4. **No NC Constraint** — ConstraintManager with ``lr_lambda=0`` (λ≡0)

    Parameters
    ----------
    scenario : str
        Scenario key.
    n_episodes : int
        Evaluation episodes per ablation variant.
    n_switches, link_rate_mbps : topology parameters.
    seed : int
        RNG seed.

    Returns
    -------
    dict
        ``{variant_key: metrics_dict}``
    """
    if scenario not in {"agv_fleet", "cobot", "plc"}:
        return {"error": f"Unknown scenario: {scenario}"}

    all_scenarios = load_scenarios_from_ontology()
    flows = list(all_scenarios[scenario])
    topology = make_line_topology(n_switches, link_rate_mbps)

    results: dict[str, dict] = {}

    # ---- 1. Full CSRL -------------------------------------------------
    print("\n[Ablation 1/4] Full CSRL (semantic + shield + NC)")

    try:
        cfg1 = CSRLConfig(
            n_switches=n_switches,
            n_ports_per_switch=4,
            n_queues=8,
            max_active_flows=max(len(flows), 1),
            hyperperiod_us=10000.0,
            link_rate_mbps=link_rate_mbps,
            frame_size_bytes=256.0,
            seed=seed,
            use_safety_shield=True,
            reward_semantic_scale=1.0,
        )
        train_env1 = TSNEnv(config=cfg1, topology=topology, flows=flows)
        shield1 = SafetyShield(topology=topology, link_rate_mbps=link_rate_mbps, enabled=True)
        cm1 = ConstraintManager(initial_lambda=0.1, lr_lambda=0.002, max_lambda=1.0)
        agent1 = CSRLAgent(env=train_env1, constraint_manager=cm1, safety_shield=shield1, device="cpu")
        agent1.train(total_timesteps=5000, nc_validation_interval=1000, log_interval=1000)

        eval_env1 = TSNEnv(config=cfg1, topology=topology, flows=flows)
        results["Full_CSRL"] = _evaluate_agent(agent1, eval_env1, flows, topology, link_rate_mbps, n_episodes, seed)
        results["Full_CSRL"]["final_lambda"] = cm1.value
        print(f"    completion_rate={results['Full_CSRL']['completion_rate']:.4f}")
    except Exception as exc:
        results["Full_CSRL"] = {"error": str(exc)}
        print(f"    ERROR: {exc}")

    # ---- 2. No Safety Shield ------------------------------------------
    print("\n[Ablation 2/4] No Safety Shield")

    try:
        cfg2 = CSRLConfig(
            n_switches=n_switches,
            n_ports_per_switch=4,
            n_queues=8,
            max_active_flows=max(len(flows), 1),
            hyperperiod_us=10000.0,
            link_rate_mbps=link_rate_mbps,
            frame_size_bytes=256.0,
            seed=seed,
            use_safety_shield=False,
            reward_semantic_scale=1.0,
        )
        train_env2 = TSNEnv(config=cfg2, topology=topology, flows=flows)
        cm2 = ConstraintManager(initial_lambda=0.1, lr_lambda=0.002, max_lambda=1.0)
        agent2 = CSRLAgent(env=train_env2, constraint_manager=cm2, safety_shield=None, device="cpu")
        agent2.train(total_timesteps=5000, nc_validation_interval=1000, log_interval=1000)

        eval_env2 = TSNEnv(config=cfg2, topology=topology, flows=flows)
        results["No_Safety_Shield"] = _evaluate_agent(agent2, eval_env2, flows, topology, link_rate_mbps, n_episodes, seed)
        results["No_Safety_Shield"]["final_lambda"] = cm2.value
        print(f"    completion_rate={results['No_Safety_Shield']['completion_rate']:.4f}")
    except Exception as exc:
        results["No_Safety_Shield"] = {"error": str(exc)}
        print(f"    ERROR: {exc}")

    # ---- 3. No Semantic ------------------------------------------------
    print("\n[Ablation 3/4] No Semantic (uniform priority)")

    try:
        cfg3 = CSRLConfig(
            n_switches=n_switches,
            n_ports_per_switch=4,
            n_queues=8,
            max_active_flows=max(len(flows), 1),
            hyperperiod_us=10000.0,
            link_rate_mbps=link_rate_mbps,
            frame_size_bytes=256.0,
            seed=seed,
            use_safety_shield=True,
            reward_semantic_scale=0.0,
        )
        train_env3 = TSNEnv(config=cfg3, topology=topology, flows=flows)
        shield3 = SafetyShield(topology=topology, link_rate_mbps=link_rate_mbps, enabled=True)
        cm3 = ConstraintManager(initial_lambda=0.1, lr_lambda=0.002, max_lambda=1.0)
        agent3 = CSRLAgent(env=train_env3, constraint_manager=cm3, safety_shield=shield3, device="cpu")
        agent3.train(total_timesteps=5000, nc_validation_interval=1000, log_interval=1000)

        eval_env3 = TSNEnv(config=cfg3, topology=topology, flows=flows)
        results["No_Semantic"] = _evaluate_agent(agent3, eval_env3, flows, topology, link_rate_mbps, n_episodes, seed)
        results["No_Semantic"]["final_lambda"] = cm3.value
        print(f"    completion_rate={results['No_Semantic']['completion_rate']:.4f}")
    except Exception as exc:
        results["No_Semantic"] = {"error": str(exc)}
        print(f"    ERROR: {exc}")

    # ---- 4. No NC Constraint -------------------------------------------
    print("\n[Ablation 4/4] No NC Constraint (λ≡0)")

    try:
        cfg4 = CSRLConfig(
            n_switches=n_switches,
            n_ports_per_switch=4,
            n_queues=8,
            max_active_flows=max(len(flows), 1),
            hyperperiod_us=10000.0,
            link_rate_mbps=link_rate_mbps,
            frame_size_bytes=256.0,
            seed=seed,
            use_safety_shield=True,
            reward_semantic_scale=1.0,
        )
        train_env4 = TSNEnv(config=cfg4, topology=topology, flows=flows)
        shield4 = SafetyShield(topology=topology, link_rate_mbps=link_rate_mbps, enabled=True)
        cm4 = ConstraintManager(initial_lambda=0.0, lr_lambda=0.0, max_lambda=0.0)
        agent4 = CSRLAgent(env=train_env4, constraint_manager=cm4, safety_shield=shield4, device="cpu")
        agent4.train(total_timesteps=5000, nc_validation_interval=1000, log_interval=1000)

        eval_env4 = TSNEnv(config=cfg4, topology=topology, flows=flows)
        results["No_NC_Constraint"] = _evaluate_agent(agent4, eval_env4, flows, topology, link_rate_mbps, n_episodes, seed)
        results["No_NC_Constraint"]["final_lambda"] = cm4.value
        print(f"    completion_rate={results['No_NC_Constraint']['completion_rate']:.4f}")
    except Exception as exc:
        results["No_NC_Constraint"] = {"error": str(exc)}
        print(f"    ERROR: {exc}")

    return results
