from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import numpy as np

from src.intent_ontology.types import (
    CriticalityLevel,
    DecayType,
    FlowSemantics,
    SemanticCompressibility,
    StreamClass,
    TaskIntent,
    UrgencyFunction,
)
from src.intent_ontology.encoder import IntentEncoder
from src.intent_ontology.mapper import QoSMapper
from src.intent_ontology.examples import (
    agv_fleet_scenario,
    cobot_scenario,
    plc_scenario,
)
from src.nc_engine.topology import (
    TSNTopology,
    make_ieee_60802_topology,
    make_line_topology,
    make_ring_topology,
)

from .tsn_env import CSRLConfig, TSNEnv
from .csrl_agent import CSRLAgent, ConstraintManager
from .safety_shield import SafetyShield


# ============================================================
# Scenario loading
# ============================================================


def generate_scarcity_flows(n_be: int = 3, seed: int = 42, shuffle_st: bool = True) -> list[FlowSemantics]:
    """Generate a *scarce* flow set: more ST flows than window slots.

    Ten ST flows share the same period (500 μs) and — with the
    single-switch scarce configuration — the same egress port.  With
    1500-byte frames (12 μs transmission) and a 100 μs deadline, at most
    eight flows can transmit back-to-back within one deadline window;
    the rest are forced to fail.  The flow order is shuffled so that the
    victim is NOT determined by list order: a semantic-aware scheduler
    must actively protect high-criticality flows.

    Combine with ``shared_st_queue=True``, ``st_window_us=100``,
    ``deadline_multiplier=0.2`` and ``frame_size_bytes=1500`` in
    CSRLConfig, plus all flows routed through one switch.
    """
    rng = np.random.RandomState(seed)
    profiles = [
        ("ST_L0", 0.98, StreamClass.SCHEDULED_TRAFFIC),
        ("ST_L1a", 0.85, StreamClass.SCHEDULED_TRAFFIC),
        ("ST_L1b", 0.75, StreamClass.SCHEDULED_TRAFFIC),
        ("ST_L2a", 0.50, StreamClass.SCHEDULED_TRAFFIC),
        ("ST_L2b", 0.45, StreamClass.SCHEDULED_TRAFFIC),
        ("ST_L2c", 0.40, StreamClass.SCHEDULED_TRAFFIC),
        ("ST_L2d", 0.35, StreamClass.SCHEDULED_TRAFFIC),
        ("ST_L2e", 0.30, StreamClass.SCHEDULED_TRAFFIC),
        ("ST_L3a", 0.25, StreamClass.SCHEDULED_TRAFFIC),
        ("ST_L3b", 0.20, StreamClass.SCHEDULED_TRAFFIC),
    ]
    if shuffle_st:
        rng.shuffle(profiles)
    flows: list[FlowSemantics] = []
    for i, (name, pw, sc) in enumerate(profiles):
        jitter = rng.uniform(0.01, 0.04)
        flows.append(FlowSemantics(
            flow_id=f"sc_{name}",
            task_id=f"task_{name}",
            priority_weight=min(1.0, max(0.0, pw + rng.uniform(-jitter, jitter))),
            delayable_boundary_us=500,  # same period → same window grid
            urgency=UrgencyFunction(DecayType.STEP, value_plateau_us=0, decay_start_us=0),
            compressibility=SemanticCompressibility(ratio=0.0),
            stream_class=sc,
        ))
    for i in range(n_be):
        flows.append(FlowSemantics(
            flow_id=f"sc_BE_{i}",
            task_id=f"task_BE_{i}",
            priority_weight=0.10 + 0.03 * rng.uniform(0, 1),
            delayable_boundary_us=10000,
            urgency=UrgencyFunction(DecayType.EXPONENTIAL, value_plateau_us=0,
                                    decay_start_us=0, decay_rate=1.0),
            compressibility=SemanticCompressibility(ratio=0.0),
            stream_class=StreamClass.BEST_EFFORT,
        ))
    return flows


def load_scenarios_from_ontology() -> dict[str, list[FlowSemantics]]:
    """Load predefined TSN scenarios from the intent ontology.

    Returns a dict of scenario_name → list of FlowSemantics.
    """
    encoder = IntentEncoder()
    mapper = QoSMapper()

    scenarios: dict[str, list[FlowSemantics]] = {}

    for name, scenario_fn in [
        ("agv_fleet", agv_fleet_scenario),
        ("cobot", cobot_scenario),
        ("plc", plc_scenario),
    ]:
        try:
            intents = scenario_fn()
            flows: list[FlowSemantics] = []
            for intent in intents:
                fs = encoder.encode(intent)
                flows.append(fs)
            scenarios[name] = flows
        except Exception as e:
            print(f"Warning: failed to load scenario '{name}': {e}")

    return scenarios


def generate_random_flows(n_flows: int = 8, seed: int = 42) -> list[FlowSemantics]:
    """Generate random flows with realistic TSN parameters from IEC 60802 profiles.

    All deadlines/periods are within one hyperperiod (10 ms), so that gate
    windows repeat within the hyperperiod and every flow is schedulable in
    principle — the completion metric is then a true measure of scheduling
    quality, not an artifact of flows with multi-hyperperiod deadlines.
    """
    rng = np.random.RandomState(seed)

    profiles = [
        # (name, priority, deadline_us, stream_class, is_periodic)
        # L0 - Safety
        ("SAFETY", 0.98, 200, StreamClass.SCHEDULED_TRAFFIC),
        # L1 - Mission Critical
        ("MISSION_HARD", 0.85, 500, StreamClass.SCHEDULED_TRAFFIC),
        ("MISSION_SOFT", 0.75, 1000, StreamClass.SCHEDULED_TRAFFIC),
        # L2 - Operational
        ("OPER_AVB", 0.50, 5000, StreamClass.RESERVED),
        ("OPER_INSPECT", 0.40, 10000, StreamClass.RESERVED),
        # L3 - Best Effort
        ("BE_TELEM", 0.20, 10000, StreamClass.BEST_EFFORT),
        ("BE_HMI", 0.15, 10000, StreamClass.BEST_EFFORT),
    ]

    flows: list[FlowSemantics] = []
    for i in range(n_flows):
        idx = rng.randint(len(profiles))
        name, pw, deadline, sc = profiles[idx]
        jitter = rng.uniform(0.01, 0.15)
        pw_jittered = min(1.0, max(0.0, pw + rng.uniform(-jitter, jitter)))

        decay_type = DecayType.STEP if pw_jittered > 0.8 else (
            DecayType.LINEAR if pw_jittered > 0.3 else DecayType.EXPONENTIAL
        )

        flows.append(FlowSemantics(
            flow_id=f"f_rand_{i:03d}",
            task_id=f"task_rand_{i:03d}",
            priority_weight=pw_jittered,
            delayable_boundary_us=int(deadline * rng.uniform(0.8, 1.2)),
            urgency=UrgencyFunction(
                decay_type=decay_type,
                value_plateau_us=int(deadline * 0.3),
                decay_start_us=int(deadline * 0.3),
                decay_rate=rng.uniform(0.5, 2.0),
            ),
            compressibility=SemanticCompressibility(
                ratio=rng.uniform(0.0, 0.4),
            ),
            stream_class=sc,
            preemption_eligible=sc != StreamClass.SCHEDULED_TRAFFIC,
        ))

    return flows


# ============================================================
# Training utilities
# ============================================================


def _log_metrics(step: int, stats: dict, scenario: str) -> None:
    """Print training metrics."""
    rewards = stats.get("rewards", [])
    violations = stats.get("wcd_violations", [])
    lambdas = stats.get("lambda_evolution", [])

    avg_reward = np.mean(rewards[-10:]) if rewards else 0.0
    avg_viol = np.mean(violations[-10:]) if violations else 0.0
    current_lambda = lambdas[-1] if lambdas else 0.0

    print(
        f"[{scenario}] step={step:6d} | "
        f"avg_reward={avg_reward:7.2f} | "
        f"avg_violations={avg_viol:4.1f} | "
        f"lambda={current_lambda:.4f}"
    )


# ============================================================
# Main training function
# ============================================================


def train(
    scenario: str = "agv_fleet",
    topology_type: str = "line",
    total_timesteps: int = 10000,
    nc_validation_interval: int = 1000,
    use_safety_shield: bool = True,
    ckpt_dir: str = "./checkpoints",
    seed: int = 42,
    n_switches: int = 3,
    link_rate_mbps: float = 1000.0,
    log_interval: int = 500,
    save_model: bool = True,
) -> dict:
    """End-to-end CSRL training pipeline.

    Parameters
    ----------
    scenario : str
        One of "agv_fleet", "cobot", "plc", or "random".
    topology_type : str
        "line", "ring", or "iec60802".
    total_timesteps : int
        Total training steps.
    nc_validation_interval : int
        Run NC engine validation every N steps.
    use_safety_shield : bool
        Enable the Safety Shield for action filtering.
    ckpt_dir : str
        Directory to save model checkpoints.
    seed : int
        Random seed.
    n_switches : int
        Number of TSN switches (line topology only).
    link_rate_mbps : float
        Link rate in Mbps.

    Returns
    -------
    stats : dict with training metrics.
    """
    # topology
    if topology_type == "line":
        topology = make_line_topology(n_switches, link_rate_mbps)
    elif topology_type == "ring":
        topology = make_ring_topology(4, link_rate_mbps)
    elif topology_type == "iec60802":
        topology = make_ieee_60802_topology(link_rate_mbps)
        # count switch nodes; TSNEnv recomputes from num_nodes anyway
        n_switches = max(topology.num_nodes - 5, 3)
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")

    # flows
    if scenario == "random":
        flows = generate_random_flows(n_flows=8, seed=seed)
    else:
        all_scenarios = load_scenarios_from_ontology()
        if scenario not in all_scenarios:
            print(f"Warning: scenario '{scenario}' not found, falling back to random.")
            flows = generate_random_flows(n_flows=8, seed=seed)
        else:
            flows = all_scenarios[scenario]

    print(f"Loaded {len(flows)} flows for scenario '{scenario}'")
    for f in flows:
        print(f"  {f.flow_id}: pw={f.priority_weight:.2f} "
              f"deadline={f.delayable_boundary_us}us "
              f"class={f.stream_class.name}")

    # config
    config = CSRLConfig(
        n_switches=n_switches,
        n_ports_per_switch=4,
        n_queues=8,
        max_active_flows=max(len(flows), 1),
        hyperperiod_us=1000.0,
        link_rate_mbps=link_rate_mbps,
        frame_size_bytes=256.0,
        seed=seed,
        use_safety_shield=use_safety_shield,
    )

    # environment
    env = TSNEnv(config=config, topology=topology, flows=flows)

    # safety shield
    shield = SafetyShield(
        topology=topology,
        link_rate_mbps=link_rate_mbps,
        enabled=use_safety_shield,
    )

    # constraint manager
    constraint_mgr = ConstraintManager(
        threshold=0.0,
        initial_lambda=0.1,
        lr_lambda=0.002,
        max_lambda=10.0,
    )

    # agent
    agent = CSRLAgent(
        env=env,
        constraint_manager=constraint_mgr,
        safety_shield=shield,
        device="cpu",
    )

    print(f"\nStarting training: {scenario} | topology={topology_type} "
          f"| timesteps={total_timesteps} | shield={use_safety_shield}\n")

    # train
    stats = agent.train(
        total_timesteps=total_timesteps,
        nc_validation_interval=nc_validation_interval,
        log_interval=log_interval,
    )

    _log_metrics(total_timesteps, stats, scenario)

    # save
    if save_model:
        os.makedirs(ckpt_dir, exist_ok=True)
        model_path = os.path.join(ckpt_dir, f"csrl_{scenario}_{topology_type}")
        agent.save(model_path)
        print(f"Model saved to {model_path}")

        metrics_path = os.path.join(ckpt_dir, f"metrics_{scenario}_{topology_type}.json")
        metrics_for_json = {}
        for k, v in stats.items():
            if hasattr(v, "__iter__"):
                metrics_for_json[k] = [float(x) for x in v]
            else:
                metrics_for_json[k] = float(v)
        with open(metrics_path, "w") as fp:
            json.dump(metrics_for_json, fp, indent=2)
        print(f"Metrics saved to {metrics_path}")

    # training curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        ax = axes[0, 0]
        if stats.get("rewards"):
            ax.plot(stats["rewards"])
        ax.set_title("Mean Episode Reward")
        ax.set_xlabel("Log Step")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        if stats.get("completion_rates"):
            ax.plot(stats["completion_rates"])
        ax.set_title("Flow Completion Rate")
        ax.set_xlabel("Log Step")
        ax.set_ylabel("Rate")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        if stats.get("wcd_violations"):
            ax.plot(stats["wcd_violations"])
        ax.set_title("WCD Violations (per NC check)")
        ax.set_xlabel("NC Check #")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if stats.get("lambda_evolution"):
            ax.plot(stats["lambda_evolution"])
        ax.set_title("Lagrangian λ Evolution")
        ax.set_xlabel("NC Check #")
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"CSRL Training — {scenario} ({topology_type})", fontsize=14)
        fig.tight_layout()
        plot_path = os.path.join(ckpt_dir, f"training_curve_{scenario}_{topology_type}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Training curves saved to {plot_path}")
    except Exception as e:
        print(f"Note: matplotlib unavailable, skipping plots: {e}")

    return stats


# ============================================================
# CLI entry
# ============================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CSRL TSN Scheduler Training")
    parser.add_argument("--scenario", type=str, default="agv_fleet",
                        choices=["agv_fleet", "cobot", "plc", "random"])
    parser.add_argument("--topology", type=str, default="line",
                        choices=["line", "ring", "iec60802"])
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--nc-interval", type=int, default=1000)
    parser.add_argument("--no-shield", action="store_true")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-switches", type=int, default=3)
    parser.add_argument("--link-rate", type=float, default=1000.0)
    parser.add_argument("--log-interval", type=int, default=500)
    args = parser.parse_args()

    train(
        scenario=args.scenario,
        topology_type=args.topology,
        total_timesteps=args.timesteps,
        nc_validation_interval=args.nc_interval,
        use_safety_shield=not args.no_shield,
        ckpt_dir=args.ckpt_dir,
        seed=args.seed,
        n_switches=args.n_switches,
        link_rate_mbps=args.link_rate,
        log_interval=args.log_interval,
    )
