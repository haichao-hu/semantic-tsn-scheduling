from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from src.intent_ontology.types import FlowSemantics
from src.nc_engine.topology import TSNTopology, make_line_topology
from src.nc_engine.delay_bounds import TASWindowSpec
from src.nc_engine.safety_validator import validate_schedule
from src.csrl.tsn_env import CSRLConfig, TSNEnv
from src.csrl.csrl_agent import CSRLAgent, ConstraintManager
from src.csrl.safety_shield import SafetyShield
from src.csrl.train import load_scenarios_from_ontology

from .baselines import FIFOCBSScheduler, PureDRLScheduler, StaticGCLScheduler


@dataclass
class ExperimentResult:
    """Structured result from a single experiment run."""

    name: str
    scenario: str
    completion_rate: float
    e2e_delays: list[float] = field(default_factory=list)
    p50_delay: float = 0.0
    p99_delay: float = 0.0
    p999_delay: float = 0.0
    wcd_violations: int = 0
    avg_jitter_us: float = 0.0
    gcl_computation_time_ms: float = 0.0
    schedule_size: int = 0
    deadline_violations: int = 0
    total_flows: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scenario": self.scenario,
            "completion_rate": self.completion_rate,
            "p50_delay_us": self.p50_delay,
            "p99_delay_us": self.p99_delay,
            "p999_delay_us": self.p999_delay,
            "wcd_violations": self.wcd_violations,
            "avg_jitter_us": self.avg_jitter_us,
            "gcl_computation_time_ms": self.gcl_computation_time_ms,
            "schedule_size": self.schedule_size,
            "deadline_violations": self.deadline_violations,
            "total_flows": self.total_flows,
            "error": self.error,
            "n_e2e_samples": len(self.e2e_delays),
        }


def _compute_percentiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 99)),
        float(np.percentile(arr, 99.9)),
    )


def _nc_validate_current_schedule(env: TSNEnv, flows: list[FlowSemantics], topology: TSNTopology | None, link_rate_mbps: float) -> int:
    """Count schedulability violations from the current env schedule."""
    from src.nc_engine.schedulability import SchedEntry, check_schedulability

    if not env.sim_flows:
        return 0

    entries: list[SchedEntry] = []
    for sf in env.sim_flows:
        if not sf.accepted:
            continue
        entries.append(SchedEntry(
            flow_id=sf.flow_id,
            queue=int(sf.queue),
            gate_start_us=float(sf.gate_start_us),
            gate_size_us=float(sf.gate_size_us),
            period_us=float(sf.period_us),
            deadline_us=float(sf.deadline_us),
            path=list(sf.path),
            task_id=sf.task_id,
            priority_weight=sf.priority_weight,
            dispatch_phase_us=float(sf.phase_offset_us),
        ))

    if not entries:
        return 0

    try:
        result = check_schedulability(
            entries,
            hyperperiod_us=env.config.hyperperiod_us,
            link_rate_mbps=link_rate_mbps,
            frame_size_bytes=env.config.frame_size_bytes,
            guard_band_us=env.config.gcl_guard_band_us,
        )
        return len(result.violations)
    except Exception:
        return 0


def run_experiment(
    name: str,
    make_scheduler: Callable[[list[FlowSemantics], CSRLConfig, TSNTopology | None], Any],
    scenario: str,
    n_episodes: int = 50,
    topology_type: str = "line",
    n_switches: int = 3,
    link_rate_mbps: float = 1000.0,
    seed: int = 42,
) -> ExperimentResult:
    """Run a single experiment and collect structured metrics.

    Parameters
    ----------
    name : str
        Human-readable label for this experiment run.
    make_scheduler : callable
        Function ``(flows, config, topology) -> scheduler`` where the
        returned scheduler object **must** expose
        ``predict(observation) -> (action, _)``.
    scenario : str
        Scenario key (``"agv_fleet"``, ``"cobot"``, ``"plc"``).
    n_episodes : int
        Number of independent evaluation episodes.
    topology_type : str
        Topology type passed to ``make_line_topology``.
    n_switches, link_rate_mbps : topology parameters.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    ExperimentResult
    """
    if scenario not in {"agv_fleet", "cobot", "plc"}:
        raise ValueError(f"Unknown scenario: {scenario}")

    try:
        all_scenarios = load_scenarios_from_ontology()
        flows = list(all_scenarios[scenario])
    except Exception as exc:
        return ExperimentResult(
            name=name, scenario=scenario, completion_rate=0.0, error=str(exc)
        )

    topology = make_line_topology(n_switches, link_rate_mbps)

    config = CSRLConfig(
        n_switches=n_switches,
        n_ports_per_switch=4,
        n_queues=8,
        max_active_flows=max(len(flows), 1),
        hyperperiod_us=10000.0,
        link_rate_mbps=link_rate_mbps,
        frame_size_bytes=256.0,
        seed=seed,
        use_safety_shield=False,
    )

    try:
        scheduler = make_scheduler(flows, config, topology)
    except Exception as exc:
        return ExperimentResult(
            name=name, scenario=scenario, completion_rate=0.0, error=f"scheduler init: {exc}"
        )

    env = TSNEnv(config=config, topology=topology, flows=flows)
    env._max_steps = max(50, n_episodes)

    all_e2e: list[float] = []
    all_jitter: list[float] = []
    total_wcd_violations = 0
    total_deadline_violations = 0
    deadline_checks = 0
    computation_time_ms = 0.0
    schedule_size = 0

    if hasattr(scheduler, "computation_time_ms"):
        computation_time_ms = float(scheduler.computation_time_ms)
    if hasattr(scheduler, "schedule_size"):
        schedule_size = int(scheduler.schedule_size)

    for ep in range(n_episodes):
        try:
            obs, _ = env.reset(seed=seed + ep)
        except Exception:
            continue

        for _ in range(50):
            try:
                action, _ = scheduler.predict(obs, deterministic=True)
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
                    total_deadline_violations += 1

        wcd_viols = _nc_validate_current_schedule(env, flows, topology, link_rate_mbps)
        total_wcd_violations += max(0, wcd_viols)

    completion_rate = (
        1.0 - total_deadline_violations / max(deadline_checks, 1)
        if deadline_checks > 0
        else 0.0
    )

    p50, p99, p999 = _compute_percentiles(all_e2e)
    avg_jitter = float(np.mean(all_jitter)) if all_jitter else 0.0

    return ExperimentResult(
        name=name,
        scenario=scenario,
        completion_rate=round(completion_rate, 6),
        e2e_delays=all_e2e,
        p50_delay=p50,
        p99_delay=p99,
        p999_delay=p999,
        wcd_violations=total_wcd_violations,
        avg_jitter_us=avg_jitter,
        gcl_computation_time_ms=computation_time_ms,
        schedule_size=schedule_size,
        deadline_violations=total_deadline_violations,
        total_flows=len(flows),
    )


def _make_csrl_scheduler(flows: list[FlowSemantics], config: CSRLConfig, topology: TSNTopology | None) -> CSRLAgent:
    """Train a full CSRL agent on the given flows and return it."""
    train_config = CSRLConfig(
        n_switches=config.n_switches,
        n_ports_per_switch=config.n_ports_per_switch,
        n_queues=config.n_queues,
        max_active_flows=config.max_active_flows,
        hyperperiod_us=config.hyperperiod_us,
        link_rate_mbps=config.link_rate_mbps,
        frame_size_bytes=config.frame_size_bytes,
        seed=config.seed,
        use_safety_shield=True,
        reward_semantic_scale=1.0,
    )
    train_env = TSNEnv(config=train_config, topology=topology, flows=flows)
    shield = SafetyShield(topology=topology, link_rate_mbps=config.link_rate_mbps, enabled=True)
    cm = ConstraintManager(initial_lambda=0.1, lr_lambda=0.002, max_lambda=1.0)
    agent = CSRLAgent(env=train_env, constraint_manager=cm, safety_shield=shield, device="cpu")
    agent.train(total_timesteps=5000, nc_validation_interval=1000, log_interval=1000)
    return agent


def _make_pure_drl_scheduler(flows: list[FlowSemantics], config: CSRLConfig, topology: TSNTopology | None) -> PureDRLScheduler:
    return PureDRLScheduler(flows, config, topology, total_timesteps=5000, seed=config.seed)


def _make_static_gcl_scheduler(flows: list[FlowSemantics], config: CSRLConfig, topology: TSNTopology | None) -> StaticGCLScheduler:
    return StaticGCLScheduler(flows, config, topology)


def _make_fifo_cbs_scheduler(flows: list[FlowSemantics], config: CSRLConfig, topology: TSNTopology | None) -> FIFOCBSScheduler:
    return FIFOCBSScheduler(flows, config, topology)


_BASELINE_FACTORIES: dict[str, tuple[str, Callable]] = {
    "CSRL": ("CSRL (semantic + shield + NC)", _make_csrl_scheduler),
    "StaticGCL": ("B1 — Static GCL (Craciunas)", _make_static_gcl_scheduler),
    "PureDRL": ("B2 — Pure DRL (no semantic/shield)", _make_pure_drl_scheduler),
    "FIFOCBS": ("B3 — FIFO + CBS (no TAS)", _make_fifo_cbs_scheduler),
}


def compare_baselines(
    scenarios: list[str] | None = None,
    n_episodes: int = 50,
    topology_type: str = "line",
    n_switches: int = 3,
    link_rate_mbps: float = 1000.0,
    seed: int = 42,
) -> dict:
    """Run all baselines + CSRL on all scenarios and return comparison data.

    Parameters
    ----------
    scenarios : list[str] or None
        Scenario keys.  Defaults to ``["agv_fleet", "cobot", "plc"]``.
    n_episodes : int
        Number of evaluation episodes per baseline per scenario.
    topology_type, n_switches, link_rate_mbps, seed : topology / RNG params.

    Returns
    -------
    dict
        Nested: ``{scenario: {baseline_key: result_dict}}``.
    """
    if scenarios is None:
        scenarios = ["agv_fleet", "cobot", "plc"]

    results: dict[str, dict] = {}

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"  Scenario: {scenario}")
        print(f"{'='*60}")
        scenario_results: dict[str, dict] = {}

        for key, (label, factory) in _BASELINE_FACTORIES.items():
            print(f"\n  [{label}]")
            try:
                result = run_experiment(
                    name=label,
                    make_scheduler=factory,
                    scenario=scenario,
                    n_episodes=n_episodes,
                    topology_type=topology_type,
                    n_switches=n_switches,
                    link_rate_mbps=link_rate_mbps,
                    seed=seed,
                )
                scenario_results[key] = result.to_dict()
                print(
                    f"    completion_rate={result.completion_rate:.4f}  "
                    f"p50={result.p50_delay:.1f}us  "
                    f"p99={result.p99_delay:.1f}us  "
                    f"wcd_violations={result.wcd_violations}"
                )
            except Exception as exc:
                scenario_results[key] = ExperimentResult(
                    name=label, scenario=scenario, completion_rate=0.0, error=str(exc)
                ).to_dict()
                print(f"    ERROR: {exc}")

        results[scenario] = scenario_results

    return results
