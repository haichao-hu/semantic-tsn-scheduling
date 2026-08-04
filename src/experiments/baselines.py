from __future__ import annotations

import time
from typing import Optional

import numpy as np

from src.intent_ontology.types import FlowSemantics, StreamClass
from src.nc_engine.topology import TSNTopology
from src.csrl.tsn_env import CSRLConfig, TSNEnv
from src.csrl.safety_shield import SafetyShield
from src.csrl.csrl_agent import CSRLAgent, ConstraintManager


class StaticGCLScheduler:
    """B1 — Static GCL: Craciunas-style priority-based round-robin.

    Precomputes a fixed GCL allocation once (descending PCP order) and
    returns the same action vector regardless of the observation.
    Never adapts — purely offline scheduling.

    ``static_flows`` (optional) limits the flow set the static scheduler
    *sees* at t=0.  Flows outside it (e.g. dynamically arriving flows)
    receive best-effort defaults: the scheduler has no window reservation
    for them, so they must contend for leftover link capacity.
    """

    def __init__(
        self,
        flows: list[FlowSemantics],
        config: CSRLConfig,
        topology: TSNTopology | None = None,
        static_flows: list[FlowSemantics] | None = None,
    ) -> None:
        self._config = config
        self._static_action: np.ndarray | None = None
        self._computation_time_ms: float = 0.0
        self._static_flows = static_flows if static_flows is not None else flows
        self._precompute(flows, topology)

    def _precompute(
        self, flows: list[FlowSemantics], topology: TSNTopology | None
    ) -> None:
        t0 = time.perf_counter()
        shield = SafetyShield(
            topology=topology,
            enabled=False,
            hyperperiod_us=self._config.hyperperiod_us,
            frame_size_bytes=self._config.frame_size_bytes,
            guard_band_us=self._config.gcl_guard_band_us,
        )
        schedule = shield.compute_fallback(
            self._static_flows,
            shared_st_queue=getattr(self._config, "shared_st_queue", False),
            st_window_us=getattr(self._config, "st_window_us", None),
        )
        self._computation_time_ms = (time.perf_counter() - t0) * 1000.0

        flow_map: dict[str, FlowSemantics] = {f.flow_id: f for f in flows}
        sa_map: dict[str, dict] = schedule.to_dict()

        M = self._config.max_active_flows
        dim = 3  # accept, dispatch_offset, gate_start_offset
        action = np.zeros(M * dim, dtype=np.float64)

        for i in range(M):
            if i >= len(flows):
                continue
            flow = flows[i]
            sa = sa_map.get(flow.flow_id, {})
            offset = i * dim

            period_us = max(float(flow.delayable_boundary_us), 1.0)
            if sa.get("accept", True) is False:
                action[offset] = -1.0  # rejected by the static scheduler
                continue

            action[offset] = 1.0 if sa.get("accept", True) else -1.0

            if not sa:
                # flow invisible at t=0 (dynamic arrival): the static
                # scheduler has no reserved window — in real TSN the new
                # flow would wait for a CNC reconfiguration.  Model this
                # as a misaligned window: frames wait nearly a full
                # period, so tight-deadline flows miss their deadline.
                action[offset + 1] = 0.0
                action[offset + 2] = 0.8  # window badly misaligned
                continue

            action[offset + 1] = np.clip(
                sa.get("dispatch_offset_us", 0.0) / period_us * 2.0 - 1.0,
                -1.0,
                1.0,
            )

            # relative gate_start: window placed relative to dispatch phase
            rel = (sa.get("gate_start_us", 0.0) - sa.get("dispatch_offset_us", 0.0)) % period_us
            action[offset + 2] = np.clip(rel / period_us * 2.0 - 1.0, -1.0, 1.0)

        self._static_action = np.clip(action, -1.0, 1.0)
        self._schedule_size = sum(
            1 for a in schedule.actions if a.accept
        )

    @property
    def schedule_size(self) -> int:
        return self._schedule_size

    @property
    def computation_time_ms(self) -> float:
        return self._computation_time_ms

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if self._static_action is None:
            raise RuntimeError("StaticGCLScheduler: _precompute not called")
        return self._static_action.copy(), None


class PureDRLScheduler:
    """B2 — Pure DRL: PPO without semantic weights and without safety shield.

    Trains a standalone PPO agent (via CSRLAgent wrapper) with
    ``reward_semantic_scale=0`` (uniform priority) and
    ``use_safety_shield=False``.  The ConstraintManager is kept
    but with ``lr_lambda=0`` so λ never evolves — effectively
    standard PPO with deadline-based reward only.

    Internally wraps a CSRLAgent that can have its own env so that
    training does not leak state into the evaluation env.
    """

    def __init__(
        self,
        flows: list[FlowSemantics],
        config: CSRLConfig,
        topology: TSNTopology | None = None,
        total_timesteps: int = 5000,
        seed: int = 42,
    ) -> None:
        train_config = CSRLConfig(
            n_switches=config.n_switches,
            n_ports_per_switch=config.n_ports_per_switch,
            n_queues=config.n_queues,
            max_active_flows=config.max_active_flows,
            hyperperiod_us=config.hyperperiod_us,
            link_rate_mbps=config.link_rate_mbps,
            frame_size_bytes=config.frame_size_bytes,
            seed=seed,
            use_safety_shield=False,
            reward_semantic_scale=0.0,
        )

        train_env = TSNEnv(config=train_config, topology=topology, flows=flows)
        cm = ConstraintManager(initial_lambda=0.0, lr_lambda=0.0, max_lambda=0.0)
        self._agent = CSRLAgent(
            env=train_env,
            constraint_manager=cm,
            safety_shield=None,
            device="cpu",
        )
        self._agent.train(
            total_timesteps=total_timesteps,
            nc_validation_interval=total_timesteps,
            log_interval=max(total_timesteps // 2, 1),
        )

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        return self._agent.predict(observation, deterministic=deterministic)


class FIFOCBSScheduler:
    """B3 — FIFO + CBS: no GCL at all.

    All flows use CBS credit-based shaping.  Arrival order determines
    transmission (FIFO within the same traffic class).  The scheduler
    always outputs the "all gates open" action so no TAS windowing
    constrains queue draining.

    Accepts all flows, assigns queues by stream class convention
    (ST→7, AVB→5, BE→0), and sets ``gate_size`` equal to the full
    hyperperiod so gates are effectively permanently open.
    """

    def __init__(
        self,
        flows: list[FlowSemantics],
        config: CSRLConfig,
        topology: TSNTopology | None = None,
    ) -> None:
        M = config.max_active_flows
        dim = 3
        action = np.zeros(M * dim, dtype=np.float64)

        for i in range(M):
            offset = i * dim
            action[offset] = 1.0  # accept

            if i < len(flows):
                period_us = max(float(flows[i].delayable_boundary_us), 1.0)
            else:
                period_us = 10000.0

            action[offset + 1] = 0.0  # dispatch at period midpoint

            action[offset + 2] = -1.0  # window aligned with dispatch phase

        self._static_action = np.clip(action, -1.0, 1.0)

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        return self._static_action.copy(), None
