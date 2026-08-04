from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from src.intent_ontology.types import (
    CBSConfig,
    CriticalityLevel,
    FlowSemantics,
    GCLConfig,
    GCLParameters,
    StreamClass,
    TSNBridgeConfig,
)

from .curves import ArrivalCurve, LeakyBucket
from .delay_bounds import (
    TASWindowSpec,
    compute_cbs_bounds,
    compute_sp_bounds,
    compute_tas_bounds,
)
from .topology import TSNTopology


# ============================================================
# Violation & ValidationResult
# ============================================================


@dataclass
class Violation:
    """A single deadline violation from the NC safety check."""

    flow_id: str
    task_id: str
    criticality: CriticalityLevel
    required_deadline_us: float
    computed_wcd_us: float

    @property
    def margin_us(self) -> float:
        """Positive = ahead of deadline, negative = violation."""
        return self.required_deadline_us - self.computed_wcd_us

    @property
    def is_violation(self) -> bool:
        return self.margin_us < 0


@dataclass
class ValidationResult:
    """Output of a single schedule validation pass."""

    is_safe: bool
    violations: list[Violation] = field(default_factory=list)
    wcd_bounds: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_safe": self.is_safe,
            "num_violations": len(self.violations),
            "violations": [
                {
                    "flow_id": v.flow_id,
                    "task_id": v.task_id,
                    "criticality": v.criticality.name,
                    "required_deadline_us": v.required_deadline_us,
                    "computed_wcd_us": v.computed_wcd_us,
                    "margin_us": v.margin_us,
                }
                for v in self.violations
            ],
            "wcd_bounds": self.wcd_bounds,
            "warnings": self.warnings,
        }


# ============================================================
# Safety Policy
# ============================================================


@dataclass
class SafetyPolicy:
    """Criticality-based safety thresholds for deadline validation.

    ┌───────────┬──────────────────────┬─────────────────────┐
    │ Level     │ Deadline Hardness    │ Overshoot Tolerance  │
    ├───────────┼──────────────────────┼─────────────────────┤
    │ L0        │ Hard real-time       │ 0% (zero tolerance) │
    │ L1        │ Hard real-time       │ 0.1%                │
    │ L2        │ Soft real-time       │ 5%                  │
    │ L3        │ Best-effort          │ No check            │
    └───────────┴──────────────────────┴─────────────────────┘
    """

    level: CriticalityLevel
    zero_tolerance: bool = False
    allowed_overshoot_pct: float = 0.0  # allowed deadline overshoot percentage

    @classmethod
    def for_level(cls, level: CriticalityLevel) -> SafetyPolicy:
        policies = {
            CriticalityLevel.L0: cls(level, zero_tolerance=True, allowed_overshoot_pct=0.0),
            CriticalityLevel.L1: cls(level, zero_tolerance=False, allowed_overshoot_pct=0.1),
            CriticalityLevel.L2: cls(level, zero_tolerance=False, allowed_overshoot_pct=5.0),
            CriticalityLevel.L3: cls(level, zero_tolerance=False, allowed_overshoot_pct=float("inf")),
        }
        return policies[level]

    def is_acceptable(self, required_us: float, computed_us: float) -> bool:
        """Check if computed WCD is within the acceptable margin."""
        if computed_us <= required_us:
            return True
        if self.zero_tolerance:
            return False
        if self.allowed_overshoot_pct == float("inf"):
            return True  # L3: always acceptable
        overshoot = (computed_us - required_us) / required_us * 100.0
        return overshoot <= self.allowed_overshoot_pct


# ============================================================
# Schedule type definitions
# ============================================================


ScheduleConfig = Union[GCLConfig, TSNBridgeConfig, Dict[str, TASWindowSpec]]


# ============================================================
# validate_schedule — single schedule
# ============================================================


def validate_schedule(
    schedule: ScheduleConfig,
    flows: list[FlowSemantics],
    topology: TSNTopology | None = None,
    link_rate_mbps: float = 1000.0,
) -> ValidationResult:
    """Validate a TSN schedule configuration against flow deadlines using NC.

    This is the Safety Shield: it computes WCD upper bounds for every flow
    and checks them against declared deadlines, respecting criticality-based
    safety policies.

    Parameters
    ----------
    schedule : GCLConfig | TSNBridgeConfig | dict
        The TSN schedule to validate. Can be a GCL configuration (TAS),
        a bridge config (mixed TAS+CBS), or a raw TASWindowSpec map.
    flows : list of FlowSemantics
        Flows with declared deadline constraints.
    topology : TSNTopology or None
        If given, E2E bounds are computed across hops. If None (single-hop),
        only the local link WCD is computed.
    link_rate_mbps : float
        Link speed (Mbps). Default 1 Gbps.

    Returns
    -------
    ValidationResult
    """
    wcd_bounds: dict[str, float] = {}
    violations: list[Violation] = []
    warnings: list[str] = []

    # Determine scheduling type from the config
    if isinstance(schedule, (GCLConfig, GCLParameters)) or (
        isinstance(schedule, dict) and all(isinstance(v, TASWindowSpec) for v in schedule.values())
    ):
        # TAS path
        spec_map = _normalize_tas_specs(schedule, flows, link_rate_mbps)
        wcd_bounds = compute_tas_bounds(flows, spec_map, link_rate_mbps)
    elif isinstance(schedule, TSNBridgeConfig):
        wcd_bounds = _validate_bridge_config(schedule, flows, link_rate_mbps, warnings)
    elif isinstance(schedule, dict):
        # Mixed or unknown — fall back to SP with default topology assumption
        wcd_bounds = compute_sp_bounds(flows, link_rate_mbps)

    # If topology is given, compute E2E
    if topology is not None:
        flow_paths = {fid: topology.get_path(fid) for fid in wcd_bounds}
        # For multi-hop, accumulate per-hop (this is a simplified path —
        # a full implementation would need per-hop configs)
        for fid in wcd_bounds:
            path = flow_paths.get(fid, [])
            n_hops = max(len(path) - 1, 1) if path else 1
            wcd_bounds[fid] *= n_hops
            if n_hops > 1:
                warnings.append(
                    f"{fid}: E2E multi-hop bound ({n_hops} hops) is a "
                    f"conservative additive estimate. For precise E2E, "
                    f"provide per-hop configurations."
                )

    # Check violations
    for flow in flows:
        wcd = wcd_bounds.get(flow.flow_id, float("inf"))
        deadline = float(flow.delayable_boundary_us)

        # Determine criticality: FlowSemantics doesn't store CriticalityLevel directly.
        # Infer from priority_weight: weights [0.98, 0.8, 0.5, 0.2] ≈ L0-L3.
        cl = _infer_criticality(flow.priority_weight)
        policy = SafetyPolicy.for_level(cl)

        if not policy.is_acceptable(deadline, wcd):
            violations.append(
                Violation(
                    flow_id=flow.flow_id,
                    task_id=flow.task_id,
                    criticality=cl,
                    required_deadline_us=deadline,
                    computed_wcd_us=wcd,
                )
            )

    is_safe = len(violations) == 0
    return ValidationResult(
        is_safe=is_safe,
        violations=violations,
        wcd_bounds=wcd_bounds,
        warnings=warnings,
    )


def _infer_criticality(priority_weight: float) -> CriticalityLevel:
    if priority_weight >= 0.95:
        return CriticalityLevel.L0
    if priority_weight >= 0.70:
        return CriticalityLevel.L1
    if priority_weight >= 0.30:
        return CriticalityLevel.L2
    return CriticalityLevel.L3


def _normalize_tas_specs(
    schedule: ScheduleConfig,
    flows: list[FlowSemantics],
    link_rate_mbps: float,
) -> dict[str, TASWindowSpec]:
    """Convert GCLConfig / TSNBridgeConfig into a flat dict of TASWindowSpec."""
    if isinstance(schedule, dict) and all(isinstance(v, TASWindowSpec) for v in schedule.values()):
        return schedule  # type: ignore[return-value]

    spec_map: dict[str, TASWindowSpec] = {}
    if isinstance(schedule, GCLParameters):
        # Single GCL entry — find the matching flow
        for f in flows:
            if f.flow_id in schedule.window_id or schedule.window_id.endswith(f.flow_id):
                spec_map[f.flow_id] = TASWindowSpec(
                    window_id=schedule.window_id,
                    flow_id=f.flow_id,
                    offset_us=schedule.offset_ns / 1000.0,
                    window_size_us=schedule.window_size_ns / 1000.0,
                    period_us=schedule.cycle_time_ns / 1000.0,
                )
    elif isinstance(schedule, TSNBridgeConfig):
        for gcl in schedule.gcl_list:
            for f in flows:
                if gcl.window_id.endswith(f.flow_id) or f.flow_id in gcl.window_id:
                    spec_map[f.flow_id] = TASWindowSpec(
                        window_id=gcl.window_id,
                        flow_id=f.flow_id,
                        offset_us=gcl.offset_ns / 1000.0,
                        window_size_us=gcl.window_size_ns / 1000.0,
                        period_us=gcl.cycle_time_ns / 1000.0,
                    )
    return spec_map


def _validate_bridge_config(
    bridge: TSNBridgeConfig,
    flows: list[FlowSemantics],
    link_rate_mbps: float,
    warnings: list[str],
) -> dict[str, float]:
    """Handle mixed TAS + CBS bridge configuration."""
    wcd: dict[str, float] = {}
    scheduled = [f for f in flows if f.stream_class == StreamClass.SCHEDULED_TRAFFIC]
    reserved = [f for f in flows if f.stream_class == StreamClass.RESERVED]
    be_flows = [f for f in flows if f.stream_class == StreamClass.BEST_EFFORT]

    # TAS windows
    if scheduled and bridge.gcl_list:
        tas_specs = _normalize_tas_specs(bridge, flows, link_rate_mbps)
        wcd.update(compute_tas_bounds(scheduled, tas_specs, link_rate_mbps))

    # CBS
    for cbs in bridge.cbs_configs:
        if reserved:
            cbs_wcd = compute_cbs_bounds(
                reserved,
                idle_slope_mbps=cbs.idle_slope_kbps / 1000.0,
                send_slope_mbps=cbs.send_slope_kbps / 1000.0,
                use_gx_server=True,
            )
            wcd.update(cbs_wcd)

    # BE: not schedulable under NC
    for f in be_flows:
        wcd[f.flow_id] = float("inf")
        warnings.append(
            f"{f.flow_id}: Best-effort flow — no deterministic WCD bound. "
            f"Marked as infinite."
        )

    return wcd


# ============================================================
# validate_batch — batch validation for RL training
# ============================================================


def validate_batch(
    schedules: Sequence[ScheduleConfig],
    flows_list: Sequence[Sequence[FlowSemantics]],
    link_rate_mbps: float = 1000.0,
) -> np.ndarray:
    """Batch-validate multiple schedules against their flow sets.

    Returns a boolean tensor: [num_schedules] where True = all safe.

    Parameters
    ----------
    schedules : Length-N sequence of schedule configs.
    flows_list : Length-N sequence of per-schedule flow lists.
    link_rate_mbps : float

    Returns
    -------
    safe_mask : np.ndarray  shape (N,) dtype=bool
    """
    results = np.zeros(len(schedules), dtype=bool)
    for i, (sched, flows) in enumerate(zip(schedules, flows_list)):
        result = validate_schedule(sched, list(flows), link_rate_mbps=link_rate_mbps)
        results[i] = result.is_safe
    return results
