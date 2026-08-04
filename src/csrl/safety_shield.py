from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from src.intent_ontology.types import (
    CriticalityLevel,
    FlowSemantics,
    GCLParameters,
    StreamClass,
)
from src.nc_engine.safety_validator import (
    SafetyPolicy,
    ValidationResult,
    Violation,
    validate_schedule,
)
from src.nc_engine.schedulability import SchedEntry, check_schedulability
from src.nc_engine.delay_bounds import TASWindowSpec
from src.nc_engine.topology import TSNTopology


# ============================================================
# Schedule output
# ============================================================


@dataclass
class ScheduleAction:
    """A validated or fallback schedule action for one flow."""

    flow_id: str
    accept: bool
    queue: int
    dispatch_offset_us: float
    gate_start_us: float
    gate_size_us: float


@dataclass
class Schedule:
    """List of validated schedule actions for the shield output."""

    actions: list[ScheduleAction] = field(default_factory=list)

    def to_dict(self) -> dict[str, dict]:
        return {
            a.flow_id: {
                "accept": a.accept,
                "queue": a.queue,
                "dispatch_offset_us": a.dispatch_offset_us,
                "gate_start_us": a.gate_start_us,
                "gate_size_us": a.gate_size_us,
            }
            for a in self.actions
        }


# ============================================================
# Safety Shield
# ============================================================


class SafetyShield:
    """Runtime safety guard that integrates the NC engine to validate RL actions.

    For each flow, checks whether the proposed GCL/CBS configuration violates
    WCD bounds from NC. Handles violations with criticality-based fallback:

    - L0: reject action, use ILP-like heuristic fallback
    - L1: attempt to adjust action (trim GCL window)
    - L2/L3: log warning, return action (acceptable risk)
    """

    MAX_ATTEMPTS = 5

    def __init__(
        self,
        topology: TSNTopology | None = None,
        link_rate_mbps: float = 1000.0,
        enabled: bool = True,
        hyperperiod_us: float = 100000.0,
        frame_size_bytes: float = 256.0,
        guard_band_us: float = 5.0,
    ):
        self.topology = topology
        self.link_rate_mbps = link_rate_mbps
        self.enabled = enabled
        self.hyperperiod_us = hyperperiod_us
        self.frame_size_bytes = frame_size_bytes
        self.guard_band_us = guard_band_us
        self.warnings: list[str] = []

    def filter_action(
        self,
        action: dict[str, dict],
        flows: list[FlowSemantics],
        state: np.ndarray | None = None,
        all_sim_flows_priority_weights: dict[str, float] | None = None,
        entries: dict[str, SchedEntry] | None = None,
    ) -> dict[str, dict]:
        """Filter an RL action through the online schedulability check.

        Parameters
        ----------
        action : dict  flow_id → {accept, queue, dispatch_offset_us, gate_start_us, gate_size_us}
        flows : list of FlowSemantics for NC validation
        state : optional observation for context
        all_sim_flows_priority_weights : optional mapping of flow_id → priority_weight
            for inferring criticality when not stored in FlowSemantics.
        entries : optional per-flow SchedEntry (path/period/dispatch phase)
            for the online schedulability check.  If None, entries are
            derived from the action and flow deadlines (single-hop).

        Returns
        -------
        safe_action : dict  same structure as input, possibly adjusted
        """
        if not self.enabled:
            return action

        self.warnings.clear()
        safe_action: dict[str, dict] = {}

        # build SchedEntry for every accepted flow
        sched_entries: list[SchedEntry] = []
        for fid, act in action.items():
            if not act.get("accept", True):
                continue
            flow = next((f for f in flows if f.flow_id == fid), None)
            if flow is None:
                continue
            ent = (entries or {}).get(fid)
            period = ent.period_us if ent is not None else float(flow.delayable_boundary_us)
            sched_entries.append(SchedEntry(
                flow_id=fid,
                queue=int(act.get("queue", 7)),
                gate_start_us=float(act.get("gate_start_us", 0.0)),
                gate_size_us=float(act.get("gate_size_us", 20.0)),
                period_us=max(period, 1.0),
                deadline_us=float(flow.delayable_boundary_us),
                path=list(ent.path) if ent is not None else [0],
                task_id=flow.task_id,
                priority_weight=flow.priority_weight,
                dispatch_phase_us=float(act.get("dispatch_offset_us", 0.0)),
            ))

        result = check_schedulability(
            sched_entries,
            hyperperiod_us=self.hyperperiod_us,
            link_rate_mbps=self.link_rate_mbps,
            frame_size_bytes=self.frame_size_bytes,
            guard_band_us=self.guard_band_us,
        )
        violated_flows = {v.flow_id for v in result.violations}

        for fid, act in action.items():
            flow = next((f for f in flows if f.flow_id == fid), None)

            if flow is None:
                safe_action[fid] = act
                continue

            if not act.get("accept", True):
                safe_action[fid] = act
                continue

            if fid not in violated_flows:
                safe_action[fid] = act
                continue

            cl = self._infer_criticality(flow, all_sim_flows_priority_weights)
            violation = next((v for v in result.violations if v.flow_id == fid), None)

            if cl == CriticalityLevel.L0:
                safe_action[fid] = self._compute_fallback_single(fid, flow, act)
                self.warnings.append(
                    f"{fid} (L0): action rejected by Safety Shield, "
                    f"fallback applied. WCD={violation.computed_wcd_us if violation else 'N/A'}us "
                    f"vs deadline={flow.delayable_boundary_us}us"
                )

            elif cl == CriticalityLevel.L1:
                adjusted = self._adjust_action(act, flow, violation)
                if adjusted is not None:
                    # re-validate
                    ent = (entries or {}).get(fid)
                    re_entries = [e for e in sched_entries if e.flow_id != fid]
                    re_entries.append(SchedEntry(
                        flow_id=fid,
                        queue=int(adjusted.get("queue", 7)),
                        gate_start_us=float(adjusted.get("gate_start_us", 0.0)),
                        gate_size_us=float(adjusted.get("gate_size_us", 20.0)),
                        period_us=max(ent.period_us if ent is not None else float(flow.delayable_boundary_us), 1.0),
                        deadline_us=float(flow.delayable_boundary_us),
                        path=list(ent.path) if ent is not None else [0],
                        task_id=flow.task_id,
                        priority_weight=flow.priority_weight,
                        dispatch_phase_us=float(adjusted.get("dispatch_offset_us", 0.0)),
                    ))
                    re_result = check_schedulability(
                        re_entries,
                        hyperperiod_us=self.hyperperiod_us,
                        link_rate_mbps=self.link_rate_mbps,
                        frame_size_bytes=self.frame_size_bytes,
                        guard_band_us=self.guard_band_us,
                    )
                    if not any(v.flow_id == fid for v in re_result.violations):
                        safe_action[fid] = adjusted
                        self.warnings.append(
                            f"{fid} (L1): gate window trimmed from "
                            f"{act.get('gate_size_us', 0):.1f} to "
                            f"{adjusted.get('gate_size_us', 0):.1f}us"
                        )
                    else:
                        safe_action[fid] = self._compute_fallback_single(fid, flow, act)
                        self.warnings.append(
                            f"{fid} (L1): adjustment insufficient, fallback applied."
                        )
                else:
                    safe_action[fid] = self._compute_fallback_single(fid, flow, act)
                    self.warnings.append(
                        f"{fid} (L1): cannot adjust, fallback applied."
                    )

            else:  # L2, L3
                safe_action[fid] = act
                self.warnings.append(
                    f"{fid} (L{cl.value}): schedulability violation accepted "
                    f"(acceptable risk). WCD={violation.computed_wcd_us if violation else 'N/A'}us"
                )

        return safe_action

    def compute_fallback(self, flows: list[FlowSemantics], topology: TSNTopology | None = None,
                         shared_st_queue: bool = False, st_window_us: float | None = None) -> Schedule:
        """Generate a static GCL schedule via greedy window allocation.

        Reference: Craciunas 2016 constraint-based approach, simplified.

        Every flow gets a per-period repeating window; windows of flows
        mapped to the same queue are mutually exclusive within the
        hyperperiod (link mutual exclusion constraint).  Flows are
        processed in descending priority order: critical flows get early
        window phases, best-effort flows get the full hyperperiod (no TAS
        isolation, i.e. they transmit in leftover link capacity).

        With ``shared_st_queue``, all ST flows share queue 7 (matching the
        scarce-resource environment), so the mutual exclusion constraint
        binds across ST flows and a flow is rejected when no slot remains.
        """
        topo = topology or self.topology
        actions: list[ScheduleAction] = []

        HP = self.hyperperiod_us
        tx_time = self.frame_size_bytes * 8.0 / self.link_rate_mbps
        min_window = tx_time + self.guard_band_us

        # sort by priority descending
        sorted_flows = sorted(flows, key=lambda f: f.priority_weight, reverse=True)

        # per-queue allocated windows: list of (start, size, period)
        queue_windows: dict[int, list[tuple[float, float, float]]] = {}
        st_idx = 0

        for flow in sorted_flows:
            if flow.stream_class == StreamClass.BEST_EFFORT:
                actions.append(ScheduleAction(
                    flow_id=flow.flow_id, accept=True, queue=0,
                    dispatch_offset_us=0.0, gate_start_us=0.0, gate_size_us=HP,
                ))
                continue

            if flow.stream_class == StreamClass.RESERVED:
                queue = 3
                window_us = max(2.0 * tx_time + self.guard_band_us, 20.0)  # AVB: wider window
            else:
                queue = 7 if shared_st_queue else 7 - min(st_idx, 4)
                st_idx += 1
                window_us = st_window_us if st_window_us is not None else min_window

            period = max(float(flow.delayable_boundary_us), 1.0)
            n_w = max(1, int(round(HP / period)))
            allocated = queue_windows.setdefault(queue, [])

            def overlaps(phase: float) -> bool:
                """Check candidate window sequence against allocated ones."""
                for (s0, w0, p0) in allocated:
                    n0 = max(1, int(round(HP / p0)))
                    for k0 in range(n0):
                        a0 = (s0 + k0 * p0) % HP
                        for k in range(n_w):
                            a1 = (phase + k * period) % HP
                            if a0 < a1 + window_us and a1 < a0 + w0:
                                return True
                return False

            # greedy first-fit phase
            found = False
            step = max(min_window, 1.0)
            phase = 0.0
            while phase < min(period, HP):
                if not overlaps(phase):
                    allocated.append((phase, window_us, period))
                    actions.append(ScheduleAction(
                        flow_id=flow.flow_id, accept=True, queue=queue,
                        dispatch_offset_us=phase % period,
                        gate_start_us=phase,
                        gate_size_us=window_us,
                    ))
                    found = True
                    break
                phase += step

            if not found:
                actions.append(ScheduleAction(
                    flow_id=flow.flow_id, accept=False, queue=0,
                    dispatch_offset_us=0.0, gate_start_us=0.0, gate_size_us=0.0,
                ))

        return Schedule(actions=actions)

    def _compute_fallback_single(
        self, flow_id: str, flow: FlowSemantics, original_action: dict[str, dict]
    ) -> dict[str, dict]:
        """Single-flow fallback for L0/L1 violations."""
        return {
            "accept": False,
            "queue": 7,
            "dispatch_offset_us": 0.0,
            "gate_start_us": 0.0,
            "gate_size_us": float(flow.delayable_boundary_us) * 0.5 if flow.delayable_boundary_us > 0 else 10.0,
        }

    def _adjust_action(
        self,
        action: dict[str, dict],
        flow: FlowSemantics,
        violation: Violation | None,
    ) -> dict[str, dict] | None:
        """Try to trim the GCL window to meet the deadline."""
        gs = action.get("gate_size_us", 20.0)
        if gs <= 5.0:
            return None

        deadline = flow.delayable_boundary_us
        if violation is not None and violation.computed_wcd_us > 0:
            ratio = deadline / violation.computed_wcd_us
            new_size = gs * ratio * 0.9
        else:
            new_size = gs * 0.5

        new_size = max(5.0, new_size)

        if new_size >= gs * 0.95:
            return None

        adjusted = dict(action)
        adjusted["gate_size_us"] = new_size
        return adjusted

    def _action_to_tas_spec(self, flow_id: str, action: dict[str, dict]) -> dict[str, TASWindowSpec]:
        """Convert action dict to TASWindowSpec map for NC validation."""
        gs = action.get("gate_start_us", 0.0)
        gw = action.get("gate_size_us", 20.0)
        return {
            flow_id: TASWindowSpec(
                window_id=f"w_{flow_id}",
                flow_id=flow_id,
                offset_us=float(gs),
                window_size_us=float(gw),
                guard_band_us=5.0,
            )
        }

    def _infer_criticality(
        self,
        flow: FlowSemantics,
        extra: dict[str, float] | None = None,
    ) -> CriticalityLevel:
        pw = flow.priority_weight
        if extra and flow.flow_id in extra:
            pw = extra[flow.flow_id]
        if pw >= 0.95:
            return CriticalityLevel.L0
        if pw >= 0.70:
            return CriticalityLevel.L1
        if pw >= 0.30:
            return CriticalityLevel.L2
        return CriticalityLevel.L3
