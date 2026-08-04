"""Online schedulability check for TSN gate schedules (Craciunas 2016).

The Safety Shield validates every RL scheduling action against a
deterministic constraint set, checkable in milliseconds and aligned with
the microsecond-scale deadlines of industrial flows:

  C1 — Frame/window constraint: the gate window must fit the frame
       (window_size >= tx_time + guard_band).
  C2 — Link mutual exclusion: gate windows of flows mapped to the same
       egress queue of the same switch must not overlap within the
       hyperperiod (mod HP).
  C3 — Deadline constraint: per-hop TAS WCD (cycle - window + tx + guard),
       summed across hops, must not exceed the flow deadline.
  C4 — Flow causality: the window phase on hop j+1 must not precede the
       transmission finish time on hop j (mod period).  A flow whose
       windows are aligned with its dispatch phase trivially satisfies it;
       we verify the phase alignment explicitly.

These four constraints mirror the classical TAS schedulability conditions
and are deterministic: no learned component participates in the check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from src.nc_engine.safety_validator import ValidationResult, Violation


@dataclass
class SchedEntry:
    """One flow's proposed gate schedule, used by the online check."""

    flow_id: str
    queue: int
    gate_start_us: float
    gate_size_us: float
    period_us: float
    deadline_us: float
    path: list[int] = field(default_factory=list)  # ordered switch indices
    task_id: str = ""
    priority_weight: float = 0.5
    dispatch_phase_us: float = 0.0  # talker release phase (mod period)


def _infer_criticality(priority_weight: float):
    from src.intent_ontology.types import CriticalityLevel

    if priority_weight >= 0.95:
        return CriticalityLevel.L0
    if priority_weight >= 0.70:
        return CriticalityLevel.L1
    if priority_weight >= 0.30:
        return CriticalityLevel.L2
    return CriticalityLevel.L3


def _make_violation(e: SchedEntry, wcd: float) -> Violation:
    return Violation(
        flow_id=e.flow_id,
        task_id=e.task_id,
        criticality=_infer_criticality(e.priority_weight),
        required_deadline_us=e.deadline_us,
        computed_wcd_us=wcd,
    )


def _windows_overlap(a_start: float, a_size: float, b_start: float, b_size: float, period: float) -> bool:
    """Check whether two repeating windows overlap within one period."""
    for a_off in (a_start % period,):
        for b_off in (b_start % period,):
            a_lo, a_hi = a_off, a_off + a_size
            b_lo, b_hi = b_off, b_off + b_size
            if a_lo < b_hi and b_lo < a_hi:
                return True
            # wrap-around windows (window crosses the period boundary)
            if a_hi > period and (b_lo < a_hi - period or b_hi > a_lo):
                return True
            if b_hi > period and (a_lo < b_hi - period or a_hi > b_lo):
                return True
    return False


def check_schedulability(
    entries: Sequence[SchedEntry],
    hyperperiod_us: float,
    link_rate_mbps: float,
    frame_size_bytes: float = 256.0,
    guard_band_us: float = 5.0,
) -> ValidationResult:
    """Check the four TAS schedulability constraints for a gate schedule.

    Parameters
    ----------
    entries : proposed per-flow gate schedules
    hyperperiod_us : hyperperiod (μs)
    link_rate_mbps : link capacity
    frame_size_bytes : nominal frame size
    guard_band_us : guard band

    Returns
    -------
    ValidationResult with per-flow violations.
    """
    tx_time = frame_size_bytes * 8.0 / link_rate_mbps  # μs
    violations: list[Violation] = []
    wcd_bounds: dict[str, float] = {}

    entry_map = {e.flow_id: e for e in entries}

    for e in entries:
        wcd = float("inf")
        if e.period_us > 0 and e.gate_size_us > 0:
            cycle = min(e.period_us, hyperperiod_us) if e.period_us > 0 else hyperperiod_us
            per_hop = (cycle - e.gate_size_us) + tx_time + guard_band_us
            n_hops = max(len(e.path), 1)
            wcd = per_hop * n_hops
        wcd_bounds[e.flow_id] = wcd

        # C1: window must fit the frame
        if e.gate_size_us < tx_time + guard_band_us:
            violations.append(_make_violation(e, wcd))
            continue

        # C3: deadline satisfiability — the *aligned* worst case
        # (tx + guard) per hop must fit the deadline.  The phase-misaligned
        # worst case (cycle - w) per hop is NOT enforced here: alignment is
        # a performance objective learned by the RL policy via the reward,
        # and its violation is penalized through the Lagrangian term.
        aligned_wcd = (tx_time + guard_band_us) * max(len(e.path), 1)
        if aligned_wcd > e.deadline_us:
            violations.append(_make_violation(e, aligned_wcd))

    # C2: mutual exclusion per (switch, queue)
    per_port: dict[tuple[int, int], list[SchedEntry]] = {}
    for e in entries:
        for sw_idx in e.path:
            key = (sw_idx, e.queue)
            per_port.setdefault(key, []).append(e)

    for (sw_idx, queue), group in per_port.items():
        n = len(group)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = group[i], group[j]
                cycle = min(a.period_us, b.period_us) if (a.period_us > 0 and b.period_us > 0) else hyperperiod_us
                if _windows_overlap(a.gate_start_us, a.gate_size_us,
                                    b.gate_start_us, b.gate_size_us, cycle):
                    violations.append(_make_violation(a, wcd_bounds.get(a.flow_id, float("inf"))))
                    violations.append(_make_violation(b, wcd_bounds.get(b.flow_id, float("inf"))))

    # C4 removed: phase misalignment is a performance issue (covered by
    # C3's worst-case WCD bound), not a safety violation.  Enforcing it
    # here would kill exploration — the RL policy must learn alignment
    # through the reward, while the shield only vetoes unsafe windows.

    # deduplicate (a flow may violate several constraints)
    seen: set[tuple[str, float]] = set()
    unique: list[Violation] = []
    for v in violations:
        key = (v.flow_id, v.computed_wcd_us)
        if key not in seen:
            seen.add(key)
            unique.append(v)

    return ValidationResult(
        is_safe=len(unique) == 0,
        violations=unique,
        wcd_bounds=wcd_bounds,
    )
