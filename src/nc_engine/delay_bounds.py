from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .curves import (
    ArrivalCurve,
    LeakyBucket,
    MaxPlusServiceCurve,
    RateLatency,
    ServiceCurve,
    aggregate_arrival,
    max_horizontal_deviation,
    residual_service,
)

from src.intent_ontology.types import (
    CBSConfig,
    FlowSemantics,
    GCLConfig,
    TSNBridgeConfig,
)


# ============================================================
# Shared helpers
# ============================================================


def _flow_to_arrival(flow: FlowSemantics, frame_size_bytes: float = 256.0) -> LeakyBucket:
    """Convert flow semantics into a leaky-bucket arrival curve.

    Uses the deadline to infer a rate:  burst = frame_size,  rate = frame_size / period.
    For event-driven flows (period=0) we treat the rate as the burst-rate equivalent.
    """
    # Use the delayable boundary as proxy for period if no explicit period is stored.
    # FlowSemantics doesn't store period — we take it from the task's temporal context.
    # For NC engine we treat each flow as sending `frame_size_bytes` per deadline interval.
    burst = frame_size_bytes  # max one frame in flight
    rate = frame_size_bytes / flow.delayable_boundary_us if flow.delayable_boundary_us > 0 else burst
    return LeakyBucket(burst=burst, rate=rate)


def _to_arrival_list(
    flows: Sequence[FlowSemantics],
    frame_size_bytes: float = 256.0,
) -> Dict[str, LeakyBucket]:
    return {f.flow_id: _flow_to_arrival(f, frame_size_bytes) for f in flows}


# ============================================================
# 1. Strict Priority (SP) — 802.1Qbu
# ============================================================


def compute_sp_bounds(
    flows: Sequence[FlowSemantics],
    link_rate_mbps: float,
    num_priorities: int = 8,
    frame_size_bytes: float = 256.0,
) -> Dict[str, float]:
    """Compute WCD per flow under Strict Priority scheduling.

    For priority level p, the residual service curve is:
        β_p = [β_link - Σ_{j > p} α_j]^+

    where β_link is the link's rate-latency service curve (zero-latency
    work-conserving server).

    Parameters
    ----------
    flows : FlowSemantics list
        Must have ``priority_weight`` set — this determines ordering.
        Higher weight → higher SP priority level.
    link_rate_mbps : float
        Link capacity in Mbps.
    num_priorities : int
        Number of SP priority levels (default 8, matching 802.1Q PCP 0-7).
    frame_size_bytes : float
        Nominal frame size for arrival curve construction.

    Returns
    -------
    wcd_map : dict  flow_id → wcd (μs)
    """
    if not flows:
        return {}

    link_rate_bpus = link_rate_mbps / 8.0
    link_service: ServiceCurve = RateLatency(rate=link_rate_bpus, latency=0.0)

    # Sort flows by priority_weight descending (higher weight = higher priority)
    sorted_flows = sorted(flows, key=lambda f: f.priority_weight, reverse=True)
    arrivals = _to_arrival_list(sorted_flows, frame_size_bytes)

    wcd_map: Dict[str, float] = {}
    higher_aggr_burst = 0.0
    higher_aggr_rate = 0.0

    for flow in sorted_flows:
        flow_arrival = arrivals[flow.flow_id]

        # Residual service after all higher-priority arrivals
        if higher_aggr_rate > 0:
            higher_aggr = LeakyBucket(burst=higher_aggr_burst, rate=higher_aggr_rate)
            residual = residual_service(link_service, higher_aggr)
        else:
            residual = link_service

        if not isinstance(residual, RateLatency):
            wcd_map[flow.flow_id] = float("inf")
        elif residual.rate <= 0:
            wcd_map[flow.flow_id] = float("inf")
        elif residual.rate <= flow_arrival.rate:
            wcd_map[flow.flow_id] = float("inf")
        else:
            wcd_map[flow.flow_id] = max_horizontal_deviation(flow_arrival, residual)

        # Accumulate for next (lower-priority) flow
        higher_aggr_burst += flow_arrival.burst
        higher_aggr_rate += flow_arrival.rate

    return wcd_map


# ============================================================
# 2. Credit-Based Shaper (CBS) — 802.1Qav
# ============================================================


def _compute_cbs_standard(
    flow_arrival: LeakyBucket,
    idle_slope_bpus: float,
    send_slope_bpus: float,
    max_low_prio_frame: float,
    link_rate_bpus: float | None = None,
) -> float:
    """Standard (flawed) CBS delay: β(t) = idleSlope · (t - T_lp)^+

    REFERENCE ONLY — mathematically invalid per Jiang 2024.
    """
    if idle_slope_bpus <= 0:
        return float("inf")
    latency = max_low_prio_frame / send_slope_bpus
    service = RateLatency(rate=idle_slope_bpus, latency=latency)
    return max_horizontal_deviation(flow_arrival, service)


def _compute_cbs_gx_server(
    flow_arrival: LeakyBucket,
    idle_slope_bpus: float,
    send_slope_bpus: float,
    max_low_prio_frame: float,
    link_rate_bpus: float,
) -> float:
    """CBS delay using Jiang 2024 g^x-server corrected model.

    The g^x-server decomposes CBS into:
      - g-server: greedy shaper respecting credit bounds
      - x-server: cross-traffic with max_low_prio_frame interference

    Delay bound (conservative, credit recovery explicitly modeled):
      D_cbs = b / IS                       (own burst drains at idle slope)
            + L_low / C                    (one low-priority frame in flight)
            + L_low * |sendSlope| / (C * IS)   (credit recovery after it)

    where IS = idleSlope, C = link capacity, sendSlope = C - IS per
    802.1Qav.  The interference terms collapse to L_low / IS, but we keep
    them separate so the sendSlope dependency is explicit and checkable.
    """
    if idle_slope_bpus <= 0:
        return float("inf")
    if flow_arrival.rate > idle_slope_bpus:
        return float("inf")
    if link_rate_bpus <= 0:
        return float("inf")

    burst_delay = flow_arrival.burst / idle_slope_bpus
    lp_tx = max_low_prio_frame / link_rate_bpus
    credit_recover = max_low_prio_frame * abs(send_slope_bpus) / (link_rate_bpus * idle_slope_bpus)
    wcd = burst_delay + lp_tx + credit_recover

    return wcd


def compute_cbs_bounds(
    flows: Sequence[FlowSemantics],
    idle_slope_mbps: float,
    send_slope_mbps: float,
    max_burst_bytes: float = 1500.0,
    frame_size_bytes: float = 256.0,
    link_rate_mbps: float = 1000.0,
    use_gx_server: bool = True,
) -> Dict[str, float]:
    """Compute WCD per flow under CBS shaping.

    Parameters
    ----------
    flows : FlowSemantics list
    idle_slope_mbps : float
        Idle slope (reserved bandwidth) in Mbps.
    send_slope_mbps : float
        Send slope in Mbps.
    max_burst_bytes : float
        Maximum low-priority / BE frame size (default MTU=1500).
    frame_size_bytes : float
        Nominal fame size for arrival curve construction.
    use_gx_server : bool
        If True, use Jiang 2024 corrected g^x-server model.
        If False, use standard (mathematically flawed) model for reference.

    Returns
    -------
    wcd_map : dict  flow_id → wcd (μs)
    """
    idle_slope_bpus = idle_slope_mbps / 8.0
    send_slope_bpus = send_slope_mbps / 8.0
    link_rate_bpus = link_rate_mbps / 8.0
    arrivals = _to_arrival_list(flows, frame_size_bytes)

    if use_gx_server:
        compute_fn = _compute_cbs_gx_server
    else:
        compute_fn = _compute_cbs_standard

    return {
        flow.flow_id: compute_fn(
            arrivals[flow.flow_id],
            idle_slope_bpus,
            send_slope_bpus,
            max_burst_bytes,
            link_rate_bpus,
        )
        for flow in flows
    }


# ============================================================
# 3. Time-Aware Shaper (TAS) — 802.1Qbv
# ============================================================


@dataclass
class TASWindowSpec:
    """A single TAS/GCL window specification used by the NC engine."""

    window_id: str
    flow_id: str
    offset_us: float
    window_size_us: float
    guard_band_us: float = 5.0  # default guard band (μs)
    period_us: float | None = None  # stream period; None → hyperperiod-wide worst case


def compute_tas_bounds(
    flows: Sequence[FlowSemantics],
    gcl_config: Dict[str, TASWindowSpec],
    link_rate_mbps: float,
    frame_size_bytes: float = 256.0,
    hyperperiod_us: float | None = None,
) -> Dict[str, float]:
    """Compute WCD per flow under TAS/GCL scheduling.

    The gate window of a stream repeats with its period inside the
    hyperperiod.  A frame that arrives just after its window closes waits
    until the next window opening of the *same period* (or the next
    hyperperiod for aperiodic flows):

        WCD = (cycle - w) + tx + guard_band

    where cycle = period (if known) else hyperperiod, w = window size,
    tx = transmission time, guard = guard band.  This is a deterministic
    upper bound: it is the worst-case phase of a frame relative to its
    gate window.

    Parameters
    ----------
    flows : FlowSemantics list
    gcl_config : dict  flow_id → TASWindowSpec
    link_rate_mbps : float
    frame_size_bytes : float
    hyperperiod_us : float | None
        Hyperperiod; used as the waiting cycle when the stream period is
        unknown.

    Returns
    -------
    wcd_map : dict  flow_id → wcd (μs)
    """
    tx_time = frame_size_bytes * 8.0 / link_rate_mbps  # μs

    result: Dict[str, float] = {}
    for flow in flows:
        spec = gcl_config.get(flow.flow_id)
        if spec is None:
            result[flow.flow_id] = float("inf")
            continue
        cycle = spec.period_us if (spec.period_us is not None and spec.period_us > 0) else (hyperperiod_us or float("inf"))
        if cycle == float("inf"):
            result[flow.flow_id] = float("inf")
            continue
        wcd = (cycle - spec.window_size_us) + tx_time + spec.guard_band_us
        result[flow.flow_id] = wcd

    return result


# ============================================================
# 4. End-to-end concatenation
# ============================================================


def compute_e2e_bounds(
    flow_path: Sequence[str],
    per_hop_bounds: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Concatenate per-hop WCD bounds into end-to-end bounds.

    Uses the pay-bursts-only-once principle: a frame's burst is paid only
    at the first hop, so the E2E WCD satisfies

        WCD_e2e ≤ (first-hop WCD) + Σ_{later hops}(rate-latency latency terms)

    For rate-latency service curves this reduces to a conservative additive
    bound when per-hop WCDs are themselves conservative.  We implement the
    additive bound (WCD_e2e ≤ Σ per-hop WCD), which is valid upper bound
    for concatenated rate-latency servers.

    Parameters
    ----------
    flow_path : list of node IDs that the flow traverses
    per_hop_bounds : dict  node_id → {flow_id → wcd}

    Returns
    -------
    e2e_wcd : dict  flow_id → e2e_wcd (μs)
    """
    # Collect all flow IDs from any hop
    all_flow_ids: set[str] = set()
    for hop_bounds in per_hop_bounds.values():
        all_flow_ids.update(hop_bounds.keys())

    result: Dict[str, float] = {}
    for fid in all_flow_ids:
        total = 0.0
        for node_id in flow_path:
            hop_wcd = per_hop_bounds.get(node_id, {}).get(fid, 0.0)
            if hop_wcd == float("inf"):
                total = float("inf")
                break
            total += hop_wcd
        result[fid] = total

    return result


def compute_e2e_tas_bounds(
    flows: Sequence[FlowSemantics],
    per_hop_specs: Dict[str, Dict[str, TASWindowSpec]],
    link_rate_mbps: float,
    frame_size_bytes: float = 256.0,
    hyperperiod_us: float | None = None,
) -> Dict[str, float]:
    """E2E TAS WCD across hops.

    Per-hop TAS WCD (cycle − w + tx + guard) is computed for every switch
    the flow traverses; the E2E bound is the sum (pay-bursts-only-once is
    not applicable to TAS gating, where each hop re-gates the frame, so the
    additive bound is the standard conservative choice).

    Parameters
    ----------
    flows : FlowSemantics list
    per_hop_specs : dict  node_id → {flow_id → TASWindowSpec}
    link_rate_mbps : float
    frame_size_bytes : float
    hyperperiod_us : float | None

    Returns
    -------
    e2e_wcd : dict  flow_id → e2e_wcd (μs)
    """
    hop_bounds: Dict[str, Dict[str, float]] = {}
    for node_id, specs in per_hop_specs.items():
        hop_bounds[node_id] = compute_tas_bounds(
            flows, specs, link_rate_mbps, frame_size_bytes, hyperperiod_us
        )
    return compute_e2e_bounds(list(per_hop_specs.keys()), hop_bounds)
