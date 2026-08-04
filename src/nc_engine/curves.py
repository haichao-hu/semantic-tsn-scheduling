from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

# ============================================================
# Unit conventions
# ============================================================
# - time:       microseconds (μs)
# - data:       bytes
# - rate:       bytes/μs  (link R_Mbps → R_bps_us = R_Mbps / 8)
# ============================================================


def mbps_to_bpus(rate_mbps: float) -> float:
    """Convert Mbps to bytes/μs.  1 Mbps = 1e6 bits/s = 1 bit/μs = 0.125 B/μs."""
    return rate_mbps / 8.0


def bpus_to_mbps(rate_bpus: float) -> float:
    return rate_bpus * 8.0


def tx_time_bytes(size_bytes: float, rate_mbps: float) -> float:
    """Transmission time in μs for `size_bytes` at `rate_mbps`."""
    return (size_bytes * 8) / rate_mbps


def tx_time_bpus(size_bytes: float, rate_bpus: float) -> float:
    return size_bytes / rate_bpus


# ============================================================
# Arrival Curves
# ============================================================


class ArrivalCurve:
    """Abstract arrival curve  α(t) ≥ 0  describing upper-bound traffic volume.

    Note: concrete subclasses should be @dataclass and define 'burst' as a field.
    """

    def __call__(self, t: float) -> float:
        raise NotImplementedError

    def evaluate_array(self, t: np.ndarray) -> np.ndarray:
        return np.array([self(ti) for ti in t])

    def get_burst(self) -> float:
        """Sustained burst (σ)."""
        return self(0.0)


@dataclass
class LeakyBucket(ArrivalCurve):
    """Leaky-bucket arrival curve:  α(t) = bur + rate·t  for t ≥ 0, zero otherwise.

    This is the workhorse of deterministic NC: every regulated flow can be
    represented this way in the single-flow case.

    Parameters
    ----------
    burst : float
        Maximum instantaneous burst in bytes (σ).
    rate : float
        Sustained average rate in bytes/μs (ρ).
    """

    burst: float
    rate: float

    def __call__(self, t: float) -> float:
        if t < 0:
            return 0.0
        return self.burst + self.rate * t

    def evaluate_array(self, t: np.ndarray) -> np.ndarray:
        out = np.where(t >= 0, self.burst + self.rate * t, 0.0)
        return out


@dataclass
class TSpec(ArrivalCurve):
    """IETF T-SPEC arrival curve for stream-based flows.

    Parameters
    ----------
    max_frame : float
        Maximum frame size (M), bytes.
    peak_rate : float
        Peak sending rate (p), bytes/μs.
    avg_rate : float
        Average sustained rate (r), bytes/μs.
    burst : float
        Maximum burst tolerance (b), bytes.
    """

    max_frame: float
    peak_rate: float
    avg_rate: float
    burst: float

    def __call__(self, t: float) -> float:
        if t < 0:
            return 0.0
        # T-SPEC: α(t) = min(M + p·t, b + r·t)
        return min(
            self.max_frame + self.peak_rate * t,
            self.burst + self.avg_rate * t,
        )

    def to_leaky_bucket(self) -> LeakyBucket:
        """Conservative approximation: drop the M+p·t ramp."""
        return LeakyBucket(burst=self.burst, rate=self.avg_rate)


# ============================================================
# Service Curves (min-plus)
# ============================================================


class ServiceCurve:
    """Abstract service curve  β(t)  — minimum guaranteed service in [0, t].

    Note: concrete subclasses should be @dataclass and define 'rate'/'latency' as fields.
    """

    def __call__(self, t: float) -> float:
        raise NotImplementedError

    def evaluate_array(self, t: np.ndarray) -> np.ndarray:
        return np.array([self(ti) for ti in t])

    def get_rate(self) -> float:
        """Long-term guaranteed rate (bytes/μs)."""
        raise NotImplementedError

    def get_latency(self) -> float:
        """Worst-case latency before first service (μs)."""
        raise NotImplementedError


@dataclass
class RateLatency(ServiceCurve):
    """Rate-latency service curve:  β(t) = R · max(0, t - T).

    This is the canonical model for a FIFO work-conserving server serving
    aggregated traffic with total arrival ≤ R.

    Parameters
    ----------
    rate : float
        Minimum guaranteed service rate (bytes/μs).
    latency : float
        Maximum latency before service starts (μs).
    """

    rate: float
    latency: float = 0.0

    def __call__(self, t: float) -> float:
        if t <= self.latency:
            return 0.0
        return self.rate * (t - self.latency)

    def evaluate_array(self, t: np.ndarray) -> np.ndarray:
        out = np.where(t > self.latency, self.rate * (t - self.latency), 0.0)
        return out


@dataclass
class Staircase(ServiceCurve):
    """Staircase service curve for TDMA / round-robin scheduling.

    β(t) = Σ  r_i · max(0, t - s_i)

    Parameters
    ----------
    rate_increments : tuple of (slot_size, slot_period, offset)
        Each triple (δ, T, φ) adds a rate-latency component with
        rate = δ/T and latency = φ.
    """

    rate_increments: tuple[tuple[float, float, float], ...] = field(default_factory=tuple)

    def __call__(self, t: float) -> float:
        out = 0.0
        for delta, period, offset in rate_increments:
            if t > offset:
                # For staircase, the service accumulates in bursts at each cycle
                n_cycles = math.floor((t - offset) / period) + 1
                out += n_cycles * delta
        return out

    @property
    def rate(self) -> float:
        return sum(d / p for d, p, _ in self.rate_increments)

    @property
    def latency(self) -> float:
        return min(off for _, _, off in self.rate_increments) if self.rate_increments else 0.0


# ============================================================
# Max-plus Service Curve (g^x-server for CBS — Jiang 2024)
# ============================================================


@dataclass
class MaxPlusServiceCurve:
    """Max-plus service curve for the g^x-server (greedy shaper with cross-traffic).

    Models the Credit-Based Shaper (CBS, IEEE 802.1Qav) in max-plus algebra.
    Standard min-plus  idleSlope·t  is **not** a valid service curve for CBS —
    the credit mechanism couples service to arrival history (Jiang 2024, Thm 1).

    The g^x-server decomposes CBS into:
      - g-server: greedy shaper respecting credit bounds
      - x-server: cross-traffic server handling non-AVB interference

    Parameters
    ----------
    idle_slope : float
        Idle slope (bytes/μs) — credit accumulation rate when AVB queue is idle.
    send_slope : float
        Send slope (bytes/μs) — credit depletion rate during AVB transmission.
        send_slope = C - idle_slope  where C is link capacity.
    max_low_prio_frame : float
        Maximum frame size of lower-priority or BE traffic (bytes).
        Sets the worst-case blocking duration.
    """

    idle_slope: float
    send_slope: float
    max_low_prio_frame: float = 1500.0

    @property
    def _credit_replenish_delay(self) -> float:
        """Extra latency from credit replenishment after low-pri interference."""
        return self.max_low_prio_frame * (1.0 / self.idle_slope - 1.0 / (self.idle_slope + self.send_slope))

    def effective_rate_latency(self, flow_burst: float) -> RateLatency:
        """Convert to equivalent min-plus rate-latency for bounding single-flow WCD.

        The resulting RateLatency curve is:
            β(t) = idle_slope · (t - δ)^+
        where  δ = (flow_burst + max_low_prio_frame) / idle_slope  (worst-case).
        """
        latency = (flow_burst + self.max_low_prio_frame) / self.idle_slope
        return RateLatency(rate=self.idle_slope, latency=latency)

    def wcd(
        self,
        flow_burst: float,
        flow_rate: float,
        residual_capacity: float | None = None,
    ) -> float:
        """Compute WCD for a single flow using the g^x-server model.

        Parameters
        ----------
        flow_burst : σ (bytes)
        flow_rate : ρ (bytes/μs)
        residual_capacity : available capacity after higher-priority traffic (bytes/μs).
            If None, uses idle_slope as the service capacity.

        Returns
        -------
        wcd : float (μs)
        """
        capacity = residual_capacity if residual_capacity is not None else self.idle_slope
        if capacity <= flow_rate:
            return float("inf")
        # The g^x-server delay bound from Jiang 2024:
        # D = (b + L_low · (sendSlope / idleSlope + 1)) / idleSlope  … simplified
        #   = b / idleSlope + L_low / idleSlope
        credit_penalty = self.max_low_prio_frame / self.idle_slope
        burst_delay = flow_burst / capacity
        return burst_delay + credit_penalty


# ============================================================
# Curve Operations (min-plus algebra)
# ============================================================


def max_horizontal_deviation(
    arrival: ArrivalCurve,
    service: ServiceCurve,
    t_max: float = 10_000.0,
    n_steps: int = 10_000,
) -> float:
    """Compute WCD as max horizontal deviation:  h(α, β) = sup_{t≥0} inf{d≥0 : α(t) ≤ β(t+d)}.

    For LeakyBucket + RateLatency the closed form is  T + σ/R — this numerical
    method validates that and handles generic curves.
    """
    if isinstance(arrival, LeakyBucket) and isinstance(service, RateLatency):
        if service.rate <= arrival.rate:
            return float("inf")
        return service.latency + arrival.burst / service.rate

    # Numerical sweep
    t_vals = np.linspace(0, t_max, n_steps)
    max_d = 0.0
    for t in t_vals:
        a_t = arrival(t)
        # Binary search for smallest d s.t. α(t) ≤ β(t+d)
        lo, hi = 0.0, t_max
        if service(t + hi) < a_t:
            return float("inf")
        for _ in range(50):
            mid = (lo + hi) / 2
            if service(t + mid) >= a_t:
                hi = mid
            else:
                lo = mid
        d = hi
        if d > max_d:
            max_d = d
    return max_d


def max_vertical_deviation(
    arrival: ArrivalCurve,
    service: ServiceCurve,
    t_max: float = 10_000.0,
    n_steps: int = 10_000,
) -> float:
    """Compute max backlog bound:  v(α, β) = sup_{t≥0} [α(t) - β(t)].

    For LeakyBucket + RateLatency: v = σ + ρ·T.
    """
    if isinstance(arrival, LeakyBucket) and isinstance(service, RateLatency):
        return arrival.burst + arrival.rate * service.latency

    t_vals = np.linspace(0, t_max, n_steps)
    diffs = arrival.evaluate_array(t_vals) - service.evaluate_array(t_vals)
    return float(np.max(diffs))


def min_plus_convolution(arrival: ArrivalCurve, service: ServiceCurve) -> Callable[[float], float]:
    """(α ⊗ β)(t) = inf_{0≤s≤t} [α(s) + β(t-s)].

    Returns a callable, not an ArrivalCurve, because the result shape may not be
    representable in the same parametric form.
    """
    return lambda t: min(arrival(s) + service(t - s) for s in np.linspace(0, t, int(t) + 1))


def min_plus_deconvolution(
    arrival_out: ArrivalCurve,
    service: ServiceCurve,
) -> Callable[[float], float]:
    """(α ⊘ β)(t) = sup_{u≥0} [α(t+u) - β(u)].

    Bounds the input arrival from observed output and known service.
    """
    return lambda t: max(arrival_out(t + u) - service(u) for u in np.linspace(0, 10_000, 1000))


def residual_service(
    service: ServiceCurve,
    higher_priority_arrival: ArrivalCurve,
) -> ServiceCurve:
    """Residual service after higher-priority traffic:  β_res = [β - α_high]^+.

    For RateLatency β(t)=R·(t-T)^+ and LeakyBucket α_high(t)=σ_h+ρ_h·t:
        β_res(t) = (R - ρ_h)·(t - (σ_h + R·T)/(R - ρ_h))^+
    """
    if isinstance(service, RateLatency) and isinstance(higher_priority_arrival, LeakyBucket):
        residual_rate = service.rate - higher_priority_arrival.rate
        if residual_rate <= 0:
            return RateLatency(rate=0.0, latency=float("inf"))
        residual_latency = (
            higher_priority_arrival.burst + service.rate * service.latency
        ) / residual_rate
        return RateLatency(rate=residual_rate, latency=residual_latency)

    # Generic numerical fallback
    def beta_res(t: float) -> float:
        return max(0.0, service(t) - higher_priority_arrival(t))

    # Fit a rate-latency approximation
    t_vals = np.linspace(0, 50_000, 5000)
    vals = np.array([max(0.0, service(t) - higher_priority_arrival(t)) for t in t_vals])
    if np.all(vals == 0):
        return RateLatency(rate=0.0, latency=float("inf"))
    # Simple linear regression on the tail
    tail_mask = t_vals > np.argmax(vals) * (t_vals[1] - t_vals[0])
    if np.any(tail_mask):
        mask = tail_mask
        t_tail = t_vals[mask]
        v_tail = vals[mask]
        if len(t_tail) > 1:
            slope = (v_tail[-1] - v_tail[0]) / (t_tail[-1] - t_tail[0])
            intercept = v_tail[0] - slope * t_tail[0]
            if slope > 0:
                return RateLatency(rate=slope, latency=-intercept / slope)
    return RateLatency(rate=0.0, latency=float("inf"))


# ============================================================
# Aggregate arrival
# ============================================================


def aggregate_arrival(curves: Sequence[ArrivalCurve]) -> ArrivalCurve:
    """Σ α_i — simple worst-case sum for non-coordinated flows."""
    if not curves:
        return LeakyBucket(burst=0.0, rate=0.0)
    total_burst = sum(c.burst for c in curves)
    total_rate = sum(getattr(c, "rate", 0.0) for c in curves)
    return LeakyBucket(burst=total_burst, rate=total_rate)
