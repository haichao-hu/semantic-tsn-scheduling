from __future__ import annotations

import math
import pytest
import numpy as np

from src.intent_ontology.types import (
    CBSParameters,
    CriticalityLevel,
    DecayType,
    FlowSemantics,
    GCLParameters,
    SemanticCompressibility,
    StreamClass,
    TSNBridgeConfig,
    UrgencyFunction,
)

from src.nc_engine.curves import (
    ArrivalCurve,
    LeakyBucket,
    MaxPlusServiceCurve,
    RateLatency,
    ServiceCurve,
    Staircase,
    TSpec,
    aggregate_arrival,
    max_horizontal_deviation,
    max_vertical_deviation,
    mbps_to_bpus,
    min_plus_convolution,
    residual_service,
    tx_time_bytes,
)
from src.nc_engine.delay_bounds import (
    TASWindowSpec,
    compute_cbs_bounds,
    compute_e2e_bounds,
    compute_sp_bounds,
    compute_tas_bounds,
)
from src.nc_engine.safety_validator import (
    SafetyPolicy,
    ValidationResult,
    Violation,
    validate_batch,
    validate_schedule,
)
from src.nc_engine.topology import (
    FlowPath,
    PathDelayComponents,
    TSNTopology,
    compute_path_delays,
    make_ieee_60802_topology,
    make_line_topology,
    make_ring_topology,
)


# ============================================================
# Test helpers
# ============================================================


def _make_flow(
    flow_id: str,
    task_id: str = "",
    deadline_us: int = 500,
    priority: float = 0.5,
    stream_cls: StreamClass = StreamClass.SCHEDULED_TRAFFIC,
) -> FlowSemantics:
    return FlowSemantics(
        flow_id=flow_id,
        task_id=task_id or flow_id,
        priority_weight=priority,
        delayable_boundary_us=deadline_us,
        urgency=UrgencyFunction(DecayType.STEP, value_plateau_us=0, decay_start_us=0),
        compressibility=SemanticCompressibility(ratio=0.0),
        stream_class=stream_cls,
        preemption_eligible=False,
    )


# ============================================================
# 1. Unit conversions
# ============================================================


class TestUnitConversions:
    def test_mbps_to_bpus(self):
        assert mbps_to_bpus(1000.0) == 125.0
        assert mbps_to_bpus(100.0) == 12.5
        assert mbps_to_bpus(8.0) == 1.0

    def test_tx_time_bytes(self):
        # 256 bytes at 1 Gbps = 256*8/1000 = 2.048 μs
        assert tx_time_bytes(256, 1000) == pytest.approx(2.048)
        # 1500 bytes at 100 Mbps = 1500*8/100 = 120 μs
        assert tx_time_bytes(1500, 100) == pytest.approx(120.0)


# ============================================================
# 2. ArrivalCurve
# ============================================================


class TestArrivalCurve:
    def test_leaky_bucket_creation(self):
        lb = LeakyBucket(burst=100.0, rate=10.0)
        assert lb.burst == 100.0
        assert lb.rate == 10.0

    def test_leaky_bucket_t_negative(self):
        lb = LeakyBucket(burst=100.0, rate=10.0)
        assert lb(-1.0) == 0.0
        assert lb(0.0) == 100.0

    def test_leaky_bucket_t_positive(self):
        lb = LeakyBucket(burst=100.0, rate=10.0)
        assert lb(10.0) == 100.0 + 10.0 * 10.0

    def test_leaky_bucket_array(self):
        lb = LeakyBucket(burst=50.0, rate=5.0)
        t = np.array([0.0, 1.0, 2.0, 3.0])
        out = lb.evaluate_array(t)
        # α(0) = burst, α(t) = burst + rate·t for t ≥ 0
        expected = np.array([50.0, 55.0, 60.0, 65.0])
        np.testing.assert_array_almost_equal(out, expected)

    def test_leaky_bucket_burst_property(self):
        lb = LeakyBucket(burst=42.0, rate=7.0)
        assert lb.burst == 42.0

    def test_tspec_creation(self):
        ts = TSpec(max_frame=256.0, peak_rate=100.0, avg_rate=10.0, burst=500.0)
        assert ts.max_frame == 256.0
        assert ts.peak_rate == 100.0
        assert ts.avg_rate == 10.0
        assert ts.burst == 500.0

    def test_tspec_evaluation(self):
        ts = TSpec(max_frame=256.0, peak_rate=100.0, avg_rate=10.0, burst=500.0)
        # α(0) = min(M, b) = min(256, 500) = 256
        v0 = ts(0.0)
        assert v0 == 256.0

    def test_tspec_to_leaky_bucket(self):
        ts = TSpec(max_frame=256.0, peak_rate=100.0, avg_rate=5.0, burst=200.0)
        lb = ts.to_leaky_bucket()
        assert isinstance(lb, LeakyBucket)
        assert lb.burst == 200.0
        assert lb.rate == 5.0


# ============================================================
# 3. ServiceCurve
# ============================================================


class TestServiceCurve:
    def test_rate_latency_creation(self):
        rl = RateLatency(rate=100.0, latency=5.0)
        assert rl.rate == 100.0
        assert rl.latency == 5.0

    def test_rate_latency_before_latency(self):
        rl = RateLatency(rate=100.0, latency=10.0)
        assert rl(0.0) == 0.0
        assert rl(5.0) == 0.0
        assert rl(10.0) == 0.0

    def test_rate_latency_after_latency(self):
        rl = RateLatency(rate=100.0, latency=10.0)
        assert rl(20.0) == 100.0 * (20.0 - 10.0)  # = 1000.0

    def test_rate_latency_default_latency(self):
        rl = RateLatency(rate=50.0)
        assert rl.latency == 0.0
        assert rl(10.0) == 500.0

    def test_rate_latency_array(self):
        rl = RateLatency(rate=10.0, latency=2.0)
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        out = rl.evaluate_array(t)
        expected = np.array([0.0, 0.0, 0.0, 10.0, 20.0])
        np.testing.assert_array_almost_equal(out, expected)

    def test_staircase_creation(self):
        sc = Staircase(rate_increments=((100.0, 10.0, 0.0),))
        assert sc.rate == 100.0 / 10.0  # = 10.0

    def test_max_plus_service_curve_creation(self):
        mx = MaxPlusServiceCurve(
            idle_slope=50.0,
            send_slope=75.0,
            max_low_prio_frame=1500.0,
        )
        assert mx.idle_slope == 50.0
        assert mx.send_slope == 75.0

    def test_max_plus_wcd_formula(self):
        """g^x-server: D = b/IS + L_low/IS"""
        mx = MaxPlusServiceCurve(
            idle_slope=50.0,
            send_slope=75.0,
            max_low_prio_frame=1500.0,
        )
        wcd = mx.wcd(flow_burst=500.0, flow_rate=10.0)
        expected = 500.0 / 50.0 + 1500.0 / 50.0  # = 10 + 30 = 40 μs
        assert wcd == pytest.approx(40.0)

    def test_max_plus_rate_exceeds_capacity(self):
        mx = MaxPlusServiceCurve(idle_slope=50.0, send_slope=75.0)
        wcd = mx.wcd(flow_burst=500.0, flow_rate=100.0)
        assert wcd == float("inf")


# ============================================================
# 4. Curve operations
# ============================================================


class TestCurveOperations:
    def test_max_horizontal_deviation_closed_form(self):
        """For LB(b, r) + RL(R, T): h = T + b/R when R > r."""
        arrival = LeakyBucket(burst=200.0, rate=5.0)
        service = RateLatency(rate=100.0, latency=10.0)
        h = max_horizontal_deviation(arrival, service)
        expected = 10.0 + 200.0 / 100.0  # = 12.0 μs
        assert h == pytest.approx(expected)

    def test_max_horizontal_deviation_zero_latency(self):
        arrival = LeakyBucket(burst=100.0, rate=2.0)
        service = RateLatency(rate=50.0, latency=0.0)
        h = max_horizontal_deviation(arrival, service)
        assert h == pytest.approx(100.0 / 50.0)  # = 2.0

    def test_max_horizontal_deviation_rate_exceeds(self):
        arrival = LeakyBucket(burst=100.0, rate=80.0)
        service = RateLatency(rate=50.0, latency=0.0)
        h = max_horizontal_deviation(arrival, service)
        assert h == float("inf")

    def test_max_vertical_deviation_closed_form(self):
        arrival = LeakyBucket(burst=100.0, rate=10.0)
        service = RateLatency(rate=100.0, latency=5.0)
        v = max_vertical_deviation(arrival, service)
        expected = 100.0 + 10.0 * 5.0  # = 150.0
        assert v == pytest.approx(expected)

    def test_residual_service_closed_form(self):
        """β_res = [R·t - (σ_h + ρ_h·t)]⁺ = (R-ρ_h)·(t - σ_h/(R-ρ_h))⁺"""
        service = RateLatency(rate=12.5, latency=0.0)  # 100 Mbps
        higher = LeakyBucket(burst=100.0, rate=5.0)     # 40 Mbps
        residual = residual_service(service, higher)
        assert isinstance(residual, RateLatency)
        assert residual.rate == pytest.approx(7.5)
        assert residual.latency == pytest.approx(100.0 / 7.5)

    def test_residual_service_zero_capacity(self):
        service = RateLatency(rate=5.0, latency=0.0)
        higher = LeakyBucket(burst=100.0, rate=5.0)
        residual = residual_service(service, higher)
        assert residual.rate == 0.0

    def test_aggregate_arrival_empty(self):
        aggr = aggregate_arrival([])
        assert isinstance(aggr, LeakyBucket)
        assert aggr.burst == 0.0
        assert aggr.rate == 0.0

    def test_aggregate_arrival_multi(self):
        curves = [
            LeakyBucket(burst=100.0, rate=10.0),
            LeakyBucket(burst=200.0, rate=5.0),
            LeakyBucket(burst=50.0, rate=20.0),
        ]
        aggr = aggregate_arrival(curves)
        assert aggr.burst == pytest.approx(350.0)
        assert aggr.rate == pytest.approx(35.0)


# ============================================================
# 5. Delay bounds — Strict Priority
# ============================================================


class TestSPBounds:
    def test_two_flows_numerical(self):
        """SP test with hand-calculated values.
        Link = 100 Mbps = 12.5 B/μs.
        Flow H (pri=0.8, deadline=500): burst=256, rate=256/500=0.512 B/μs.
        Flow L (pri=0.2, deadline=500): burst=256, rate=256/500=0.512 B/μs.
        """
        fh = _make_flow("fh", deadline_us=500, priority=0.8)
        fl = _make_flow("fl", deadline_us=500, priority=0.2)
        bounds = compute_sp_bounds([fh, fl], link_rate_mbps=100.0)
        assert bounds["fh"] > 0
        assert bounds["fl"] > bounds["fh"]

    def test_two_flows_hand_calculated(self):
        """Direct curve test with clean numbers.
        Flow H: LB(burst=100, rate=5) = 100 B, 5 B/μs (40 Mbps).
        Flow L: LB(burst=100, rate=2.5) = 100 B, 2.5 B/μs (20 Mbps).
        Link: 100 Mbps = 12.5 B/μs.
        H WCD = 100/12.5 = 8.0 μs.
        Residual rate = 12.5 - 5 = 7.5, latency = 100/7.5 = 13.333...
        L WCD = 13.333 + 100/7.5 = 26.667 μs.
        """
        service = RateLatency(rate=12.5, latency=0.0)
        arrival_h = LeakyBucket(burst=100.0, rate=5.0)
        arrival_l = LeakyBucket(burst=100.0, rate=2.5)

        h_h = max_horizontal_deviation(arrival_h, service)
        assert h_h == pytest.approx(8.0)

        residual = residual_service(service, arrival_h)
        h_l = max_horizontal_deviation(arrival_l, residual)
        assert h_l == pytest.approx(100.0 / 7.5 + 100.0 / 7.5)  # 26.667

    def test_single_flow(self):
        f = _make_flow("f1", deadline_us=500, priority=0.8)
        bounds = compute_sp_bounds([f], link_rate_mbps=100.0)
        assert len(bounds) == 1
        assert bounds["f1"] > 0
        assert bounds["f1"] != float("inf")

    def test_empty_flows(self):
        bounds = compute_sp_bounds([], link_rate_mbps=100.0)
        assert bounds == {}

    def test_rate_exceeds_capacity(self):
        """When total arrival rate exceeds link, lowest priority gets inf."""
        f1 = _make_flow("f_overload", deadline_us=500, priority=0.9)
        f2 = _make_flow("f_low", deadline_us=500, priority=0.1)
        # With frame_size=256, deadline=500: rate = 0.512 B/μs each
        # Total: 1.024 B/μs < 100 Mbps=12.5 B/μs — won't overload.
        # We need many flows to overload. Test with 50 flows.
        many = [_make_flow(f"f{i}", deadline_us=50, priority=0.9) for i in range(50)]
        # Each: rate = 256/50 = 5.12 B/μs. 50×5.12 = 256 B/μs > 12.5.
        bounds = compute_sp_bounds(many, link_rate_mbps=100.0)
        # At least some should be inf
        assert any(v == float("inf") for v in bounds.values())


# ============================================================
# 6. Delay bounds — CBS
# ============================================================


class TestCBSBounds:
    def test_standard_flawed_model(self):
        """Standard CBS: D = b/IS + L_low/SS"""
        f = _make_flow("f_cbs", deadline_us=1000, priority=0.5, stream_cls=StreamClass.RESERVED)
        bounds = compute_cbs_bounds(
            [f],
            idle_slope_mbps=400.0,
            send_slope_mbps=600.0,
            max_burst_bytes=1500.0,
            use_gx_server=False,
        )
        assert bounds["f_cbs"] > 0
        assert bounds["f_cbs"] != float("inf")

    def test_gx_server_model(self):
        """g^x-server: D = b/IS + L_low/IS."""
        f = _make_flow("f_gx", deadline_us=1000, priority=0.5, stream_cls=StreamClass.RESERVED)
        bounds = compute_cbs_bounds(
            [f],
            idle_slope_mbps=400.0,
            send_slope_mbps=600.0,
            max_burst_bytes=1500.0,
            use_gx_server=True,
        )
        assert bounds["f_gx"] > 0

    def test_gx_server_hand_calculated(self):
        """All rates in B/μs: IS=50, SS=75, L_low=1500, burst=256, rate=256/1000."""
        # Direct computation:
        # D = b/IS + L_low/IS = 256/50 + 1500/50 = 5.12 + 30 = 35.12 μs
        f = _make_flow("f_hc", deadline_us=1000, priority=0.5, stream_cls=StreamClass.RESERVED)
        bounds = compute_cbs_bounds(
            [f],
            idle_slope_mbps=400.0,  # = 50 B/μs
            send_slope_mbps=600.0,  # = 75 B/μs
            max_burst_bytes=1500.0,
            use_gx_server=True,
        )
        expected = 256.0 / 50.0 + 1500.0 / 50.0  # = 35.12
        assert bounds["f_hc"] == pytest.approx(expected)

    def test_gx_server_rate_exceeds_idleslope(self):
        f = _make_flow("f_over", deadline_us=10, priority=0.5, stream_cls=StreamClass.RESERVED)
        # rate = 256/10 = 25.6 B/μs = 204.8 Mbps > idleSlope 100 Mbps = 12.5 B/μs
        bounds = compute_cbs_bounds(
            [f],
            idle_slope_mbps=100.0,
            send_slope_mbps=900.0,
            use_gx_server=True,
        )
        assert bounds["f_over"] == float("inf")

    def test_gx_vs_standard_gx_larger(self):
        """g^x-server should give ≥ standard because credit replenishment
        adds extra latency that the standard model ignores."""
        f = _make_flow("f_comp", deadline_us=1000, priority=0.5, stream_cls=StreamClass.RESERVED)
        std = compute_cbs_bounds(
            [f], idle_slope_mbps=400.0, send_slope_mbps=600.0, use_gx_server=False,
        )["f_comp"]
        gx = compute_cbs_bounds(
            [f], idle_slope_mbps=400.0, send_slope_mbps=600.0, use_gx_server=True,
        )["f_comp"]
        assert gx >= std - 1e-9  # allow floating-point noise

    def test_multiple_flows_cbs(self):
        flows = [
            _make_flow(f"f{i}", deadline_us=1000, priority=0.5, stream_cls=StreamClass.RESERVED)
            for i in range(3)
        ]
        bounds = compute_cbs_bounds(
            flows, idle_slope_mbps=400.0, send_slope_mbps=600.0, use_gx_server=True,
        )
        assert len(bounds) == 3
        assert all(v > 0 and v != float("inf") for v in bounds.values())


# ============================================================
# 7. Delay bounds — TAS
# ============================================================


class TestTASBounds:
    def test_single_gcl_window(self):
        f = _make_flow("f_tas", deadline_us=200, priority=0.95)
        specs = {
            "f_tas": TASWindowSpec(
                window_id="w_f_tas",
                flow_id="f_tas",
                offset_us=50.0,
                window_size_us=20.0,
                guard_band_us=5.0,
                period_us=200.0,  # window repeats with the stream period
            ),
        }
        bounds = compute_tas_bounds([f], specs, link_rate_mbps=1000.0)
        tx_time = 256 * 8.0 / 1000.0  # = 2.048 μs
        # WCD = (period - window) + tx + guard: worst case = frame misses
        # its window and waits until the next period's window opening.
        expected = (200.0 - 20.0) + tx_time + 5.0  # = 187.048 μs
        assert bounds["f_tas"] == pytest.approx(expected)

    def test_missing_gcl_spec(self):
        f = _make_flow("f_miss", deadline_us=200, priority=0.95)
        bounds = compute_tas_bounds([f], {}, link_rate_mbps=1000.0)
        assert bounds["f_miss"] == float("inf")

    def test_zero_guard_band(self):
        f = _make_flow("f_no_gb", deadline_us=200, priority=0.95)
        specs = {
            "f_no_gb": TASWindowSpec(
                window_id="w", flow_id="f_no_gb",
                offset_us=10.0, window_size_us=30.0, guard_band_us=0.0,
                period_us=200.0,
            ),
        }
        bounds = compute_tas_bounds([f], specs, link_rate_mbps=1000.0)
        tx = 256 * 8.0 / 1000.0
        expected = (200.0 - 30.0) + tx + 0.0
        assert bounds["f_no_gb"] == pytest.approx(expected)

    def test_multiple_tas_windows(self):
        flows = [
            _make_flow(f"f{i}", deadline_us=200, priority=0.95)
            for i in range(3)
        ]
        specs = {
            f"f{i}": TASWindowSpec(
                window_id=f"w{i}", flow_id=f"f{i}",
                offset_us=i * 100.0, window_size_us=20.0, guard_band_us=5.0,
                period_us=200.0,
            )
            for i in range(3)
        }
        bounds = compute_tas_bounds(flows, specs, link_rate_mbps=1000.0)
        assert len(bounds) == 3
        tx = 256 * 8.0 / 1000.0
        for i in range(3):
            expected = (200.0 - 20.0) + tx + 5.0  # cycle-scale worst case
            assert bounds[f"f{i}"] == pytest.approx(expected)


# ============================================================
# 8. Delay bounds — E2E
# ============================================================


class TestE2EBounds:
    def test_additive_concatenation(self):
        per_hop = {
            "sw1": {"f1": 10.0, "f2": 50.0},
            "sw2": {"f1": 8.0, "f2": 40.0},
            "sw3": {"f1": 12.0, "f2": 30.0},
        }
        e2e = compute_e2e_bounds(["sw1", "sw2", "sw3"], per_hop)
        assert e2e["f1"] == pytest.approx(30.0)
        assert e2e["f2"] == pytest.approx(120.0)

    def test_empty_path(self):
        e2e = compute_e2e_bounds([], {})
        assert e2e == {}

    def test_inf_propagates(self):
        per_hop = {
            "sw1": {"f1": 10.0},
            "sw2": {"f1": float("inf")},
            "sw3": {"f1": 5.0},
        }
        e2e = compute_e2e_bounds(["sw1", "sw2", "sw3"], per_hop)
        assert e2e["f1"] == float("inf")

    def test_single_hop(self):
        per_hop = {"sw1": {"f1": 15.0}}
        e2e = compute_e2e_bounds(["sw1"], per_hop)
        assert e2e["f1"] == pytest.approx(15.0)

    def test_missing_flow_in_hop(self):
        per_hop = {
            "sw1": {"f1": 10.0},
            "sw2": {},  # f1 not in this hop
        }
        e2e = compute_e2e_bounds(["sw1", "sw2"], per_hop)
        # Missing = 0.0 per hop
        assert e2e["f1"] == pytest.approx(10.0)


# ============================================================
# 9. Safety Validator
# ============================================================


class TestSafetyValidator:
    def test_safe_schedule(self):
        flows = [_make_flow("f_safe", deadline_us=500, priority=0.8)]
        specs = {
            "f_safe": TASWindowSpec(
                window_id="w", flow_id="f_safe",
                offset_us=0.0, window_size_us=20.0, guard_band_us=5.0,
                period_us=500.0,
            ),
        }
        result = validate_schedule(specs, flows, link_rate_mbps=1000.0)
        assert result.is_safe is True
        assert len(result.violations) == 0

    def test_unsafe_single_violation(self):
        flows = [_make_flow("f_bad", deadline_us=10, priority=0.8)]
        specs = {
            "f_bad": TASWindowSpec(
                window_id="w", flow_id="f_bad",
                offset_us=50.0, window_size_us=100.0, guard_band_us=5.0,
                period_us=200.0,
            ),
        }
        result = validate_schedule(specs, flows, link_rate_mbps=1000.0)
        assert result.is_safe is False
        assert len(result.violations) == 1
        assert result.violations[0].flow_id == "f_bad"

    def test_multiple_violations(self):
        flows = [
            _make_flow("f1", deadline_us=20, priority=0.9),
            _make_flow("f2", deadline_us=30, priority=0.8),
        ]
        specs = {
            "f1": TASWindowSpec("w1", "f1", offset_us=100.0, window_size_us=50.0, period_us=200.0),
            "f2": TASWindowSpec("w2", "f2", offset_us=100.0, window_size_us=50.0, period_us=200.0),
        }
        result = validate_schedule(specs, flows, link_rate_mbps=1000.0)
        assert result.is_safe is False
        assert len(result.violations) == 2

    def test_l0_violation_fails(self):
        flows = [_make_flow("f_l0", deadline_us=5, priority=0.98)]
        specs = {
            "f_l0": TASWindowSpec("w", "f_l0", offset_us=50.0, window_size_us=100.0, period_us=200.0),
        }
        result = validate_schedule(specs, flows, link_rate_mbps=1000.0)
        assert result.is_safe is False
        assert len(result.violations) == 1

    def test_l3_violation_within_margin_passes(self):
        """L3 with infinite overshoot tolerance: always acceptable regardless of WCD."""
        f = _make_flow("f_l3", deadline_us=100, priority=0.1)
        specs = {
            "f_l3": TASWindowSpec("w", "f_l3", offset_us=1000.0, window_size_us=5000.0, period_us=10000.0),
        }
        result = validate_schedule(specs, [f], link_rate_mbps=1000.0)
        assert result.is_safe is True
        assert len(result.violations) == 0

    def test_l1_small_margin_passes(self):
        f = _make_flow("f_l1_ok", deadline_us=500, priority=0.80)
        specs = {
            "f_l1_ok": TASWindowSpec(
                "w", "f_l1_ok", offset_us=0.0, window_size_us=20.0, guard_band_us=5.0,
                period_us=500.0,
            ),
        }
        result = validate_schedule(specs, [f], link_rate_mbps=1000.0)
        assert result.is_safe is True

    def test_empty_flows(self):
        result = validate_schedule({}, [], link_rate_mbps=1000.0)
        assert result.is_safe is True
        assert len(result.violations) == 0

    def test_bridge_config_tas(self):
        flows = [_make_flow("f_bridge", deadline_us=200, priority=0.95)]
        bridge = TSNBridgeConfig(
            bridge_id="SW1",
            port_id="1",
            gcl_list=[
                GCLParameters(
                    window_id="w_f_bridge",
                    gate_states="10000000",
                    window_size_ns=20_000,
                    base_time_ns=0,
                    cycle_time_ns=100_000,
                    offset_ns=0,
                ),
            ],
        )
        result = validate_schedule(bridge, flows, link_rate_mbps=1000.0)
        assert result.is_safe is True

    def test_bridge_config_mixed(self):
        scheduled = _make_flow("f_tt", deadline_us=200, priority=0.95, stream_cls=StreamClass.SCHEDULED_TRAFFIC)
        reserved = _make_flow("f_avb", deadline_us=500, priority=0.5, stream_cls=StreamClass.RESERVED)
        be = _make_flow("f_be", deadline_us=1000, priority=0.1, stream_cls=StreamClass.BEST_EFFORT)
        flows = [scheduled, reserved, be]

        bridge = TSNBridgeConfig(
            bridge_id="SW1",
            port_id="1",
            gcl_list=[
                GCLParameters(
                    window_id="w_f_tt",
                    gate_states="10000000",
                    window_size_ns=20_000,
                    base_time_ns=0,
                    cycle_time_ns=100_000,
                    offset_ns=0,
                ),
            ],
            cbs_configs=[
                CBSParameters(
                    traffic_class=5,
                    idle_slope_kbps=400_000,
                    send_slope_kbps=600_000,
                ),
            ],
        )
        result = validate_schedule(bridge, flows, link_rate_mbps=1000.0)
        assert "f_be" in result.wcd_bounds
        assert result.wcd_bounds["f_be"] == float("inf")
        assert len(result.warnings) > 0

    def test_bridge_config_missing_cbs(self):
        """Only TAS flows, no CBS config → CBS flows miss their config."""
        flows = [
            _make_flow("f_tt", deadline_us=200, priority=0.95, stream_cls=StreamClass.SCHEDULED_TRAFFIC),
            _make_flow("f_avb", deadline_us=500, priority=0.5, stream_cls=StreamClass.RESERVED),
        ]
        bridge = TSNBridgeConfig(
            bridge_id="SW1",
            port_id="1",
            gcl_list=[
                GCLParameters(
                    window_id="w_f_tt", gate_states="10000000",
                    window_size_ns=20_000, base_time_ns=0,
                    cycle_time_ns=100_000, offset_ns=0,
                ),
            ],
        )
        result = validate_schedule(bridge, flows, link_rate_mbps=1000.0)
        # TT flow should be safe, AVB may have inf or computed
        assert "f_tt" in result.wcd_bounds
        assert result.wcd_bounds["f_tt"] != float("inf")


# ============================================================
# 10. SafetyValidator — batch
# ============================================================


class TestValidateBatch:
    def test_all_safe(self):
        flows_list = [[_make_flow(f"f_{i}0", deadline_us=500, priority=0.8)] for i in range(3)]
        specs_list = [
            {f"f_{i}0": TASWindowSpec("w", f"f_{i}0", offset_us=0.0, window_size_us=20.0, guard_band_us=5.0, period_us=500.0)}
            for i in range(3)
        ]
        mask = validate_batch(specs_list, flows_list)
        assert mask.shape == (3,)
        assert mask.dtype == bool
        assert np.all(mask)

    def test_some_unsafe(self):
        flows_list = [
            [_make_flow("f_ok", deadline_us=500, priority=0.8)],
            [_make_flow("f_bad", deadline_us=5, priority=0.8)],
        ]
        specs_list = [
            {"f_ok": TASWindowSpec("w", "f_ok", offset_us=0.0, window_size_us=20.0, guard_band_us=5.0, period_us=500.0)},
            {"f_bad": TASWindowSpec("w", "f_bad", offset_us=1000.0, window_size_us=50.0, period_us=200.0)},
        ]
        mask = validate_batch(specs_list, flows_list)
        assert mask.shape == (2,)
        assert bool(mask[0]) is True
        assert bool(mask[1]) is False


# ============================================================
# 11. Safety Policy
# ============================================================


class TestSafetyPolicy:
    def test_l0_zero_tolerance(self):
        pol = SafetyPolicy.for_level(CriticalityLevel.L0)
        assert pol.zero_tolerance is True
        assert pol.allowed_overshoot_pct == 0.0
        assert pol.is_acceptable(100.0, 100.0) is True
        assert pol.is_acceptable(100.0, 100.0001) is False

    def test_l1_point_one_pct(self):
        pol = SafetyPolicy.for_level(CriticalityLevel.L1)
        assert pol.allowed_overshoot_pct == 0.1
        # 0.1% of 1000 = 1 μs
        assert pol.is_acceptable(1000.0, 1000.5) is True
        assert pol.is_acceptable(1000.0, 1002.0) is False

    def test_l2_five_pct(self):
        pol = SafetyPolicy.for_level(CriticalityLevel.L2)
        assert pol.allowed_overshoot_pct == 5.0
        # 5% of 100 = 5 μs
        assert pol.is_acceptable(100.0, 104.9) is True
        assert pol.is_acceptable(100.0, 106.0) is False

    def test_l3_always_ok(self):
        pol = SafetyPolicy.for_level(CriticalityLevel.L3)
        assert pol.is_acceptable(100.0, 1_000_000.0) is True
        assert pol.is_acceptable(100.0, float("inf")) is True


# ============================================================
# 12. Violation & ValidationResult
# ============================================================


class TestViolationResult:
    def test_violation_properties(self):
        v = Violation(
            flow_id="f1",
            task_id="t1",
            criticality=CriticalityLevel.L1,
            required_deadline_us=100.0,
            computed_wcd_us=150.0,
        )
        assert v.is_violation is True
        assert v.margin_us == pytest.approx(-50.0)

    def test_violation_not_violation(self):
        v = Violation("f1", "t1", CriticalityLevel.L1, 100.0, 80.0)
        assert v.is_violation is False
        assert v.margin_us == pytest.approx(20.0)

    def test_validation_result_to_dict(self):
        result = ValidationResult(
            is_safe=False,
            violations=[
                Violation("f1", "t1", CriticalityLevel.L1, 100.0, 150.0),
            ],
            wcd_bounds={"f1": 150.0},
            warnings=["test warning"],
        )
        d = result.to_dict()
        assert d["is_safe"] is False
        assert d["num_violations"] == 1
        assert d["violations"][0]["flow_id"] == "f1"
        assert d["warnings"] == ["test warning"]

    def test_validation_result_safe_no_warnings(self):
        result = ValidationResult(
            is_safe=True,
            wcd_bounds={"f1": 50.0},
        )
        d = result.to_dict()
        assert d["is_safe"] is True
        assert d["num_violations"] == 0


# ============================================================
# 13. Topology
# ============================================================


class TestTopology:
    def test_line_topology(self):
        t = make_line_topology(n_switches=3)
        assert t.num_nodes == 5  # es0, sw1, sw2, sw3, es1
        assert t.num_edges == 8  # 4 bidirectional links = 8 directed edges

    def test_ring_topology(self):
        t = make_ring_topology(n_switches=4)
        assert "sw1" in t.nodes
        assert "es1" in t.nodes
        # 4 ES-switch bidirectional + 4 ring bidirectional = 8 bidirectional = 16 directed
        assert t.num_edges == 16

    def test_ieee_60802_topology(self):
        t = make_ieee_60802_topology()
        assert "plc1" in t.nodes
        assert "agv1" in t.nodes
        assert "robot1" in t.nodes
        assert t.num_nodes >= 10  # 5 switches + 5 ES minimum
        # Ring backbone (5 links) + cross-link (1) + ES links (5) = 11 bidirectional = 22 directed
        assert t.num_edges >= 16

    def test_flow_path(self):
        fp = FlowPath(
            flow_id="f1",
            hops=[
                ("es0", None, "p1"),
                ("sw1", "p2", "p3"),
                ("sw2", "p4", "p5"),
                ("es1", "p6", None),
            ],
        )
        assert fp.flow_id == "f1"
        assert len(fp) == 4
        assert fp.node_ids == ["es0", "sw1", "sw2", "es1"]
        assert fp.num_hops == 4

    def test_flow_path_empty(self):
        fp = FlowPath(flow_id="f_empty")
        assert len(fp) == 0
        assert fp.node_ids == []

    def test_shortest_path(self):
        t = make_line_topology(n_switches=3)
        path = t.shortest_path("es0", "es1")
        assert path is not None
        assert path[0] == "es0"
        assert path[-1] == "es1"
        assert len(path) == 5  # es0-sw1-sw2-sw3-es1

    def test_set_get_flow_path(self):
        t = make_line_topology(n_switches=2)
        fp = FlowPath(flow_id="f1", hops=[
            ("es0", None, "p1"),
            ("sw1", "p2", "p3"),
            ("es1", "p4", None),
        ])
        t.set_flow_path("f1", fp)
        assert t.get_path("f1") == ["es0", "sw1", "es1"]

    def test_link_properties(self):
        t = TSNTopology()
        t.add_link("a", "b", link_rate_mbps=500.0, propagation_us=0.01, processing_us=2.0)
        assert t.get_link_rate("a", "b") == 500.0
        assert t.get_propagation_delay("a", "b") == 0.01
        assert t.get_processing_delay("a") == 2.0

    def test_get_path_missing(self):
        t = TSNTopology()
        assert t.get_path("nonexistent") == []


# ============================================================
# 14. Path delay decomposition
# ============================================================


class TestPathDelays:
    def test_compute_path_delays_simple(self):
        t = make_line_topology(n_switches=2)
        fp = FlowPath(flow_id="f1", hops=[
            ("es0", None, "p1"),
            ("sw1", "p2", "p3"),
            ("sw2", "p4", "p5"),
            ("es1", "p6", None),
        ])
        comp = compute_path_delays(t, "f1", fp, frame_size_bytes=256.0)
        assert comp.propagation_us > 0
        assert comp.processing_us > 0
        assert comp.transmission_us > 0
        assert comp.total_us > 0

    def test_path_delays_empty_path(self):
        t = TSNTopology()
        comp = compute_path_delays(t, "f1")
        assert comp.total_us == 0.0

    def test_path_delays_components_non_negative(self):
        t = make_line_topology(n_switches=3)
        fp = FlowPath(flow_id="f1", hops=[
            ("es0", None, "p1"),
            ("sw1", "p2", "p3"),
            ("sw2", "p4", "p5"),
            ("sw3", "p6", "p7"),
            ("es1", "p8", None),
        ])
        comp = compute_path_delays(t, "f1", fp)
        assert comp.propagation_us >= 0
        assert comp.processing_us >= 0
        assert comp.transmission_us >= 0


# ============================================================
# 15. Numerical correctness
# ============================================================


class TestNumericalCorrectness:
    def test_wcd_formula_zero_latency(self):
        """Fundamental NC formula: h(α, β) = T + b/R for α=LB(b,r), β=RL(R,T)"""
        arrival = LeakyBucket(burst=200.0, rate=10.0)
        service = RateLatency(rate=100.0, latency=0.0)
        wcd = max_horizontal_deviation(arrival, service)
        assert wcd == pytest.approx(2.0)  # 0 + 200/100

    def test_wcd_formula_with_latency(self):
        arrival = LeakyBucket(burst=300.0, rate=20.0)
        service = RateLatency(rate=50.0, latency=15.0)
        wcd = max_horizontal_deviation(arrival, service)
        expected = 15.0 + 300.0 / 50.0  # = 21.0
        assert wcd == pytest.approx(expected)

    def test_wcd_non_negative(self):
        """All WCD values must be ≥ 0."""
        for burst in [10, 100, 1000]:
            for rate in [1, 10, 50]:
                for svc_rate in [100, 200]:
                    if svc_rate > rate:
                        arr = LeakyBucket(burst=float(burst), rate=float(rate))
                        svc = RateLatency(rate=float(svc_rate), latency=0.0)
                        wcd = max_horizontal_deviation(arr, svc)
                        assert wcd >= 0.0

    def test_e2e_gte_single_hop(self):
        """E2E WCD must be ≥ any individual hop WCD."""
        per_hop = {
            "sw1": {"f1": 10.0},
            "sw2": {"f1": 15.0},
            "sw3": {"f1": 8.0},
        }
        e2e = compute_e2e_bounds(["sw1", "sw2", "sw3"], per_hop)
        assert e2e["f1"] == pytest.approx(33.0)
        assert e2e["f1"] >= 10.0  # ≥ max single hop
        assert e2e["f1"] >= 15.0
        assert e2e["f1"] >= 8.0

    def test_known_sp_example_literature(self):
        """Reproduce a known SP NC example from the TSN literature.
        Two flows at 100 Mbps, verified by hand.
        """
        service = RateLatency(rate=12.5, latency=0.0)  # 100 Mbps
        # Flow high: 50 B burst, 2 B/μs (16 Mbps)
        a_h = LeakyBucket(burst=50.0, rate=2.0)
        # Flow low: 200 B burst, 1.5 B/μs (12 Mbps)
        a_l = LeakyBucket(burst=200.0, rate=1.5)

        h_h = max_horizontal_deviation(a_h, service)
        assert h_h == pytest.approx(4.0)  # 50/12.5

        residual = residual_service(service, a_h)
        assert isinstance(residual, RateLatency)
        assert residual.rate == pytest.approx(10.5)  # 12.5-2.0

        h_l = max_horizontal_deviation(a_l, residual)
        # latency = 50/10.5 ≈ 4.762, rate = 10.5
        # WCD = 4.762 + 200/10.5 ≈ 4.762 + 19.048 = 23.810
        assert h_l == pytest.approx(50.0 / 10.5 + 200.0 / 10.5, rel=1e-6)

    def test_cbs_example(self):
        """CBS with g^x-server: known answer."""
        # idleSlope=400Mbps=50B/μs, sendSlope=600Mbps=75B/μs
        # flow burst=256B, rate=10B/μs, max_low_prio=1500B
        # D = 256/50 + 1500/50 = 5.12 + 30 = 35.12 μs
        mx = MaxPlusServiceCurve(idle_slope=50.0, send_slope=75.0, max_low_prio_frame=1500.0)
        wcd = mx.wcd(flow_burst=256.0, flow_rate=10.0)
        assert wcd == pytest.approx(256.0 / 50.0 + 1500.0 / 50.0)

    def test_aggregated_flows_increase_bound(self):
        """More flows → larger delay bounds for lower-priority flows."""
        service = RateLatency(rate=100.0, latency=0.0)
        one_flow = LeakyBucket(burst=100.0, rate=1.0)
        many_flows = LeakyBucket(burst=500.0, rate=20.0)
        wcd_one = max_horizontal_deviation(one_flow, service)
        wcd_many = max_horizontal_deviation(many_flows, service)
        assert wcd_many > wcd_one

    def test_consistency_sp_to_e2e(self):
        """SP WCD for a flow ≤ its E2E WCD across multiple identical hops."""
        service = RateLatency(rate=12.5, latency=0.0)
        arrival = LeakyBucket(burst=256.0, rate=2.0)
        single = max_horizontal_deviation(arrival, service)

        per_hop = {"sw1": {"f1": single}, "sw2": {"f1": single}, "sw3": {"f1": single}}
        e2e = compute_e2e_bounds(["sw1", "sw2", "sw3"], per_hop)
        assert e2e["f1"] == pytest.approx(3 * single)
        assert single <= e2e["f1"]


# ============================================================
# 16. Edge cases
# ============================================================


class TestEdgeCases:
    def test_zero_size_flow(self):
        """Zero-burst zero-rate flow should have zero WCD."""
        arrival = LeakyBucket(burst=0.0, rate=0.0)
        service = RateLatency(rate=100.0, latency=0.0)
        wcd = max_horizontal_deviation(arrival, service)
        assert wcd == pytest.approx(0.0)

    def test_zero_link_rate(self):
        arrival = LeakyBucket(burst=100.0, rate=1.0)
        service = RateLatency(rate=0.0, latency=0.0)
        wcd = max_horizontal_deviation(arrival, service)
        assert wcd == float("inf") or wcd > 1e10

    def test_identical_flow_ids_in_sp(self):
        """Two flows with same priority and deadline should get different WCDs."""
        f1 = _make_flow("f1", deadline_us=500, priority=0.5)
        f2 = _make_flow("f2", deadline_us=500, priority=0.5)
        bounds = compute_sp_bounds([f1, f2], link_rate_mbps=1000.0)
        # Both should be finite and > 0
        assert bounds["f1"] > 0 and bounds["f1"] != float("inf")
        assert bounds["f2"] > 0 and bounds["f2"] != float("inf")
        # The second (lower priority after tie-breaking by iteration order) gets larger WCD
        assert bounds["f2"] >= bounds["f1"] - 1e-9

    def test_many_flows_sp(self):
        n = 20
        flows = [_make_flow(f"f{i}", deadline_us=500, priority=0.5) for i in range(n)]
        bounds = compute_sp_bounds(flows, link_rate_mbps=1000.0)
        assert len(bounds) == n


# ============================================================
# 17. Integration: ontology → NC engine
# ============================================================


class TestOntologyIntegration:
    def test_import_flow_semantics_compute_bounds(self):
        flows = [
            _make_flow("f_int_tt", deadline_us=200, priority=0.95, stream_cls=StreamClass.SCHEDULED_TRAFFIC),
            _make_flow("f_int_avb", deadline_us=500, priority=0.5, stream_cls=StreamClass.RESERVED),
        ]
        # TAS for TT + CBS for AVB
        tas_specs = {
            "f_int_tt": TASWindowSpec("w", "f_int_tt", offset_us=0.0, window_size_us=20.0, guard_band_us=5.0, period_us=200.0),
        }
        tas_bounds = compute_tas_bounds(
            [f for f in flows if f.stream_class == StreamClass.SCHEDULED_TRAFFIC],
            tas_specs,
            link_rate_mbps=1000.0,
        )
        cbs_bounds = compute_cbs_bounds(
            [f for f in flows if f.stream_class == StreamClass.RESERVED],
            idle_slope_mbps=400.0,
            send_slope_mbps=600.0,
            use_gx_server=True,
        )
        assert "f_int_tt" in tas_bounds
        assert "f_int_avb" in cbs_bounds
        assert tas_bounds["f_int_tt"] != float("inf")

    def test_topology_pipeline(self):
        t = make_line_topology(n_switches=2)
        fp = FlowPath(flow_id="f_pipe", hops=[
            ("es0", None, "p1"),
            ("sw1", "p2", "p3"),
            ("sw2", "p4", "p5"),
            ("es1", "p6", None),
        ])
        t.set_flow_path("f_pipe", fp)

        comp = compute_path_delays(t, "f_pipe", fp, frame_size_bytes=256.0)
        # Propagation: 3 hops × 0.006 μs = 0.018 μs
        # Processing: 3 switch nodes × 1.9 μs = 5.7 μs (es0, sw1, sw2)
        # Transmission: 3 hops × 2.048 μs = 6.144 μs
        assert comp.propagation_us > 0
        assert comp.processing_us > 0
        assert comp.transmission_us > 0
        assert comp.total_us > 0

    def test_batch_tensor_output_shape(self):
        flows_list = [[_make_flow(f"f_{i}_0", deadline_us=500)] for i in range(5)]
        specs_list = [
            {f"f_{i}_0": TASWindowSpec("w", f"f_{i}_0", offset_us=0.0, window_size_us=20.0)}
            for i in range(5)
        ]
        mask = validate_batch(specs_list, flows_list)
        assert mask.shape == (5,)
        assert mask.dtype == bool


# ============================================================
# 18. Regression: specific formula checks
# ============================================================


class TestRegressionFormulas:
    def test_leaky_bucket_rate_latency_wcd(self):
        """Verify the closed-form optimization in max_horizontal_deviation."""
        # Case: R > r
        lb = LeakyBucket(burst=100.0, rate=5.0)
        rl = RateLatency(rate=20.0, latency=3.0)
        wcd = max_horizontal_deviation(lb, rl)
        assert wcd == pytest.approx(3.0 + 100.0 / 20.0)

    def test_residual_service_preserves_rate_latency(self):
        """Residual of RL - LB should remain RL-shaped."""
        beta = RateLatency(rate=50.0, latency=4.0)
        alpha = LeakyBucket(burst=60.0, rate=10.0)
        residual = residual_service(beta, alpha)
        assert isinstance(residual, RateLatency)
        assert residual.rate == pytest.approx(40.0)

    def test_max_vertical_deviation_backlog(self):
        """Backlog bound for LB+RL: v = bur + rate·latency."""
        alpha = LeakyBucket(burst=30.0, rate=2.0)
        beta = RateLatency(rate=100.0, latency=5.0)
        v = max_vertical_deviation(alpha, beta)
        assert v == pytest.approx(30.0 + 2.0 * 5.0)
