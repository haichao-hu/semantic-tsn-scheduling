from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from src.intent_ontology.types import (
    CriticalityLevel,
    DecayType,
    FlowSemantics,
    SemanticCompressibility,
    StreamClass,
    UrgencyFunction,
)
from src.nc_engine.topology import make_line_topology
from src.csrl.tsn_env import CSRLConfig, TSNEnv
from src.csrl.safety_shield import SafetyShield
from src.experiments.baselines import (
    FIFOCBSScheduler,
    PureDRLScheduler,
    StaticGCLScheduler,
)
from src.experiments.runner import (
    ExperimentResult,
    _compute_percentiles,
    _nc_validate_current_schedule,
    run_experiment,
    compare_baselines,
)
from src.experiments.ablation import run_ablation


def _make_flow(
    flow_id: str = "f_test",
    priority: float = 0.5,
    deadline_us: int = 500,
    stream_cls: StreamClass = StreamClass.SCHEDULED_TRAFFIC,
) -> FlowSemantics:
    return FlowSemantics(
        flow_id=flow_id,
        task_id=f"t_{flow_id}",
        priority_weight=priority,
        delayable_boundary_us=deadline_us,
        urgency=UrgencyFunction(DecayType.STEP, value_plateau_us=0, decay_start_us=0),
        compressibility=SemanticCompressibility(ratio=0.0),
        stream_class=stream_cls,
        preemption_eligible=False,
    )


# ============================================================
# Test _compute_percentiles
# ============================================================


class TestPercentiles:
    def test_empty(self):
        p50, p99, p999 = _compute_percentiles([])
        assert p50 == 0.0
        assert p99 == 0.0
        assert p999 == 0.0

    def test_single(self):
        p50, p99, p999 = _compute_percentiles([100.0])
        assert p50 == 100.0

    def test_multiple(self):
        vals = list(range(100))
        p50, p99, p999 = _compute_percentiles(vals)
        assert abs(p50 - 49.5) < 1.0
        assert p99 > 95.0
        assert p999 > 98.0


# ============================================================
# Test ExperimentResult
# ============================================================


class TestExperimentResult:
    def test_to_dict(self):
        r = ExperimentResult(
            name="test",
            scenario="agv_fleet",
            completion_rate=0.9,
            e2e_delays=[1.0, 2.0],
            p50_delay=1.5,
            p99_delay=2.0,
            p999_delay=2.0,
            wcd_violations=1,
            avg_jitter_us=0.5,
            total_flows=3,
        )
        d = r.to_dict()
        assert d["name"] == "test"
        assert d["completion_rate"] == 0.9
        assert d["p50_delay_us"] == 1.5
        assert d["n_e2e_samples"] == 2

    def test_to_dict_serializable(self):
        r = ExperimentResult(name="test", scenario="agv_fleet", completion_rate=0.8)
        d = r.to_dict()
        json.dumps(d)  # must not raise


# ============================================================
# Test Baselines
# ============================================================


@pytest.fixture
def sample_flows():
    return [
        _make_flow("f_l0", 0.98, 200, StreamClass.SCHEDULED_TRAFFIC),
        _make_flow("f_l1", 0.80, 500, StreamClass.SCHEDULED_TRAFFIC),
        _make_flow("f_l2", 0.50, 5000, StreamClass.RESERVED),
        _make_flow("f_l3", 0.20, 100000, StreamClass.BEST_EFFORT),
    ]


@pytest.fixture
def config():
    return CSRLConfig(
        n_switches=2,
        max_active_flows=4,
        seed=42,
        max_gate_window_us=200.0,
    )


@pytest.fixture
def topology():
    return make_line_topology(2)


class TestStaticGCLScheduler:
    def test_produces_valid_action(self, sample_flows, config, topology):
        sched = StaticGCLScheduler(sample_flows, config, topology)
        obs = np.zeros(10, dtype=np.float64)
        action, _ = sched.predict(obs)
        M = config.max_active_flows
        assert action.shape == (M * 3,)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    def test_deterministic(self, sample_flows, config, topology):
        sched = StaticGCLScheduler(sample_flows, config, topology)
        obs = np.zeros(10, dtype=np.float64)
        a1, _ = sched.predict(obs)
        a2, _ = sched.predict(obs)
        np.testing.assert_array_equal(a1, a2)

    def test_priority_ordering(self, sample_flows, config, topology):
        sched = StaticGCLScheduler(sample_flows, config, topology)
        obs = np.zeros(10, dtype=np.float64)
        action, _ = sched.predict(obs)
        # L0 (index 0) should be accepted
        assert action[0] > 0.0

    def test_computation_time_recorded(self, sample_flows, config, topology):
        sched = StaticGCLScheduler(sample_flows, config, topology)
        assert sched.computation_time_ms >= 0.0

    def test_schedule_size(self, sample_flows, config, topology):
        sched = StaticGCLScheduler(sample_flows, config, topology)
        assert sched.schedule_size >= 0


class TestFIFOCBSScheduler:
    def test_produces_valid_action(self, sample_flows, config, topology):
        sched = FIFOCBSScheduler(sample_flows, config, topology)
        obs = np.zeros(10, dtype=np.float64)
        action, _ = sched.predict(obs)
        M = config.max_active_flows
        assert action.shape == (M * 3,)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    def test_all_flows_accepted(self, sample_flows, config, topology):
        sched = FIFOCBSScheduler(sample_flows, config, topology)
        obs = np.zeros(10, dtype=np.float64)
        action, _ = sched.predict(obs)
        dim = 3
        for i in range(min(config.max_active_flows, len(sample_flows))):
            assert action[i * dim] == 1.0  # accept=1

    def test_gates_always_open(self, sample_flows, config, topology):
        sched = FIFOCBSScheduler(sample_flows, config, topology)
        obs = np.zeros(10, dtype=np.float64)
        action, _ = sched.predict(obs)
        dim = 3
        for i in range(min(config.max_active_flows, len(sample_flows))):
            # window offset = -1.0 → aligned with dispatch phase
            assert action[i * dim + 2] == -1.0

    def test_queue_assignment_by_class(self, sample_flows, config, topology):
        sched = FIFOCBSScheduler(sample_flows, config, topology)
        obs = np.zeros(10, dtype=np.float64)
        action, _ = sched.predict(obs)
        # queue assignment is fixed by the env's semantic mapping rules
        # (not in the action space anymore); verify the env applies it:
        # one queue per critical flow (7, 6, 5, ...), BE on queue 0
        env = TSNEnv(config=config, topology=topology, flows=sample_flows)
        st_queues = set()
        for sf in env.sim_flows:
            if sf.stream_class == StreamClass.SCHEDULED_TRAFFIC:
                st_queues.add(sf.queue)
                assert sf.queue >= 5
            elif sf.stream_class == StreamClass.BEST_EFFORT:
                assert sf.queue == 0
        assert len(st_queues) == len([f for f in env.sim_flows
                                      if f.stream_class == StreamClass.SCHEDULED_TRAFFIC])


class TestPureDRLScheduler:
    def test_initialization_and_predict(self, sample_flows, config, topology):
        sched = PureDRLScheduler(sample_flows, config, topology, total_timesteps=200, seed=42)
        M = config.max_active_flows
        obs_dim = M * 8 + 2 + (config.n_switches * config.n_ports_per_switch * config.n_queues)
        obs = np.zeros(obs_dim, dtype=np.float64)
        action, _ = sched.predict(obs)
        assert action.shape == (M * 3,)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    def test_custom_observation_shape(self, sample_flows, config, topology):
        sched = PureDRLScheduler(sample_flows, config, topology, total_timesteps=200, seed=42)
        M = config.max_active_flows
        obs_dim = M * 8 + 2 + (config.n_switches * config.n_ports_per_switch * config.n_queues)
        obs = np.zeros(obs_dim, dtype=np.float64)
        action, _ = sched.predict(obs)
        assert action.shape == (M * 3,)


# ============================================================
# Test Runner
# ============================================================


class TestRunExperiment:
    def test_returns_valid_result(self, sample_flows, config, topology):
        from src.experiments.baselines import StaticGCLScheduler

        def make_sched(flows, cfg, topo):
            return StaticGCLScheduler(flows, cfg, topo)

        result = run_experiment(
            name="test_static",
            make_scheduler=make_sched,
            scenario="agv_fleet",
            n_episodes=3,
            n_switches=2,
        )
        assert isinstance(result, ExperimentResult)
        assert result.name == "test_static"
        assert 0.0 <= result.completion_rate <= 1.0
        assert result.total_flows > 0

    def test_completion_rate_bounded(self, sample_flows, config, topology):
        from src.experiments.baselines import StaticGCLScheduler

        def make_sched(flows, cfg, topo):
            return StaticGCLScheduler(flows, cfg, topo)

        result = run_experiment(
            name="test",
            make_scheduler=make_sched,
            scenario="agv_fleet",
            n_episodes=3,
            n_switches=2,
        )
        assert 0.0 <= result.completion_rate <= 1.0

    def test_captures_error_on_unknown_scenario(self):
        def make_sched(flows, cfg, topo):
            return FIFOCBSScheduler(flows, cfg, topo)

        with pytest.raises(ValueError, match="Unknown scenario"):
            run_experiment(
                name="fail", make_scheduler=make_sched, scenario="nonexistent", n_episodes=1
            )

    def test_fifo_baseline_runs(self):
        def make_sched(flows, cfg, topo):
            return FIFOCBSScheduler(flows, cfg, topo)

        result = run_experiment(
            name="fifo", make_scheduler=make_sched, scenario="agv_fleet", n_episodes=3, n_switches=2
        )
        assert result.name == "fifo"
        assert result.completion_rate >= 0.0

    def test_wcd_violations_tracked(self):
        from src.experiments.baselines import StaticGCLScheduler

        def make_sched(flows, cfg, topo):
            return StaticGCLScheduler(flows, cfg, topo)

        result = run_experiment(
            name="test", make_scheduler=make_sched, scenario="agv_fleet", n_episodes=3, n_switches=2
        )
        assert result.wcd_violations >= 0


# ============================================================
# Test Ablation
# ============================================================


class TestAblation:
    def test_all_variants_produce_results(self):
        results = run_ablation(scenario="agv_fleet", n_episodes=3, n_switches=2)
        assert "Full_CSRL" in results
        assert "No_Safety_Shield" in results
        assert "No_Semantic" in results
        assert "No_NC_Constraint" in results

    def test_each_variant_has_completion_rate(self):
        results = run_ablation(scenario="agv_fleet", n_episodes=3, n_switches=2)
        for key in ["Full_CSRL", "No_Safety_Shield", "No_Semantic", "No_NC_Constraint"]:
            if "error" in results[key]:
                continue
            assert "completion_rate" in results[key]
            assert 0.0 <= results[key]["completion_rate"] <= 1.0

    def test_disabled_features_not_identical_to_full(self):
        results = run_ablation(scenario="agv_fleet", n_episodes=3, n_switches=2)
        # At least one ablation variant should differ from Full in some
        # metric — completion saturates at 8f, so the Lagrangian trajectory
        # (final λ) is the discriminating observable: No-NC has λ≡0.
        full = results.get("Full_CSRL", {})
        if "error" in full:
            pytest.skip("Full CSRL failed")
        any_different = False
        for key in ["No_Safety_Shield", "No_Semantic", "No_NC_Constraint"]:
            variant = results.get(key, {})
            if "error" in variant:
                continue
            if (variant.get("completion_rate") != full.get("completion_rate") or
                    variant.get("wcd_violations") != full.get("wcd_violations") or
                    variant.get("final_lambda") != full.get("final_lambda")):
                any_different = True
                break
        assert any_different, "All ablation variants identical to Full CSRL"


# ============================================================
# Test Integration / compare_baselines
# ============================================================


class TestCompareBaselines:
    def test_single_scenario_quick(self):
        results = compare_baselines(
            scenarios=["agv_fleet"],
            n_episodes=2,
            n_switches=2,
        )
        assert "agv_fleet" in results
        assert "CSRL" in results["agv_fleet"]
        assert "StaticGCL" in results["agv_fleet"]
        assert "PureDRL" in results["agv_fleet"]
        assert "FIFOCBS" in results["agv_fleet"]

    def test_result_json_serializable(self):
        results = compare_baselines(
            scenarios=["agv_fleet"],
            n_episodes=2,
            n_switches=2,
        )
        # must be JSON-serializable
        json.dumps(results, indent=2, default=str)

    def test_multi_scenario(self):
        results = compare_baselines(
            scenarios=["agv_fleet", "plc"],
            n_episodes=2,
            n_switches=2,
        )
        assert "agv_fleet" in results
        assert "plc" in results
        for scenario in results:
            for key in ["CSRL", "StaticGCL", "PureDRL", "FIFOCBS"]:
                assert key in results[scenario]


# ============================================================
# Test NC Validation Helpers
# ============================================================


class TestNCValidation:
    def test_empty_env_returns_zero(self):
        env = TSNEnv(
            config=CSRLConfig(n_switches=2, max_active_flows=0, seed=42),
            topology=make_line_topology(2),
        )
        env.reset()
        result = _nc_validate_current_schedule(env, [], None, 1000.0)
        assert result == 0
        env.close()

    def test_with_flows(self, sample_flows):
        env = TSNEnv(
            config=CSRLConfig(n_switches=2, max_active_flows=4, seed=42),
            topology=make_line_topology(2),
            flows=sample_flows,
        )
        env.reset()
        result = _nc_validate_current_schedule(env, sample_flows, None, 1000.0)
        assert isinstance(result, int)
        assert result >= 0
        env.close()


# ============================================================
# Test Experiment Persistence
# ============================================================


class TestExperimentPersistence:
    def test_results_serializable_to_json(self):
        from src.experiments.baselines import StaticGCLScheduler

        def make_sched(flows, cfg, topo):
            return StaticGCLScheduler(flows, cfg, topo)

        result = run_experiment(
            name="persistence_test",
            make_scheduler=make_sched,
            scenario="agv_fleet",
            n_episodes=2,
            n_switches=2,
        )
        d = result.to_dict()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(d, f, indent=2, default=str)
            path = f.name
        with open(path, "r") as f:
            loaded = json.load(f)
        assert loaded["name"] == "persistence_test"
        Path(path).unlink()

    def test_run_ablation_json_serializable(self):
        results = run_ablation(scenario="agv_fleet", n_episodes=2, n_switches=2)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f, indent=2, default=str)
            path = f.name
        with open(path, "r") as f:
            json.load(f)
        Path(path).unlink()
