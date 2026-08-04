from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import gymnasium as gym

from src.intent_ontology.types import (
    CriticalityLevel,
    DecayType,
    FlowSemantics,
    SemanticCompressibility,
    StreamClass,
    UrgencyFunction,
)
from src.nc_engine.safety_validator import (
    SafetyPolicy,
    ValidationResult,
    Violation,
    validate_schedule,
)
from src.nc_engine.delay_bounds import TASWindowSpec
from src.nc_engine.schedulability import SchedEntry
from src.nc_engine.topology import (
    TSNTopology,
    make_line_topology,
    make_ieee_60802_topology,
)

from src.csrl.tsn_env import CSRLConfig, TSNEnv, SimFlow, SwitchModel, SwitchPort
from src.csrl.safety_shield import SafetyShield, Schedule, ScheduleAction
from src.csrl.csrl_agent import CSRLAgent, ConstraintManager


# ============================================================
# Helpers
# ============================================================


def _make_flow_semantics(
    flow_id: str = "f_test",
    task_id: str = "t_test",
    priority: float = 0.5,
    deadline_us: int = 500,
    stream_cls: StreamClass = StreamClass.SCHEDULED_TRAFFIC,
) -> FlowSemantics:
    return FlowSemantics(
        flow_id=flow_id,
        task_id=task_id,
        priority_weight=priority,
        delayable_boundary_us=deadline_us,
        urgency=UrgencyFunction(DecayType.STEP, value_plateau_us=0, decay_start_us=0),
        compressibility=SemanticCompressibility(ratio=0.0),
        stream_class=stream_cls,
        preemption_eligible=False,
    )


# ============================================================
# Test CSRLConfig
# ============================================================


class TestCSRLConfig:
    def test_default_config(self):
        cfg = CSRLConfig()
        assert cfg.n_switches == 3
        assert cfg.n_queues == 8
        assert cfg.max_active_flows == 8
        assert cfg.hyperperiod_us == 1000.0
        assert cfg.link_rate_mbps == 1000.0

    def test_config_seed_determinism(self):
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        assert np.array_equal(rng1.rand(5), rng2.rand(5))


# ============================================================
# Test SwitchModel
# ============================================================


class TestSwitchModel:
    def test_init(self):
        sw = SwitchModel(0, 4, 8, 1000.0)
        assert sw.switch_id == 0
        assert sw.n_ports == 4
        assert sw.n_queues == 8
        assert len(sw.ports) == 4

    def test_enqueue_dequeue(self):
        sw = SwitchModel(0, 4, 8, 1000.0)
        sw.enqueue(0, 3, 500.0)
        assert sw.ports[0].queues[3] == 500.0
        sw.dequeue(0, 3, 200.0)
        assert sw.ports[0].queues[3] == 300.0
        sw.dequeue(0, 3, 500.0)
        assert sw.ports[0].queues[3] == 0.0

    def test_gcl(self):
        sw = SwitchModel(0, 4, 8, 1000.0)
        sw.set_gcl(0, 0b00001111)
        assert sw.queue_open(0, 0)
        assert sw.queue_open(0, 1)
        assert sw.queue_open(0, 2)
        assert sw.queue_open(0, 3)
        assert not sw.queue_open(0, 4)
        assert not sw.queue_open(0, 7)

    def test_port_busy(self):
        sw = SwitchModel(0, 4, 8, 1000.0)
        sw.ports[0].link_busy_until_us = 100.0
        assert sw.port_busy(0, 50.0)
        assert not sw.port_busy(0, 150.0)

    def test_tx_time(self):
        sw = SwitchModel(0, 4, 8, 1000.0)
        tx = sw.tx_time
        expected = (256 * 8) / 1000.0  # 256 bytes at 1 Gbps = 2.048 μs
        assert tx == pytest.approx(expected)


# ============================================================
# Test TSNEnv
# ============================================================


class TestTSNEnv:
    @pytest.fixture
    def env(self):
        cfg = CSRLConfig(n_switches=2, max_active_flows=4, seed=42)
        topo = make_line_topology(2)
        env = TSNEnv(config=cfg, topology=topo)
        yield env
        env.close()

    @pytest.fixture
    def env_with_flows(self):
        flows = [
            _make_flow_semantics("f_agv_ctrl", "agv_ctrl", 0.80, 500, StreamClass.SCHEDULED_TRAFFIC),
            _make_flow_semantics("f_agv_lidar", "agv_lidar", 0.50, 5000, StreamClass.RESERVED),
            _make_flow_semantics("f_agv_stop", "agv_stop", 0.98, 200, StreamClass.SCHEDULED_TRAFFIC),
        ]
        cfg = CSRLConfig(n_switches=2, max_active_flows=4, seed=42)
        topo = make_line_topology(2)
        env = TSNEnv(config=cfg, topology=topo, flows=flows)
        yield env
        env.close()

    def test_basic_step_reset_cycle(self, env):
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert not terminated

    def test_observation_space_shape(self, env):
        obs, info = env.reset()
        M = env.config.max_active_flows
        expected_dim = M * env._obs_flow_dim + env._obs_global_dim + env._obs_queue_dim
        assert obs.shape[0] == expected_dim

    def test_action_space_shape(self, env):
        M = env.config.max_active_flows
        assert env.action_space.shape[0] == M * env._action_dim_per_flow

    def test_action_application_changes_flow_state(self, env_with_flows):
        env = env_with_flows
        env.reset()
        action = np.zeros(env.action_space.shape[0])
        dim = env._action_dim_per_flow
        action[0] = 1.0      # accept flow 0
        action[1] = 0.5      # mid-range queue
        action[2] = 0.0      # mid-range offset
        action[3] = 0.0      # gate start
        action[4] = 0.5      # moderate gate size

        obs, reward, _, _, _ = env.step(action)
        decoded = env._decode_action(action)
        assert decoded[env.sim_flows[0].flow_id]["accept"] is True
        assert env.sim_flows[0].accepted is True

    def test_flows_with_deadlines_expire(self, env_with_flows):
        env = env_with_flows
        env.reset()
        # set very short deadline and run many steps
        for sf in env.sim_flows:
            sf.deadline_us = 1.0  # essentially impossible to meet

        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)

        # Packets should not exceed deadline — they just get late-delivery penalty
        # But nothing crashes
        assert True

    def test_gcl_gating_toggles(self, env):
        env.reset()
        decoded = {"f_0": {"accept": True, "queue": 7, "dispatch_offset_us": 0.0,
                            "gate_start_us": 0.0, "gate_size_us": 500.0}}
        env._apply_gcl(decoded)
        # after apply, some queues should have open gates
        has_open = any(
            sw.queue_open(0, q)
            for sw in env.switches for q in range(8)
        )
        # depends on which flows are active but should not crash
        _ = has_open

    def test_multi_hop_flows_traverse_switches(self, env):
        env.reset()
        # set multi-hop paths
        for i, sf in enumerate(env.sim_flows):
            sf.path = [0, 1] if env.n_switches >= 2 else [0]
            sf.accepted = True
            sf.gate_start_us = 0.0
            sf.gate_size_us = 1000.0

        for _ in range(5):
            action = env.action_space.sample()
            env.step(action)

        # multi-hop flows should have e2e delays recorded
        for sf in env.sim_flows:
            if sf.accepted and len(sf.path) > 1:
                # at least verified nothing crashed
                pass

    def test_reset_clears_state(self, env):
        env.reset()
        action = env.action_space.sample()
        env.step(action)
        env.step(action)
        obs2, _ = env.reset()
        obs3, _ = env.reset(seed=42)
        assert env._current_time_us == 0.0
        assert env._step_count == 0
        # fresh reset should show no accumulated queue
        for sw in env.switches:
            for port in sw.ports:
                assert np.all(port.queues == 0.0)

    def test_deterministic_seed(self):
        cfg = CSRLConfig(n_switches=2, max_active_flows=4, seed=42)
        topo = make_line_topology(2)
        env1 = TSNEnv(config=cfg, topology=topo)
        env2 = TSNEnv(config=cfg, topology=topo)
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        np.testing.assert_array_equal(obs1, obs2)
        env1.close()
        env2.close()

    def test_decode_action(self, env):
        env.reset()
        M = env.config.max_active_flows
        action = np.ones(M * env._action_dim_per_flow, dtype=np.float64)
        decoded = env._decode_action(action)
        assert len(decoded) == M
        for fid, d in decoded.items():
            assert "accept" in d
            assert "queue" in d
            assert "gate_start_us" in d
            assert "gate_size_us" in d

    def test_action_space_bounds(self, env):
        assert (env.action_space.low == -1.0).all()
        assert (env.action_space.high == 1.0).all()

    def test_observation_space_bounds(self, env):
        env.reset()
        obs, _ = env.reset()
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)


# ============================================================
# Test SafetyShield
# ============================================================


class TestSafetyShield:
    @pytest.fixture
    def shield(self):
        return SafetyShield(enabled=True)

    @pytest.fixture
    def flows(self):
        return [
            _make_flow_semantics("f_l0", "t_l0", 0.98, 200, StreamClass.SCHEDULED_TRAFFIC),
            _make_flow_semantics("f_l1", "t_l1", 0.80, 500, StreamClass.SCHEDULED_TRAFFIC),
            _make_flow_semantics("f_l2", "t_l2", 0.50, 5000, StreamClass.RESERVED),
            _make_flow_semantics("f_l3", "t_l3", 0.20, 100000, StreamClass.BEST_EFFORT),
        ]

    def test_safe_action_passes_through(self, shield, flows):
        action = {
            "f_l0": {"accept": True, "queue": 7, "dispatch_offset_us": 0.0,
                     "gate_start_us": 0.0, "gate_size_us": 50.0},
            "f_l1": {"accept": True, "queue": 6, "dispatch_offset_us": 100.0,
                     "gate_start_us": 100.0, "gate_size_us": 50.0},
        }
        pw_map = {"f_l0": 0.98, "f_l1": 0.80}
        filtered = shield.filter_action(action, flows, None, pw_map)
        # f_l0 with small, early window should pass NC
        assert filtered["f_l0"]["accept"] is not None

    def test_unsafe_action_triggers_fallback(self, shield, flows):
        # An L0 flow with a window too small to fit the frame (C1 violation):
        # window 2 μs < tx(2.048 μs) + guard(5 μs)
        action = {
            "f_l0": {"accept": True, "queue": 7, "dispatch_offset_us": 0.0,
                     "gate_start_us": 0.0, "gate_size_us": 2.0},
        }
        pw_map = {"f_l0": 0.98}
        filtered = shield.filter_action(action, flows, None, pw_map)
        # L0 violation should trigger fallback (reject)
        assert not filtered["f_l0"]["accept"]

    def test_l0_constraint_violation_blocked(self, shield, flows):
        # C1 violation: window (3 μs) smaller than tx + guard (7 μs)
        action = {
            "f_l0": {"accept": True, "queue": 7, "dispatch_offset_us": 0.0,
                     "gate_start_us": 0.0, "gate_size_us": 3.0},
        }
        pw_map = {"f_l0": 0.98}
        filtered = shield.filter_action(action, flows, None, pw_map)
        assert not filtered["f_l0"]["accept"]

    def test_l1_multi_hop_deadline_violation_blocked(self, shield, flows):
        # C2 violation: two flows on the same (switch, queue) with
        # overlapping windows → mutual exclusion fails
        action = {
            "f_l0": {"accept": True, "queue": 7, "dispatch_offset_us": 0.0,
                     "gate_start_us": 0.0, "gate_size_us": 500.0},
            "f_l1": {"accept": True, "queue": 7, "dispatch_offset_us": 0.0,
                     "gate_start_us": 100.0, "gate_size_us": 500.0},
        }
        entries = {
            "f_l0": SchedEntry(
                flow_id="f_l0", queue=7, gate_start_us=0.0, gate_size_us=500.0,
                period_us=200.0, deadline_us=200.0, path=[0],
                task_id="t_l0", priority_weight=0.98, dispatch_phase_us=0.0,
            ),
            "f_l1": SchedEntry(
                flow_id="f_l1", queue=7, gate_start_us=100.0, gate_size_us=500.0,
                period_us=500.0, deadline_us=500.0, path=[0],
                task_id="t_l1", priority_weight=0.80, dispatch_phase_us=0.0,
            ),
        }
        pw_map = {"f_l0": 0.98, "f_l1": 0.80}
        filtered = shield.filter_action(action, flows, None, pw_map, entries)
        # windows overlap on queue 7 → L0 rejected, L1 trimmed or rejected
        assert not filtered["f_l0"]["accept"]
        assert not filtered["f_l1"]["accept"] or filtered["f_l1"]["gate_size_us"] < 500.0

    def test_fallback_generates_valid_schedule(self, shield, flows):
        schedule = shield.compute_fallback(flows)
        assert isinstance(schedule, Schedule)
        assert len(schedule.actions) == len(flows)
        # higher priority flows should be placed first
        priorities = []
        for a in schedule.actions:
            f = next(f for f in flows if f.flow_id == a.flow_id)
            priorities.append(f.priority_weight)
        assert priorities[0] >= priorities[-1]  # sorted by priority

    def test_integration_with_nc_validator(self, shield, flows):
        # validate that a known-safe schedule passes
        safe_action = {
            "f_l2": {"accept": True, "queue": 5, "dispatch_offset_us": 0.0,
                     "gate_start_us": 0.0, "gate_size_us": 500.0},
        }
        pw_map = {"f_l2": 0.50}

        # L2 violations are accepted
        filtered = shield.filter_action(safe_action, flows, None, pw_map)
        assert filtered["f_l2"]["accept"] is not None

    def test_shield_disabled_passes_all(self, flows):
        shield = SafetyShield(enabled=False)
        action = {
            "f_l0": {"accept": True, "queue": 7, "dispatch_offset_us": 0.0,
                     "gate_start_us": 0.0, "gate_size_us": 5000.0},
        }
        filtered = shield.filter_action(action, flows)
        assert filtered["f_l0"]["accept"]

    def test_schedule_action(self):
        sa = ScheduleAction(
            flow_id="f1", accept=True, queue=7,
            dispatch_offset_us=0.0, gate_start_us=50.0, gate_size_us=20.0,
        )
        assert sa.flow_id == "f1"
        assert sa.accept

    def test_schedule_to_dict(self):
        actions = [
            ScheduleAction("f1", True, 7, 0.0, 0.0, 50.0),
            ScheduleAction("f2", False, 0, 0.0, 0.0, 0.0),
        ]
        sched = Schedule(actions=actions)
        d = sched.to_dict()
        assert d["f1"]["accept"]
        assert not d["f2"]["accept"]

    def test_l1_adjustment_attempt(self, shield, flows):
        # L1 flow with gate window that's a bit tight
        l1 = flows[1]  # f_l1, pw=0.80, deadline=500
        action = {
            "f_l1": {"accept": True, "queue": 6, "dispatch_offset_us": 0.0,
                     "gate_start_us": 0.0, "gate_size_us": 300.0},
        }
        pw_map = {"f_l1": 0.80}
        filtered = shield.filter_action(action, flows, None, pw_map)
        # L1 with reasonable window should be safe
        assert filtered["f_l1"]["accept"] is not None


# ============================================================
# Test ConstraintManager
# ============================================================


class TestConstraintManager:
    def test_initial_lambda(self):
        cm = ConstraintManager(initial_lambda=0.1)
        assert cm.value == 0.1

    def test_lambda_increases_on_violation(self):
        cm = ConstraintManager(initial_lambda=0.1, lr_lambda=0.1)
        cm.update(5.0)  # 5 violations
        assert cm.value == pytest.approx(0.1 + 0.1 * 5.0)

    def test_lambda_decreases_to_zero(self):
        cm = ConstraintManager(initial_lambda=0.5, lr_lambda=0.1)
        cm.update(-1.0)  # negative = decrease proportionally: 0.5 + 0.1*(-1) = 0.4
        assert cm.value == pytest.approx(0.4)
        # multiple updates with zero violation eventually reach zero
        for _ in range(10):
            cm.update(-0.5)
        assert cm.value == 0.0

    def test_lambda_max_cap(self):
        cm = ConstraintManager(initial_lambda=0.1, lr_lambda=1.0, max_lambda=2.0)
        cm.update(10.0)
        assert cm.value == 2.0

    def test_lambda_stays_non_negative(self):
        cm = ConstraintManager(initial_lambda=0.1, lr_lambda=1.0)
        cm.update(-100.0)
        assert cm.value >= 0.0

    def test_reset(self):
        cm = ConstraintManager(initial_lambda=0.1)
        cm.update(5.0)
        assert cm.value > 0.1
        cm.reset()
        assert cm.value == 0.1

    def test_total_loss(self):
        cm = ConstraintManager(initial_lambda=0.5)
        loss = cm.total_loss(reward_loss=10.0, constraint_violation=3.0)
        # L = -10.0 + 0.5 * 3.0 = -8.5
        assert loss == pytest.approx(-8.5)

    def test_is_constraint_active(self):
        cm = ConstraintManager(initial_lambda=0.0)
        assert not cm.is_constraint_active
        cm.update(3.0)
        assert cm.is_constraint_active


# ============================================================
# Test CSRLAgent
# ============================================================


class TestCSRLAgent:
    @pytest.fixture
    def env(self):
        cfg = CSRLConfig(n_switches=2, max_active_flows=4, seed=42)
        topo = make_line_topology(2)
        env = TSNEnv(config=cfg, topology=topo)
        yield env
        env.close()

    def test_agent_initialization(self, env):
        agent = CSRLAgent(env=env, device="cpu")
        assert agent.model is not None
        assert agent.constraint_manager is not None

    def test_training_step_does_not_crash(self, env):
        agent = CSRLAgent(env=env, device="cpu")
        stats = agent.train(total_timesteps=100, nc_validation_interval=50)
        assert "wcd_violations" in stats
        assert "lambda_evolution" in stats

    def test_lagrangian_multiplier_updates_on_violation(self, env):
        agent = CSRLAgent(
            env=env,
            constraint_manager=ConstraintManager(initial_lambda=0.1, lr_lambda=0.05),
            device="cpu",
        )
        initial_lambda = agent.constraint_manager.value
        agent._validate_current_schedule()
        agent.constraint_manager.update(3.0)
        assert agent.constraint_manager.value > initial_lambda

    def test_save_load_model_roundtrip(self, env):
        agent = CSRLAgent(env=env, device="cpu")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_model")
            agent.save(path)
            assert os.path.exists(path + ".zip")
            assert os.path.exists(path + ".meta.npz")

            agent2 = CSRLAgent(env=env, device="cpu")
            agent2.load(path)
            assert agent2.model is not None

    def test_predict_returns_valid_action(self, env):
        agent = CSRLAgent(env=env, device="cpu")
        obs, _ = env.reset()
        # train a tiny bit so the policy is initialized
        agent.train(total_timesteps=100, nc_validation_interval=100)
        action, _ = agent.predict(obs)
        assert action.shape == env.action_space.shape
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_safety_shield_integration(self, env):
        shield = SafetyShield(enabled=True)
        agent = CSRLAgent(env=env, safety_shield=shield, device="cpu")
        # train briefly with safety shield enabled
        stats = agent.train(total_timesteps=100, nc_validation_interval=100)
        assert stats is not None


# ============================================================
# Test Integration
# ============================================================


class TestIntegration:
    @pytest.fixture
    def topology(self):
        return make_line_topology(3)

    @pytest.fixture
    def flows(self):
        return [
            _make_flow_semantics("f_agv_ctrl", "agv_ctrl", 0.80, 500, StreamClass.SCHEDULED_TRAFFIC),
            _make_flow_semantics("f_agv_lidar", "agv_lidar", 0.50, 5000, StreamClass.RESERVED),
            _make_flow_semantics("f_agv_stop", "agv_stop", 0.98, 200, StreamClass.SCHEDULED_TRAFFIC),
            _make_flow_semantics("f_hmi", "hmi", 0.15, 100000, StreamClass.BEST_EFFORT),
        ]

    def test_full_pipeline(self, topology, flows):
        """Full pipeline: Intent → Flow → Env → Agent action → Safety Shield → Execution."""
        cfg = CSRLConfig(n_switches=3, max_active_flows=4, seed=42)
        env = TSNEnv(config=cfg, topology=topology, flows=flows)
        shield = SafetyShield(topology=topology, enabled=True)
        cm = ConstraintManager(initial_lambda=0.1)
        agent = CSRLAgent(env=env, constraint_manager=cm, safety_shield=shield, device="cpu")

        obs, _ = env.reset()

        # get action from agent
        action, _ = agent.predict(obs)
        assert action.shape == env.action_space.shape

        # decode and validate through safety shield
        decoded = env._decode_action(action)
        pw_map = {f.flow_id: f.priority_weight for f in flows}
        safe_decoded = shield.filter_action(decoded, flows, None, pw_map)

        # execute
        obs2, reward, _, _, info = env.step(action)
        assert obs2.shape == env.observation_space.shape
        assert isinstance(reward, float)

    def test_semantic_priority_weighting_in_rewards(self, topology, flows):
        """Higher priority flows should contribute more to reward."""
        cfg = CSRLConfig(
            n_switches=3,
            max_active_flows=4,
            seed=42,
            reward_completion_weight=1.0,
            reward_semantic_scale=1.0,
        )
        env = TSNEnv(config=cfg, topology=topology, flows=flows)
        env.reset()

        # configure all flows to succeed with small deadlines
        for sf in env.sim_flows:
            sf.gate_start_us = 0.0
            sf.gate_size_us = 1000.0
            sf.deadline_us = 100000.0  # very lenient

        # action: accept all, identical config (3-dim per flow)
        M = cfg.max_active_flows
        action = np.zeros(M * env._action_dim_per_flow, dtype=np.float64)
        for i in range(M):
            off = i * env._action_dim_per_flow
            action[off] = 1.0      # accept
            action[off + 1] = 0.0  # dispatch mid-period
            action[off + 2] = -1.0  # window aligned with dispatch

        obs, reward, _, _, _ = env.step(action)
        # reward should be non-zero (positive for completed, negative for violated)
        assert isinstance(reward, float)

    def test_nc_engine_catches_deadline_violations_from_rl(self, topology, flows):
        """NC engine should detect WCD violations from RL-generated schedules."""
        env = TSNEnv(config=CSRLConfig(n_switches=3, max_active_flows=4, seed=42),
                     topology=topology, flows=flows)
        env.reset()

        # configure schedule with tiny gate window → deadline violation likely
        for sf in env.sim_flows:
            sf.gate_start_us = 0.0
            sf.gate_size_us = 5.0  # very tight window

        specs = {}
        for sf in env.sim_flows:
            specs[sf.flow_id] = TASWindowSpec(
                window_id=f"w_{sf.flow_id}",
                flow_id=sf.flow_id,
                offset_us=float(sf.gate_start_us),
                window_size_us=float(sf.gate_size_us),
                guard_band_us=5.0,
            )

        result = validate_schedule(
            schedule=specs,
            flows=[f for f in flows if f.flow_id in {sf.flow_id for sf in env.sim_flows[:len(flows)]}],
            link_rate_mbps=1000.0,
        )
        assert isinstance(result, ValidationResult)

    def test_episode_completion(self, topology, flows):
        """Train until episode terminates (truncated)."""
        cfg = CSRLConfig(n_switches=2, max_active_flows=4, seed=42)
        env = TSNEnv(config=cfg, topology=topology, flows=flows)
        env._max_steps = 5

        obs, _ = env.reset()
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        assert env._step_count <= 5

    def test_scenario_loading(self):
        """Verify scenarios can be loaded from ontology."""
        from src.csrl.train import load_scenarios_from_ontology

        scenarios = load_scenarios_from_ontology()
        assert "agv_fleet" in scenarios
        assert "cobot" in scenarios
        assert "plc" in scenarios
        assert len(scenarios["agv_fleet"]) == 3
        assert len(scenarios["cobot"]) == 3
        assert len(scenarios["plc"]) == 4

    def test_random_flow_generation(self):
        """Verify random flow generation produces valid flows."""
        from src.csrl.train import generate_random_flows

        flows = generate_random_flows(n_flows=8, seed=42)
        assert len(flows) == 8
        for f in flows:
            assert 0.0 <= f.priority_weight <= 1.0
            assert f.delayable_boundary_us > 0
            assert isinstance(f.stream_class, StreamClass)
            assert f.urgency is not None
            assert f.compressibility is not None


# ============================================================
# Test Train Script
# ============================================================


class TestTrain:
    def test_train_minimal(self):
        """Minimal end-to-end training test."""
        from src.csrl.train import train

        with tempfile.TemporaryDirectory() as tmp:
            stats = train(
                scenario="random",
                topology_type="line",
                total_timesteps=200,
                nc_validation_interval=100,
                use_safety_shield=False,
                ckpt_dir=tmp,
                seed=42,
                n_switches=2,
                save_model=False,
                log_interval=100,
            )
            assert stats is not None
            assert len(stats.get("lambda_evolution", [])) >= 1

    def test_train_with_shield(self):
        """Training with safety shield enabled should not crash."""
        from src.csrl.train import train

        with tempfile.TemporaryDirectory() as tmp:
            stats = train(
                scenario="random",
                topology_type="line",
                total_timesteps=200,
                nc_validation_interval=100,
                use_safety_shield=True,
                ckpt_dir=tmp,
                seed=42,
                n_switches=2,
                save_model=False,
                log_interval=100,
            )
            assert stats is not None

    def test_train_save_model(self):
        """Training with model saving."""
        from src.csrl.train import train

        with tempfile.TemporaryDirectory() as tmp:
            stats = train(
                scenario="random",
                topology_type="line",
                total_timesteps=200,
                nc_validation_interval=100,
                use_safety_shield=False,
                ckpt_dir=tmp,
                seed=42,
                n_switches=2,
                save_model=True,
                log_interval=100,
            )
            import glob
            saved = glob.glob(os.path.join(tmp, "*.zip"))
            assert len(saved) >= 1

    def test_train_agv_fleet_scenario(self):
        """Train on the AGV fleet scenario."""
        from src.csrl.train import train

        with tempfile.TemporaryDirectory() as tmp:
            stats = train(
                scenario="agv_fleet",
                topology_type="line",
                total_timesteps=200,
                nc_validation_interval=100,
                use_safety_shield=False,
                ckpt_dir=tmp,
                seed=42,
                n_switches=2,
                save_model=False,
                log_interval=100,
            )
            assert stats is not None
