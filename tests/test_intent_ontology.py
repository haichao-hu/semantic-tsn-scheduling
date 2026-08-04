from __future__ import annotations

import json
import math

import pytest

from src.intent_ontology.types import (
    CriticalityLevel,
    CriticalityProfile,
    DataDependency,
    DataDependencyGraph,
    DDGEdge,
    DecayType,
    DependencyType,
    EscalationRule,
    FlowSemantics,
    SemanticCompressibility,
    StreamClass,
    TaskIntent,
    TaskType,
    TemporalConstraints,
    UrgencyFunction,
)
from src.intent_ontology.encoder import (
    CRITICALITY_PCP,
    CRITICALITY_TC,
    CRITICALITY_QUEUE_PRIORITY,
    IntentEncoder,
)
from src.intent_ontology.mapper import QoSMapper, _compute_gcl_window_ns
from src.intent_ontology.examples import (
    load_all_scenarios,
    parse_scenario_yaml,
    scenario_to_yaml,
    agv_fleet_scenario,
    cobot_scenario,
    plc_scenario,
    agv_path_tracking,
    agv_lidar_scan,
    agv_emergency_stop,
    AGV_FLEET_SCENARIO_YAML,
    COBOT_SCENARIO_YAML,
    PLC_SCENARIO_YAML,
)


# ============================================================
# 辅助
# ============================================================


def _make_intent(
    task_id: str = "test_001",
    task_type: TaskType = TaskType.PERIODIC_CONTROL,
    criticality: CriticalityLevel = CriticalityLevel.L1,
    deadline_us: int = 500,
    period_us: int = 1000,
    dependencies: list[DataDependency] | None = None,
) -> TaskIntent:
    return TaskIntent(
        task_id=task_id,
        task_type=task_type,
        agent_id="test_agent",
        criticality=CriticalityProfile(base_level=criticality),
        temporal=TemporalConstraints(period_us=period_us, deadline_us=deadline_us),
        dependencies=dependencies or [],
    )


# ============================================================
# 1. Table 6.2: Criticality → PCP / TC / Queue mapping
# ============================================================


class TestCriticalityMapping:
    def test_l0_maps_to_pcp7(self):
        assert CRITICALITY_PCP[CriticalityLevel.L0] == 7
        assert CRITICALITY_TC[CriticalityLevel.L0] == 7
        assert CRITICALITY_QUEUE_PRIORITY[CriticalityLevel.L0] == "highest"

    def test_l1_maps_to_pcp6(self):
        assert CRITICALITY_PCP[CriticalityLevel.L1] == 6
        assert CRITICALITY_TC[CriticalityLevel.L1] == 6
        assert CRITICALITY_QUEUE_PRIORITY[CriticalityLevel.L1] == "high"

    def test_l2_maps_to_pcp5(self):
        assert CRITICALITY_PCP[CriticalityLevel.L2] == 5
        assert CRITICALITY_TC[CriticalityLevel.L2] == 5
        assert CRITICALITY_QUEUE_PRIORITY[CriticalityLevel.L2] == "medium"

    def test_l3_maps_to_pcp1(self):
        assert CRITICALITY_PCP[CriticalityLevel.L3] == 1
        assert CRITICALITY_TC[CriticalityLevel.L3] == 0
        assert CRITICALITY_QUEUE_PRIORITY[CriticalityLevel.L3] == "low"


# ============================================================
# 2. Urgency function decay
# ============================================================


class TestUrgencyFunction:
    def test_step_decay_within_deadline(self):
        uf = UrgencyFunction(
            decay_type=DecayType.STEP,
            value_plateau_us=200,
            decay_start_us=200,
        )
        assert uf.evaluate(elapsed_us=0, deadline_us=200) == 1.0
        assert uf.evaluate(elapsed_us=150, deadline_us=200) == 1.0
        assert uf.evaluate(elapsed_us=200, deadline_us=200) == 1.0

    def test_step_decay_beyond_deadline(self):
        uf = UrgencyFunction(
            decay_type=DecayType.STEP,
            value_plateau_us=200,
            decay_start_us=200,
        )
        assert uf.evaluate(elapsed_us=201, deadline_us=200) == 0.0
        assert uf.evaluate(elapsed_us=1000, deadline_us=200) == 0.0

    def test_linear_decay_symmetric(self):
        uf = UrgencyFunction(
            decay_type=DecayType.LINEAR,
            value_plateau_us=0,
            decay_start_us=0,
            decay_rate=1.0,
        )
        assert uf.evaluate(elapsed_us=0, deadline_us=500) == pytest.approx(1.0)
        assert uf.evaluate(elapsed_us=250, deadline_us=500) == pytest.approx(0.5)
        assert uf.evaluate(elapsed_us=500, deadline_us=500) == pytest.approx(0.0)
        assert uf.evaluate(elapsed_us=600, deadline_us=500) == pytest.approx(0.0)

    def test_linear_decay_with_plateau(self):
        uf = UrgencyFunction(
            decay_type=DecayType.LINEAR,
            value_plateau_us=50,
            decay_start_us=50,
            decay_rate=1.0,
        )
        assert uf.evaluate(elapsed_us=25, deadline_us=100) == pytest.approx(1.0)
        assert uf.evaluate(elapsed_us=50, deadline_us=100) == pytest.approx(1.0)
        mid = uf.evaluate(elapsed_us=75, deadline_us=100)
        assert 0.0 < mid < 1.0

    def test_exponential_decay_monotonic(self):
        uf = UrgencyFunction(
            decay_type=DecayType.EXPONENTIAL,
            value_plateau_us=0,
            decay_start_us=0,
            decay_rate=0.01,
        )
        v0 = uf.evaluate(elapsed_us=0, deadline_us=1000)
        v1 = uf.evaluate(elapsed_us=500, deadline_us=1000)
        v2 = uf.evaluate(elapsed_us=1000, deadline_us=1000)
        assert v0 == pytest.approx(1.0)
        assert 0 < v1 < 1
        assert 0 < v2 < v1

    def test_exponential_lambda_sensitivity(self):
        uf_fast = UrgencyFunction(DecayType.EXPONENTIAL, 0, 0, decay_rate=10.0)
        uf_slow = UrgencyFunction(DecayType.EXPONENTIAL, 0, 0, decay_rate=0.001)
        elapsed = 500
        assert uf_fast.evaluate(elapsed, 1000) < uf_slow.evaluate(elapsed, 1000)

    def test_bound_zero_deadline(self):
        uf = UrgencyFunction(DecayType.LINEAR, 0, 0, 1.0)
        assert uf.evaluate(elapsed_us=10, deadline_us=0) == 0.0


# ============================================================
# 3. Scenario YAML deserialization
# ============================================================


class TestScenarioYAML:
    def test_agv_fleet_parses_correctly(self):
        intents = parse_scenario_yaml(AGV_FLEET_SCENARIO_YAML)
        assert len(intents) == 3
        task_ids = {i.task_id for i in intents}
        assert "agv_001_path_tracking" in task_ids
        assert "agv_001_lidar_scan" in task_ids
        assert "agv_001_emergency_stop" in task_ids

    def test_agv_path_tracking_type(self):
        intents = parse_scenario_yaml(AGV_FLEET_SCENARIO_YAML)
        pt = next(i for i in intents if i.task_id == "agv_001_path_tracking")
        assert pt.task_type == TaskType.PERIODIC_CONTROL
        assert pt.criticality.base_level == CriticalityLevel.L1
        assert pt.criticality.escalatable is True
        assert pt.temporal.deadline_us == 500
        assert pt.temporal.period_us == 1000
        assert len(pt.dependencies) == 1
        assert pt.dependencies[0].depends_on == "agv_001_lidar_scan"
        assert pt.dependencies[0].dependency_type == DependencyType.SOFT

    def test_agv_emergency_stop_event_driven(self):
        intents = parse_scenario_yaml(AGV_FLEET_SCENARIO_YAML)
        es = next(i for i in intents if i.task_id == "agv_001_emergency_stop")
        assert es.task_type == TaskType.EMERGENCY_STOP
        assert es.criticality.base_level == CriticalityLevel.L0
        assert es.temporal.period_us == 0
        assert es.temporal.deadline_us == 200
        assert es.temporal.max_consecutive_drop == 0

    def test_agv_lidar_escalation(self):
        intents = parse_scenario_yaml(AGV_FLEET_SCENARIO_YAML)
        lidar = next(i for i in intents if i.task_id == "agv_001_lidar_scan")
        assert lidar.criticality.base_level == CriticalityLevel.L2
        assert lidar.criticality.escalatable is True
        assert len(lidar.criticality.escalation_rules) == 1
        rule = lidar.criticality.escalation_rules[0]
        assert rule.condition == "obstacle_detected"
        assert rule.new_level == CriticalityLevel.L1

    def test_cobot_scenario_parses(self):
        intents = parse_scenario_yaml(COBOT_SCENARIO_YAML)
        assert len(intents) == 3
        sync = next(i for i in intents if i.task_id == "arm_001_collaboration_sync")
        assert sync.task_type == TaskType.COLLABORATION
        assert sync.dependencies[0].dependency_type == DependencyType.HARD
        assert sync.dependencies[0].max_skip == 0

    def test_plc_scenario_parses(self):
        intents = parse_scenario_yaml(PLC_SCENARIO_YAML)
        assert len(intents) == 4
        plc = next(i for i in intents if i.task_id == "plc_01_position_loop")
        assert plc.task_type == TaskType.PERIODIC_CONTROL
        assert plc.temporal.deadline_us == 100
        assert plc.temporal.period_us == 500

    def test_load_all_scenarios(self):
        all_scenarios = load_all_scenarios()
        assert "agv_fleet" in all_scenarios
        assert "cobot" in all_scenarios
        assert "plc" in all_scenarios
        assert len(all_scenarios["agv_fleet"]) == 3
        assert len(all_scenarios["cobot"]) == 3
        assert len(all_scenarios["plc"]) == 4

    def test_roundtrip_yaml_serialization(self):
        intents = agv_fleet_scenario()
        yaml_str = scenario_to_yaml(intents)
        parsed = parse_scenario_yaml(yaml_str)
        assert len(parsed) == 3
        for orig, p in zip(intents, parsed):
            assert orig.task_id == p.task_id
            assert orig.task_type == p.task_type
            assert orig.criticality.base_level == p.criticality.base_level

    def test_roundtrip_json_serialization(self):
        intent = agv_path_tracking()
        d = intent.to_dict()
        assert d["task_id"] == "agv_001_path_tracking"
        assert d["task_type"] == "PERIODIC_CONTROL"
        restored = TaskIntent.from_dict(d)
        assert restored.task_id == intent.task_id
        assert restored.task_type == intent.task_type
        assert restored.temporal.deadline_us == intent.temporal.deadline_us
        assert restored.criticality.base_level == intent.criticality.base_level

    def test_criticality_escalation_in_yaml(self):
        intent = agv_path_tracking()
        intent.criticality.escalate("obstacle_detected")
        assert intent.criticality.effective_level == CriticalityLevel.L0
        d = intent.to_dict()
        restored = TaskIntent.from_dict(d)
        assert restored.criticality.effective_level == CriticalityLevel.L0


# ============================================================
# 4. Dependency graph validation
# ============================================================


class TestDependencyGraph:
    def test_topological_sort_linear_chain(self):
        dg = DataDependencyGraph(edges=[
            DDGEdge("a", "b", DependencyType.HARD),
            DDGEdge("b", "c", DependencyType.HARD),
            DDGEdge("c", "d", DependencyType.HARD),
        ])
        order = dg.topological_sort()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")
        assert order.index("c") < order.index("d")

    def test_topological_sort_diamond(self):
        dg = DataDependencyGraph(edges=[
            DDGEdge("a", "b", DependencyType.HARD),
            DDGEdge("a", "c", DependencyType.HARD),
            DDGEdge("b", "d", DependencyType.HARD),
            DDGEdge("c", "d", DependencyType.HARD),
        ])
        order = dg.topological_sort()
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_topological_sort_empty(self):
        dg = DataDependencyGraph()
        assert dg.topological_sort() == []

    def test_topological_sort_single_node(self):
        dg = DataDependencyGraph(edges=[
            DDGEdge("a", "b", DependencyType.HARD),
        ])
        order = dg.topological_sort()
        assert order == ["a", "b"]

    def test_upstreams_downstreams(self):
        dg = DataDependencyGraph(edges=[
            DDGEdge("a", "b", DependencyType.HARD),
            DDGEdge("b", "c", DependencyType.SOFT),
        ])
        assert dg.upstreams("b") == ["a"]
        assert dg.upstreams("c") == ["b"]
        assert dg.upstreams("a") == []
        assert dg.downstreams("a") == ["b"]
        assert dg.downstreams("b") == ["c"]

    def test_validate_hard_dependencies(self):
        dg = DataDependencyGraph(edges=[
            DDGEdge("a", "b", DependencyType.HARD),
            DDGEdge("c", "b", DependencyType.SOFT),
        ])
        assert dg.validate({"a", "c"}, "b") is True
        assert dg.validate({"c"}, "b") is False
        assert dg.validate(set(), "b") is False
        assert dg.validate({"a", "c", "x"}, "b") is True

    def test_validate_trigger_ignored(self):
        dg = DataDependencyGraph(edges=[
            DDGEdge("a", "b", DependencyType.TRIGGER),
        ])
        assert dg.validate(set(), "b") is True

    def test_agv_scenario_ddg(self):
        dg = DataDependencyGraph(edges=[
            DDGEdge("agv_001_lidar_scan", "agv_001_path_tracking", DependencyType.SOFT, 3),
            DDGEdge("agv_001_lidar_scan", "agv_001_emergency_stop", DependencyType.TRIGGER),
        ])
        assert "agv_001_lidar_scan" in dg.upstreams("agv_001_path_tracking")
        assert "agv_001_lidar_scan" in dg.upstreams("agv_001_emergency_stop")
        order = dg.topological_sort()
        assert order.index("agv_001_lidar_scan") < order.index("agv_001_path_tracking")
        assert order.index("agv_001_lidar_scan") < order.index("agv_001_emergency_stop")

    def test_ddg_edge_serialization(self):
        edge = DDGEdge("a", "b", DependencyType.HARD, 3)
        d = edge.to_dict()
        assert d["src_flow_id"] == "a"
        assert d["dep_type"] == "HARD"
        restored = DDGEdge.from_dict(d)
        assert restored.src_flow_id == edge.src_flow_id
        assert restored.dep_type == edge.dep_type


# ============================================================
# 5. Round-trip encoding: TaskIntent → FlowSemantics → TSNBridgeConfig
# ============================================================


class TestRoundTripEncoding:
    def test_encode_periodic_control(self):
        intent = _make_intent("ctrl_001", TaskType.PERIODIC_CONTROL, CriticalityLevel.L1)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        assert fs.flow_id == "f_ctrl_001"
        assert fs.task_id == "ctrl_001"
        assert fs.stream_class == StreamClass.SCHEDULED_TRAFFIC
        assert fs.preemption_eligible is False
        assert fs.urgency.decay_type == DecayType.LINEAR

    def test_encode_emergency_stop(self):
        intent = _make_intent("estop_001", TaskType.EMERGENCY_STOP, CriticalityLevel.L0)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        assert fs.stream_class == StreamClass.SCHEDULED_TRAFFIC
        assert fs.preemption_eligible is False
        assert fs.urgency.decay_type == DecayType.STEP
        assert fs.priority_weight == pytest.approx(0.98)

    def test_encode_telemetry(self):
        intent = _make_intent("tele_001", TaskType.TELEMETRY, CriticalityLevel.L3)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        assert fs.stream_class == StreamClass.BEST_EFFORT
        assert fs.preemption_eligible is True
        assert fs.urgency.decay_type == DecayType.EXPONENTIAL
        assert fs.priority_weight == pytest.approx(0.20)

    def test_encode_inspection(self):
        intent = _make_intent("insp_001", TaskType.INSPECTION, CriticalityLevel.L2)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        assert fs.stream_class == StreamClass.RESERVED
        assert fs.preemption_eligible is True
        assert fs.urgency.decay_type == DecayType.EXPONENTIAL
        assert fs.priority_weight == pytest.approx(0.50)
        assert fs.compressibility.ratio == pytest.approx(0.3)

    def test_encode_collaboration(self):
        intent = _make_intent("collab_001", TaskType.COLLABORATION, CriticalityLevel.L1)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        assert fs.stream_class == StreamClass.RESERVED
        assert fs.preemption_eligible is True
        assert fs.urgency.decay_type == DecayType.LINEAR
        assert fs.priority_weight == pytest.approx(0.80)

    def test_encode_reconfiguration(self):
        intent = _make_intent("reconf_001", TaskType.RECONFIGURATION, CriticalityLevel.L2)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        assert fs.stream_class == StreamClass.RESERVED
        assert fs.preemption_eligible is True
        assert fs.urgency.decay_type == DecayType.STEP

    def test_encode_all_task_types_yield_valid_semantics(self):
        encoder = IntentEncoder()
        for task_type in TaskType:
            for level in CriticalityLevel:
                intent = _make_intent(f"t_{task_type.name}_{level.name}", task_type, level)
                fs = encoder.encode(intent)
                assert 0.0 <= fs.priority_weight <= 1.0
                assert fs.delayable_boundary_us > 0 or fs.delayable_boundary_us == 0
                assert fs.compressibility.ratio >= 0.0
                assert fs.compressibility.ratio <= 1.0

    def test_gcl_window_calculation(self):
        encoder = IntentEncoder(link_rate_gbps=1.0)
        window_ns = encoder.calculate_gcl_window_ns(
            frame_size_bytes=256,
            compressibility=SemanticCompressibility(ratio=0.0),
        )
        expected = math.ceil((256 * 8) / 1.0)
        assert window_ns == expected

    def test_gcl_window_with_semantic_compression(self):
        encoder = IntentEncoder(link_rate_gbps=1.0)
        window_no_comp = encoder.calculate_gcl_window_ns(
            frame_size_bytes=256,
            compressibility=SemanticCompressibility(ratio=0.0),
        )
        window_with_comp = encoder.calculate_gcl_window_ns(
            frame_size_bytes=256,
            compressibility=SemanticCompressibility(ratio=0.5),
        )
        assert window_with_comp < window_no_comp

    def test_encode_with_escalation(self):
        intent = agv_path_tracking()
        encoder = IntentEncoder()
        fs_base = encoder.encode(intent)
        assert fs_base.priority_weight == pytest.approx(0.80)
        intent.criticality.escalate("obstacle_detected")
        fs_escalated = encoder.encode(intent)
        assert fs_escalated.priority_weight == pytest.approx(0.98)

    def test_batch_encode(self):
        encoder = IntentEncoder()
        intents = agv_fleet_scenario()
        flows = encoder.batch_encode(intents)
        assert len(flows) == 3
        flow_ids = {f.flow_id for f in flows}
        assert "f_agv_001_path_tracking" in flow_ids
        assert "f_agv_001_lidar_scan" in flow_ids
        assert "f_agv_001_emergency_stop" in flow_ids

    def test_mapping_metadata_contains_required_keys(self):
        encoder = IntentEncoder()
        intent = _make_intent("meta_test", TaskType.PERIODIC_CONTROL, CriticalityLevel.L1)
        meta = encoder.get_mapping_metadata(intent)
        for key in [
            "task_type", "criticality_effective", "stream_class",
            "shaping", "gating", "pcp", "traffic_class",
            "queue_priority", "preemption_eligible", "urgency_decay_type",
            "compressibility_ratio",
        ]:
            assert key in meta, f"Missing key: {key}"


# ============================================================
# 6. QoS Mapper
# ============================================================


class TestQoSMapper:
    def test_map_gcl_for_scheduled_traffic(self):
        intent = _make_intent("ctrl_001", TaskType.PERIODIC_CONTROL, CriticalityLevel.L1)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        mapper = QoSMapper(link_rate_gbps=1.0, frame_size_bytes=256)
        gcl = mapper.map_gcl(fs)
        assert gcl is not None
        assert gcl.window_id == "w_f_ctrl_001"
        assert gcl.gate_states is not None
        assert len(gcl.gate_states) == 8
        assert gcl.window_size_ns > 0

    def test_map_gcl_returns_none_for_non_scheduled(self):
        intent = _make_intent("tele_001", TaskType.TELEMETRY, CriticalityLevel.L3)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        mapper = QoSMapper()
        assert mapper.map_gcl(fs) is None

    def test_map_cbs_for_reserved(self):
        intent = _make_intent("insp_001", TaskType.INSPECTION, CriticalityLevel.L2)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        mapper = QoSMapper(link_rate_gbps=1.0)
        cbs = mapper.map_cbs(fs)
        assert cbs is not None
        assert cbs.traffic_class == 5
        assert cbs.idle_slope_kbps > 0
        assert cbs.send_slope_kbps > 0

    def test_map_psfp(self):
        intent = _make_intent("ctrl_001", TaskType.PERIODIC_CONTROL, CriticalityLevel.L1)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        mapper = QoSMapper()
        psfp = mapper.map_psfp(fs)
        assert psfp.stream_filter_id == "sf_f_ctrl_001"
        assert psfp.stream_handle_id == "sh_f_ctrl_001"
        assert psfp.flow_meter is not None
        assert psfp.flow_meter.cir_kbps > 0

    def test_map_queue(self):
        intent = _make_intent("ctrl_001", TaskType.PERIODIC_CONTROL, CriticalityLevel.L1)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        mapper = QoSMapper()
        queue = mapper.map_queue(fs)
        assert queue.pcp == 6
        assert queue.traffic_class == 6
        assert queue.queue_priority == "high"

    def test_map_preemption_non_preemptible(self):
        intent = _make_intent("ctrl_001", TaskType.PERIODIC_CONTROL, CriticalityLevel.L1)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        mapper = QoSMapper()
        preempt = mapper.map_preemption(fs)
        assert preempt.preemptible is False
        assert preempt.hold_advance_bytes == 64

    def test_map_single_flow_bridge(self):
        intent = _make_intent("ctrl_001", TaskType.PERIODIC_CONTROL, CriticalityLevel.L1)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        mapper = QoSMapper()
        bridge = mapper.map_single_flow_bridge(fs, bridge_id="SW1", port_id="3")
        assert bridge.bridge_id == "SW1"
        assert bridge.port_id == "3"
        assert len(bridge.gcl_list) == 1
        assert len(bridge.queue_map) == 1
        assert bridge.preemption is not None
        assert bridge.preemption.preemptible is False

    def test_map_bridge_multi_flow(self):
        intents = agv_fleet_scenario()
        encoder = IntentEncoder()
        flows = encoder.batch_encode(intents)
        mapper = QoSMapper(link_rate_gbps=1.0, frame_size_bytes=256)
        bridge = mapper.map_bridge("SW1", "1", flows)

        scheduled = [f for f in flows if f.stream_class == StreamClass.SCHEDULED_TRAFFIC]
        reserved = [f for f in flows if f.stream_class == StreamClass.RESERVED]

        assert len(bridge.gcl_list) == len(scheduled)
        assert len(bridge.cbs_configs) == len(reserved)
        assert len(bridge.psfp_rules) == len(flows)
        assert len(bridge.queue_map) == len(flows)
        assert bridge.preemption is not None

    def test_map_bridge_gcl_offsets_non_overlapping(self):
        intents = [
            _make_intent("a", TaskType.PERIODIC_CONTROL, CriticalityLevel.L1),
            _make_intent("b", TaskType.PERIODIC_CONTROL, CriticalityLevel.L1),
        ]
        encoder = IntentEncoder()
        flows = encoder.batch_encode(intents)
        mapper = QoSMapper(link_rate_gbps=1.0, frame_size_bytes=64)
        bridge = mapper.map_bridge("SW1", "1", flows)
        assert len(bridge.gcl_list) == 2
        w0 = bridge.gcl_list[0]
        w1 = bridge.gcl_list[1]
        assert w1.offset_ns >= w0.offset_ns + w0.window_size_ns

    def test_flow_semantics_serialization_roundtrip(self):
        intent = _make_intent("ctrl_001", TaskType.PERIODIC_CONTROL, CriticalityLevel.L1)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        d = fs.to_dict()
        assert d["flow_id"] == "f_ctrl_001"
        restored = FlowSemantics.from_dict(d)
        assert restored.flow_id == fs.flow_id
        assert restored.task_id == fs.task_id
        assert restored.priority_weight == pytest.approx(fs.priority_weight)
        assert restored.stream_class == fs.stream_class

    def test_bridge_config_serialization_roundtrip(self):
        intent = _make_intent("ctrl_001", TaskType.PERIODIC_CONTROL, CriticalityLevel.L1)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        mapper = QoSMapper()
        bridge = mapper.map_single_flow_bridge(fs)
        d = bridge.to_dict()
        assert d["bridge_id"] == "SW1"
        j = json.dumps(d)
        assert "bridge_id" in j


# ============================================================
# 7. GCL compute function
# ============================================================


class TestGCLWindowComputation:
    def test_no_compression(self):
        ns = _compute_gcl_window_ns(256, 1.0, compressibility_ratio=0.0)
        assert ns == 2048

    def test_with_compression(self):
        ns_no = _compute_gcl_window_ns(256, 1.0, compressibility_ratio=0.0)
        ns_cmp = _compute_gcl_window_ns(256, 1.0, compressibility_ratio=0.5)
        assert ns_cmp < ns_no

    def test_gbps_sensitivity(self):
        ns_1g = _compute_gcl_window_ns(256, 1.0)
        ns_10g = _compute_gcl_window_ns(256, 10.0)
        assert ns_10g < ns_1g


# ============================================================
# 8. Priority weight update
# ============================================================


class TestPriorityWeightUpdate:
    def test_base_weight_from_criticality(self):
        intent = _make_intent("t", TaskType.PERIODIC_CONTROL, CriticalityLevel.L1)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        assert fs.priority_weight == pytest.approx(0.80)

    def test_weight_increases_near_deadline(self):
        intent = _make_intent("t", TaskType.PERIODIC_CONTROL, CriticalityLevel.L1)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        fs.update_priority(CriticalityLevel.L1, elapsed_us=400, upstream_lost=False)
        assert fs.priority_weight > 0.80

    def test_weight_decreases_on_upstream_loss(self):
        intent = _make_intent("t", TaskType.PERIODIC_CONTROL, CriticalityLevel.L1)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        fs.update_priority(CriticalityLevel.L1, elapsed_us=0, upstream_lost=True)
        assert fs.priority_weight < 0.80

    def test_weight_clamped_zero_one(self):
        intent = _make_intent("t", TaskType.TELEMETRY, CriticalityLevel.L3)
        encoder = IntentEncoder()
        fs = encoder.encode(intent)
        fs.update_priority(CriticalityLevel.L3, elapsed_us=1000, upstream_lost=True)
        assert fs.priority_weight >= 0.0
        assert fs.priority_weight <= 1.0


# ============================================================
# 9. Criticality escalation
# ============================================================


class TestCriticalityEscalation:
    def test_escalate_matching_rule(self):
        profile = CriticalityProfile(
            base_level=CriticalityLevel.L1,
            escalatable=True,
            escalation_rules=[
                EscalationRule("obstacle", CriticalityLevel.L0, 5000),
            ],
        )
        profile.escalate("obstacle")
        assert profile.effective_level == CriticalityLevel.L0

    def test_escalate_no_match(self):
        profile = CriticalityProfile(
            base_level=CriticalityLevel.L1,
            escalation_rules=[
                EscalationRule("obstacle", CriticalityLevel.L0, 5000),
            ],
        )
        profile.escalate("unknown_condition")
        assert profile.effective_level == CriticalityLevel.L1

    def test_deescalate(self):
        profile = CriticalityProfile(
            base_level=CriticalityLevel.L1,
            escalation_rules=[
                EscalationRule("obstacle", CriticalityLevel.L0, 5000),
            ],
        )
        profile.escalate("obstacle")
        assert profile.effective_level == CriticalityLevel.L0
        profile.deescalate()
        assert profile.effective_level == CriticalityLevel.L1

    def test_effective_level_falls_back(self):
        profile = CriticalityProfile(base_level=CriticalityLevel.L2)
        assert profile.effective_level == CriticalityLevel.L2


# ============================================================
# 10. Enum string conversion
# ============================================================


class TestEnumConversion:
    def test_task_type_from_str(self):
        assert TaskType.from_str("PERIODIC_CONTROL") == TaskType.PERIODIC_CONTROL
        assert TaskType.from_str("emergency_stop") == TaskType.EMERGENCY_STOP
        with pytest.raises(ValueError):
            TaskType.from_str("unknown")

    def test_criticality_level_from_str(self):
        assert CriticalityLevel.from_str("L0") == CriticalityLevel.L0
        assert CriticalityLevel.from_str("l3") == CriticalityLevel.L3

    def test_dependency_type_from_str(self):
        assert DependencyType.from_str("HARD") == DependencyType.HARD
        assert DependencyType.from_str("soft") == DependencyType.SOFT
        assert DependencyType.from_str("TRIGGER") == DependencyType.TRIGGER

    def test_decay_type_from_str(self):
        assert DecayType.from_str("step_decay") == DecayType.STEP
        assert DecayType.from_str("LINEAR") == DecayType.LINEAR

    def test_stream_class_from_str(self):
        assert StreamClass.from_str("SCHEDULED_TRAFFIC") == StreamClass.SCHEDULED_TRAFFIC
        assert StreamClass.from_str("best_effort") == StreamClass.BEST_EFFORT
