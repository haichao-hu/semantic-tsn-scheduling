from __future__ import annotations

from typing import Any

from .types import (
    CriticalityLevel,
    CriticalityProfile,
    DataDependency,
    DependencyType,
    EscalationRule,
    SpatialContext,
    TaskIntent,
    TaskType,
    TemporalConstraints,
)


# ============================================================
# 场景一：AGV 编队（Inspection + Periodic Control + Emergency Stop）
# ============================================================

AGV_FLEET_SCENARIO_YAML = """
# AGV 编队 — 路径跟踪控制
task_intent:
  task_id: "agv_001_path_tracking"
  task_type: "PERIODIC_CONTROL"
  agent_id: "agv_001"
  criticality:
    base_level: "L1"
    escalatable: true
    escalation_rules:
      - condition: "obstacle_detected"
        new_level: "L0"
        max_duration_ms: 5000
  temporal:
    period_us: 1000
    deadline_us: 500
    max_jitter_us: 1
    max_consecutive_drop: 2
    phase_offset_us: 0
    aoi_max_us: 2000
  dependencies:
    - depends_on: "agv_001_lidar_scan"
      dependency_type: "SOFT"
      max_skip: 3
  spatial:
    switch_port: "SW1.1"
    burst_radius_m: 5.0
  semantic_notes: "AGV path tracking closed-loop control, <1ms cycle, <0.5ms deadline"

---
# AGV 编队 — 激光雷达巡检
task_intent:
  task_id: "agv_001_lidar_scan"
  task_type: "INSPECTION"
  agent_id: "agv_001"
  criticality:
    base_level: "L2"
    escalatable: true
    escalation_rules:
      - condition: "obstacle_detected"
        new_level: "L1"
        max_duration_ms: 5000
  temporal:
    period_us: 25000
    deadline_us: 5000
    max_jitter_us: 50
    max_consecutive_drop: 5
  spatial:
    switch_port: "SW1.1"

---
# AGV 编队 — 急停（事件驱动）
task_intent:
  task_id: "agv_001_emergency_stop"
  task_type: "EMERGENCY_STOP"
  agent_id: "agv_001"
  criticality:
    base_level: "L0"
    escalatable: false
  temporal:
    period_us: 0
    deadline_us: 200
    max_jitter_us: 0.1
    max_consecutive_drop: 0
  dependencies:
    - depends_on: "agv_001_lidar_scan"
      dependency_type: "TRIGGER"
"""

# ============================================================
# 场景二：协作机械臂 + 视觉引导
# ============================================================

COBOT_SCENARIO_YAML = """
# 协作机械臂 — 视觉位姿遥测
task_intent:
  task_id: "arm_001_vision_telemetry"
  task_type: "TELEMETRY"
  agent_id: "arm_001"
  criticality:
    base_level: "L2"
  temporal:
    period_us: 16667
    deadline_us: 10000
    max_jitter_us: 500
    max_consecutive_drop: 3
    aoi_max_us: 50000
  spatial:
    switch_port: "SW2.1"

---
# 协作机械臂 — 双臂协作同步
task_intent:
  task_id: "arm_001_collaboration_sync"
  task_type: "COLLABORATION"
  agent_id: "arm_001"
  criticality:
    base_level: "L1"
  temporal:
    period_us: 1000
    deadline_us: 500
    max_jitter_us: 5
    max_consecutive_drop: 2
  dependencies:
    - depends_on: "arm_002_collaboration_sync"
      dependency_type: "HARD"
      max_skip: 0

---
# 协作机械臂 — 紧急停止
task_intent:
  task_id: "arm_001_emergency_stop"
  task_type: "EMERGENCY_STOP"
  agent_id: "arm_001"
  criticality:
    base_level: "L0"
  temporal:
    period_us: 0
    deadline_us: 100
    max_jitter_us: 0.1
    max_consecutive_drop: 0
"""

# ============================================================
# 场景三：PLC 控制回路 + HMI + 数字孪生
# ============================================================

PLC_SCENARIO_YAML = """
# PLC 控制回路
task_intent:
  task_id: "plc_01_position_loop"
  task_type: "PERIODIC_CONTROL"
  agent_id: "plc_01"
  criticality:
    base_level: "L1"
  temporal:
    period_us: 500
    deadline_us: 100
    max_jitter_us: 0.1
    max_consecutive_drop: 1
    aoi_max_us: 500

---
# HMI 操作员监视面板
task_intent:
  task_id: "hmi_status_polling"
  task_type: "TELEMETRY"
  agent_id: "hmi_01"
  criticality:
    base_level: "L3"
  temporal:
    period_us: 500000
    deadline_us: 100000
    max_jitter_us: 5000
    max_consecutive_drop: 10

---
# 数字孪生同步（周期批量上传）
task_intent:
  task_id: "dt_sync_batch"
  task_type: "TELEMETRY"
  agent_id: "edge_server_01"
  criticality:
    base_level: "L2"
  temporal:
    period_us: 1000000
    deadline_us: 500000
    max_jitter_us: 50000
    max_consecutive_drop: 2
    aoi_max_us: 2000000

---
# 网络重配置（按需）
task_intent:
  task_id: "net_reconfig_gcl_update"
  task_type: "RECONFIGURATION"
  agent_id: "cnc_01"
  criticality:
    base_level: "L2"
  temporal:
    period_us: 0
    deadline_us: 50000
    max_jitter_us: 100
    max_consecutive_drop: 0
"""

# ============================================================
# 所有 YAML 场景汇总
# ============================================================

ALL_SCENARIOS_YAML: dict[str, str] = {
    "agv_fleet": AGV_FLEET_SCENARIO_YAML,
    "cobot": COBOT_SCENARIO_YAML,
    "plc": PLC_SCENARIO_YAML,
}


# ============================================================
# Python 对象构造
# ============================================================


def agv_fleet_scenario() -> list[TaskIntent]:
    return [
        agv_path_tracking(),
        agv_lidar_scan(),
        agv_emergency_stop(),
    ]


def agv_path_tracking() -> TaskIntent:
    return TaskIntent(
        task_id="agv_001_path_tracking",
        task_type=TaskType.PERIODIC_CONTROL,
        agent_id="agv_001",
        criticality=CriticalityProfile(
            base_level=CriticalityLevel.L1,
            escalatable=True,
            escalation_rules=[
                EscalationRule(
                    condition="obstacle_detected",
                    new_level=CriticalityLevel.L0,
                    max_duration_ms=5000,
                )
            ],
        ),
        temporal=TemporalConstraints(
            period_us=1000,
            deadline_us=500,
            max_jitter_us=1,
            max_consecutive_drop=2,
            phase_offset_us=0,
            aoi_max_us=2000,
        ),
        dependencies=[
            DataDependency(
                depends_on="agv_001_lidar_scan",
                dependency_type=DependencyType.SOFT,
                max_skip=3,
            )
        ],
        spatial=SpatialContext(switch_port="SW1.1", burst_radius_m=5.0),
        semantic_notes="AGV path tracking closed-loop control, <1ms cycle, <0.5ms deadline",
    )


def agv_lidar_scan() -> TaskIntent:
    return TaskIntent(
        task_id="agv_001_lidar_scan",
        task_type=TaskType.INSPECTION,
        agent_id="agv_001",
        criticality=CriticalityProfile(
            base_level=CriticalityLevel.L2,
            escalatable=True,
            escalation_rules=[
                EscalationRule(
                    condition="obstacle_detected",
                    new_level=CriticalityLevel.L1,
                    max_duration_ms=5000,
                )
            ],
        ),
        temporal=TemporalConstraints(
            period_us=25000,
            deadline_us=5000,
            max_jitter_us=50,
            max_consecutive_drop=5,
        ),
        spatial=SpatialContext(switch_port="SW1.1"),
    )


def agv_emergency_stop() -> TaskIntent:
    return TaskIntent(
        task_id="agv_001_emergency_stop",
        task_type=TaskType.EMERGENCY_STOP,
        agent_id="agv_001",
        criticality=CriticalityProfile(
            base_level=CriticalityLevel.L0,
            escalatable=False,
        ),
        temporal=TemporalConstraints(
            period_us=0,
            deadline_us=200,
            max_jitter_us=0.1,
            max_consecutive_drop=0,
        ),
        dependencies=[
            DataDependency(
                depends_on="agv_001_lidar_scan",
                dependency_type=DependencyType.TRIGGER,
            )
        ],
    )


def cobot_scenario() -> list[TaskIntent]:
    return [
        arm_vision_telemetry(),
        arm_collaboration_sync(),
        arm_emergency_stop(),
    ]


def arm_vision_telemetry() -> TaskIntent:
    return TaskIntent(
        task_id="arm_001_vision_telemetry",
        task_type=TaskType.TELEMETRY,
        agent_id="arm_001",
        criticality=CriticalityProfile(base_level=CriticalityLevel.L2),
        temporal=TemporalConstraints(
            period_us=16667,
            deadline_us=10000,
            max_jitter_us=500,
            max_consecutive_drop=3,
            aoi_max_us=50000,
        ),
        spatial=SpatialContext(switch_port="SW2.1"),
    )


def arm_collaboration_sync() -> TaskIntent:
    return TaskIntent(
        task_id="arm_001_collaboration_sync",
        task_type=TaskType.COLLABORATION,
        agent_id="arm_001",
        criticality=CriticalityProfile(base_level=CriticalityLevel.L1),
        temporal=TemporalConstraints(
            period_us=1000,
            deadline_us=500,
            max_jitter_us=5,
            max_consecutive_drop=2,
        ),
        dependencies=[
            DataDependency(
                depends_on="arm_002_collaboration_sync",
                dependency_type=DependencyType.HARD,
                max_skip=0,
            )
        ],
    )


def arm_emergency_stop() -> TaskIntent:
    return TaskIntent(
        task_id="arm_001_emergency_stop",
        task_type=TaskType.EMERGENCY_STOP,
        agent_id="arm_001",
        criticality=CriticalityProfile(base_level=CriticalityLevel.L0),
        temporal=TemporalConstraints(
            period_us=0,
            deadline_us=100,
            max_jitter_us=0.1,
            max_consecutive_drop=0,
        ),
    )


def plc_scenario() -> list[TaskIntent]:
    return [
        plc_position_loop(),
        hmi_status_polling(),
        dt_sync_batch(),
        net_reconfig_gcl_update(),
    ]


def plc_position_loop() -> TaskIntent:
    return TaskIntent(
        task_id="plc_01_position_loop",
        task_type=TaskType.PERIODIC_CONTROL,
        agent_id="plc_01",
        criticality=CriticalityProfile(base_level=CriticalityLevel.L1),
        temporal=TemporalConstraints(
            period_us=500,
            deadline_us=100,
            max_jitter_us=0.1,
            max_consecutive_drop=1,
            aoi_max_us=500,
        ),
    )


def hmi_status_polling() -> TaskIntent:
    return TaskIntent(
        task_id="hmi_status_polling",
        task_type=TaskType.TELEMETRY,
        agent_id="hmi_01",
        criticality=CriticalityProfile(base_level=CriticalityLevel.L3),
        temporal=TemporalConstraints(
            period_us=500000,
            deadline_us=100000,
            max_jitter_us=5000,
            max_consecutive_drop=10,
        ),
    )


def dt_sync_batch() -> TaskIntent:
    return TaskIntent(
        task_id="dt_sync_batch",
        task_type=TaskType.TELEMETRY,
        agent_id="edge_server_01",
        criticality=CriticalityProfile(base_level=CriticalityLevel.L2),
        temporal=TemporalConstraints(
            period_us=1_000_000,
            deadline_us=500_000,
            max_jitter_us=50_000,
            max_consecutive_drop=2,
            aoi_max_us=2_000_000,
        ),
    )


def net_reconfig_gcl_update() -> TaskIntent:
    return TaskIntent(
        task_id="net_reconfig_gcl_update",
        task_type=TaskType.RECONFIGURATION,
        agent_id="cnc_01",
        criticality=CriticalityProfile(base_level=CriticalityLevel.L2),
        temporal=TemporalConstraints(
            period_us=0,
            deadline_us=50_000,
            max_jitter_us=100,
            max_consecutive_drop=0,
        ),
    )


# ============================================================
# YAML 辅助
# ============================================================


def parse_scenario_yaml(yaml_str: str) -> list[TaskIntent]:
    import yaml

    docs = list(yaml.safe_load_all(yaml_str))
    intents: list[TaskIntent] = []
    for doc in docs:
        if doc is None or "task_intent" not in doc:
            continue
        ti_data = doc["task_intent"]
        intents.append(TaskIntent.from_dict(ti_data))
    return intents


def scenario_to_yaml(intents: list[TaskIntent]) -> str:
    import yaml

    parts: list[str] = []
    for intent in intents:
        data = {"task_intent": intent.to_dict()}
        parts.append(
            yaml.dump(
                data,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            ).rstrip()
        )
    return "\n---\n".join(parts) + "\n"


def load_all_scenarios() -> dict[str, list[TaskIntent]]:
    return {
        name: parse_scenario_yaml(yaml_str)
        for name, yaml_str in ALL_SCENARIOS_YAML.items()
    }
