from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, auto
from typing import Any


# ============================================================
# 枚举定义
# ============================================================


class TaskType(Enum):
    INSPECTION = auto()
    COLLABORATION = auto()
    EMERGENCY_STOP = auto()
    PERIODIC_CONTROL = auto()
    TELEMETRY = auto()
    RECONFIGURATION = auto()

    @classmethod
    def from_str(cls, s: str) -> TaskType:
        mapping = {
            "INSPECTION": cls.INSPECTION,
            "COLLABORATION": cls.COLLABORATION,
            "EMERGENCY_STOP": cls.EMERGENCY_STOP,
            "PERIODIC_CONTROL": cls.PERIODIC_CONTROL,
            "TELEMETRY": cls.TELEMETRY,
            "RECONFIGURATION": cls.RECONFIGURATION,
        }
        if s.upper() not in mapping:
            raise ValueError(f"Unknown TaskType: {s}")
        return mapping[s.upper()]


class CriticalityLevel(Enum):
    L0 = 0
    L1 = 1
    L2 = 2
    L3 = 3

    @classmethod
    def from_str(cls, s: str) -> CriticalityLevel:
        mapping = {
            "L0": cls.L0,
            "L1": cls.L1,
            "L2": cls.L2,
            "L3": cls.L3,
        }
        if s.upper() not in mapping:
            raise ValueError(f"Unknown CriticalityLevel: {s}")
        return mapping[s.upper()]


class StreamClass(Enum):
    SCHEDULED_TRAFFIC = auto()
    RESERVED = auto()
    BEST_EFFORT = auto()

    @classmethod
    def from_str(cls, s: str) -> StreamClass:
        mapping = {
            "SCHEDULED_TRAFFIC": cls.SCHEDULED_TRAFFIC,
            "RESERVED": cls.RESERVED,
            "BEST_EFFORT": cls.BEST_EFFORT,
            "scheduled_traffic": cls.SCHEDULED_TRAFFIC,
            "reserved": cls.RESERVED,
            "best_effort": cls.BEST_EFFORT,
        }
        if s not in mapping:
            raise ValueError(f"Unknown StreamClass: {s}")
        return mapping[s]


class DependencyType(Enum):
    HARD = auto()
    SOFT = auto()
    TRIGGER = auto()

    @classmethod
    def from_str(cls, s: str) -> DependencyType:
        mapping = {
            "HARD": cls.HARD,
            "SOFT": cls.SOFT,
            "TRIGGER": cls.TRIGGER,
        }
        if s.upper() not in mapping:
            raise ValueError(f"Unknown DependencyType: {s}")
        return mapping[s.upper()]


class DecayType(Enum):
    STEP = auto()
    LINEAR = auto()
    EXPONENTIAL = auto()

    @classmethod
    def from_str(cls, s: str) -> DecayType:
        mapping = {
            "step_decay": cls.STEP,
            "linear_decay": cls.LINEAR,
            "exponential_decay": cls.EXPONENTIAL,
            "STEP": cls.STEP,
            "LINEAR": cls.LINEAR,
            "EXPONENTIAL": cls.EXPONENTIAL,
        }
        if s not in mapping:
            raise ValueError(f"Unknown DecayType: {s}")
        return mapping[s]


# ============================================================
# 序列化辅助
# ============================================================


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Enum):
        return obj.name
    elif is_dataclass(obj) and not isinstance(obj, type):
        result: dict[str, Any] = {}
        for f in fields(obj):
            val = getattr(obj, f.name)
            result[f.name] = _dataclass_to_dict(val)
        return result
    elif isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def _dataclass_from_dict(cls: type, data: dict[str, Any]) -> Any:
    if cls == bool and isinstance(data, str):
        return data.lower() in ("true", "1", "yes")
    if issubclass(cls, Enum) and isinstance(data, str):
        return getattr(cls, data)
    if not hasattr(cls, "__dataclass_fields__"):
        return data
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        key = f.name
        if key not in data:
            continue
        val = data[key]
        if isinstance(f.type, str):
            kwargs[key] = val
        elif hasattr(f.type, "__origin__") and f.type.__origin__ is list:
            item_type = f.type.__args__[0] if f.type.__args__ else str
            kwargs[key] = [_dataclass_from_dict(item_type, v) if isinstance(v, dict) else v for v in val]
        elif hasattr(f.type, "__origin__") and f.type.__origin__ is dict:
            kwargs[key] = val
        elif isinstance(val, dict):
            kwargs[key] = _dataclass_from_dict(f.type, val)
        elif isinstance(val, str) and hasattr(f.type, "__dataclass_fields__"):
            kwargs[key] = val
        else:
            kwargs[key] = val
    return cls(**kwargs)


# ============================================================
# Intent 层
# ============================================================


@dataclass
class EscalationRule:
    condition: str
    new_level: CriticalityLevel
    max_duration_ms: int

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EscalationRule:
        return EscalationRule(
            condition=data["condition"],
            new_level=CriticalityLevel.from_str(data["new_level"]),
            max_duration_ms=data["max_duration_ms"],
        )


@dataclass
class CriticalityProfile:
    base_level: CriticalityLevel
    escalatable: bool = False
    escalation_rules: list[EscalationRule] = field(default_factory=list)
    current_level: CriticalityLevel | None = None

    def escalate(self, condition: str) -> CriticalityProfile:
        for rule in self.escalation_rules:
            if rule.condition == condition:
                self.current_level = rule.new_level
                return self
        return self

    def deescalate(self) -> CriticalityProfile:
        self.current_level = self.base_level
        return self

    @property
    def effective_level(self) -> CriticalityLevel:
        return self.current_level if self.current_level else self.base_level

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CriticalityProfile:
        return CriticalityProfile(
            base_level=CriticalityLevel.from_str(data["base_level"]),
            escalatable=data.get("escalatable", False),
            escalation_rules=[
                EscalationRule.from_dict(r) for r in data.get("escalation_rules", [])
            ],
            current_level=(
                CriticalityLevel.from_str(data["current_level"])
                if data.get("current_level") else None
            ),
        )


@dataclass
class TemporalConstraints:
    period_us: int
    deadline_us: int
    max_jitter_us: int = 1
    max_consecutive_drop: int = 0
    phase_offset_us: int = 0
    aoi_max_us: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalConstraints:
        return TemporalConstraints(
            period_us=data["period_us"],
            deadline_us=data["deadline_us"],
            max_jitter_us=data.get("max_jitter_us", 1),
            max_consecutive_drop=data.get("max_consecutive_drop", 0),
            phase_offset_us=data.get("phase_offset_us", 0),
            aoi_max_us=data.get("aoi_max_us", 0),
        )


@dataclass
class DataDependency:
    depends_on: str
    dependency_type: DependencyType
    max_skip: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataDependency:
        return DataDependency(
            depends_on=data["depends_on"],
            dependency_type=DependencyType.from_str(data["dependency_type"]),
            max_skip=data.get("max_skip", 0),
        )


@dataclass
class SpatialContext:
    switch_port: str
    burst_radius_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialContext:
        return SpatialContext(
            switch_port=data["switch_port"],
            burst_radius_m=data.get("burst_radius_m", 0.0),
        )


@dataclass
class TaskIntent:
    task_id: str
    task_type: TaskType
    agent_id: str
    criticality: CriticalityProfile
    temporal: TemporalConstraints
    dependencies: list[DataDependency] = field(default_factory=list)
    spatial: SpatialContext | None = None
    semantic_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskIntent:
        return TaskIntent(
            task_id=data["task_id"],
            task_type=TaskType.from_str(data["task_type"]),
            agent_id=data["agent_id"],
            criticality=CriticalityProfile.from_dict(data["criticality"]),
            temporal=TemporalConstraints.from_dict(data["temporal"]),
            dependencies=[
                DataDependency.from_dict(d) for d in data.get("dependencies", [])
            ],
            spatial=(
                SpatialContext.from_dict(data["spatial"])
                if data.get("spatial") else None
            ),
            semantic_notes=data.get("semantic_notes", ""),
        )


# ============================================================
# Semantic 层
# ============================================================


@dataclass
class UrgencyFunction:
    decay_type: DecayType
    value_plateau_us: int
    decay_start_us: int
    decay_rate: float = 1.0

    def evaluate(self, elapsed_us: int, deadline_us: int) -> float:
        """
        计算经过 elapsed_us 微秒后，该帧的语义价值（[0, 1]）。
        """
        if self.decay_type == DecayType.STEP:
            return 1.0 if elapsed_us <= deadline_us else 0.0
        if elapsed_us <= self.decay_start_us:
            return 1.0
        elapsed = elapsed_us - self.decay_start_us
        remaining = deadline_us - self.decay_start_us
        if remaining <= 0:
            return 0.0
        if self.decay_type == DecayType.LINEAR:
            return max(0.0, 1.0 - elapsed / remaining)
        if self.decay_type == DecayType.EXPONENTIAL:
            return math.exp(-self.decay_rate * elapsed / 1e3)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": {
                DecayType.STEP: "step_decay",
                DecayType.LINEAR: "linear_decay",
                DecayType.EXPONENTIAL: "exponential_decay",
            }[self.decay_type],
            "parameters": {
                "value_plateau_us": self.value_plateau_us,
                "decay_start_us": self.decay_start_us,
                "decay_rate": self.decay_rate,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UrgencyFunction:
        if "decay_type" in data:
            return UrgencyFunction(
                decay_type=DecayType.from_str(data["decay_type"]),
                value_plateau_us=data.get("value_plateau_us", 0),
                decay_start_us=data.get("decay_start_us", 0),
                decay_rate=data.get("decay_rate", 1.0),
            )
        params = data.get("parameters", {})
        return UrgencyFunction(
            decay_type=DecayType.from_str(data["type"]),
            value_plateau_us=params.get("value_plateau_us", 0),
            decay_start_us=params.get("decay_start_us", 0),
            decay_rate=params.get("decay_rate", 1.0),
        )


@dataclass
class CompressionStrategy:
    strategy_type: str
    safe_drop_bits: int = 0
    safe_skip_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompressionStrategy:
        return CompressionStrategy(
            strategy_type=data["strategy_type"],
            safe_drop_bits=data.get("safe_drop_bits", 0),
            safe_skip_ratio=data.get("safe_skip_ratio", 0.0),
        )


@dataclass(frozen=True)
class SemanticCompressibility:
    ratio: float
    strategies: list[CompressionStrategy] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticCompressibility:
        if isinstance(data, dict):
            return SemanticCompressibility(
                ratio=data.get("ratio", 0.0),
                strategies=[
                    CompressionStrategy.from_dict(s) for s in data.get("strategies", [])
                ],
            )
        return SemanticCompressibility(ratio=data)


@dataclass
class FlowSemantics:
    flow_id: str
    task_id: str
    priority_weight: float
    delayable_boundary_us: int
    urgency: UrgencyFunction
    compressibility: SemanticCompressibility
    stream_class: StreamClass
    preemption_eligible: bool = False

    def update_priority(
        self,
        effective_level: CriticalityLevel,
        elapsed_us: int,
        upstream_lost: bool = False,
    ) -> FlowSemantics:
        base = {
            CriticalityLevel.L0: 0.98,
            CriticalityLevel.L1: 0.80,
            CriticalityLevel.L2: 0.50,
            CriticalityLevel.L3: 0.20,
        }[effective_level]
        deadline_ratio = (
            elapsed_us / self.delayable_boundary_us
            if self.delayable_boundary_us > 0 else 0.0
        )
        proximity_bonus = 0.2 * min(deadline_ratio, 1.0)
        dep_penalty = -0.3 if upstream_lost else 0.0
        self.priority_weight = max(0.0, min(1.0, base + proximity_bonus + dep_penalty))
        return self

    def to_dict(self) -> dict[str, Any]:
        result = _dataclass_to_dict(self)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowSemantics:
        return FlowSemantics(
            flow_id=data["flow_id"],
            task_id=data["task_id"],
            priority_weight=data["priority_weight"],
            delayable_boundary_us=data["delayable_boundary_us"],
            urgency=UrgencyFunction.from_dict(data["urgency"]),
            compressibility=SemanticCompressibility.from_dict(data["compressibility"]),
            stream_class=StreamClass.from_str(data["stream_class"]),
            preemption_eligible=data.get("preemption_eligible", False),
        )


# ============================================================
# Parameter 层
# ============================================================


@dataclass
class GCLParameters:
    window_id: str
    gate_states: str
    window_size_ns: int
    base_time_ns: int
    cycle_time_ns: int
    offset_ns: int = 0
    admin_control_list_length: int = 1

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GCLParameters:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


GCLConfig = GCLParameters


@dataclass
class CBSParameters:
    traffic_class: int
    idle_slope_kbps: int
    send_slope_kbps: int
    hi_credit_bytes: int = 1500
    lo_credit_bytes: int = -1500

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CBSParameters:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


CBSConfig = CBSParameters


@dataclass
class FlowMeter:
    cir_kbps: int
    cbs_bytes: int
    eir_kbps: int = 0
    ebs_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowMeter:
        return FlowMeter(
            cir_kbps=data["cir_kbps"],
            cbs_bytes=data["cbs_bytes"],
            eir_kbps=data.get("eir_kbps", 0),
            ebs_bytes=data.get("ebs_bytes", 0),
        )


@dataclass
class StreamGate:
    admin_gate_state: str = "OPEN"
    admin_ipv: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamGate:
        return StreamGate(
            admin_gate_state=data.get("admin_gate_state", "OPEN"),
            admin_ipv=data.get("admin_ipv", 0),
        )


@dataclass
class PSFPParameters:
    stream_filter_id: str
    stream_handle_id: str
    max_sdu_bytes: int
    stream_gate: StreamGate = field(default_factory=StreamGate)
    flow_meter: FlowMeter | None = None

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PSFPParameters:
        return PSFPParameters(
            stream_filter_id=data["stream_filter_id"],
            stream_handle_id=data["stream_handle_id"],
            max_sdu_bytes=data["max_sdu_bytes"],
            stream_gate=(
                StreamGate.from_dict(data["stream_gate"])
                if data.get("stream_gate") else StreamGate()
            ),
            flow_meter=(
                FlowMeter.from_dict(data["flow_meter"])
                if data.get("flow_meter") else None
            ),
        )


PSFPConfig = PSFPParameters


@dataclass
class QueueAssignment:
    pcp: int
    traffic_class: int
    queue_priority: str
    num_queues: int = 8

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueueAssignment:
        return QueueAssignment(
            pcp=data["pcp"],
            traffic_class=data["traffic_class"],
            queue_priority=data["queue_priority"],
            num_queues=data.get("num_queues", 8),
        )


QueueConfig = QueueAssignment


@dataclass
class PreemptionConfig:
    preemptible: bool
    hold_advance_bytes: int = 64

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreemptionConfig:
        return PreemptionConfig(
            preemptible=data["preemptible"],
            hold_advance_bytes=data.get("hold_advance_bytes", 64),
        )


@dataclass
class TSNBridgeConfig:
    bridge_id: str
    port_id: str
    gcl_list: list[GCLParameters] = field(default_factory=list)
    cbs_configs: list[CBSParameters] = field(default_factory=list)
    psfp_rules: list[PSFPParameters] = field(default_factory=list)
    queue_map: list[QueueAssignment] = field(default_factory=list)
    preemption: PreemptionConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TSNBridgeConfig:
        gcl_data = data.get("gcl", data.get("gcl_list", []))
        return TSNBridgeConfig(
            bridge_id=data["bridge_id"],
            port_id=data["port_id"],
            gcl_list=[
                (
                    GCLParameters.from_dict(g)
                    if isinstance(g, dict) else g
                )
                for g in (gcl_data if isinstance(gcl_data, list) else [])
            ],
            cbs_configs=[
                (
                    CBSParameters.from_dict(c)
                    if isinstance(c, dict) else c
                )
                for c in data.get("cbs_configs", [])
            ],
            psfp_rules=[
                (
                    PSFPParameters.from_dict(p)
                    if isinstance(p, dict) else p
                )
                for p in data.get("psfp_rules", [])
            ],
            queue_map=[
                (
                    QueueAssignment.from_dict(q)
                    if isinstance(q, dict) else q
                )
                for q in data.get("queue_map", [])
            ],
            preemption=(
                PreemptionConfig.from_dict(data["preemption"])
                if data.get("preemption") else None
            ),
        )


# ============================================================
# 数据依赖图
# ============================================================


@dataclass
class DDGEdge:
    src_flow_id: str
    tgt_flow_id: str
    dep_type: DependencyType
    max_skip: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DDGEdge:
        return DDGEdge(
            src_flow_id=data["src_flow_id"],
            tgt_flow_id=data["tgt_flow_id"],
            dep_type=DependencyType.from_str(data["dep_type"]),
            max_skip=data.get("max_skip", 0),
        )


@dataclass
class DataDependencyGraph:
    edges: list[DDGEdge] = field(default_factory=list)

    def upstreams(self, flow_id: str) -> list[str]:
        return [e.src_flow_id for e in self.edges if e.tgt_flow_id == flow_id]

    def downstreams(self, flow_id: str) -> list[str]:
        return [e.tgt_flow_id for e in self.edges if e.src_flow_id == flow_id]

    def validate(self, arrived: set[str], required: str) -> bool:
        for e in self.edges:
            if e.tgt_flow_id == required and e.dep_type == DependencyType.HARD:
                if e.src_flow_id not in arrived:
                    return False
        return True

    def topological_sort(self) -> list[str]:
        try:
            import networkx as nx
        except ImportError:
            return self._topological_sort_fallback()

        g = nx.DiGraph()
        for e in self.edges:
            g.add_edge(e.src_flow_id, e.tgt_flow_id)
        try:
            return list(nx.topological_sort(g))
        except nx.NetworkXUnfeasible:
            return []

    def _topological_sort_fallback(self) -> list[str]:
        nodes: set[str] = set()
        for e in self.edges:
            nodes.add(e.src_flow_id)
            nodes.add(e.tgt_flow_id)
        in_degree: dict[str, int] = {n: 0 for n in nodes}
        adj: dict[str, list[str]] = {n: [] for n in nodes}
        for e in self.edges:
            adj[e.src_flow_id].append(e.tgt_flow_id)
            in_degree[e.tgt_flow_id] += 1
        queue = [n for n in nodes if in_degree[n] == 0]
        result: list[str] = []
        while queue:
            u = queue.pop(0)
            result.append(u)
            for v in adj.get(u, []):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"edges": [e.to_dict() for e in self.edges]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataDependencyGraph:
        return DataDependencyGraph(
            edges=[DDGEdge.from_dict(e) for e in data.get("edges", [])]
        )
