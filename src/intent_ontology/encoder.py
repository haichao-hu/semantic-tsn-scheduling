from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from .types import (
    CompressionStrategy,
    CriticalityLevel,
    DecayType,
    FlowSemantics,
    SemanticCompressibility,
    StreamClass,
    TaskIntent,
    TaskType,
    TemporalConstraints,
    UrgencyFunction,
)


# ============================================================
# 映射查找表（Table 6.1）
# ============================================================

TASK_TYPE_STREAM_CLASS: dict[TaskType, StreamClass] = {
    TaskType.EMERGENCY_STOP: StreamClass.SCHEDULED_TRAFFIC,
    TaskType.PERIODIC_CONTROL: StreamClass.SCHEDULED_TRAFFIC,
    TaskType.COLLABORATION: StreamClass.RESERVED,
    TaskType.INSPECTION: StreamClass.RESERVED,
    TaskType.RECONFIGURATION: StreamClass.RESERVED,
    TaskType.TELEMETRY: StreamClass.BEST_EFFORT,
}

TASK_TYPE_PREEMPTION: dict[TaskType, bool] = {
    TaskType.EMERGENCY_STOP: False,
    TaskType.PERIODIC_CONTROL: False,
    TaskType.COLLABORATION: True,
    TaskType.INSPECTION: True,
    TaskType.RECONFIGURATION: True,
    TaskType.TELEMETRY: True,
}

TASK_TYPE_SHAPING: dict[TaskType, str] = {
    TaskType.EMERGENCY_STOP: "TAS_Qbv_Qbu",
    TaskType.PERIODIC_CONTROL: "TAS_Qbv",
    TaskType.COLLABORATION: "CBS_Qav",
    TaskType.INSPECTION: "CBS_Qav",
    TaskType.RECONFIGURATION: "CBS_Qav",
    TaskType.TELEMETRY: "FIFO",
}

TASK_TYPE_GATING: dict[TaskType, str] = {
    TaskType.EMERGENCY_STOP: "event_driven",
    TaskType.PERIODIC_CONTROL: "periodic",
    TaskType.COLLABORATION: "n/a",
    TaskType.INSPECTION: "n/a",
    TaskType.RECONFIGURATION: "n/a",
    TaskType.TELEMETRY: "n/a",
}

# ============================================================
# 关键性 → 队列优先级 → PCP（Table 6.2）
# ============================================================

CRITICALITY_PCP: dict[CriticalityLevel, int] = {
    CriticalityLevel.L0: 7,
    CriticalityLevel.L1: 6,
    CriticalityLevel.L2: 5,
    CriticalityLevel.L3: 1,
}

CRITICALITY_TC: dict[CriticalityLevel, int] = {
    CriticalityLevel.L0: 7,
    CriticalityLevel.L1: 6,
    CriticalityLevel.L2: 5,
    CriticalityLevel.L3: 0,
}

CRITICALITY_QUEUE_PRIORITY: dict[CriticalityLevel, str] = {
    CriticalityLevel.L0: "highest",
    CriticalityLevel.L1: "high",
    CriticalityLevel.L2: "medium",
    CriticalityLevel.L3: "low",
}

# ============================================================
# 优先级权重基础值（Section 4.2）
# ============================================================

CRITICALITY_BASE_WEIGHT: dict[CriticalityLevel, float] = {
    CriticalityLevel.L0: 0.98,
    CriticalityLevel.L1: 0.80,
    CriticalityLevel.L2: 0.50,
    CriticalityLevel.L3: 0.20,
}

# ============================================================
# 紧迫性函数分配规则
# ============================================================

TASK_TYPE_URGENCY: dict[TaskType, DecayType] = {
    TaskType.EMERGENCY_STOP: DecayType.STEP,
    TaskType.PERIODIC_CONTROL: DecayType.LINEAR,
    TaskType.COLLABORATION: DecayType.LINEAR,
    TaskType.INSPECTION: DecayType.EXPONENTIAL,
    TaskType.TELEMETRY: DecayType.EXPONENTIAL,
    TaskType.RECONFIGURATION: DecayType.STEP,
}

# ============================================================
# 语义压缩基础比（按任务类型）
# ============================================================

TASK_TYPE_COMPRESSIBILITY: dict[TaskType, float] = {
    TaskType.EMERGENCY_STOP: 0.0,
    TaskType.PERIODIC_CONTROL: 0.05,
    TaskType.COLLABORATION: 0.1,
    TaskType.INSPECTION: 0.3,
    TaskType.TELEMETRY: 0.5,
    TaskType.RECONFIGURATION: 0.1,
}

TASK_TYPE_COMPRESSION_STRATEGIES: dict[TaskType, list[CompressionStrategy]] = {
    TaskType.INSPECTION: [
        CompressionStrategy(strategy_type="temporal_subsampling", safe_skip_ratio=0.25),
    ],
    TaskType.TELEMETRY: [
        CompressionStrategy(strategy_type="quantization", safe_drop_bits=2),
        CompressionStrategy(strategy_type="temporal_subsampling", safe_skip_ratio=0.25),
    ],
    TaskType.COLLABORATION: [],
    TaskType.PERIODIC_CONTROL: [],
    TaskType.EMERGENCY_STOP: [],
    TaskType.RECONFIGURATION: [],
}

# ============================================================
# CBS idleSlope 占链路百分比（Table 6.2）
# ============================================================

CRITICALITY_CBS_IDLE_SLOPE_PCT: dict[CriticalityLevel, int] = {
    CriticalityLevel.L0: 0,
    CriticalityLevel.L1: 0,
    CriticalityLevel.L2: 25,
    CriticalityLevel.L3: 8,
}

# ============================================================
# Protocol for v2 extension
# ============================================================


@runtime_checkable
class IntentEncoderV2(Protocol):
    def encode(self, intent: TaskIntent) -> FlowSemantics:
        ...

    def batch_encode(self, intents: list[TaskIntent]) -> list[FlowSemantics]:
        ...


# ============================================================
# IntentEncoder v1
# ============================================================


class IntentEncoder:
    """v1 规则驱动编码器：将 TaskIntent 转换为 FlowSemantics。

    基于第 6 节中的映射表，以确定性查找规则将任务层的意图字段
    转换为语义层的流语义字段。v2 将用 attention / LLM 编码器替代。
    """

    def __init__(self, link_rate_gbps: float = 1.0, base_time_ns: int = 1_000_000):
        self.link_rate_gbps = link_rate_gbps
        self.base_time_ns = base_time_ns

    def encode(self, intent: TaskIntent) -> FlowSemantics:
        effective_level = intent.criticality.effective_level
        flow_id = self._flow_id_from_task(intent.task_id)
        urgency = self._build_urgency(intent.task_type, intent.temporal)
        compressibility = self._build_compressibility(intent.task_type)
        stream_class = TASK_TYPE_STREAM_CLASS[intent.task_type]
        preemptible = TASK_TYPE_PREEMPTION[intent.task_type]
        base_weight = CRITICALITY_BASE_WEIGHT[effective_level]

        return FlowSemantics(
            flow_id=flow_id,
            task_id=intent.task_id,
            priority_weight=base_weight,
            delayable_boundary_us=intent.temporal.deadline_us,
            urgency=urgency,
            compressibility=compressibility,
            stream_class=stream_class,
            preemption_eligible=preemptible,
        )

    def encode_with_context(
        self,
        intent: TaskIntent,
        elapsed_us: int = 0,
        upstream_lost: bool = False,
    ) -> FlowSemantics:
        fs = self.encode(intent)
        fs.update_priority(
            effective_level=intent.criticality.effective_level,
            elapsed_us=elapsed_us,
            upstream_lost=upstream_lost,
        )
        return fs

    def batch_encode(self, intents: list[TaskIntent]) -> list[FlowSemantics]:
        return [self.encode(it) for it in intents]

    def _flow_id_from_task(self, task_id: str) -> str:
        return f"f_{task_id}"

    def _build_urgency(self, task_type: TaskType, temporal: TemporalConstraints) -> UrgencyFunction:
        decay_type = TASK_TYPE_URGENCY[task_type]
        deadline = temporal.deadline_us
        value_plateau = 0
        decay_start = 0
        decay_rate = 1.0

        if decay_type == DecayType.STEP:
            value_plateau = deadline
            decay_start = deadline
        elif decay_type == DecayType.LINEAR:
            value_plateau = int(deadline * 0.1)
            decay_start = value_plateau
            decay_rate = 1.0
        elif decay_type == DecayType.EXPONENTIAL:
            decay_start = int(deadline * 0.1)
            decay_rate = 0.01

        return UrgencyFunction(
            decay_type=decay_type,
            value_plateau_us=value_plateau,
            decay_start_us=decay_start,
            decay_rate=decay_rate,
        )

    def _build_compressibility(self, task_type: TaskType) -> SemanticCompressibility:
        ratio = TASK_TYPE_COMPRESSIBILITY[task_type]
        strategies = list(TASK_TYPE_COMPRESSION_STRATEGIES.get(task_type, []))
        return SemanticCompressibility(ratio=ratio, strategies=strategies)

    def calculate_gcl_window_ns(
        self,
        frame_size_bytes: int,
        compressibility: SemanticCompressibility,
        achievable_ratio: float = 0.8,
    ) -> int:
        """根据语义压缩计算所需 GCL 窗口大小（Table 6.5）。

        GCL.window_size_ns = ceil((frame_size_compressed_bytes × 8) / link_rate_gbps)
        frame_size_compressed = original × (1 - semantic_compressibility × achievable_ratio)
        """
        compressed_bytes = frame_size_bytes * (
            1 - compressibility.ratio * achievable_ratio
        )
        bits = compressed_bytes * 8
        return math.ceil(bits / self.link_rate_gbps)

    def get_mapping_metadata(self, intent: TaskIntent) -> dict:
        """返回编码过程中查表得到的元数据，用于调试与验证。"""
        return {
            "task_type": intent.task_type.name,
            "criticality_effective": intent.criticality.effective_level.name,
            "stream_class": TASK_TYPE_STREAM_CLASS[intent.task_type].name,
            "shaping": TASK_TYPE_SHAPING[intent.task_type],
            "gating": TASK_TYPE_GATING[intent.task_type],
            "pcp": CRITICALITY_PCP[intent.criticality.effective_level],
            "traffic_class": CRITICALITY_TC[intent.criticality.effective_level],
            "queue_priority": CRITICALITY_QUEUE_PRIORITY[intent.criticality.effective_level],
            "preemption_eligible": TASK_TYPE_PREEMPTION[intent.task_type],
            "urgency_decay_type": TASK_TYPE_URGENCY[intent.task_type].name,
            "compressibility_ratio": TASK_TYPE_COMPRESSIBILITY[intent.task_type],
        }


# ============================================================
# v2 Stub (Attention / LLM encoder)
# ============================================================


class AttentionIntentEncoder:
    """v2 占位：基于注意力机制的 TaskIntent → FlowSemantics 编码器。

    由 Section 12.1「Intent 编码器原型」驱动。当前为 stub 实现，
    内部委托给规则编码器。
    """

    def __init__(self, link_rate_gbps: float = 1.0, model_path: str | None = None):
        self.v1_encoder = IntentEncoder(link_rate_gbps=link_rate_gbps)
        self.model_path = model_path

    def encode(self, intent: TaskIntent) -> FlowSemantics:
        raise NotImplementedError(
            "AttentionIntentEncoder is a v2 stub. "
            "Use IntentEncoder for rule-based encoding."
        )

    def batch_encode(self, intents: list[TaskIntent]) -> list[FlowSemantics]:
        raise NotImplementedError(
            "AttentionIntentEncoder is a v2 stub. "
            "Use IntentEncoder.batch_encode for rule-based encoding."
        )
