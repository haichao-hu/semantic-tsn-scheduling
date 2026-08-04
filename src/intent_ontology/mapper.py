from __future__ import annotations

import math
from typing import Sequence

from .types import (
    CBSParameters,
    CriticalityLevel,
    FlowMeter,
    FlowSemantics,
    GCLParameters,
    PSFPParameters,
    PreemptionConfig,
    QueueAssignment,
    StreamClass,
    StreamGate,
    TSNBridgeConfig,
)
from .encoder import (
    CRITICALITY_CBS_IDLE_SLOPE_PCT,
    CRITICALITY_PCP,
    CRITICALITY_QUEUE_PRIORITY,
    CRITICALITY_TC,
)


# ============================================================
# 默认链路参数
# ============================================================

DEFAULT_LINK_RATE_GBPS = 1.0
DEFAULT_REFERENCE_FRAME_SIZE_BYTES = 256
DEFAULT_CYCLE_TIME_NS = 1_000_000
DEFAULT_BASE_TIME_NS = 1_000_000
DEFAULT_MAX_SDU_BYTES = 256


# ============================================================
# GCL 窗口最小保护带（ns）
# ============================================================

_GUARD_BAND_NS = 5_000


def _compute_gcl_window_ns(
    frame_size_bytes: int,
    link_rate_gbps: float = DEFAULT_LINK_RATE_GBPS,
    compressibility_ratio: float = 0.0,
    achievable_ratio: float = 0.8,
) -> int:
    """Table 6.5: 计算压缩后的 GCL 窗口大小（纳秒）。"""
    compressed = frame_size_bytes * (1 - compressibility_ratio * achievable_ratio)
    bits = compressed * 8
    return math.ceil(bits / link_rate_gbps)


# ============================================================
# QoS 映射器
# ============================================================


class QoSMapper:
    """将 FlowSemantics 映射为 TSN 桥配置（GCL / CBS / PSFP / Queue / Preemption）。

    基于 Section 5 和 Section 6 中的映射表。
    """

    def __init__(
        self,
        link_rate_gbps: float = DEFAULT_LINK_RATE_GBPS,
        base_time_ns: int = DEFAULT_BASE_TIME_NS,
        cycle_time_ns: int = DEFAULT_CYCLE_TIME_NS,
        frame_size_bytes: int = DEFAULT_REFERENCE_FRAME_SIZE_BYTES,
        max_sdu_bytes: int = DEFAULT_MAX_SDU_BYTES,
    ):
        self.link_rate_gbps = link_rate_gbps
        self.base_time_ns = base_time_ns
        self.cycle_time_ns = cycle_time_ns
        self.frame_size_bytes = frame_size_bytes
        self.max_sdu_bytes = max_sdu_bytes

    # ---------------------------------------------------------
    # 单流映射
    # ---------------------------------------------------------

    def map_gcl(self, fs: FlowSemantics, offset_ns: int = 0) -> GCLParameters | None:
        """Table 6.1: SCHEDULED_TRAFFIC 流生成 GCL 配置。"""
        if fs.stream_class != StreamClass.SCHEDULED_TRAFFIC:
            return None

        window_ns = _compute_gcl_window_ns(
            self.frame_size_bytes,
            self.link_rate_gbps,
            compressibility_ratio=fs.compressibility.ratio,
        )

        tc = CRITICALITY_TC.get(
            self._effective_criticality(fs), CRITICALITY_TC[CriticalityLevel.L1]
        )
        gate_states = self._gate_states_from_tc(tc)

        return GCLParameters(
            window_id=f"w_{fs.flow_id}",
            gate_states=gate_states,
            window_size_ns=int(window_ns),
            base_time_ns=self.base_time_ns,
            cycle_time_ns=self.cycle_time_ns,
            offset_ns=offset_ns,
            admin_control_list_length=1,
        )

    def map_cbs(self, fs: FlowSemantics, send_slope_ratio: float = 1.0) -> CBSParameters | None:
        """Table 6.1 + 6.2: RESERVED 流生成 CBS 配置。"""
        if fs.stream_class not in (StreamClass.RESERVED, StreamClass.BEST_EFFORT):
            return None

        level = self._effective_criticality(fs)
        tc = CRITICALITY_TC.get(level, 5)
        idle_pct = CRITICALITY_CBS_IDLE_SLOPE_PCT.get(level, 10)
        send_slope_kbps = int(self.link_rate_gbps * 1_000_000 * send_slope_ratio)
        idle_slope_kbps = int(send_slope_kbps * idle_pct / 100 * send_slope_ratio)

        return CBSParameters(
            traffic_class=tc,
            idle_slope_kbps=idle_slope_kbps,
            send_slope_kbps=send_slope_kbps,
            hi_credit_bytes=1500,
            lo_credit_bytes=-1500,
        )

    def map_psfp(self, fs: FlowSemantics) -> PSFPParameters:
        """生成流过滤与计量配置。"""
        cir_kbps = int(self.link_rate_gbps * 1_000_000 * 0.01)
        flow_meter = FlowMeter(
            cir_kbps=cir_kbps,
            cbs_bytes=1280,
            eir_kbps=int(cir_kbps * 0.1),
            ebs_bytes=256,
        )

        return PSFPParameters(
            stream_filter_id=f"sf_{fs.flow_id}",
            stream_handle_id=f"sh_{fs.flow_id}",
            max_sdu_bytes=self.max_sdu_bytes,
            stream_gate=StreamGate(admin_gate_state="OPEN", admin_ipv=CRITICALITY_PCP.get(
                self._effective_criticality(fs), 5
            ) * 1000),
            flow_meter=flow_meter,
        )

    def map_queue(self, fs: FlowSemantics) -> QueueAssignment:
        """Table 6.2: 关键性 → PCP → TC → 队列优先级。"""
        level = self._effective_criticality(fs)
        return QueueAssignment(
            pcp=CRITICALITY_PCP.get(level, 5),
            traffic_class=CRITICALITY_TC.get(level, 5),
            queue_priority=CRITICALITY_QUEUE_PRIORITY.get(level, "medium"),
        )

    def map_preemption(self, fs: FlowSemantics) -> PreemptionConfig:
        """Table 6.1: 流是否可被抢占。"""
        return PreemptionConfig(
            preemptible=fs.preemption_eligible,
            hold_advance_bytes=64,
        )

    # ---------------------------------------------------------
    # 批量映射 → TSNBridgeConfig
    # ---------------------------------------------------------

    def map_bridge(
        self,
        bridge_id: str,
        port_id: str,
        flows: list[FlowSemantics],
        base_offset_ns: int = 0,
    ) -> TSNBridgeConfig:
        gcl_list: list[GCLParameters] = []
        cbs_list: list[CBSParameters] = []
        psfp_list: list[PSFPParameters] = []
        queue_list: list[QueueAssignment] = []

        # 计算 GCL 窗口偏移：为每条 SCHEDULED_TRAFFIC 流分配时隙
        current_offset = base_offset_ns
        for fs in flows:
            if fs.stream_class == StreamClass.SCHEDULED_TRAFFIC:
                gcl = self.map_gcl(fs, offset_ns=current_offset)
                if gcl:
                    gcl_list.append(gcl)
                    current_offset += gcl.window_size_ns + _GUARD_BAND_NS
            else:
                cbs = self.map_cbs(fs)
                if cbs:
                    cbs_list.append(cbs)
            psfp_list.append(self.map_psfp(fs))
            queue_list.append(self.map_queue(fs))

        # 抢占配置：取所有流中最严格的约束
        preemption = PreemptionConfig(
            preemptible=all(fs.preemption_eligible for fs in flows),
            hold_advance_bytes=64,
        )

        return TSNBridgeConfig(
            bridge_id=bridge_id,
            port_id=port_id,
            gcl_list=gcl_list,
            cbs_configs=cbs_list,
            psfp_rules=psfp_list,
            queue_map=queue_list,
            preemption=preemption,
        )

    def map_single_flow_bridge(
        self,
        fs: FlowSemantics,
        bridge_id: str = "SW1",
        port_id: str = "1",
        offset_ns: int = 0,
    ) -> TSNBridgeConfig:
        gcl = self.map_gcl(fs, offset_ns=offset_ns)
        cbs = self.map_cbs(fs)
        psfp = self.map_psfp(fs)
        queue = self.map_queue(fs)
        preemption = self.map_preemption(fs)

        return TSNBridgeConfig(
            bridge_id=bridge_id,
            port_id=port_id,
            gcl_list=[gcl] if gcl else [],
            cbs_configs=[cbs] if cbs else [],
            psfp_rules=[psfp],
            queue_map=[queue],
            preemption=preemption,
        )

    # ---------------------------------------------------------
    # 辅助
    # ---------------------------------------------------------

    @staticmethod
    def _effective_criticality(fs: FlowSemantics) -> CriticalityLevel:
        if fs.priority_weight >= 0.95:
            return CriticalityLevel.L0
        elif fs.priority_weight >= 0.70:
            return CriticalityLevel.L1
        elif fs.priority_weight >= 0.30:
            return CriticalityLevel.L2
        return CriticalityLevel.L3

    @staticmethod
    def _gate_states_from_tc(tc: int) -> str:
        mask = ["0"] * 8
        if 0 <= tc < 8:
            mask[7 - tc] = "1"
        return "".join(mask)
