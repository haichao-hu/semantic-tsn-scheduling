from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.intent_ontology.types import (
    CriticalityLevel,
    DecayType,
    FlowSemantics,
    SemanticCompressibility,
    StreamClass,
    UrgencyFunction,
)
from src.nc_engine.topology import TSNTopology, make_line_topology


# ============================================================
# Config
# ============================================================


@dataclass
class CSRLConfig:
    n_switches: int = 3
    n_ports_per_switch: int = 4
    n_queues: int = 8
    max_active_flows: int = 8
    hyperperiod_us: float = 1000.0
    link_rate_mbps: float = 1000.0
    frame_size_bytes: float = 256.0
    seed: int = 42
    # reward weights
    reward_completion_weight: float = 1.0
    # deadline penalty: mild (paper eq. latency term is α=0.1 scaled);
    # a harsh penalty makes "reject everything" the optimal policy during
    # the initial exploration phase when window alignment is rare.
    reward_deadline_penalty: float = -0.3
    # rejecting a task that should be scheduled is itself a failure
    # (task completion semantics): penalty per rejected flow per step.
    # Disabled (0.0) for dynamic-arrival scenarios where flows may not
    # have arrived yet.
    reward_reject_penalty: float = -0.2
    reward_jitter_bonus: float = 0.1
    reward_semantic_scale: float = 1.0
    # GCL config
    gcl_guard_band_us: float = 5.0
    # max gate window: by default allow opening the gate for the full
    # hyperperiod (equivalent to "no TAS isolation").
    max_gate_window_us: float | None = None
    # safety shield (default on)
    use_safety_shield: bool = True
    # Lagrangian penalty applied to the reward: total_reward -= λ * violations
    constraint_penalty: float = 0.0
    # deployment-aware mutual-exclusion penalty: overlapping windows of
    # flows in the same queue violate the deployment-time schedulability
    # check (C2) and would be vetoed by the Safety Shield.  Penalizing
    # overlap in training closes the train/deploy gap; the penalty scales
    # with the product of the flows' priority weights so the semantic
    # layer steers the policy toward protecting critical flows from
    # overlap (a vetoed critical flow completes 0%).
    overlap_penalty_scale: float = 0.0
    # dynamic arrival: flows arrive mid-episode (5 base + N arriving)
    # — training with this flag matches the dynamic evaluation distribution
    dynamic_arrival: bool = False
    n_base_flows: int = 5
    arrival_interval: int = 20
    # scarcity configuration: makes gate-window slots a scarce resource
    # so that "who fails" is a forced choice and semantic weighting
    # decides it.
    deadline_multiplier: float = 1.0     # deadline = mult × period (<1: tight)
    shared_st_queue: bool = False        # ST flows share queue 7 (mutual exclusion)
    st_window_us: float | None = None    # ST window size (large ⇒ saturation)
    single_switch: bool = False          # route all flows through switch 0
                                         # (max link contention)

    def __post_init__(self) -> None:
        if self.max_gate_window_us is None:
            self.max_gate_window_us = self.hyperperiod_us


# ============================================================
# Flow data for simulation
# ============================================================


@dataclass
class SimFlow:
    """Runtime flow state in the TSN simulator."""
    flow_id: str
    task_id: str
    priority_weight: float
    deadline_us: float
    period_us: float
    frame_size_bytes: float
    stream_class: StreamClass
    queue: int = 0
    path: list[int] = field(default_factory=list)  # list of switch indices
    accepted: bool = False
    # timing state
    next_dispatch_us: float = 0.0
    phase_offset_us: float = 0.0
    gate_start_us: float = 0.0
    gate_size_us: float = 20.0
    # statistics
    packets_sent: int = 0
    packets_dropped: int = 0
    e2e_delays: list[float] = field(default_factory=list)
    jitter_samples: list[float] = field(default_factory=list)

    @property
    def criticality(self) -> CriticalityLevel:
        if self.priority_weight >= 0.95:
            return CriticalityLevel.L0
        if self.priority_weight >= 0.70:
            return CriticalityLevel.L1
        if self.priority_weight >= 0.30:
            return CriticalityLevel.L2
        return CriticalityLevel.L3


# ============================================================
# Switch model
# ============================================================


@dataclass
class SwitchPort:
    """One output port of a TSN switch with 8 priority queues."""
    queues: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    gcl_mask: int = 0xFF  # 8-bit: 1=open, 0=closed
    link_busy_until_us: float = 0.0
    port_id: int = 0


class SwitchModel:
    def __init__(self, switch_id: int, n_ports: int, n_queues: int, link_rate_mbps: float):
        self.switch_id = switch_id
        self.n_ports = n_ports
        self.n_queues = n_queues
        self.link_rate_mbps = link_rate_mbps
        self.ports: list[SwitchPort] = [SwitchPort(port_id=i) for i in range(n_ports)]

    @property
    def tx_time(self) -> float:
        """Time to transmit one frame at link rate (μs)."""
        return (CSRLConfig.frame_size_bytes * 8) / self.link_rate_mbps

    def enqueue(self, port: int, queue: int, packet_size: float) -> None:
        self.ports[port].queues[queue] += packet_size

    def dequeue(self, port: int, queue: int, packet_size: float) -> None:
        self.ports[port].queues[queue] = max(0.0, self.ports[port].queues[queue] - packet_size)

    def set_gcl(self, port: int, mask: int) -> None:
        self.ports[port].gcl_mask = mask

    def queue_open(self, port: int, queue: int) -> bool:
        return bool(self.ports[port].gcl_mask & (1 << queue))

    def port_busy(self, port: int, current_time_us: float) -> bool:
        return current_time_us < self.ports[port].link_busy_until_us


# ============================================================
# TSNEnv — Gym Environment
# ============================================================


class TSNEnv(gym.Env):
    """TSN simulation environment for Constrained Safe RL scheduling.

    Observation space:
      - Per-flow features: [priority, deadline/period, stream_class_onehot(3), queue/8]
      - Global: [GCL phase, n_active_flows/max]
      - Queue occupancies: n_switches × n_ports_per_switch × n_queues

    Action space (Box, [-1, 1], per-flow):
      - dispatch_offset: [-1,1] → [0, period]
      - gate_start_offset: [-1,1] → [0, period] (window relative to dispatch)
    (the accept dimension is retained for API stability but the policy
    always admits: admission control is the Safety Shield's job at
    deployment, where it vetoes schedulability-violating actions with
    formal guarantees — learned rejection was a shortcut that hurt
    completion.)
    Queue and gate size are NOT learned: they are derived by the semantic
    mapping rules (Parameter Layer of the intent ontology) — one queue
    per critical flow (Craciunas Configuration 3), BEST_EFFORT on queue 0
    with an always-open gate.  The RL scheduler focuses on the temporal
    decision (phase alignment, window placement), which keeps the action
    space learnable at 8+ flows.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: CSRLConfig | None = None,
        topology: TSNTopology | None = None,
        flows: list[FlowSemantics] | None = None,
    ):
        super().__init__()
        self.config = config or CSRLConfig()
        self._rng = np.random.RandomState(self.config.seed)

        # build or accept topology
        self.topology = topology or make_line_topology(self.config.n_switches)
        self.n_switches = self.config.n_switches = self.topology.num_nodes - 2

        # switch models
        self.switches: list[SwitchModel] = []
        for i in range(self.n_switches):
            sw = SwitchModel(
                switch_id=i,
                n_ports=self.config.n_ports_per_switch,
                n_queues=self.config.n_queues,
                link_rate_mbps=self.config.link_rate_mbps,
            )
            self.switches.append(sw)

        # flows
        self._flows_semantics: list[FlowSemantics] = flows or []
        self.sim_flows: list[SimFlow] = []
        self._current_time_us: float = 0.0
        self._episode_done: bool = False
        self._step_count: int = 0
        self._max_steps: int = 500

        # action dimension per flow: accept, dispatch_offset, gate_start_offset
        self._action_dim_per_flow = 3
        M = self.config.max_active_flows

        # observation dimensions
        self._obs_flow_dim = 8   # priority, deadline_ratio, sc_1hot(3), queue_norm, accepted, age
        self._obs_global_dim = 2  # GCL phase, active_ratio
        self._obs_queue_dim = self.n_switches * self.config.n_ports_per_switch * self.config.n_queues

        obs_dim = M * self._obs_flow_dim + self._obs_global_dim + self._obs_queue_dim

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(M * self._action_dim_per_flow,), dtype=np.float64
        )

        self._setup_flows()

    def _setup_flows(self) -> None:
        """Initialize SimFlow from FlowSemantics or generate random flows."""
        M = self.config.max_active_flows
        self.sim_flows = []
        tx_time = self.config.frame_size_bytes * 8.0 / self.config.link_rate_mbps
        min_window = tx_time + self.config.gcl_guard_band_us

        # Craciunas Configuration 3, generalized: one egress queue per
        # critical flow (ST and RESERVED), in descending priority order
        # (queues 7, 6, 5, ...); BEST_EFFORT uses queue 0 with the gate
        # always open.  With ``shared_st_queue``, all ST flows share
        # queue 7 so their windows must be mutually exclusive — gate
        # slots become a scarce resource.
        q_idx = 0

        def _queue_and_window(sc: StreamClass) -> tuple[int, float]:
            nonlocal q_idx
            if sc == StreamClass.SCHEDULED_TRAFFIC:
                if self.config.shared_st_queue:
                    q = 7
                else:
                    q = 7 - min(q_idx, 6)
                q_idx += 1
                if self.config.st_window_us is not None:
                    return q, self.config.st_window_us
                return q, max(min_window, 25.0)
            if sc == StreamClass.RESERVED:
                q = 7 - min(q_idx, 6)
                q_idx += 1
                return q, max(2.0 * tx_time + self.config.gcl_guard_band_us, 100.0)
            return 0, self.config.hyperperiod_us

        if self._flows_semantics:
            for i, fs in enumerate(self._flows_semantics[:M]):
                q, gw = _queue_and_window(fs.stream_class)
                sf = SimFlow(
                    flow_id=fs.flow_id,
                    task_id=fs.task_id,
                    priority_weight=fs.priority_weight,
                    deadline_us=float(fs.delayable_boundary_us) * self.config.deadline_multiplier,
                    period_us=max(float(fs.delayable_boundary_us), 1.0),
                    frame_size_bytes=self.config.frame_size_bytes,
                    stream_class=fs.stream_class,
                    queue=q,
                    gate_size_us=gw,
                    path=self._flow_path(fs.stream_class),
                    accepted=True,
                )
                self.sim_flows.append(sf)

            # pad with random flows
            n_remaining = M - len(self._flows_semantics)
            for i in range(max(0, n_remaining)):
                self.sim_flows.append(self._random_flow(f"f_random_{i}"))
        else:
            for i in range(M):
                self.sim_flows.append(self._random_flow(f"f_{i}"))

        self._previous_e2e: dict[str, float] = {}
        self._last_violation_count: int = 0
        self._arrival_count: int = 0

    def _flow_path(self, stream_class: StreamClass) -> list[int]:
        """Assign a path by stream class.

        Safety/mission flows take the shortest path (1 hop), operational
        flows 2 hops, best-effort flows traverse the whole line (3 hops).
        This mirrors real TSN network planning where critical streams are
        deployed on direct links.
        """
        n = self.n_switches
        if n <= 0:
            return []
        if self.config.single_switch:
            return [0]
        if stream_class == StreamClass.SCHEDULED_TRAFFIC:
            n_hops = 1
        elif stream_class == StreamClass.RESERVED:
            n_hops = min(2, n)
        else:
            n_hops = min(3, n)
        switches = list(range(n))
        rng = np.random.RandomState(self._rng.randint(2**31 - 1))
        rng.shuffle(switches)
        return switches[:n_hops]

    def _random_flow(self, flow_id: str) -> SimFlow:
        profiles = [
            (0.98, 200.0, 200.0, StreamClass.SCHEDULED_TRAFFIC, 7),
            (0.80, 500.0, 1000.0, StreamClass.SCHEDULED_TRAFFIC, 6),
            (0.50, 5000.0, 25000.0, StreamClass.RESERVED, 3),
            (0.20, 10000.0, 100000.0, StreamClass.BEST_EFFORT, 0),
        ]
        idx = self._rng.randint(len(profiles))
        pw, deadline, period, sc, q = profiles[idx]
        tx_time = self.config.frame_size_bytes * 8.0 / self.config.link_rate_mbps
        min_window = tx_time + self.config.gcl_guard_band_us
        if sc == StreamClass.SCHEDULED_TRAFFIC:
            if self.config.st_window_us is not None:
                gw = self.config.st_window_us
            else:
                gw = max(min_window, 25.0)
        elif sc == StreamClass.RESERVED:
            gw = max(2.0 * tx_time + self.config.gcl_guard_band_us, 100.0)
        else:
            gw = self.config.hyperperiod_us
        return SimFlow(
            flow_id=flow_id,
            task_id=flow_id,
            priority_weight=pw + self._rng.uniform(-0.02, 0.02),
            deadline_us=deadline * self.config.deadline_multiplier * self._rng.uniform(0.8, 1.2),
            period_us=period * self._rng.uniform(0.8, 1.2),
            frame_size_bytes=self.config.frame_size_bytes,
            stream_class=sc,
            queue=q,
            gate_size_us=gw,
            path=self._flow_path(sc),
            accepted=True,
            phase_offset_us=self._rng.uniform(0, period),
        )

    # ── Observation ────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        M = self.config.max_active_flows
        obs_flows = np.zeros((M, self._obs_flow_dim), dtype=np.float64)

        for i, sf in enumerate(self.sim_flows):
            if i >= M:
                break
            obs_flows[i, 0] = sf.priority_weight
            if sf.deadline_us > 0:
                obs_flows[i, 1] = min(1.0, sf.deadline_us / 100000.0)
            # stream class one-hot
            sc_idx = {StreamClass.SCHEDULED_TRAFFIC: 0, StreamClass.RESERVED: 1, StreamClass.BEST_EFFORT: 2}[sf.stream_class]
            obs_flows[i, 2 + sc_idx] = 1.0
            obs_flows[i, 5] = sf.queue / 8.0
            obs_flows[i, 6] = 1.0 if sf.accepted else 0.0
            # age (time since last dispatch / deadline)
            age = (self._current_time_us - sf.next_dispatch_us) / max(sf.deadline_us, 1.0)
            obs_flows[i, 7] = np.clip(age, -1.0, 1.0)

        obs_global = np.array([
            self._current_time_us % self.config.hyperperiod_us / max(self.config.hyperperiod_us, 1.0),
            sum(1 for f in self.sim_flows if f.accepted) / max(M, 1),
        ], dtype=np.float64)

        obs_queues = np.zeros(self._obs_queue_dim, dtype=np.float64)
        idx = 0
        q_max = max(self.config.frame_size_bytes * 4, 1.0)
        for sw in self.switches:
            for port in sw.ports:
                for q in range(self.config.n_queues):
                    if idx < self._obs_queue_dim:
                        obs_queues[idx] = min(1.0, port.queues[q] / q_max)
                    idx += 1

        return np.concatenate([obs_flows.flatten(), obs_global, obs_queues])

    def _get_info(self) -> dict:
        return {
            "time_us": self._current_time_us,
            "n_accepted": sum(1 for f in self.sim_flows if f.accepted),
            "n_completed": sum(1 for f in self.sim_flows if len(f.e2e_delays) > 0),
        }

    # ── Action decoding ────────────────────────────────────

    def _decode_action(self, action: np.ndarray) -> dict[str, dict]:
        """Decode [-1,1] action vector to per-flow schedule parameters.

        Queue and gate size are fixed per flow (semantic mapping rules:
        one queue per ST stream per Craciunas Configuration 3); the RL
        policy controls acceptance, dispatch phase, and window offset
        relative to the dispatch phase.
        """
        M = self.config.max_active_flows
        decoded: dict[str, dict] = {}
        for i in range(M):
            offset = i * self._action_dim_per_flow
            if offset + self._action_dim_per_flow > len(action):
                break
            a = action[offset:offset + self._action_dim_per_flow]
            flow = self.sim_flows[i]

            # always admit: admission control belongs to the Safety Shield
            # (deployment-time schedulability veto)
            dispatch_offset = (a[1] + 1) / 2 * flow.period_us
            # window placed relative to the dispatch phase:
            # a[2] = -1 → window exactly on the packet phase (aligned)
            gate_start = (dispatch_offset + (a[2] + 1) / 2 * flow.period_us) % self.config.hyperperiod_us

            decoded[flow.flow_id] = {
                "accept": True,
                "queue": flow.queue,
                "dispatch_offset_us": float(dispatch_offset),
                "gate_start_us": float(gate_start),
                "gate_size_us": flow.gate_size_us,
            }
        return decoded

    # ── Step ───────────────────────────────────────────────

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1

        # dynamic arrival: admit the next flow when its time comes
        if self.config.dynamic_arrival:
            n_total = self.config.max_active_flows
            next_idx = self.config.n_base_flows + self._arrival_count
            if (next_idx < n_total and self._step_count > 0
                    and self._step_count % self.config.arrival_interval == 0):
                self.sim_flows[next_idx].accepted = True
                self._arrival_count += 1

        decoded = self._decode_action(action)

        # apply schedule to flows
        for sf in self.sim_flows:
            if sf.flow_id in decoded:
                d = decoded[sf.flow_id]
                sf.accepted = d["accept"]
                sf.queue = d["queue"]
                sf.phase_offset_us = d["dispatch_offset_us"]
                sf.gate_start_us = d["gate_start_us"]
                sf.gate_size_us = d["gate_size_us"]

        # GCL setup
        self._apply_gcl(decoded)

        # simulate one hyperperiod
        reward = self._simulate_hyperperiod()

        # check termination
        truncated = self._step_count >= self._max_steps
        terminated = False

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _apply_gcl(self, decoded: dict[str, dict]) -> None:
        """Configure GCL masks on all switches based on decoded actions.

        Each flow gets a per-period repeating window: window j of flow f
        opens at ``(gate_start_us + j * period_us) mod HP`` and stays open
        for ``gate_size_us``.  This mirrors IEEE 802.1Qbv GCL semantics
        where a stream's gate entries repeat with its period inside the
        hyperperiod.
        """
        HP = self.config.hyperperiod_us

        for sw in self.switches:
            for port in sw.ports:
                port.gcl_mask = 0x00  # all closed by default

        for sf in self.sim_flows:
            if not sf.accepted or not sf.path or sf.period_us <= 0:
                continue
            n_w = max(1, int(round(HP / sf.period_us)))
            for hop_idx, sw_idx in enumerate(sf.path):
                if sw_idx >= len(self.switches):
                    continue
                sw = self.switches[sw_idx]
                for port in sw.ports:
                    for j in range(n_w):
                        gs = (sf.gate_start_us + j * sf.period_us) % HP
                        ge = (gs + sf.gate_size_us) % HP
                        phase = self._current_time_us % HP
                        if gs < ge:
                            if gs <= phase <= ge:
                                port.gcl_mask |= (1 << sf.queue)
                        else:
                            if phase >= gs or phase <= ge:
                                port.gcl_mask |= (1 << sf.queue)

    def _window_sequence(self, sf: SimFlow, HP: float) -> list[tuple[float, float]]:
        """Return the repeating gate windows of a flow inside one hyperperiod.

        Returns a list of (start, end) phases.  The window repeats with
        ``sf.period_us``; the first window opens at ``sf.gate_start_us``.
        """
        if sf.gate_size_us <= 0 or sf.period_us <= 0:
            return []
        n_w = max(1, int(round(HP / sf.period_us)))
        gs0 = sf.gate_start_us % HP
        windows: list[tuple[float, float]] = []
        for j in range(n_w):
            start = (gs0 + j * sf.period_us) % HP
            end = start + sf.gate_size_us
            windows.append((start, end))  # end may exceed HP (wrap-around)
        return windows

    def _in_window(self, phase: float, sf: SimFlow, HP: float) -> bool:
        for start, end in self._window_sequence(sf, HP):
            if end <= HP:
                if start <= phase < end:
                    return True
            else:  # window wraps across the hyperperiod boundary
                if phase >= start or phase < end - HP:
                    return True
        return False

    def _next_window_wait(self, phase: float, sf: SimFlow, HP: float) -> float:
        """Waiting time from ``phase`` until the next window opens."""
        windows = self._window_sequence(sf, HP)
        if not windows:
            return HP
        best = HP
        for start, _end in windows:
            wait = (start - phase) % HP
            best = min(best, wait)
        return best

    def _simulate_hyperperiod(self) -> float:
        """Run one hyperperiod of flow simulation. Returns the reward.

        Discrete-event model: within one hyperperiod every accepted flow
        releases one frame per period (``HP / period`` frames per step),
        and the GCL gate of each flow opens a window every period.  Frames
        wait for their window and for the egress link to become idle.
        """
        HP = self.config.hyperperiod_us
        tx_time = self.config.frame_size_bytes * 8.0 / self.config.link_rate_mbps
        total_reward = 0.0
        violation_count = 0
        self._current_time_us += HP

        # rejecting a schedulable task is a failure: penalize rejected flows
        # (in dynamic mode, only flows that have already arrived count)
        if self.config.reward_reject_penalty < 0.0:
            arrived_up_to = self.config.n_base_flows + self._arrival_count if self.config.dynamic_arrival else len(self.sim_flows)
            for i, sf in enumerate(self.sim_flows):
                if not sf.accepted and sf.period_us > 0 and i < arrived_up_to:
                    total_reward += self.config.reward_reject_penalty * sf.priority_weight

        # deployment-aware mutual-exclusion penalty: windows of flows in
        # the same queue must not overlap (C2, enforced at deployment by
        # the Safety Shield).  Scale by the product of priority weights.
        if self.config.overlap_penalty_scale > 0.0:
            flows_by_queue: dict[int, list] = {}
            for sf in self.sim_flows:
                if sf.accepted and sf.gate_size_us > 0 and sf.period_us > 0:
                    flows_by_queue.setdefault(sf.queue, []).append(sf)
            for group in flows_by_queue.values():
                n = len(group)
                for i in range(n):
                    for j in range(i + 1, n):
                        a, b = group[i], group[j]
                        cycle = min(a.period_us, b.period_us)
                        if _windows_overlap_mod(a.gate_start_us, a.gate_size_us,
                                                b.gate_start_us, b.gate_size_us, cycle):
                            total_reward -= (self.config.overlap_penalty_scale
                                             * a.priority_weight * b.priority_weight)

        # release frames inside this hyperperiod
        events: list[tuple[float, int, SimFlow, int, list[int], float]] = []
        seq = 0
        for sf in self.sim_flows:
            if not sf.accepted or sf.period_us <= 0:
                continue
            n_pkt = max(1, int(round(HP / sf.period_us)))
            for k in range(n_pkt):
                phase = (sf.phase_offset_us + k * sf.period_us) % HP
                dispatch_time = self._current_time_us - HP + phase
                events.append((dispatch_time, seq, sf, 0, list(sf.path), dispatch_time))
                seq += 1

        # process per-hop store-and-forward events in chronological order.
        # A min-heap keeps later-hop events ordered even when their time is
        # earlier than pending first-hop events of other flows.
        heapq.heapify(events)
        while events:
            t, _seq, sf, hop_idx, path, dispatch_time = heapq.heappop(events)

            if hop_idx >= len(path):
                continue
            sw_idx = path[hop_idx]
            if sw_idx >= len(self.switches):
                continue
            sw = self.switches[sw_idx]
            port = sw.ports[sf.queue % len(sw.ports)]

            # wait for the per-period GCL window
            if not self._in_window(t % HP, sf, HP):
                t += self._next_window_wait(t % HP, sf, HP)

            # wait for the egress link (queueing behind earlier frames)
            if port.link_busy_until_us > t:
                t = port.link_busy_until_us

            port.link_busy_until_us = t + tx_time
            sf.packets_sent += 1

            if hop_idx + 1 < len(path):
                heapq.heappush(events, (t + tx_time, seq, sf, hop_idx + 1, path, dispatch_time))
                seq += 1
            else:
                e2e = t + tx_time - dispatch_time
                sf.e2e_delays.append(e2e)

                if len(sf.e2e_delays) >= 2:
                    jitter = abs(sf.e2e_delays[-1] - sf.e2e_delays[-2])
                    sf.jitter_samples.append(jitter)

                # completion reward
                if e2e <= sf.deadline_us:
                    total_reward += self.config.reward_completion_weight * sf.priority_weight * self.config.reward_semantic_scale
                else:
                    total_reward += self.config.reward_deadline_penalty * sf.priority_weight
                    violation_count += 1

                # jitter bonus
                if sf.stream_class == StreamClass.RESERVED and len(sf.jitter_samples) > 0:
                    latest_jitter = sf.jitter_samples[-1]
                    if latest_jitter < 50.0:  # < 50 μs for AVB
                        total_reward += self.config.reward_jitter_bonus * sf.priority_weight

        # Lagrangian constraint penalty: total_reward -= λ * violations
        if self.config.constraint_penalty > 0.0 and violation_count > 0:
            total_reward -= self.config.constraint_penalty * violation_count

        self._last_violation_count = violation_count
        return total_reward

    # ── Reset ──────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self._current_time_us = 0.0
        self._episode_done = False
        self._step_count = 0
        self._arrival_count = 0

        # reset switches
        for sw in self.switches:
            for port in sw.ports:
                port.queues.fill(0.0)
                port.link_busy_until_us = 0.0
                port.gcl_mask = 0xFF

        # reset flows
        self._setup_flows()
        for sf in self.sim_flows:
            sf.e2e_delays.clear()
            sf.jitter_samples.clear()
            sf.packets_sent = 0
            sf.packets_dropped = 0
            sf.next_dispatch_us = 0.0

        # dynamic arrival: only base flows are admitted at episode start
        if self.config.dynamic_arrival:
            for i, sf in enumerate(self.sim_flows):
                sf.accepted = i < self.config.n_base_flows

        return self._get_obs(), self._get_info()

    def close(self) -> None:
        pass


def _windows_overlap_mod(s1: float, w1: float, s2: float, w2: float, period: float) -> bool:
    """Check whether two repeating windows overlap within one period."""
    if period <= 0:
        return False
    a_lo, a_hi = s1 % period, (s1 % period) + w1
    b_lo, b_hi = s2 % period, (s2 % period) + w2
    if a_lo < b_hi and b_lo < a_hi:
        return True
    # wrap-around windows (window crosses the period boundary)
    if a_hi > period and (b_lo < a_hi - period or b_hi > a_lo):
        return True
    if b_hi > period and (a_lo < b_hi - period or a_hi > b_lo):
        return True
    return False
