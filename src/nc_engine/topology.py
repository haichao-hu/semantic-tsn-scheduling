from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx


# ============================================================
# FlowPath
# ============================================================


@dataclass
class FlowPath:
    """Ordered list of (node, in_port, out_port) hops for a flow.

    Parameters
    ----------
    flow_id : str
    hops : list of (node_id, in_port, out_port)
    """

    flow_id: str
    hops: list[tuple[str, Optional[str], Optional[str]]] = field(default_factory=list)

    @property
    def node_ids(self) -> list[str]:
        return [h[0] for h in self.hops]

    @property
    def num_hops(self) -> int:
        return len(self.hops)

    def __len__(self) -> int:
        return len(self.hops)


# ============================================================
# TSNTopology
# ============================================================


@dataclass
class TSNTopology:
    """Network topology model for NC analysis.

    Wraps a networkx DiGraph with TSN-specific metadata on edges.
    Each edge represents a unidirectional link between two nodes
    (end-stations or switches).

    Edge attributes:
      - link_rate_mbps : float
      - propagation_us : float (delay)
      - port_out / port_in : str
    """

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    flow_paths: dict[str, FlowPath] = field(default_factory=dict)
    _node_index: dict[str, int] = field(default_factory=dict)

    # ── Construction ──────────────────────────────────────

    def add_node(self, node_id: str, node_type: str = "switch") -> None:
        self.graph.add_node(node_id, type=node_type)
        if node_id not in self._node_index:
            self._node_index[node_id] = len(self._node_index)

    def add_link(
        self,
        src: str,
        dst: str,
        link_rate_mbps: float = 1000.0,
        propagation_us: float = 0.006,  # 6 ns = Xue 2024 measurement
        processing_us: float = 1.9,     # 1.9 μs = Xue 2024 measurement
    ) -> None:
        """Add a directed link between two nodes."""
        self.add_node(src)
        self.add_node(dst)
        self.graph.add_edge(
            src, dst,
            link_rate_mbps=link_rate_mbps,
            propagation_us=propagation_us,
            processing_us=processing_us,
        )

    def add_bidirectional_link(
        self,
        node_a: str,
        node_b: str,
        link_rate_mbps: float = 1000.0,
        propagation_us: float = 0.006,
        processing_us: float = 1.9,
    ) -> None:
        self.add_link(node_a, node_b, link_rate_mbps, propagation_us, processing_us)
        self.add_link(node_b, node_a, link_rate_mbps, propagation_us, processing_us)

    # ── Path management ───────────────────────────────────

    def set_flow_path(self, flow_id: str, path: FlowPath) -> None:
        """Register a flow's path through the topology."""
        self.flow_paths[flow_id] = path

    def get_path(self, flow_id: str) -> list[str]:
        """Return ordered list of nodes the flow traverses."""
        p = self.flow_paths.get(flow_id)
        return p.node_ids if p else []

    # ── Queries ───────────────────────────────────────────

    def get_link_rate(self, src: str, dst: str) -> float:
        return self.graph.edges[src, dst].get("link_rate_mbps", 1000.0)

    def get_propagation_delay(self, src: str, dst: str) -> float:
        return self.graph.edges[src, dst].get("propagation_us", 0.006)

    def get_processing_delay(self, node: str) -> float:
        """Average per-hop processing delay (switching fabric + lookup)."""
        edges = list(self.graph.out_edges(node, data=True))
        if edges:
            return float(edges[0][2].get("processing_us", 1.9))
        return 1.9

    def all_links_for_node(self, node: str) -> list[tuple[str, str]]:
        """Return all (src, dst) links incident to `node`."""
        return [
            (u, v) for u, v in self.graph.edges() if u == node or v == node
        ]

    @property
    def nodes(self) -> list[str]:
        return list(self.graph.nodes)

    @property
    def edges(self) -> list[tuple[str, str]]:
        return list(self.graph.edges)

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    # ── Shortest-path routing ─────────────────────────────

    def shortest_path(self, src: str, dst: str) -> list[str] | None:
        """Shortest path by hop count (weight=1 per edge)."""
        try:
            return nx.shortest_path(self.graph, src, dst, weight=None)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None


# ============================================================
# Path delay decomposition
# ============================================================


@dataclass
class PathDelayComponents:
    """Decomposed per-hop delay components for NC analysis."""

    propagation_us: float = 0.0    # Σ(link propagation) across all hops
    processing_us: float = 0.0     # Σ(switch processing) at each intermediate node
    transmission_us: float = 0.0   # Σ(tx_time) for each hop = frame_size / link_rate
    queuing_us: float = 0.0        # queuing delay bound from NC

    @property
    def total_us(self) -> float:
        return self.propagation_us + self.processing_us + self.transmission_us + self.queuing_us


def compute_path_delays(
    topology: TSNTopology,
    flow_id: str,
    flow_path: FlowPath | None = None,
    frame_size_bytes: float = 256.0,
    link_delays: dict[tuple[str, str], float] | None = None,
    num_queuing_stages: int | None = None,
) -> PathDelayComponents:
    """Compute decomposed per-hop delay components for a flow.

    Parameters
    ----------
    topology : TSNTopology
    flow_id : str
    flow_path : FlowPath or None
        If None, reads from topology.flow_paths.
    frame_size_bytes : float
    link_delays : dict or None
        Per-link (src, dst) → delay_us. Overrides topology data.
    num_queuing_stages : int or None
        If None, uses hop count − 1.

    Returns
    -------
    PathDelayComponents
    """
    path = flow_path if flow_path is not None else topology.flow_paths.get(flow_id)
    if path is None or len(path.hops) < 2:
        return PathDelayComponents()

    comp = PathDelayComponents()
    hops = path.hops
    for i in range(len(hops) - 1):
        src_node, _, out_port = hops[i]
        dst_node, in_port, _ = hops[i + 1]

        # Propagation
        prop = topology.get_propagation_delay(src_node, dst_node)
        comp.propagation_us += prop

        # Processing at the sender node (switch fabric)
        comp.processing_us += topology.get_processing_delay(src_node)

        # Transmission time
        link_rate = topology.get_link_rate(src_node, dst_node)
        comp.transmission_us += (frame_size_bytes * 8.0) / link_rate

    # Queuing: per switch = the NC bound at each egress port.
    # This is filled by the delay_bounds module.
    # We reserve the default but accept override from caller.
    n_queues = num_queuing_stages if num_queuing_stages is not None else (len(hops) - 1)
    if link_delays:
        comp.propagation_us = sum(link_delays.values())

    return comp


# ============================================================
# Topology factories
# ============================================================


def make_line_topology(n_switches: int = 3, link_rate_mbps: float = 1000.0) -> TSNTopology:
    """Create a linear topology:  ES0 — SW1 — SW2 — ... — SW{N} — ES1.

    End-stations are 'es0' and 'es1', switches are 'sw1' through 'sw{n}'.
    """
    t = TSNTopology()
    t.add_node("es0", node_type="end-station")
    t.add_node("es1", node_type="end-station")

    prev = "es0"
    for i in range(1, n_switches + 1):
        sw = f"sw{i}"
        t.add_node(sw, node_type="switch")
        t.add_bidirectional_link(prev, sw, link_rate_mbps)
        prev = sw
    t.add_bidirectional_link(prev, "es1", link_rate_mbps)

    return t


def make_ring_topology(n_switches: int = 4, link_rate_mbps: float = 1000.0) -> TSNTopology:
    """Create a ring topology:  SW1 — SW2 — ... — SW{N} — SW1.

    Each switch also connects to a local end-station 'es{i}'.
    """
    t = TSNTopology()
    for i in range(1, n_switches + 1):
        t.add_node(f"sw{i}", node_type="switch")
        t.add_node(f"es{i}", node_type="end-station")
        t.add_bidirectional_link(f"es{i}", f"sw{i}", link_rate_mbps)
    for i in range(1, n_switches):
        t.add_bidirectional_link(f"sw{i}", f"sw{i+1}", link_rate_mbps)
    t.add_bidirectional_link(f"sw{n_switches}", f"sw1", link_rate_mbps)

    return t


def make_ieee_60802_topology(link_rate_mbps: float = 1000.0) -> TSNTopology:
    """IEC/IEEE 60802 industrial topology: 5 switches, redundant ring backbone.

    Layout (simplified from 60802 use case):
        PLC1 — SW1 — SW2 — AGV1
                |    |
              SW3 — SW4 — Robot1
                |
              HMI1

    Each switch connects to 1–2 end-stations.
    """
    t = TSNTopology()

    # Switches
    for i in range(1, 6):
        t.add_node(f"sw{i}", node_type="switch")

    # End-stations
    end_stations = {
        "plc1": "sw1",
        "agv1": "sw2",
        "robot1": "sw4",
        "hmi1": "sw3",
        "plc2": "sw5",
    }
    for es_name, parent_sw in end_stations.items():
        t.add_node(es_name, node_type="end-station")
        t.add_bidirectional_link(es_name, parent_sw, link_rate_mbps)

    # Switch-to-switch links (ring backbone)
    t.add_bidirectional_link("sw1", "sw2", link_rate_mbps)
    t.add_bidirectional_link("sw2", "sw3", link_rate_mbps)
    t.add_bidirectional_link("sw3", "sw4", link_rate_mbps)
    t.add_bidirectional_link("sw4", "sw5", link_rate_mbps)
    t.add_bidirectional_link("sw5", "sw1", link_rate_mbps)

    # Cross-link for redundancy (IEC 60802 topology)
    t.add_bidirectional_link("sw2", "sw4", link_rate_mbps)

    return t
