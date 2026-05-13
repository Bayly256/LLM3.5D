"""
Communication cost model for LaMoSys3.5D simulator.

Models the two-level interconnect from LaMoSys3.5D §IV-A:
    - NoC (intra-chiplet): 200 GB/s aggregate, 2D mesh over PEs
    - NoP (inter-chiplet): 800 GB/s aggregate, AIB 2.0 PHY over interposer

Core formula (paper §V-C):
    t_comm = α · m + β · h
where α = 1/bandwidth (time per byte) and β = per-hop latency.

Provides:
    - p2p latency/energy for arbitrary message sizes and hop counts
    - Ring all-reduce (for TP collectives)
    - Tree multicast / reduce (for activation broadcast & reductions)
    - KV cache transfer (PD-disaggregated serving)
    - Two-level mesh hop accounting (intra + inter chiplet)

Author: <you>
References:
    - LaMoSys3.5D, arXiv:2512.08731, §IV-A, §V-B/C, §VI-A
    - AIB 2.0 spec (CHIPS Alliance): PHY for D2D links
"""
import math
from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Link characterization
# ---------------------------------------------------------------------------

@dataclass
class LinkConfig:
    """Communication link characteristics.

    bandwidth_GBs        peak point-to-point bandwidth on one link
    per_hop_latency_ns   router + PHY traversal cost per hop
    energy_pJ_per_bit    per-bit transmission energy (per hop)
    """
    name: str
    bandwidth_GBs: float
    per_hop_latency_ns: float
    energy_pJ_per_bit: float


# Reference link configs (LaMoSys3.5D §VI-A)
NOC = LinkConfig(
    name='NoC',
    bandwidth_GBs=200.0,
    per_hop_latency_ns=5.0,
    energy_pJ_per_bit=0.3,
)

NOP = LinkConfig(
    name='NoP',
    bandwidth_GBs=800.0,
    per_hop_latency_ns=20.0,
    energy_pJ_per_bit=0.5,
)


# ---------------------------------------------------------------------------
# Point-to-point
# ---------------------------------------------------------------------------

def p2p_latency_ns(message_bytes: float, hop_count: int,
                    link: LinkConfig) -> float:
    """Send `message_bytes` over `hop_count` hops on `link`.

    Implements t_comm = α · m + β · h from LaMoSys3.5D §V-C.
        α = 1e9 / (BW_GBs · 1e9)  ns per byte
        β = per_hop_latency_ns
    """
    if message_bytes <= 0:
        return 0.0
    alpha_ns_per_byte = 1.0 / link.bandwidth_GBs   # GB/s ⇒ ns/B (units cancel)
    return alpha_ns_per_byte * message_bytes + link.per_hop_latency_ns * hop_count


def p2p_energy_pJ(message_bytes: float, hop_count: int,
                   link: LinkConfig) -> float:
    """Per-bit energy × bits × hops (each hop consumes energy)."""
    return message_bytes * 8 * link.energy_pJ_per_bit * max(1, hop_count)


# ---------------------------------------------------------------------------
# 2D mesh topology helpers
# ---------------------------------------------------------------------------

def manhattan_distance(coord1: Tuple[int, int], coord2: Tuple[int, int]) -> int:
    """Hops between two grid coordinates."""
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


def mesh_diameter(grid_shape: Tuple[int, int]) -> int:
    """Worst-case hops in an R×C mesh."""
    return (grid_shape[0] - 1) + (grid_shape[1] - 1)


def mesh_avg_distance(grid_shape: Tuple[int, int]) -> float:
    """Average hops between random pairs.

    For an R×C mesh: avg ≈ (R + C) / 3 (good approximation for R, C ≥ 3).
    """
    return (grid_shape[0] + grid_shape[1]) / 3.0


# ---------------------------------------------------------------------------
# Collective operations
# ---------------------------------------------------------------------------

def ring_all_reduce_ns(tensor_bytes: float, n_devices: int,
                        link: LinkConfig, avg_hops_per_step: int = 1) -> float:
    """Ring all-reduce latency.

    Algorithm: each device sends its chunk to the next neighbor in a ring;
    after 2(N-1) steps each device has the full reduced tensor.

    Per-step traffic = tensor_bytes / N per device.
    avg_hops_per_step: 1 for true ring topology; for ring-embedded-in-mesh,
                      pass the average hops per step.
    """
    if n_devices <= 1 or tensor_bytes <= 0:
        return 0.0
    chunk = tensor_bytes / n_devices
    n_steps = 2 * (n_devices - 1)
    step_t = p2p_latency_ns(chunk, avg_hops_per_step, link)
    return n_steps * step_t


def ring_all_reduce_energy_pJ(tensor_bytes: float, n_devices: int,
                               link: LinkConfig, avg_hops_per_step: int = 1) -> float:
    """Aggregate energy across the ring."""
    if n_devices <= 1 or tensor_bytes <= 0:
        return 0.0
    chunk = tensor_bytes / n_devices
    n_steps = 2 * (n_devices - 1)
    return n_devices * n_steps * p2p_energy_pJ(chunk, avg_hops_per_step, link)


def tree_multicast_ns(tensor_bytes: float, n_devices: int,
                       link: LinkConfig, avg_hops_per_step: int = 1) -> float:
    """Tree multicast: 1 root → N receivers in ceil(log2(N)) steps.

    Each step doubles the number of devices holding the tensor; the bottleneck
    is the single longest forward chain. Approximated as log2(N) × p2p.
    """
    if n_devices <= 1 or tensor_bytes <= 0:
        return 0.0
    n_steps = max(1, math.ceil(math.log2(n_devices)))
    return n_steps * p2p_latency_ns(tensor_bytes, avg_hops_per_step, link)


def tree_reduce_ns(tensor_bytes: float, n_devices: int,
                    link: LinkConfig, avg_hops_per_step: int = 1) -> float:
    """Tree reduce: N senders → 1 root, same shape as multicast in reverse."""
    return tree_multicast_ns(tensor_bytes, n_devices, link, avg_hops_per_step)


def all_reduce_via_rd_mc_ns(tensor_bytes: float, n_devices: int,
                             link: LinkConfig, avg_hops_per_step: int = 1) -> float:
    """All-reduce as REDUCE + MULTICAST.

    LaMoSys3.5D §V-B notes that the runtime scheduler decomposes
    all-reduce into a reduce (RD) then a multicast (MC), so the two
    halves can be overlapped with compute independently.
    """
    return (tree_reduce_ns(tensor_bytes, n_devices, link, avg_hops_per_step) +
            tree_multicast_ns(tensor_bytes, n_devices, link, avg_hops_per_step))


# ---------------------------------------------------------------------------
# Two-level mesh: intra-chiplet (NoC) + inter-chiplet (NoP)
# ---------------------------------------------------------------------------

def two_level_mesh_latency_ns(message_bytes: float,
                               src_chiplet: Tuple[int, int],
                               dst_chiplet: Tuple[int, int],
                               src_pe: Tuple[int, int],
                               dst_pe: Tuple[int, int],
                               noc: LinkConfig = NOC,
                               nop: LinkConfig = NOP,
                               edge_pe_hop: int = 1) -> float:
    """Latency from one PE to another on a two-level mesh.

    Same chiplet: NoC hops between PEs.
    Cross chiplet: NoC to chiplet edge + NoP between chiplets + NoC to dst PE.

    edge_pe_hop: hops from a PE to the chiplet edge (rough approximation;
                 better: compute from actual PE coords vs chiplet edge).
    """
    if src_chiplet == dst_chiplet:
        hops = manhattan_distance(src_pe, dst_pe)
        return p2p_latency_ns(message_bytes, hops, noc)

    chiplet_hops = manhattan_distance(src_chiplet, dst_chiplet)
    return (p2p_latency_ns(message_bytes, edge_pe_hop, noc) +
            p2p_latency_ns(message_bytes, chiplet_hops, nop) +
            p2p_latency_ns(message_bytes, edge_pe_hop, noc))


# ---------------------------------------------------------------------------
# KV cache transfer (PD-disaggregated serving)
# ---------------------------------------------------------------------------

def kv_transfer_ns(kv_bytes: float, n_parallel_streams: int = 1,
                    link: LinkConfig = NOP, avg_hops: int = 2) -> float:
    """Transfer KV cache from prefill chiplet(s) to decode chiplet(s).

    n_parallel_streams: number of NoP links used in parallel (typically
                        equal to TP degree on the decode side).
    avg_hops: average NoP hops between PC and DC chiplets in the package.
    """
    per_stream = kv_bytes / max(1, n_parallel_streams)
    return p2p_latency_ns(per_stream, avg_hops, link)


def kv_transfer_overlapped_ns(kv_bytes: float, compute_ns: float,
                               n_parallel_streams: int = 1,
                               link: LinkConfig = NOP,
                               avg_hops: int = 2) -> float:
    """KV transfer overlapped with downstream compute.

    Paper §V-B: the dynamic scheduler pipelines KV forwarding with the
    next stage's QKV-projection. Effective latency is max(transfer, compute).
    """
    t_xfer = kv_transfer_ns(kv_bytes, n_parallel_streams, link, avg_hops)
    return max(t_xfer, compute_ns)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_test():
    print("=" * 70)
    print("LaMoSys3.5D communication cost model — sanity checks")
    print("=" * 70)

    # --- Link summary ---
    for link in (NOC, NOP):
        print(f"\n[{link.name}] BW={link.bandwidth_GBs} GB/s, "
              f"β={link.per_hop_latency_ns} ns/hop, E={link.energy_pJ_per_bit} pJ/bit")
        print(f"  Point-to-point (1 hop):")
        print(f"    {'size':>8s} {'latency':>12s} {'achieved BW':>14s}")
        for sz_kb in (1, 64, 1024, 16384):
            t = p2p_latency_ns(sz_kb * 1024, 1, link)
            ach = sz_kb * 1024 / (t * 1e-9) / 1e9 if t > 0 else 0
            print(f"    {sz_kb:>5d} KB {t:>10.1f} ns {ach:>11.1f} GB/s")

    # --- Ring all-reduce sweep ---
    print(f"\n--- Ring all-reduce: 1 MB tensor ---")
    print(f"  {'devices':>8s} {'NoC (μs)':>10s} {'NoP (μs)':>10s}")
    for n in (2, 4, 8, 16, 32):
        t_noc = ring_all_reduce_ns(1024**2, n, NOC) / 1000
        t_nop = ring_all_reduce_ns(1024**2, n, NOP) / 1000
        print(f"  {n:>8d} {t_noc:>10.2f} {t_nop:>10.2f}")

    # --- Tree multicast sweep ---
    print(f"\n--- Tree multicast: 1 MB tensor ---")
    print(f"  {'devices':>8s} {'NoC (μs)':>10s} {'NoP (μs)':>10s}")
    for n in (2, 4, 8, 16, 32):
        t_noc = tree_multicast_ns(1024**2, n, NOC) / 1000
        t_nop = tree_multicast_ns(1024**2, n, NOP) / 1000
        print(f"  {n:>8d} {t_noc:>10.2f} {t_nop:>10.2f}")

    # --- Realistic LLM operations ---
    print(f"\n--- Realistic LLM serving operations (GPT3-13B, H=5120) ---")

    # TP=8 AR after FFN (prefill): activation tensor M×H = 4*1024 × 5120 × 2 = 40 MB
    ar_prefill = 4 * 1024 * 5120 * 2
    t = ring_all_reduce_ns(ar_prefill, 8, NOP, avg_hops_per_step=1)
    print(f"  Prefill TP=8 AR after FFN (bs=4, seq=1024):")
    print(f"    tensor = {ar_prefill/1e6:6.1f} MB → {t/1000:7.2f} μs over NoP")

    # Decode TP=4 AR: activation tensor M×H = 4*1 × 5120 × 2 = 40 KB
    ar_decode = 4 * 5120 * 2
    t = ring_all_reduce_ns(ar_decode, 4, NOP)
    print(f"  Decode TP=4 AR (bs=4, per-token):")
    print(f"    tensor = {ar_decode/1024:6.1f} KB → {t/1000:7.2f} μs over NoP")

    # KV cache transfer per layer (MHA: 2 × bs × seq × n_heads × d_head × 2)
    kv_layer = 2 * 4 * 1024 * 40 * 128 * 2
    t = kv_transfer_ns(kv_layer, n_parallel_streams=8, link=NOP, avg_hops=2)
    print(f"  KV transfer per layer (MHA, bs=4, seq=1024, 8-way):")
    print(f"    KV = {kv_layer/1e6:6.1f} MB → {t/1000:7.2f} μs over NoP")

    # KV with GQA shrinks dramatically (LLaMA3-70B: n_kv_heads=8, d_head=128)
    kv_gqa = 2 * 4 * 1024 * 8 * 128 * 2
    t = kv_transfer_ns(kv_gqa, n_parallel_streams=8, link=NOP, avg_hops=2)
    print(f"  KV transfer per layer (GQA n_kv=8, bs=4, seq=1024, 8-way):")
    print(f"    KV = {kv_gqa/1e6:6.2f} MB → {t/1000:7.2f} μs over NoP")

    # --- Two-level mesh example ---
    print(f"\n--- Two-level mesh latency (1 MB tensor) ---")
    # Same chiplet, PE (0,0) → (3,3): 6 NoC hops
    t1 = two_level_mesh_latency_ns(
        1024**2, src_chiplet=(0, 0), dst_chiplet=(0, 0),
        src_pe=(0, 0), dst_pe=(3, 3))
    # Different chiplet (0,0)→(2,2), 4 NoP hops + 2 NoC edges
    t2 = two_level_mesh_latency_ns(
        1024**2, src_chiplet=(0, 0), dst_chiplet=(2, 2),
        src_pe=(0, 0), dst_pe=(0, 0))
    print(f"  Same chiplet,  PE(0,0)→(3,3), 6 NoC hops:   {t1/1000:.2f} μs")
    print(f"  Diff chiplets, (0,0)→(2,2), 4 NoP + 2 NoC:  {t2/1000:.2f} μs")

    # --- Day 4 acceptance ---
    print(f"\n--- Day 4 acceptance check ---")
    t = ring_all_reduce_ns(1024**2, 8, NOP) / 1000
    status = "✓" if 1.0 <= t <= 10.0 else "✗"
    print(f"  1 MB ring AR, 8 devices, NoP: {t:.2f} μs  (expected 1–10 μs)  {status}")

    # --- Compute vs comm balance check ---
    # Compare AR time to FFN compute time (from compute.py: ~16.4 ms for FFN_up on PC)
    print(f"\n--- Compute/comm balance (prefill FFN on PC, bs=4, seq=1024) ---")
    ffn_compute_us = 16384  # μs, from compute.py output
    ar_comm_us = ring_all_reduce_ns(ar_prefill, 8, NOP) / 1000
    print(f"  FFN compute: {ffn_compute_us:8.1f} μs")
    print(f"  AR comm:     {ar_comm_us:8.2f} μs")
    print(f"  Comm/compute ratio: {ar_comm_us/ffn_compute_us*100:.2f}%  "
          f"(should be small; if >20% the comm is the bottleneck)")


if __name__ == "__main__":
    _run_self_test()