"""
D³ (Direct-DRAM-Delivery) intra-PE dataflow search — paper Algorithm 1.

Exhaustively searches over (tile_M, tile_N, tile_K, RU, loop_order, 2D core split)
for one (op_key, chiplet_cfg). Pareto-prunes on (latency, energy, sram_bytes).

Key extension beyond paper: also searches chiplet-internal 2D core split
(cores_M, cores_N), because Week 1 found decode requires 2D split to avoid
GEMV degradation.

The cost model interface is the `CostEvaluator` Protocol. Wire Week 1 by
implementing it in cost_model/real_evaluator.py. MockCostEvaluator is
provided for standalone testing.
"""

from __future__ import annotations
from typing import Protocol
import time

from mapping_lib import (
    OpKey, MappingCandidate, OpEntry, SearchStats,
)

LOOP_ORDERS: tuple = ("MNK", "MKN", "NMK", "NKM", "KMN", "KNM")
RU_POLICIES: tuple = ("IRU", "WRU", "ORU", "ARU")


# ============================================================
# CostEvaluator protocol — Week 1 wire-in point
# ============================================================

class CostEvaluator(Protocol):
    """Interface Week 1's cost model must implement.

    Wire-in: write cost_model/real_evaluator.py with a class implementing
    .evaluate(...) returning the dict below. build_mapping_lib.py auto-detects.
    """
    def evaluate(self, op_key: OpKey, tile, ru, loop_order, cores_split, chiplet_cfg) -> dict:
        """Returns dict with keys:
            latency_us, energy_uJ, sram_bytes,
            p_mpu_W, p_vpu_W, p_dram_W, p_noc_W, p_avg_W, p_peak_W,
            bw_utilization, mpu_utilization, sa_active_rows_frac,
            is_compute_bound, provenance
        """
        ...


# ============================================================
# MockCostEvaluator — for standalone testing only
# ============================================================

class MockCostEvaluator:
    """Toy cost model so D³ search runs end-to-end without Week 1 wired.

    Uses simplistic max(compute_time, mem_time) for latency (known ≤30% error
    per project's M4 ledger). Numbers are NOT meaningful — only used to verify
    search-logic correctness.
    """

    def __init__(self):
        self.method = "mock_M0"

    def evaluate(self, op_key, tile, ru, loop_order, cores_split, chiplet_cfg):
        M, N, K = op_key.M, op_key.N, op_key.K
        T_M, T_N, T_K = tile
        cM, cN, cK = cores_split
        cores = cM * cN * cK

        sa_rows = chiplet_cfg.get("sa_rows", 64)
        sa_cols = chiplet_cfg.get("sa_cols", 32)
        freq_hz = chiplet_cfg.get("freq_hz", 800e6)
        bw_Bps = chiplet_cfg.get("bw_Bps", 26.2e12)  # bytes/s (PC=10.6e12, DC=26.2e12)
        bytes_per_elem = 2  # fp16
        tile_overhead_cycles = 8  # pipeline fill — penalizes degenerate tiny tiles

        # GEMV degradation: SA rows underused when T_M < sa_rows
        sa_active = min(T_M, sa_rows)
        sa_active_rows_frac = sa_active / sa_rows if sa_rows > 0 else 0.0

        # Tile MACs and cycles (with per-tile overhead so tile_size=1 isn't optimal)
        mac_per_tile = T_M * T_N * T_K
        cycles_per_tile = tile_overhead_cycles + (
            mac_per_tile / max(sa_active * sa_cols, 1) if sa_active > 0 else 1e12
        )

        # Number of tiles
        n_tiles_M = max(1, (M + T_M - 1) // T_M)
        n_tiles_N = max(1, (N + T_N - 1) // T_N)
        n_tiles_K = max(1, (K + T_K - 1) // T_K)
        n_tiles_total = n_tiles_M * n_tiles_N * n_tiles_K

        n_tiles_per_core = n_tiles_total / max(cores, 1)
        compute_time_s = (cycles_per_tile * n_tiles_per_core) / freq_hz

        # Memory traffic by RU policy (rough — Week 1 has the real version)
        if ru == "WRU":  # stage B, reload A and C
            traffic_bytes = (M * K + M * N + N * K) * bytes_per_elem
        elif ru == "IRU":
            traffic_bytes = (M * K + 2 * N * K + M * N) * bytes_per_elem
        elif ru == "ORU":
            traffic_bytes = (M * K * n_tiles_N + N * K * n_tiles_M + M * N) * bytes_per_elem
        else:  # ARU
            traffic_bytes = (M * K + N * K + M * N) * bytes_per_elem

        mem_time_s = traffic_bytes / bw_Bps
        latency_s = max(compute_time_s, mem_time_s)
        is_compute_bound = compute_time_s > mem_time_s

        # Power model (rough)
        p_mpu_max = chiplet_cfg.get("p_mpu_max", 200)
        p_dram_max = chiplet_cfg.get("p_dram_max", 100)
        p_noc_max = chiplet_cfg.get("p_noc_max", 30)
        total_cores = chiplet_cfg.get("total_cores", 256)
        p_mpu = p_mpu_max * (cores / total_cores) * sa_active_rows_frac
        p_dram = p_dram_max * min(1.0, mem_time_s / max(latency_s, 1e-12))
        p_noc = p_noc_max * 0.3
        p_vpu = 5.0
        p_avg = p_mpu + p_dram + p_noc + p_vpu
        p_peak = p_avg * 1.5

        # SRAM footprint per RU
        ru_sram = {
            "IRU": T_M * T_K * bytes_per_elem,
            "WRU": T_N * T_K * bytes_per_elem,
            "ORU": T_M * T_N * bytes_per_elem,
            "ARU": (T_M * T_K + T_N * T_K + T_M * T_N) * bytes_per_elem,
        }[ru]

        return {
            "latency_us": latency_s * 1e6,
            "energy_uJ": p_avg * latency_s * 1e6,
            "sram_bytes": ru_sram,
            "p_mpu_W": p_mpu, "p_vpu_W": p_vpu,
            "p_dram_W": p_dram, "p_noc_W": p_noc,
            "p_avg_W": p_avg, "p_peak_W": p_peak,
            "bw_utilization": min(1.0, traffic_bytes / (bw_Bps * max(latency_s, 1e-12))),
            "mpu_utilization": min(1.0, compute_time_s / max(latency_s, 1e-12)) * sa_active_rows_frac,
            "sa_active_rows_frac": sa_active_rows_frac,
            "is_compute_bound": is_compute_bound,
            "provenance": {
                "latency_us": ["mock:M0"],
                "energy_uJ": ["mock:M0"],
                "p_dram_W": ["mock:M0"],
            },
        }


# ============================================================
# Enumeration helpers
# ============================================================

def smart_divisors(n: int, max_count: int = 8) -> list:
    """Sample up to max_count divisors of n, biased toward powers of 2.
    Always includes 1 and n."""
    if n <= 1:
        return [max(1, n)]
    raw = {1, n}
    i = 2
    while i * i <= n:
        if n % i == 0:
            raw.add(i)
            raw.add(n // i)
        i += 1
    p = 2
    while p <= n:
        raw.add(p)
        p *= 2
    sorted_d = sorted(raw)
    if len(sorted_d) <= max_count:
        return sorted_d
    indices = sorted({
        int(round(i * (len(sorted_d) - 1) / (max_count - 1)))
        for i in range(max_count)
    })
    return [sorted_d[i] for i in indices]


def enumerate_core_splits(total_cores: int, M: int, N: int,
                          max_count: int = 8) -> list:
    """2D core splits (cM, cN, cK=1) where cM*cN ≤ total_cores and cM ≤ M, cN ≤ N.
    Decode (M small) prefers cM=1; prefill prefers balanced."""
    splits = []
    cK = 1
    for cM in range(1, min(total_cores, M) + 1):
        if total_cores % cM != 0:
            continue
        cN = total_cores // cM
        if cN < 1 or cN > N:
            continue
        splits.append((cM, cN, cK))
    if not splits:
        splits.append((1, 1, 1))
    if len(splits) > max_count:
        indices = sorted({
            int(round(i * (len(splits) - 1) / (max_count - 1)))
            for i in range(max_count)
        })
        splits = [splits[i] for i in indices]
    return splits


def ru_feasible(ru: str, T_M: int, T_N: int, T_K: int,
                sram_budget_bytes: int, bytes_per_elem: int = 2) -> bool:
    if ru == "IRU":
        return T_M * T_K * bytes_per_elem <= sram_budget_bytes
    if ru == "WRU":
        return T_N * T_K * bytes_per_elem <= sram_budget_bytes
    if ru == "ORU":
        return T_M * T_N * bytes_per_elem <= sram_budget_bytes
    if ru == "ARU":
        total = (T_M * T_K + T_N * T_K + T_M * T_N) * bytes_per_elem
        return total <= sram_budget_bytes
    return False


# ============================================================
# Main D³ search
# ============================================================

def d3_search(
    op_key: OpKey,
    chiplet_cfg: dict,
    evaluator,
    loop_orders=LOOP_ORDERS,
    ru_policies=RU_POLICIES,
    max_tiles_per_dim: int = 6,
    max_core_splits: int = 6,
    bytes_per_elem: int = 2,
    verbose: bool = False,
) -> OpEntry:
    """Exhaustive D³ search for one OpKey. Returns OpEntry with Pareto front."""
    from pareto import pareto_filter_3d  # late import to avoid cycle

    M, N, K = op_key.M, op_key.N, op_key.K
    sram_budget = chiplet_cfg.get("sram_budget_bytes", 1024 * 1024)
    total_cores = chiplet_cfg.get("total_cores", 256)

    t_start = time.time()
    stats = SearchStats()

    tiles_M = smart_divisors(M, max_tiles_per_dim)
    tiles_N = smart_divisors(N, max_tiles_per_dim)
    tiles_K = smart_divisors(K, max_tiles_per_dim)
    core_splits = enumerate_core_splits(total_cores, M, N, max_core_splits)

    if verbose:
        print(f"[d3_search] {op_key.canonical_str()}")
        print(f"  tiles_M={tiles_M}, tiles_N={tiles_N}, tiles_K={tiles_K}")
        print(f"  core_splits={core_splits}")

    all_candidates = []

    for T_M in tiles_M:
        for T_N in tiles_N:
            for T_K in tiles_K:
                feasible_rus = [
                    ru for ru in ru_policies
                    if ru_feasible(ru, T_M, T_N, T_K, sram_budget, bytes_per_elem)
                ]
                if not feasible_rus:
                    stats.n_pruned_sram += len(ru_policies) * len(loop_orders) * len(core_splits)
                    continue

                for ru in feasible_rus:
                    for loop_order in loop_orders:
                        for split in core_splits:
                            stats.n_evaluated += 1
                            try:
                                cost = evaluator.evaluate(
                                    op_key=op_key,
                                    tile=(T_M, T_N, T_K),
                                    ru=ru, loop_order=loop_order,
                                    cores_split=split,
                                    chiplet_cfg=chiplet_cfg,
                                )
                            except Exception as e:
                                if verbose:
                                    print(f"  eval error: {e}")
                                continue

                            cand = MappingCandidate(
                                mapping_id=MappingCandidate.make_id(
                                    split[0], split[1], split[2],
                                    T_M, T_N, T_K, ru, loop_order
                                ),
                                cores_M=split[0], cores_N=split[1], cores_K=split[2],
                                T_M=T_M, T_N=T_N, T_K=T_K,
                                RU=ru, loop_order=loop_order,
                                latency_us=cost["latency_us"],
                                energy_uJ=cost["energy_uJ"],
                                sram_bytes=cost["sram_bytes"],
                                p_mpu_W=cost["p_mpu_W"], p_vpu_W=cost["p_vpu_W"],
                                p_dram_W=cost["p_dram_W"], p_noc_W=cost["p_noc_W"],
                                p_avg_W=cost["p_avg_W"], p_peak_W=cost["p_peak_W"],
                                is_compute_bound=cost["is_compute_bound"],
                                bw_utilization=cost["bw_utilization"],
                                mpu_utilization=cost["mpu_utilization"],
                                sa_active_rows_frac=cost["sa_active_rows_frac"],
                                provenance=cost.get("provenance", {}),
                            )
                            all_candidates.append(cand)
                            stats.n_feasible += 1

    pareto = pareto_filter_3d(all_candidates)
    for c in pareto:
        c.pareto_rank = 0
    stats.n_pareto = len(pareto)
    stats.search_time_s = time.time() - t_start

    if verbose:
        print(f"  evaluated={stats.n_evaluated}, feasible={stats.n_feasible}, "
              f"pareto={stats.n_pareto}, time={stats.search_time_s:.2f}s")

    return OpEntry(op_key=op_key, pareto_front=pareto, stats=stats)
