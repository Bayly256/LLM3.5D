"""
D³ (Direct-DRAM-Delivery) intra-PE dataflow search — paper Algorithm 1.

Exhaustively searches over (tile_M, tile_N, tile_K, RU, loop_order, 2D core
split) for one (op_key, chiplet_cfg). Pareto-prunes on
(latency, energy, sram_bytes).

KNOWN LIMITATIONS under Week 1 cost model (see real_evaluator.py docstring):
  - loop_order has no effect; restricted to ('MNK',) by default to avoid 6×
    redundant Pareto points
  - tile size affects ONLY SRAM feasibility, not latency/energy
  - Pareto front collapses to "best feasible RU per (op, chiplet, split)"
"""

from __future__ import annotations
from typing import Protocol
import time

from mapping_lib import OpKey, MappingCandidate, OpEntry, SearchStats

# All 6 loop orders are valid; restrict to MNK by default because Week 1
# cost model doesn't differentiate them.
LOOP_ORDERS_ALL: tuple = ("MNK", "MKN", "NMK", "NKM", "KMN", "KNM")
LOOP_ORDERS_DEFAULT: tuple = ("MNK",)

# Include STREAM as feasibility fallback (always feasible — no SRAM staging)
RU_POLICIES: tuple = ("IRU", "WRU", "ORU", "ARU", "STREAM")


# ============================================================
# CostEvaluator protocol
# ============================================================

class CostEvaluator(Protocol):
    """Interface that cost evaluators must implement.

    Default implementation: cost_model.real_evaluator.Week1CostEvaluator.
    """

    def evaluate(self, op_key: OpKey, tile, ru, loop_order,
                 cores_split, chiplet_cfg) -> dict:
        """Returns dict with keys:
            latency_us, energy_uJ, sram_bytes,
            p_mpu_W, p_vpu_W, p_dram_W, p_noc_W, p_avg_W, p_peak_W,
            bw_utilization, mpu_utilization, sa_active_rows_frac,
            is_compute_bound, feasible, provenance

        chiplet_cfg is a Week 1 memory.ChipletConfig dataclass.
        """
        ...


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
    if max_count <= 1:
        return [sorted_d[0]]
    indices = sorted({
        int(round(i * (len(sorted_d) - 1) / (max_count - 1)))
        for i in range(max_count)
    })
    return [sorted_d[i] for i in indices]


def enumerate_core_splits(total_cores: int, M: int, N: int,
                          max_count: int = 8) -> list:
    """2D core splits (cM, cN, cK=1) where cM*cN ≤ total_cores and cM ≤ M, cN ≤ N.
    Mirrors Week 1's _split_MN logic when cM·cN = total_cores."""
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
    if ru == "STREAM":
        return True
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
    chiplet_cfg,                       # Week 1's ChipletConfig dataclass
    evaluator,                         # implements CostEvaluator
    loop_orders=LOOP_ORDERS_DEFAULT,   # only MNK by default (Week 1 limitation)
    ru_policies=RU_POLICIES,
    max_tiles_per_dim: int = 5,
    max_core_splits: int = 6,
    bytes_per_elem: int = 2,
    verbose: bool = False,
) -> OpEntry:
    """Exhaustive D³ search for one OpKey. Returns OpEntry with Pareto front."""
    from pareto import pareto_filter_3d

    M, N, K = op_key.M, op_key.N, op_key.K
    sram_budget = chiplet_cfg.SRAM_per_core_KB * 1024
    total_cores = chiplet_cfg.n_PE * chiplet_cfg.n_cores_per_PE

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

                            if not cost.get("feasible", True):
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
