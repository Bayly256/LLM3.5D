"""
Week 1 → Week 2 CostEvaluator adapter.

Wraps Week 1's compute.py / memory.py / operators.py primitives into the
`CostEvaluator` Protocol consumed by d3_search.py.

Architectural note (carry forward to error ledger):
  Week 1's cost model is per-(op, RU) at the chiplet level. It does NOT
  model tile-aware reuse: DRAM traffic is computed from operator_dram_bytes
  using the outer-loop assumption, and compute time depends only on the
  per-core (M, N, K) shape after the chiplet 2D split.

  Therefore, in this evaluator:
    - loop_order has NO effect on cost. Different loop orders return identical
      values. (D³ search restricts to MNK to avoid 6× redundant Pareto points.)
    - tile (T_M, T_N, T_K) affects ONLY SRAM feasibility, not latency/energy.
    - The Pareto front collapses mostly to "best feasible RU per (op, chiplet)".

  This is documented as a Week 1 limitation; tile-aware reuse is a Week 2.5
  task (or M5 cost-model upgrade).

Power decomposition:
  Week 1 returns total compute_energy_pJ (MAC only) and memory_energy_pJ
  (DRAM only). We split into per-component time-averaged rates:
    p_mpu_W = compute_energy_pJ / t_comp_ns × 1e-3        (during compute)
    p_dram_W = memory_energy_pJ / t_mem_ns × 1e-3         (during DRAM access)
    p_noc_W, p_vpu_W = fixed estimates                    (not modeled in Wk1)
    p_avg_W = (e_comp + e_mem) / latency_ns × 1e-3 + p_noc + p_vpu
    p_peak_W = max(p_mpu, p_dram) + p_noc + p_vpu

  This is good enough for the M2 thermal proxy. Calibrate in M4-M5.
"""

from __future__ import annotations
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from memory import (
    ChipletConfig,
    memory_access_latency_ns, memory_energy_pJ,
)
from compute import gemm_cycles, compute_energy_pJ
from operators import GemmShape, operator_dram_bytes, operator_sram_bytes_needed


# ============================================================
# 2D core split mirror of Week 1 chiplet_2d_split
# ============================================================

def per_core_shape(shape: GemmShape, total_cores: int,
                   cores_M: int, cores_N: int) -> GemmShape:
    """Compute per-core (M, N, K, heads) under a (cores_M, cores_N) split.

    Mirrors build_lut.chiplet_2d_split's logic for heads-first splitting, but
    allows the caller to specify (cores_M, cores_N) for within-head distribution.
    """
    if shape.n_heads > 1:
        if shape.n_heads >= total_cores:
            heads_per_core = math.ceil(shape.n_heads / total_cores)
            return GemmShape(
                M=shape.M, N=shape.N, K=shape.K,
                n_heads=heads_per_core, op_name=shape.op_name,
            )
        # Fewer heads than cores: distribute heads, then split inside head
        cores_per_head = max(1, total_cores // shape.n_heads)
        # Cap requested split to within-head budget
        cM = min(cores_M, shape.M, cores_per_head)
        cM = max(1, cM)
        cN = max(1, cores_per_head // cM)
        return GemmShape(
            M=math.ceil(shape.M / cM),
            N=math.ceil(shape.N / cN),
            K=shape.K, n_heads=1, op_name=shape.op_name,
        )

    # Linear op (n_heads == 1)
    cM = max(1, min(cores_M, shape.M))
    cN = max(1, min(cores_N, shape.N))
    if cM * cN > total_cores:
        # Renormalize while keeping aspect ratio
        scale = math.sqrt((cM * cN) / total_cores)
        cM = max(1, int(cM / scale))
        cN = max(1, total_cores // cM)
    return GemmShape(
        M=math.ceil(shape.M / cM),
        N=math.ceil(shape.N / cN),
        K=shape.K, n_heads=1, op_name=shape.op_name,
    )


# ============================================================
# DRAM bytes including STREAM
# ============================================================

def dram_bytes_for_ru(shape: GemmShape, RU: str, dtype_bytes: int = 2) -> float:
    """Like operator_dram_bytes but also handles STREAM (no SRAM staging)."""
    if RU == "STREAM":
        M, N, K = shape.M, shape.N, shape.K
        h = shape.n_heads
        return (M * K + K * N + M * N) * dtype_bytes * h
    return operator_dram_bytes(shape, RU, dtype_bytes)


# ============================================================
# Tile feasibility per RU
# ============================================================

def tile_feasible(T_M: int, T_N: int, T_K: int, RU: str,
                  sram_budget_bytes: int, dtype_bytes: int = 2) -> bool:
    """SRAM footprint check for one PE under tile (T_M, T_N, T_K) + RU."""
    if RU == "STREAM":
        return True
    if RU == "IRU":
        return T_M * T_K * dtype_bytes <= sram_budget_bytes
    if RU == "WRU":
        return T_N * T_K * dtype_bytes <= sram_budget_bytes
    if RU == "ORU":
        return T_M * T_N * dtype_bytes <= sram_budget_bytes
    if RU == "ARU":
        return (T_M * T_K + T_N * T_K + T_M * T_N) * dtype_bytes <= sram_budget_bytes
    return False


# ============================================================
# Main evaluator
# ============================================================

class Week1CostEvaluator:
    """Implements the CostEvaluator Protocol using Week 1 primitives.

    Limitations (see module docstring):
      - loop_order ignored
      - tile size affects SRAM feasibility only
      - power decomposition is rate-based (e/t per component)
      - p_noc, p_vpu are fixed estimates (Week 1 doesn't model them)
    """

    def __init__(
        self,
        temp_C: float = 65.0,
        p_noc_W_estimate: float = 10.0,
        p_vpu_W_estimate: float = 5.0,
        peak_to_avg_ratio: float = 1.5,
    ):
        self.temp_C = temp_C
        self.p_noc_W = p_noc_W_estimate
        self.p_vpu_W = p_vpu_W_estimate
        self.peak_to_avg = peak_to_avg_ratio

    def evaluate(self, op_key, tile, ru, loop_order, cores_split, chiplet_cfg) -> dict:
        T_M, T_N, T_K = tile
        cores_M, cores_N, _cores_K = cores_split
        cfg: ChipletConfig = chiplet_cfg
        total_cores = cfg.n_PE * cfg.n_cores_per_PE
        sram_budget = cfg.SRAM_per_core_KB * 1024
        dtype_bytes = 2  # fp16

        # Reconstruct the full GemmShape from OpKey
        full_shape = GemmShape(
            M=op_key.M, N=op_key.N, K=op_key.K,
            n_heads=op_key.n_heads, op_name=op_key.op_type,
        )

        # Per-core shape under requested (cores_M, cores_N) split
        pc_shape = per_core_shape(full_shape, total_cores, cores_M, cores_N)

        # SRAM feasibility (per PE) for this RU.
        # Week 1 cost model has no tile-aware reuse: operator_dram_bytes returns
        # bytes for the full operand based on RU. So the "real" SRAM requirement
        # is the per-core operand size under the RU's staging policy, NOT the
        # tile size. (Tile param is honored for forward-compat with M5
        # tile-aware reuse, but doesn't enter cost or feasibility in M2.)
        if ru == "STREAM":
            sram_needed = 0
            feasible = True
        else:
            sram_needed = operator_sram_bytes_needed(pc_shape, ru, dtype_bytes=dtype_bytes)
            feasible = sram_needed <= sram_budget

        # === Chiplet-level latency ===
        # Compute: single-core gemm cycles on per-core shape
        if pc_shape.n_heads > 1:
            cyc = pc_shape.n_heads * gemm_cycles(pc_shape.M, pc_shape.N, pc_shape.K, cfg)
        else:
            cyc = gemm_cycles(pc_shape.M, pc_shape.N, pc_shape.K, cfg)
        t_comp_ns = cyc / cfg.freq_MHz * 1000.0

        # Memory: aggregate chiplet DRAM bandwidth applied to total DRAM bytes
        total_dram_bytes = dram_bytes_for_ru(full_shape, ru, dtype_bytes)
        t_mem_ns = memory_access_latency_ns(total_dram_bytes, cfg, self.temp_C)

        latency_ns = max(t_mem_ns, t_comp_ns)
        is_compute_bound = t_comp_ns >= t_mem_ns

        # === Energy ===
        e_comp_pJ = compute_energy_pJ(full_shape, cfg)
        e_mem_pJ = memory_energy_pJ(total_dram_bytes, cfg, self.temp_C)
        # NoC + VPU energy added as rate × time (not in Week 1 LUT)
        e_noc_pJ = self.p_noc_W * latency_ns * 1e-3 * 1e9  # W·ns → pJ: W*s=J; ns·W*1e-9 = J; *1e12 = pJ
        # Simpler: pJ = W × ns × 1000  (since 1W = 1e9 nW = 1e12 pW; 1W·ns = 1e-9 J = 1e3 pJ)
        e_noc_pJ = self.p_noc_W * latency_ns * 1e3
        e_vpu_pJ = self.p_vpu_W * latency_ns * 1e3

        total_energy_pJ = e_comp_pJ + e_mem_pJ + e_noc_pJ + e_vpu_pJ

        # === Power decomposition (W) ===
        # p_X_W = e_X_pJ / latency_ns × 1e-3
        if latency_ns > 0:
            p_mpu_W = e_comp_pJ / latency_ns * 1e-3
            p_dram_W = e_mem_pJ / latency_ns * 1e-3
        else:
            p_mpu_W = p_dram_W = 0.0
        p_noc_W = self.p_noc_W
        p_vpu_W = self.p_vpu_W
        p_avg_W = p_mpu_W + p_dram_W + p_noc_W + p_vpu_W
        p_peak_W = (max(p_mpu_W, p_dram_W) * self.peak_to_avg + p_noc_W + p_vpu_W)

        # === Utilization / regime ===
        bw_Bps = cfg.peak_bw_TBs * 1e12
        bw_util = (total_dram_bytes / (bw_Bps * latency_ns * 1e-9)
                   if latency_ns > 0 and bw_Bps > 0 else 0.0)
        bw_util = min(1.0, bw_util)
        # MPU peak in MACs/ns = SA_rows × SA_cols × freq_MHz / 1000
        peak_macs_per_ns = cfg.SA_rows * cfg.SA_cols * cfg.freq_MHz / 1000.0
        # Achieved MACs/ns aggregated over all cores (parallel)
        total_macs = full_shape.M * full_shape.N * full_shape.K * full_shape.n_heads
        achieved_per_core = total_macs / (latency_ns * total_cores) if latency_ns > 0 else 0
        mpu_util = min(1.0, achieved_per_core / peak_macs_per_ns) if peak_macs_per_ns > 0 else 0.0
        # SA active rows: per-core M vs SA_rows (GEMV indicator)
        sa_active = min(pc_shape.M, cfg.SA_rows) / cfg.SA_rows if cfg.SA_rows > 0 else 0

        return {
            "latency_us": latency_ns / 1000.0,
            "energy_uJ": total_energy_pJ / 1e6,
            "sram_bytes": sram_needed,
            "p_mpu_W": p_mpu_W, "p_vpu_W": p_vpu_W,
            "p_dram_W": p_dram_W, "p_noc_W": p_noc_W,
            "p_avg_W": p_avg_W, "p_peak_W": p_peak_W,
            "bw_utilization": bw_util,
            "mpu_utilization": mpu_util,
            "sa_active_rows_frac": sa_active,
            "is_compute_bound": is_compute_bound,
            "feasible": feasible,
            "provenance": {
                "latency_us": ["sim:M2"],
                "energy_uJ": ["sim:M2"],
                "p_mpu_W": ["sim:M2"],
                "p_dram_W": ["sim:M2"],
                "p_noc_W": ["est:M9"],  # not really computed
                "p_vpu_W": ["est:M9"],
            },
        }


def _tile_sram(T_M: int, T_N: int, T_K: int, RU: str, dtype_bytes: int = 2) -> int:
    if RU == "IRU":
        return T_M * T_K * dtype_bytes
    if RU == "WRU":
        return T_N * T_K * dtype_bytes
    if RU == "ORU":
        return T_M * T_N * dtype_bytes
    if RU == "ARU":
        return (T_M * T_K + T_N * T_K + T_M * T_N) * dtype_bytes
    return 0
