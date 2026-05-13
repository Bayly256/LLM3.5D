"""
Systolic array compute cost model for LaMoSys3.5D simulator.

Models the partitioned MPU design from LaMoSys3.5D §IV-D:
    - Full SA mode: SA_rows × SA_cols cells, used when M >= SA_rows.
    - baseSA mode: each baseSA is baseSA_rows × SA_cols, used when M < SA_rows.
      Up to (SA_rows / baseSA_rows) baseSAs run in parallel, splitting the K
      dimension among them — recovers GEMV/small-batch throughput lost on the
      full SA.

Headline behaviors this model reproduces:
    - M = 1 GEMV on full SA: utilization ≈ 1/SA_rows (very poor)
    - M = 1 GEMV with baseSAs K-split: throughput recovered up to
      (SA_rows / baseSA_rows)×, matching LaMoSys3.5D Fig. 13(a) trends.

Author: <you>
References:
    - LaMoSys3.5D, arXiv:2512.08731, §IV-D and §VI-D
    - ScaleSim, https://github.com/scalesim-project/scale-sim-v2
"""
import math
from typing import Optional, Tuple

from memory import ChipletConfig, PC_CONFIG, DC_CONFIG
from operators import GemmShape


# ---------------------------------------------------------------------------
# Core: cycle count on a single SA of arbitrary shape
# ---------------------------------------------------------------------------

def sa_compute_cycles(M: int, N: int, K: int,
                      sa_rows: int, sa_cols: int) -> int:
    """Cycles for one GEMM(M, N, K) on a single `sa_rows × sa_cols` SA.

    Weight-stationary dataflow. Effective MACs/cycle once pipeline is full:
        rows_used × cols_used  ≤  sa_rows × sa_cols
    where the slack from `M < sa_rows` or `N < sa_cols` is wasted cells.

    Plus a one-time pipeline fill of (sa_rows + sa_cols - 2) cycles.

    This formula reproduces:
      - Full utilization when M >= sa_rows and N >= sa_cols
      - 1/SA_rows utilization for GEMV (M=1) on a full SA → the GEMV problem
        that motivates the baseSA design.
    """
    if M <= 0 or N <= 0 or K <= 0:
        return 0
    rows_used = min(M, sa_rows)
    cols_used = min(N, sa_cols)
    eff_throughput = rows_used * cols_used  # MACs/cycle
    total_macs = M * N * K
    ideal = math.ceil(total_macs / eff_throughput)
    pipeline = sa_rows + sa_cols - 2
    return ideal + pipeline


# ---------------------------------------------------------------------------
# MPU mode selection (full SA vs baseSA + K-split parallelism)
# ---------------------------------------------------------------------------

def select_mpu_mode(M: int, cfg: ChipletConfig) -> Tuple[str, int]:
    """Decide MPU mode for a given problem M.

    Returns (mode, n_parallel):
        mode = 'full'   → use full SA (sa_rows × sa_cols), n_parallel = 1
        mode = 'baseSA' → use baseSAs in parallel splitting K,
                          n_parallel = SA_rows // baseSA_rows

    Rule of thumb: when M >= SA_rows the full SA is well utilized and uses
    fewer K-tiles than the baseSA mode would. When M < SA_rows the full SA
    wastes most rows, so we switch to baseSA + K-split.
    """
    if M >= cfg.SA_rows:
        return 'full', 1
    n_baseSA = max(1, cfg.SA_rows // cfg.baseSA_rows)
    return 'baseSA', n_baseSA


def gemm_cycles(M: int, N: int, K: int, cfg: ChipletConfig,
                force_mode: Optional[str] = None) -> int:
    """Cycle count for one core's MPU to compute GEMM(M, N, K).

    `force_mode` ∈ {None, 'full', 'baseSA'} lets you override the auto-
    selection for benchmarking / sweeps.
    """
    if force_mode is not None:
        mode = force_mode
        n_par = max(1, cfg.SA_rows // cfg.baseSA_rows) if mode == 'baseSA' else 1
    else:
        mode, n_par = select_mpu_mode(M, cfg)

    if mode == 'full':
        return sa_compute_cycles(M, N, K, cfg.SA_rows, cfg.SA_cols)

    # baseSA mode with K-split: each baseSA handles K/n_par of K dim.
    K_per_baseSA = math.ceil(K / n_par)
    per_baseSA = sa_compute_cycles(M, N, K_per_baseSA,
                                    cfg.baseSA_rows, cfg.SA_cols)
    # Cross-baseSA reduction in the VPU (n_par - 1 adds per output element)
    reduction_ops = max(0, (n_par - 1) * M * N)
    reduction_cycles = math.ceil(reduction_ops / cfg.SA_cols) if reduction_ops else 0
    return per_baseSA + reduction_cycles


# ---------------------------------------------------------------------------
# Multi-core scaling within one PE
# ---------------------------------------------------------------------------

def compute_latency_ns(shape: GemmShape, cfg: ChipletConfig,
                        n_cores_used: Optional[int] = None) -> float:
    """Latency to compute `shape` on `n_cores_used` cores of a single PE.

    Distribution:
      - Linear GEMM (n_heads == 1): split M across cores.
      - Batched per-head (n_heads > 1): one head per core when possible.
        (Falls back gracefully when fewer cores than heads.)
    """
    if n_cores_used is None:
        n_cores_used = cfg.n_cores_per_PE
    n_cores_used = max(1, n_cores_used)

    if shape.n_heads > 1:
        # Distribute heads across cores; ceil ensures we cover all heads.
        heads_per_core = math.ceil(shape.n_heads / n_cores_used)
        cyc = heads_per_core * gemm_cycles(shape.M, shape.N, shape.K, cfg)
    else:
        # Split M dimension; each core handles ceil(M / n_cores) rows.
        M_per_core = math.ceil(shape.M / n_cores_used)
        cyc = gemm_cycles(M_per_core, shape.N, shape.K, cfg)

    return cyc / cfg.freq_MHz * 1000.0  # cycles / MHz → ns


def compute_energy_pJ(shape: GemmShape, cfg: ChipletConfig) -> float:
    """MAC energy only. SRAM/DRAM access energy is accounted for elsewhere."""
    n_macs = shape.flops / 2  # GemmShape.flops counts 2 ops per MAC
    return n_macs * cfg.mac_energy_pJ_op


# ---------------------------------------------------------------------------
# Peak / utilization helpers
# ---------------------------------------------------------------------------

def single_core_peak_TFLOPS(cfg: ChipletConfig) -> float:
    """Peak TFLOPS per core = SA_rows × SA_cols × 2 × freq."""
    return (cfg.SA_rows * cfg.SA_cols * 2 * cfg.freq_MHz * 1e6) / 1e12


def per_chiplet_peak_TFLOPS(cfg: ChipletConfig) -> float:
    """Aggregate peak across all PEs × all cores."""
    return single_core_peak_TFLOPS(cfg) * cfg.n_PE * cfg.n_cores_per_PE


def utilization(shape: GemmShape, cfg: ChipletConfig,
                n_cores_used: Optional[int] = None) -> float:
    """Achieved fraction of (n_cores × single_core_peak)."""
    if n_cores_used is None:
        n_cores_used = cfg.n_cores_per_PE
    t_ns = compute_latency_ns(shape, cfg, n_cores_used)
    achieved_TFLOPS = shape.flops / (t_ns * 1e-9) / 1e12
    peak_TFLOPS = single_core_peak_TFLOPS(cfg) * n_cores_used
    return achieved_TFLOPS / peak_TFLOPS if peak_TFLOPS > 0 else 0.0


# ---------------------------------------------------------------------------
# Self-tests (run `python compute.py`)
# ---------------------------------------------------------------------------

def _run_self_test():
    print("=" * 70)
    print("LaMoSys3.5D compute cost model — sanity checks")
    print("=" * 70)

    # --- Chiplet peak sanity ---
    for cfg in (PC_CONFIG, DC_CONFIG):
        n_baseSA = cfg.SA_rows // cfg.baseSA_rows
        peak_core = single_core_peak_TFLOPS(cfg)
        peak_chip = per_chiplet_peak_TFLOPS(cfg)
        ratio = peak_chip / cfg.peak_TFLOPS
        print(f"\n[{cfg.name}] SA={cfg.SA_rows}×{cfg.SA_cols}, "
              f"baseSA_rows={cfg.baseSA_rows}, #baseSA per MPU = {n_baseSA}")
        print(f"  per-core peak: {peak_core:6.2f} TFLOPS")
        print(f"  per-chip peak (computed from SA): {peak_chip:7.1f} TFLOPS")
        print(f"  per-chip peak (paper target):     {cfg.peak_TFLOPS:7.1f} TFLOPS   "
              f"(computed/paper = {ratio:.2f}×)")
        if abs(ratio - 1.0) > 0.3:
            print(f"  ⚠  off by >30%. To match paper exactly, try SA={int(cfg.SA_rows/math.sqrt(ratio))}"
                  f"×{int(cfg.SA_cols/math.sqrt(ratio))}.")

    PC = PC_CONFIG

    # --- GEMM sweep over M: shows full SA → baseSA transition ---
    print(f"\n--- GEMM cycle sweep vs M  (PC, single core, N=K=1024) ---")
    print(f"{'M':>5s} {'mode':>10s} {'cycles':>12s} {'TFLOPS':>9s} {'util%':>7s}")
    print("-" * 50)
    for M in (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024):
        mode, n_par = select_mpu_mode(M, PC)
        mode_str = "full" if n_par == 1 else f"baseSA×{n_par}"
        cyc = gemm_cycles(M, 1024, 1024, PC)
        t_ns = cyc / PC.freq_MHz * 1000.0
        tflops = 2 * M * 1024 * 1024 / (t_ns * 1e-9) / 1e12
        util = tflops / single_core_peak_TFLOPS(PC) * 100
        print(f"{M:>5d} {mode_str:>10s} {cyc:>12d} {tflops:>9.2f} {util:>6.1f}%")

    # --- Forced comparison: full SA vs baseSA mode ---
    print(f"\n--- baseSA speedup over full SA  (PC, N=K=1024) ---")
    n_baseSA = PC.SA_rows // PC.baseSA_rows
    print(f"  Forced comparison: full SA vs {n_baseSA}× parallel baseSAs (K-split)")
    print(f"{'M':>5s} {'fullSA cyc':>12s} {'baseSA cyc':>12s} {'speedup':>9s}")
    print("-" * 45)
    for M in (1, 2, 4, 8, 16, 32, 64, 128):
        full = gemm_cycles(M, 1024, 1024, PC, force_mode='full')
        base = gemm_cycles(M, 1024, 1024, PC, force_mode='baseSA')
        speedup = full / base if base > 0 else 0
        print(f"{M:>5d} {full:>12d} {base:>12d} {speedup:>8.2f}×")

    # --- Operator-level latency: prefill vs decode on PC and DC ---
    from operators import GPT3_13B, operator_shapes
    print(f"\n--- GPT3-13B operator latency per PE ---")
    for phase, bs, sl in [('prefill', 4, 1024), ('decode', 4, 1024)]:
        print(f"\n  {phase}, bs={bs}, seq={sl}:")
        shapes = operator_shapes(GPT3_13B, bs, sl, phase)
        print(f"    {'op':12s} {'PC t(μs)':>10s} {'DC t(μs)':>10s} "
              f"{'PC util':>9s} {'DC util':>9s}")
        for name, sh in shapes.items():
            t_pc = compute_latency_ns(sh, PC_CONFIG) / 1000
            t_dc = compute_latency_ns(sh, DC_CONFIG) / 1000
            u_pc = utilization(sh, PC_CONFIG) * 100
            u_dc = utilization(sh, DC_CONFIG) * 100
            print(f"    {name:12s} {t_pc:>10.2f} {t_dc:>10.2f} "
                  f"{u_pc:>8.1f}% {u_dc:>8.1f}%")

    # --- Total layer latencies under several workloads ---
    print(f"\n--- Total per-layer compute time on PC (single PE) ---")
    for phase, bs, sl in [('prefill', 4, 1024), ('prefill', 4, 4096),
                           ('decode', 4, 1024), ('decode', 16, 1024)]:
        shapes = operator_shapes(GPT3_13B, bs, sl, phase)
        total_ns = sum(compute_latency_ns(s, PC_CONFIG) for s in shapes.values())
        total_flops = sum(s.flops for s in shapes.values())
        print(f"  {phase:7s} bs={bs:2d} seq={sl:5d}: "
              f"{total_ns/1000:8.1f} μs   ({total_flops/1e9:7.1f} GFLOPs)")


if __name__ == "__main__":
    _run_self_test()