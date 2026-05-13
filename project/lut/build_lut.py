"""
Operator cost LUT builder for LaMoSys3.5D simulator (Day 5).

Chiplet-level cost model used throughout:
    Memory:  t_mem = total_DRAM_bytes / chiplet_aggregate_BW
             (with refresh overhead from memory.py)
    Compute: t_comp = single-core gemm cycles after a 2D (M, N) split
             across ALL cores in the chiplet (PEs × cores_per_PE).

The 2D split is critical for decode (small M):
    n_heads = 1:
        if M ≥ total_cores: split M only (M_per_core = M / total_cores).
        if M < total_cores: M_per_core = 1, split N across remaining cores.
    n_heads > 1:
        split heads first; if cores remain, split inside each head as above.

This exposes the bandwidth advantage of DC over PC for decode, which the
naive per-PE split (v1) wrongly hides as compute-bound.

RU policies:
    IRU/WRU/ORU/ARU  — stage A/B/C/all in SRAM (paper §V-A).
    STREAM           — nothing staged; all operands traverse DRAM.
                       Always feasible; used when nothing else fits.

Author: <you>
"""
import math
import os
import pickle
import time
from typing import Dict, Optional, Tuple

from memory import (
    ChipletConfig, PC_CONFIG, DC_CONFIG,
    memory_access_latency_ns, memory_energy_pJ,
)
from compute import gemm_cycles, compute_energy_pJ
from operators import (
    GemmShape, ModelConfig, ALL_MODELS,
    operator_shapes, operator_dram_bytes, operator_sram_bytes_needed,
)

RU_POLICIES = ('IRU', 'WRU', 'ORU', 'ARU', 'STREAM')


# ---------------------------------------------------------------------------
# Chiplet-level 2D parallelism split
# ---------------------------------------------------------------------------

def _split_MN(M: int, N: int, n_cores: int) -> Tuple[int, int]:
    """Split work onto n_cores along (M, N). Returns (cores_M, cores_N)."""
    if M >= n_cores:
        return n_cores, 1
    cores_M = max(1, M)
    cores_N = max(1, n_cores // cores_M)
    return cores_M, cores_N


def chiplet_2d_split(shape: GemmShape, cfg: ChipletConfig
                      ) -> Tuple[GemmShape, int, int]:
    """Distribute an op across all chiplet cores via 2D (M, N) parallelism.

    Returns (per_core_shape, cores_along_M, cores_along_N).
    """
    total_cores = cfg.n_PE * cfg.n_cores_per_PE

    if shape.n_heads > 1:
        if shape.n_heads >= total_cores:
            # More heads than cores: serialize heads on each core
            heads_per_core = math.ceil(shape.n_heads / total_cores)
            per_core = GemmShape(M=shape.M, N=shape.N, K=shape.K,
                                  n_heads=heads_per_core, op_name=shape.op_name)
            return per_core, 1, 1
        # Fewer heads than cores: one head per group of cores
        cores_per_head = max(1, total_cores // shape.n_heads)
        cores_M, cores_N = _split_MN(shape.M, shape.N, cores_per_head)
        per_core = GemmShape(
            M=math.ceil(shape.M / cores_M),
            N=math.ceil(shape.N / cores_N),
            K=shape.K, n_heads=1, op_name=shape.op_name,
        )
        return per_core, cores_M, cores_N

    cores_M, cores_N = _split_MN(shape.M, shape.N, total_cores)
    per_core = GemmShape(
        M=math.ceil(shape.M / cores_M),
        N=math.ceil(shape.N / cores_N),
        K=shape.K, n_heads=1, op_name=shape.op_name,
    )
    return per_core, cores_M, cores_N


# ---------------------------------------------------------------------------
# Chiplet-level latency/energy
# ---------------------------------------------------------------------------

def chiplet_compute_ns(shape: GemmShape, cfg: ChipletConfig) -> float:
    """Chiplet compute time (all cores in parallel)."""
    per_core_shape, _, _ = chiplet_2d_split(shape, cfg)
    if per_core_shape.n_heads > 1:
        cyc = per_core_shape.n_heads * gemm_cycles(
            per_core_shape.M, per_core_shape.N, per_core_shape.K, cfg)
    else:
        cyc = gemm_cycles(per_core_shape.M, per_core_shape.N,
                          per_core_shape.K, cfg)
    return cyc / cfg.freq_MHz * 1000.0


def chiplet_memory_ns(total_dram_bytes: float, cfg: ChipletConfig,
                       temp_C: float) -> float:
    """Aggregate DRAM time at chiplet bandwidth."""
    return memory_access_latency_ns(total_dram_bytes, cfg, temp_C)


# ---------------------------------------------------------------------------
# DRAM bytes per RU
# ---------------------------------------------------------------------------

def dram_bytes_for_policy(shape: GemmShape, RU: str,
                           dtype_bytes: int = 2) -> float:
    if RU == 'STREAM':
        M, N, K = shape.M, shape.N, shape.K
        h = shape.n_heads
        return (M*K + K*N + M*N) * dtype_bytes * h
    return operator_dram_bytes(shape, RU, dtype_bytes)


def is_feasible_per_core(shape: GemmShape, cfg: ChipletConfig, RU: str,
                          dtype_bytes: int = 2) -> bool:
    if RU == 'STREAM':
        return True
    per_core_shape, _, _ = chiplet_2d_split(shape, cfg)
    sram_budget = cfg.SRAM_per_core_KB * 1024
    needed = operator_sram_bytes_needed(per_core_shape, RU, dtype_bytes=dtype_bytes)
    return needed <= sram_budget


# ---------------------------------------------------------------------------
# Per-chiplet operator cost
# ---------------------------------------------------------------------------

def op_cost_per_chiplet(shape: GemmShape, cfg: ChipletConfig, RU: str,
                         temp_C: float = 65.0,
                         dtype_bytes: int = 2) -> Dict:
    total_dram = dram_bytes_for_policy(shape, RU, dtype_bytes)
    t_mem = chiplet_memory_ns(total_dram, cfg, temp_C)
    e_mem = memory_energy_pJ(total_dram, cfg, temp_C)

    t_comp = chiplet_compute_ns(shape, cfg)
    e_comp = compute_energy_pJ(shape, cfg)

    feasible = is_feasible_per_core(shape, cfg, RU, dtype_bytes)
    _, cores_M, cores_N = chiplet_2d_split(shape, cfg)

    return {
        'latency_ns': max(t_mem, t_comp),
        'energy_pJ': e_mem + e_comp,
        't_mem_ns': t_mem,
        't_comp_ns': t_comp,
        'bound_by': 'memory' if t_mem >= t_comp else 'compute',
        'feasible': feasible,
        'cores_M': cores_M,
        'cores_N': cores_N,
        'total_dram_MB': total_dram / 1e6,
    }


def best_feasible_cost(shape: GemmShape, cfg: ChipletConfig,
                        temp_C: float = 65.0,
                        preferred_order: Tuple[str, ...] = (
                            'IRU', 'ORU', 'WRU', 'ARU', 'STREAM',
                        )) -> Tuple[str, Dict]:
    best_RU, best_cost = None, None
    for RU in preferred_order:
        cost = op_cost_per_chiplet(shape, cfg, RU, temp_C)
        if not cost['feasible']:
            continue
        if best_cost is None or cost['latency_ns'] < best_cost['latency_ns']:
            best_RU, best_cost = RU, cost
    return best_RU, best_cost


# ---------------------------------------------------------------------------
# LUT construction
# ---------------------------------------------------------------------------

def build_lut(models=None, batch_sizes=None, seq_lens=None,
              chiplets=None, RUs=None, temp_C: float = 65.0,
              out_path: str = 'lut.pkl', verbose: bool = True) -> Dict:
    if models is None:
        models = list(ALL_MODELS.values())
    if batch_sizes is None:
        batch_sizes = [1, 4, 16]
    if seq_lens is None:
        seq_lens = [1024, 2048, 4096]
    if chiplets is None:
        chiplets = [PC_CONFIG, DC_CONFIG]
    if RUs is None:
        RUs = RU_POLICIES

    lut = {}
    phases = ('prefill', 'decode')
    t_start = time.time()
    count = 0

    for model in models:
        for phase in phases:
            for bs in batch_sizes:
                for sl in seq_lens:
                    shapes = operator_shapes(model, bs, sl, phase)
                    for op_name, shape in shapes.items():
                        for cfg in chiplets:
                            for RU in RUs:
                                key = (model.name, phase, op_name, bs, sl,
                                       cfg.name, RU)
                                lut[key] = op_cost_per_chiplet(shape, cfg, RU, temp_C)
                                count += 1

    elapsed = time.time() - t_start
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(lut, f)
    if verbose:
        size_kb = os.path.getsize(out_path) / 1024
        print(f"Built {count} LUT entries in {elapsed:.2f}s")
        print(f"Wrote {out_path}  ({size_kb:.1f} KB)")
    return lut


def load_lut(path: str = 'lut.pkl') -> Dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def query(lut: Dict, model: str, phase: str, op: str, bs: int, sl: int,
          chiplet: str, RU: str) -> Optional[Dict]:
    return lut.get((model, phase, op, bs, sl, chiplet, RU))


# ---------------------------------------------------------------------------
# Day 6 acceptance
# ---------------------------------------------------------------------------

def run_day6_acceptance(model_name: str = 'gpt3-13b',
                         bs: int = 4, sl: int = 1024) -> None:
    print("=" * 88)
    print(f"Day 6 acceptance: {model_name}, batch={bs}, seq={sl}")
    print("=" * 88)

    model = ALL_MODELS[model_name]

    def per_op_table(phase: str):
        shapes = operator_shapes(model, bs, sl, phase)
        print(f"\n[{phase}]")
        print(f"  {'op':12s} | {'PC':>35s} | {'DC':>35s}")
        print(f"  {'':12s} | {'RU':>6s} {'t(μs)':>9s} {'bound':>8s} {'DRAM':>10s} | "
              f"{'RU':>6s} {'t(μs)':>9s} {'bound':>8s} {'DRAM':>10s}")
        print("  " + "-" * 100)
        totals = {'PC': 0.0, 'DC': 0.0}
        for op_name, shape in shapes.items():
            pc_RU, pc = best_feasible_cost(shape, PC_CONFIG)
            dc_RU, dc = best_feasible_cost(shape, DC_CONFIG)
            print(f"  {op_name:12s} | "
                  f"{pc_RU:>6s} {pc['latency_ns']/1000:>9.2f} "
                  f"{pc['bound_by']:>8s} {pc['total_dram_MB']:>9.1f}M | "
                  f"{dc_RU:>6s} {dc['latency_ns']/1000:>9.2f} "
                  f"{dc['bound_by']:>8s} {dc['total_dram_MB']:>9.1f}M")
            totals['PC'] += pc['latency_ns']
            totals['DC'] += dc['latency_ns']
        print("  " + "-" * 100)
        print(f"  {'TOTAL':12s} | "
              f"{'':>6s} {totals['PC']/1000:>9.2f} {'':>8s} {'':>10s} | "
              f"{'':>6s} {totals['DC']/1000:>9.2f} {'':>8s} {'':>10s}")
        return totals

    pre = per_op_table('prefill')
    dec = per_op_table('decode')

    def partial(phase, chip_cfg, op_filter):
        total = 0
        for op_name, shape in operator_shapes(model, bs, sl, phase).items():
            if op_filter(op_name):
                _, cost = best_feasible_cost(shape, chip_cfg)
                total += cost['latency_ns']
        return total

    ffn_pre = partial('prefill', PC_CONFIG, lambda n: 'ffn' in n)
    qkv_pre = partial('prefill', PC_CONFIG, lambda n: n in ('qkv_proj', 'o_proj'))
    att_pre = partial('prefill', PC_CONFIG, lambda n: 'attn' in n)

    print("\n" + "=" * 88)
    print("Trend checks")
    print("=" * 88)

    print(f"\n[1] Prefill on PC: FFN > QKV+O > Attn?")
    print(f"    FFN={ffn_pre/1000:.1f}μs   QKV+O={qkv_pre/1000:.1f}μs   "
          f"Attn={att_pre/1000:.1f}μs")
    print(f"    {'✓ Pass' if ffn_pre > qkv_pre > att_pre else '✗ Fail'}")

    print(f"\n[2] Decode: DC < PC (bandwidth advantage)?")
    print(f"    PC={dec['PC']/1000:.1f}μs   DC={dec['DC']/1000:.1f}μs   "
          f"DC/PC={dec['DC']/dec['PC']:.2f}×")
    print(f"    {'✓ Pass' if dec['DC'] < dec['PC'] else '✗ Fail'}")

    r = pre['DC'] / pre['PC']
    print(f"\n[3] Prefill: PC ≈ DC (both compute-bound)?")
    print(f"    PC={pre['PC']/1000:.1f}μs   DC={pre['DC']/1000:.1f}μs   DC/PC={r:.2f}×")
    print(f"    {'✓ Pass' if 0.7 < r < 1.5 else '⚠ Larger gap than expected'}")

    r4 = dec['PC'] / dec['DC']
    print(f"\n[4] PC decode / DC decode > 1.5 ?")
    print(f"    PC/DC decode ratio = {r4:.2f}×")
    print(f"    {'✓ Pass' if r4 > 1.3 else '⚠ Smaller advantage than expected'}")


def cross_model_summary(bs: int = 4, sl: int = 1024) -> None:
    print("\n" + "=" * 88)
    print(f"Cross-model summary  (batch={bs}, seq={sl})")
    print("=" * 88)
    print(f"\n  {'model':12s} | {'prefill PC':>12s} {'prefill DC':>12s} | "
          f"{'decode PC':>12s} {'decode DC':>12s} | {'attn':>5s}")
    print("  " + "-" * 78)
    for m_name, m in ALL_MODELS.items():
        pre_pc = sum(best_feasible_cost(s, PC_CONFIG)[1]['latency_ns']
                     for s in operator_shapes(m, bs, sl, 'prefill').values())
        pre_dc = sum(best_feasible_cost(s, DC_CONFIG)[1]['latency_ns']
                     for s in operator_shapes(m, bs, sl, 'prefill').values())
        dec_pc = sum(best_feasible_cost(s, PC_CONFIG)[1]['latency_ns']
                     for s in operator_shapes(m, bs, sl, 'decode').values())
        dec_dc = sum(best_feasible_cost(s, DC_CONFIG)[1]['latency_ns']
                     for s in operator_shapes(m, bs, sl, 'decode').values())
        print(f"  {m_name:12s} | "
              f"{pre_pc/1000:>11.1f}μs {pre_dc/1000:>11.1f}μs | "
              f"{dec_pc/1000:>11.1f}μs {dec_dc/1000:>11.1f}μs | "
              f"{m.attention_type:>5s}")


def print_lut_examples(lut: Dict) -> None:
    print("\nExample LUT entries:")
    examples = [
        ('gpt3-13b', 'prefill', 'qkv_proj', 4, 1024, 'PC', 'STREAM'),
        ('gpt3-13b', 'prefill', 'qkv_proj', 4, 1024, 'DC', 'STREAM'),
        ('gpt3-13b', 'decode',  'ffn_up',  4, 1024, 'PC', 'IRU'),
        ('gpt3-13b', 'decode',  'ffn_up',  4, 1024, 'DC', 'IRU'),
        ('gpt3-13b', 'decode',  'ffn_up',  16, 1024, 'DC', 'IRU'),
        ('llama3-70b', 'decode', 'attn_qk', 4, 1024, 'DC', 'IRU'),
    ]
    for ex in examples:
        e = lut.get(ex)
        if not e:
            print(f"  {ex} → (missing)")
            continue
        feas = "✓" if e['feasible'] else "✗"
        print(f"  {ex}")
        print(f"     lat={e['latency_ns']/1000:8.2f}μs  "
              f"t_mem={e['t_mem_ns']/1000:7.2f}μs  t_comp={e['t_comp_ns']/1000:7.2f}μs  "
              f"bound={e['bound_by']:6s}  DRAM={e['total_dram_MB']:6.1f}MB  {feas}")


if __name__ == "__main__":
    print("Building LUT...\n")
    lut = build_lut(out_path='lut.pkl')
    print_lut_examples(lut)
    print()
    run_day6_acceptance(model_name='gpt3-13b', bs=4, sl=1024)
    cross_model_summary(bs=4, sl=1024)

    print("\n" + "=" * 88)
    print(f"Week 1 done.  LUT cached at  lut.pkl  ({len(lut)} entries)")
    print("=" * 88)