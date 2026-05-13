"""
LLM operator shape modeling for LaMoSys3.5D simulator.

Provides:
    - ModelConfig: hidden, heads, FFN dims, attention/FFN type
    - GemmShape: M, N, K, n_heads, op_name; with .flops property
    - operator_shapes(): per-layer GEMM shapes for prefill or decode
    - operator_dram_bytes(): DRAM bytes per RU policy {IRU, WRU, ORU, ARU}
    - operator_sram_bytes_needed(): SRAM footprint of staged tile
    - kv_cache_size_bytes(), model_param_bytes(): capacity sanity checks

Models covered (LaMoSys3.5D §VI-A):
    GPT3-13B (MHA, dense FFN)
    QwQ-32B  (GQA, SwiGLU)
    LLaMA3-70B (GQA, SwiGLU)

Author: <you>
References:
    - LaMoSys3.5D, arXiv:2512.08731, §III + §VI
    - GPT-3 paper for 13B params; LLaMA-3 paper for 70B params
"""
from dataclasses import dataclass
from typing import Dict


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str
    n_layers: int          # transformer layers
    hidden: int            # H — model dim
    n_heads: int           # query heads
    d_head: int            # per-head dim
    n_kv_heads: int        # = n_heads for MHA, smaller for GQA
    ffn_intermediate: int  # F — FFN intermediate dim
    vocab: int = 50000
    attention_type: str = 'MHA'   # 'MHA' | 'GQA' | 'MQA'
    ffn_type: str = 'dense'       # 'dense' | 'GLU'


GPT3_13B = ModelConfig(
    name='gpt3-13b',
    n_layers=40, hidden=5120,
    n_heads=40, d_head=128, n_kv_heads=40,
    ffn_intermediate=20480,
    attention_type='MHA', ffn_type='dense',
)

QWQ_32B = ModelConfig(
    name='qwq-32b',
    n_layers=64, hidden=5120,
    n_heads=40, d_head=128, n_kv_heads=8,
    ffn_intermediate=27648,
    attention_type='GQA', ffn_type='GLU',
)

LLAMA3_70B = ModelConfig(
    name='llama3-70b',
    n_layers=80, hidden=8192,
    n_heads=64, d_head=128, n_kv_heads=8,
    ffn_intermediate=28672,
    attention_type='GQA', ffn_type='GLU',
)

ALL_MODELS = {m.name: m for m in (GPT3_13B, QWQ_32B, LLAMA3_70B)}


# ---------------------------------------------------------------------------
# Shape representation
# ---------------------------------------------------------------------------

@dataclass
class GemmShape:
    """A (batched) matrix multiplication: C[M,N] += A[M,K] @ B[K,N].

    n_heads is the batch dim for per-head attention GEMMs (=1 for linear layers).
    """
    M: int
    N: int
    K: int
    op_name: str
    n_heads: int = 1

    @property
    def flops(self) -> int:
        return 2 * self.M * self.N * self.K * self.n_heads

    @property
    def is_gemv(self) -> bool:
        """True if M is small enough that the op degenerates to a vector op."""
        return self.M <= 4

    def __repr__(self) -> str:
        head_str = f" (×{self.n_heads}h)" if self.n_heads > 1 else ""
        return (f"GemmShape[{self.op_name}: M={self.M}, N={self.N}, "
                f"K={self.K}{head_str}, {self.flops/1e9:.2f} GFLOPs]")


# ---------------------------------------------------------------------------
# Operator shape generation
# ---------------------------------------------------------------------------

def operator_shapes(model: ModelConfig, batch: int, seq: int,
                    phase: str = 'prefill') -> Dict[str, GemmShape]:
    """Per-layer GEMM shapes for one transformer layer.

    phase='prefill': processes all `seq` tokens at once.
                     M = batch * seq for linear layers.
    phase='decode':  processes 1 new token per request.
                     M = batch.

    Returns a dict mapping operator name → GemmShape.
    """
    H = model.hidden
    h_q, h_kv = model.n_heads, model.n_kv_heads
    d = model.d_head
    F = model.ffn_intermediate

    if phase == 'prefill':
        M_lin = batch * seq
    elif phase == 'decode':
        M_lin = batch
    else:
        raise ValueError(f"Unknown phase: {phase!r}, expected 'prefill' or 'decode'")

    # QKV fused projection: input H → (Q | K | V) of total (h_q + 2*h_kv)*d
    # For MHA: h_q == h_kv, so output = 3*H. For GQA: smaller.
    qkv_out = (h_q + 2 * h_kv) * d
    q_out = h_q * d   # matches H for standard MHA

    shapes = {
        'qkv_proj': GemmShape(M=M_lin, N=qkv_out, K=H, op_name='qkv_proj'),
        'o_proj':   GemmShape(M=M_lin, N=H,        K=q_out, op_name='o_proj'),
    }

    # FFN: dense (2 matmuls) or GLU (3 matmuls: gate, up, down)
    if model.ffn_type == 'dense':
        shapes['ffn_up']   = GemmShape(M=M_lin, N=F, K=H, op_name='ffn_up')
        shapes['ffn_down'] = GemmShape(M=M_lin, N=H, K=F, op_name='ffn_down')
    elif model.ffn_type == 'GLU':
        shapes['ffn_gate'] = GemmShape(M=M_lin, N=F, K=H, op_name='ffn_gate')
        shapes['ffn_up']   = GemmShape(M=M_lin, N=F, K=H, op_name='ffn_up')
        shapes['ffn_down'] = GemmShape(M=M_lin, N=H, K=F, op_name='ffn_down')
    else:
        raise ValueError(f"Unknown ffn_type: {model.ffn_type!r}")

    # Attention: per-head batched GEMM
    #   QK^T: (M_attn, d) @ (d, S_kv) → (M_attn, S_kv)
    #   AV:   (M_attn, S_kv) @ (S_kv, d) → (M_attn, d)
    # where M_attn = batch*seq for prefill, batch for decode.
    # S_kv is the KV-cache length. For prefill we assume full attention on `seq`;
    # for decode we use `seq` as the average context length (approximation).
    M_attn = batch * seq if phase == 'prefill' else batch
    S_kv = seq

    shapes['attn_qk'] = GemmShape(
        M=M_attn, N=S_kv, K=d, op_name='attn_qk', n_heads=h_q
    )
    shapes['attn_av'] = GemmShape(
        M=M_attn, N=d, K=S_kv, op_name='attn_av', n_heads=h_q
    )

    return shapes


# ---------------------------------------------------------------------------
# DRAM / SRAM byte accounting per RU policy
# ---------------------------------------------------------------------------

def operator_dram_bytes(shape: GemmShape, RU: str,
                        dtype_bytes: int = 2) -> float:
    """DRAM bytes read+written, given a Reuse policy.

    RU policy (LaMoSys3.5D §V-A):
        IRU: stage A (input) in SRAM       → fetch B, write C from/to DRAM
        WRU: stage B (weight) in SRAM      → fetch A, write C
        ORU: stage C (output) in SRAM      → fetch A, B; write C once
        ARU: stage all in SRAM             → no DRAM (if it fits)

    Note: for attention (n_heads > 1), this is a conservative count.
    GQA-style KV sharing across query heads is not exploited here;
    refine in M5 if needed.
    """
    M, N, K = shape.M, shape.N, shape.K
    A = M * K * dtype_bytes
    B = K * N * dtype_bytes
    C = M * N * dtype_bytes
    h = shape.n_heads

    if RU == 'IRU':
        return (B + C) * h
    if RU == 'WRU':
        return (A + C) * h
    if RU == 'ORU':
        return (A + B) * h
    if RU == 'ARU':
        return 0.0
    raise ValueError(f"Unknown RU policy: {RU!r}")


def operator_sram_bytes_needed(shape: GemmShape, RU: str,
                                tile_M: int = None, tile_N: int = None,
                                tile_K: int = None,
                                dtype_bytes: int = 2) -> float:
    """SRAM footprint of the tile that gets staged under this RU policy.

    Used to check feasibility against per-core SRAM budget.
    If tile_* is None, uses the full tensor dimension (no tiling).
    """
    M = tile_M if tile_M is not None else shape.M
    N = tile_N if tile_N is not None else shape.N
    K = tile_K if tile_K is not None else shape.K

    if RU == 'IRU':
        return M * K * dtype_bytes
    if RU == 'WRU':
        return K * N * dtype_bytes
    if RU == 'ORU':
        return M * N * dtype_bytes
    if RU == 'ARU':
        return (M*K + K*N + M*N) * dtype_bytes
    raise ValueError(f"Unknown RU policy: {RU!r}")


def feasible_RU_policies(shape: GemmShape, sram_budget_bytes: float,
                          dtype_bytes: int = 2) -> list:
    """Return RU policies that fit in `sram_budget_bytes` (no tiling assumed).

    For Day 5 this is a coarse filter — D³ search in Week 2 will refine via tiling.
    """
    feasible = []
    for ru in ('IRU', 'WRU', 'ORU', 'ARU'):
        if operator_sram_bytes_needed(shape, ru, dtype_bytes=dtype_bytes) <= sram_budget_bytes:
            feasible.append(ru)
    return feasible


# ---------------------------------------------------------------------------
# Capacity-related helpers (used by serving sim to size pools)
# ---------------------------------------------------------------------------

def kv_cache_size_bytes(model: ModelConfig, batch: int, seq: int,
                         dtype_bytes: int = 2) -> float:
    """KV cache for ONE layer.

    Stored: K and V → 2 × batch × seq × n_kv_heads × d_head × dtype_bytes
    GQA significantly reduces this (small n_kv_heads).
    """
    return 2 * batch * seq * model.n_kv_heads * model.d_head * dtype_bytes


def total_kv_cache_bytes(model: ModelConfig, batch: int, seq: int,
                          dtype_bytes: int = 2) -> float:
    """KV cache across all layers."""
    return kv_cache_size_bytes(model, batch, seq, dtype_bytes) * model.n_layers


def model_param_bytes(model: ModelConfig, dtype_bytes: int = 2) -> float:
    """Total weight bytes (all layers).

    Embedding/output layers are ignored — they're small relative to layer stack.
    """
    H = model.hidden
    h_q, h_kv = model.n_heads, model.n_kv_heads
    d = model.d_head
    F = model.ffn_intermediate

    qkv = H * (h_q + 2 * h_kv) * d
    o = H * h_q * d
    ffn = (3 if model.ffn_type == 'GLU' else 2) * H * F
    per_layer = qkv + o + ffn
    return per_layer * model.n_layers * dtype_bytes


# ---------------------------------------------------------------------------
# Self-tests (run `python operators.py`)
# ---------------------------------------------------------------------------

def _print_shapes(model: ModelConfig, batch: int, seq: int, phase: str):
    print(f"\n--- {model.name} | batch={batch}, seq={seq}, {phase} ---")
    shapes = operator_shapes(model, batch, seq, phase)
    print(f"{'op':12s} {'M':>10s} {'N':>8s} {'K':>8s} {'heads':>6s} {'GFLOPs':>10s}")
    print("-" * 60)
    total_flops = 0
    for name, s in shapes.items():
        total_flops += s.flops
        print(f"{name:12s} {s.M:>10d} {s.N:>8d} {s.K:>8d} "
              f"{s.n_heads:>6d} {s.flops/1e9:>10.2f}")
    print(f"{'TOTAL':12s} {'':>10s} {'':>8s} {'':>8s} {'':>6s} {total_flops/1e9:>10.2f}")


def _run_self_test():
    print("=" * 70)
    print("LaMoSys3.5D operator shape model — sanity checks")
    print("=" * 70)

    # 1) Model sizes
    print("\n--- Model summary ---")
    for m in (GPT3_13B, QWQ_32B, LLAMA3_70B):
        params_gb = model_param_bytes(m) / 1e9
        kv_gb = total_kv_cache_bytes(m, batch=4, seq=2048) / 1e9
        print(f"  {m.name:12s} | {m.attention_type}/{m.ffn_type:5s} | "
              f"{m.n_layers}L, H={m.hidden}, h={m.n_heads}/{m.n_kv_heads}, F={m.ffn_intermediate}")
        print(f"  {'':12s}   params: {params_gb:5.1f} GB | "
              f"KV cache @ bs=4, seq=2048: {kv_gb:5.2f} GB")

    # 2) Shape printouts for GPT3-13B
    _print_shapes(GPT3_13B, batch=4, seq=1024, phase='prefill')
    _print_shapes(GPT3_13B, batch=4, seq=1024, phase='decode')

    # 3) GQA effect: QwQ vs GPT3
    print("\n--- GQA effect on qkv_proj output dim ---")
    for m in (GPT3_13B, QWQ_32B, LLAMA3_70B):
        s = operator_shapes(m, batch=1, seq=1, phase='decode')['qkv_proj']
        # For MHA: should equal 3*H. For GQA: smaller.
        ratio = s.N / m.hidden
        print(f"  {m.name:12s} qkv_proj N={s.N:5d}  (= {ratio:.2f}×H)  "
              f"[expected: MHA=3.00, GQA<3]")

    # 4) DRAM bytes by RU policy
    print("\n--- DRAM bytes by RU policy ---")
    print("    (GPT3-13B, ffn_up, bs=4, seq=1024, prefill)")
    s = operator_shapes(GPT3_13B, batch=4, seq=1024, phase='prefill')['ffn_up']
    print(f"    shape: M={s.M}, N={s.N}, K={s.K}")
    for ru in ('IRU', 'WRU', 'ORU', 'ARU'):
        b = operator_dram_bytes(s, ru)
        sram = operator_sram_bytes_needed(s, ru)
        print(f"    {ru}: DRAM {b/1e6:8.1f} MB  |  SRAM tile {sram/1e6:7.2f} MB")

    # 5) Feasibility under PC's per-core SRAM (256 KB)
    print("\n--- Feasible RU under 256 KB SRAM (no tiling) ---")
    print("    (will be tighter than reality; tiling relaxes this in Week 2)")
    sram_budget = 256 * 1024
    shapes = operator_shapes(GPT3_13B, batch=4, seq=1024, phase='prefill')
    for name, s in shapes.items():
        feas = feasible_RU_policies(s, sram_budget)
        marker = "" if feas else "  ← needs tiling"
        print(f"    {name:12s} feasible: {feas}{marker}")

    # 6) Prefill vs decode flops ratio (sanity)
    print("\n--- Prefill vs decode total FLOPs (GPT3-13B, bs=4) ---")
    for seq in (256, 1024, 4096):
        sp = operator_shapes(GPT3_13B, 4, seq, 'prefill')
        sd = operator_shapes(GPT3_13B, 4, seq, 'decode')
        fp = sum(x.flops for x in sp.values()) / 1e9
        fd = sum(x.flops for x in sd.values()) / 1e9
        print(f"    seq={seq:5d}: prefill {fp:8.1f} GFLOPs, decode {fd:7.2f} GFLOPs  "
              f"(ratio {fp/fd:6.1f}×)")


if __name__ == "__main__":
    _run_self_test()