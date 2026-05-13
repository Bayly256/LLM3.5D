"""
Day 12 driver: build the full thermal-labeled mapping library.

Sweeps {models} × {batch_sizes} × {seq_lens} × {ops} × {chiplets},
runs D³ search per cell, attaches thermal labels, tags Pareto modes,
saves pkl + parquet.

Wire-in:
  - configs:    make_chiplet_configs() — reads outputs/configs/PC.yaml / DC.yaml
                                          (falls back to defaults if absent)
  - evaluator:  get_evaluator() — looks for cost_model/real_evaluator.py
                                  (falls back to MockCostEvaluator)

Without Week 1 wired, this script still runs end-to-end but produces
plumbing-only numbers (not meaningful for paper comparisons).
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cost_model"))

from mapping_lib import OpKey, MappingLibrary
from d3_search import d3_search, MockCostEvaluator
from thermal_proxy import LinearThermalProxy
from pareto import tag_modes


# =============================================================
# Model shape table  
# =============================================================
# M = batch * seq_len (prefill) or batch * 1 (decode)
# Convention: GEMM is C[M,N] = A[M,K] @ B[K,N]
# So qkv_proj: input [M, K=hidden], weight [K=hidden, N=qkv_out], output [M, N]

MODEL_SHAPES = {
    "gpt3-13b":   {"hidden": 5120, "hidden_inner": 20480, "n_q_heads": 40, "n_kv_heads": 40, "head_dim": 128},
    "qwq-32b":    {"hidden": 5120, "hidden_inner": 27648, "n_q_heads": 40, "n_kv_heads": 8,  "head_dim": 128},
    "llama3-70b": {"hidden": 8192, "hidden_inner": 28672, "n_q_heads": 64, "n_kv_heads": 8,  "head_dim": 128},
}


def op_shapes(model: str, batch: int, seq_len: int, phase: str = "prefill"):
    """Return list of (op_type, M, N, K) for one layer.

    Note: attn_logit and attn_attend depend on phase and need per-head treatment.
    For M2 we treat them at the head-batch level: M = batch * n_q_heads.
    """
    s = MODEL_SHAPES[model]
    H = s["hidden"]
    Hi = s["hidden_inner"]
    Hd = s["head_dim"]
    n_q = s["n_q_heads"]
    n_kv = s["n_kv_heads"]

    M = batch * seq_len if phase == "prefill" else batch
    qkv_out = (n_q + 2 * n_kv) * Hd  # fused QKV projection

    return [
        ("qkv_proj",   M,          qkv_out,    H),
        ("attn_logit", batch*n_q,  seq_len,    Hd),  # Q @ K^T: [bs*nq, hd] @ [hd, seq_len]
        ("attn_attend", batch*n_q, Hd,         seq_len),  # softmax @ V
        ("o_proj",     M,          H,          H),
        ("ffn_up",     M,          Hi,         H),
        ("ffn_gate",   M,          Hi,         H),
        ("ffn_down",   M,          H,          Hi),
    ]


# =============================================================
# Chiplet configs — WIRE-IN-WEEK-1: replace with YAML loader  
# =============================================================

def make_chiplet_configs():
    """Build (PC, DC) chiplet configs.

    WIRE-IN-WEEK-1: replace with:
        import yaml
        with open(ROOT / "configs" / "PC.yaml") as f: pc = yaml.safe_load(f)
        with open(ROOT / "configs" / "DC.yaml") as f: dc = yaml.safe_load(f)
        return {"PC": pc, "DC": dc}
    """
    yaml_pc = ROOT / "configs" / "PC.yaml"
    yaml_dc = ROOT / "configs" / "DC.yaml"
    if yaml_pc.exists() and yaml_dc.exists():
        try:
            import yaml
            with open(yaml_pc) as f:
                pc = yaml.safe_load(f)
            with open(yaml_dc) as f:
                dc = yaml.safe_load(f)
            print(f"[INFO] Loaded chiplet configs from {yaml_pc} and {yaml_dc}")
            return {"PC": pc, "DC": dc}
        except Exception as e:
            print(f"[WARN] YAML load failed ({e}); using defaults")

    print("[INFO] Using hardcoded chiplet config defaults (no YAML found)")
    return {
        "PC": {
            "sa_rows": 64, "sa_cols": 32,
            "freq_hz": 800e6,
            "bw_Bps": 10.6e12,                    # 10.6 TB/s
            "sram_budget_bytes": 256 * 1024,      # 256 KB per PE
            "total_cores": 256,                   # 16 PEs × 16 cores
            "p_mpu_max": 250, "p_dram_max": 60, "p_noc_max": 30,
            "n_dram_layers": 4,
        },
        "DC": {
            "sa_rows": 64, "sa_cols": 32,
            "freq_hz": 800e6,
            "bw_Bps": 26.2e12,                    # 26.2 TB/s
            "sram_budget_bytes": 256 * 1024,
            "total_cores": 256,                   # 32 PEs × 8 cores
            "p_mpu_max": 200, "p_dram_max": 100, "p_noc_max": 30,
            "n_dram_layers": 8,
        },
    }


# =============================================================
# CostEvaluator wire-in  
# =============================================================

def get_evaluator():
    """Try Week 1's real evaluator; fall back to MockCostEvaluator.

    WIRE-IN-WEEK-1: create cost_model/real_evaluator.py with class
        class Week1CostEvaluator:
            def evaluate(self, op_key, tile, ru, loop_order, cores_split, chiplet_cfg) -> dict:
                ...
    See d3_search.CostEvaluator Protocol for the required return dict.
    """
    try:
        from real_evaluator import Week1CostEvaluator  # type: ignore
        ev = Week1CostEvaluator()
        print(f"[INFO] Using real Week 1 evaluator: {type(ev).__name__}")
        return ev
    except ImportError:
        print("[WARN] real_evaluator.py not found in cost_model/; using MockCostEvaluator")
        print("       Wire-in instructions: see README_week2.md §Week 1 wire-in")
        return MockCostEvaluator()


# =============================================================
# Build  
# =============================================================

def build(
    models=None,
    batch_sizes=None,
    seq_lens=None,
    phases=None,
    output_pkl=None,
    output_parquet=None,
    max_tiles=5,
    max_splits=5,
    verbose=False,
):
    models = models or ["gpt3-13b", "qwq-32b", "llama3-70b"]
    batch_sizes = batch_sizes or [1, 4, 16]
    seq_lens = seq_lens or [1024, 4096]
    phases = phases or ["prefill"]
    output_pkl = output_pkl or str(ROOT / "cost_model" / "mapping_library_v0.1.pkl")
    output_parquet = output_parquet or str(ROOT / "cost_model" / "mapping_library_v0.1.parquet")

    cfgs = make_chiplet_configs()
    evaluator = get_evaluator()
    proxy = LinearThermalProxy()

    lib = MappingLibrary(version="v0.1", cost_model_version="M2")
    lib.meta["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    lib.meta["models"] = models
    lib.meta["batch_sizes"] = batch_sizes
    lib.meta["seq_lens"] = seq_lens
    lib.meta["phases"] = phases
    lib.meta["evaluator"] = type(evaluator).__name__

    t_start = time.time()
    n_entries = 0
    seen_keys = set()

    for model in models:
        for bs in batch_sizes:
            for sl in seq_lens:
                for phase in phases:
                    shapes = op_shapes(model, bs, sl, phase)
                    for op_type, M, N, K in shapes:
                        if M < 1 or N < 1 or K < 1:
                            continue
                        for chiplet in ("PC", "DC"):
                            key = OpKey(
                                op_type=op_type, M=M, N=N, K=K,
                                dtype="fp16", chiplet_type=chiplet,
                                batch_size=bs,
                            )
                            if key in seen_keys:
                                continue
                            seen_keys.add(key)
                            entry = d3_search(
                                key, cfgs[chiplet], evaluator,
                                max_tiles_per_dim=max_tiles,
                                max_core_splits=max_splits,
                                verbose=verbose,
                            )
                            lib.entries[key] = entry
                            n_entries += 1
                            if n_entries % 20 == 0:
                                elapsed = time.time() - t_start
                                print(f"  [{n_entries} entries done, {elapsed:.1f}s elapsed]")

    print(f"\nD³ search complete: {n_entries} entries in {time.time()-t_start:.1f}s")

    print(f"Attaching thermal labels (method={proxy.method_tag()})...")
    lib.attach_thermal(proxy)

    print("Tagging Pareto modes...")
    for entry in lib.entries.values():
        tag_modes(entry.pareto_front)

    print(f"\nLibrary stats:")
    for k, v in lib.stats().items():
        print(f"  {k}: {v}")

    Path(output_pkl).parent.mkdir(parents=True, exist_ok=True)
    lib.save(output_pkl)
    print(f"\nSaved pkl: {output_pkl}")
    try:
        lib.export_parquet(output_parquet)
        print(f"Saved parquet: {output_parquet}")
    except Exception as e:
        print(f"[WARN] Parquet export skipped: {e}")

    return lib


# =============================================================
# CLI  
# =============================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="Smaller sweep for testing (qwq-32b, bs=1,4, sl=1024)")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    p.add_argument("--seq-lens", nargs="+", type=int, default=None)
    p.add_argument("--max-tiles", type=int, default=5)
    p.add_argument("--max-splits", type=int, default=5)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.quick:
        build(
            models=["qwq-32b"],
            batch_sizes=[1, 4],
            seq_lens=[1024],
            max_tiles=4,
            max_splits=4,
            verbose=args.verbose,
        )
    else:
        build(
            models=args.models,
            batch_sizes=args.batch_sizes,
            seq_lens=args.seq_lens,
            max_tiles=args.max_tiles,
            max_splits=args.max_splits,
            verbose=args.verbose,
        )
