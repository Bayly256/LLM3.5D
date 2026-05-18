"""
Day 12 driver: build the full thermal-labeled mapping library.

Sweeps {models} × {batch_sizes} × {seq_lens} × {phases} × {ops} × {chiplets},
runs D³ search per cell, attaches thermal labels, tags Pareto modes, saves
pkl + parquet.

Uses Week 1's operator_shapes() (so op naming, n_heads, attn-vs-linear logic
is consistent), Week 1's load_chiplet() for YAML configs, and the
Week1CostEvaluator adapter.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cost_model"))
sys.path.insert(0, str(ROOT))

from configs import load_chiplet  # type: ignore
from operators import operator_shapes, ALL_MODELS  # type: ignore

from mapping_lib import OpKey, MappingLibrary
from d3_search import d3_search
from thermal_proxy import LinearThermalProxy
from pareto import tag_modes
from real_evaluator import Week1CostEvaluator


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
    max_splits=6,
    temp_C: float = 65.0,
    verbose=False,
):
    models = models or ["gpt3-13b", "qwq-32b", "llama3-70b"]
    batch_sizes = batch_sizes or [1, 4, 16]
    seq_lens = seq_lens or [1024, 4096]
    phases = phases or ["prefill", "decode"]
    output_pkl = output_pkl or str(ROOT / "cost_model" / "mapping_library_v0.1.pkl")
    output_parquet = output_parquet or str(ROOT / "cost_model" / "mapping_library_v0.1.parquet")

    print("Loading chiplet configs from YAML...")
    cfgs = {"PC": load_chiplet("PC"), "DC": load_chiplet("DC")}
    print(f"  PC: BW={cfgs['PC'].peak_bw_TBs} TB/s, "
          f"{cfgs['PC'].n_PE}×{cfgs['PC'].n_cores_per_PE} cores, "
          f"SRAM/core={cfgs['PC'].SRAM_per_core_KB} KB")
    print(f"  DC: BW={cfgs['DC'].peak_bw_TBs} TB/s, "
          f"{cfgs['DC'].n_PE}×{cfgs['DC'].n_cores_per_PE} cores, "
          f"SRAM/core={cfgs['DC'].SRAM_per_core_KB} KB")

    evaluator = Week1CostEvaluator(temp_C=temp_C)
    proxy = LinearThermalProxy(ambient_C=45.0)

    lib = MappingLibrary(version="v0.1", cost_model_version="M2")
    lib.meta["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    lib.meta["models"] = models
    lib.meta["batch_sizes"] = batch_sizes
    lib.meta["seq_lens"] = seq_lens
    lib.meta["phases"] = phases
    lib.meta["temp_C_for_cost"] = temp_C
    lib.meta["evaluator"] = type(evaluator).__name__

    t_start = time.time()
    n_entries = 0
    seen_keys = set()

    for model_name in models:
        if model_name not in ALL_MODELS:
            print(f"[WARN] unknown model {model_name}, skip")
            continue
        model = ALL_MODELS[model_name]
        for bs in batch_sizes:
            for sl in seq_lens:
                for phase in phases:
                    shapes = operator_shapes(model, bs, sl, phase)
                    for op_name, shape in shapes.items():
                        if shape.M < 1 or shape.N < 1 or shape.K < 1:
                            continue
                        for chiplet in ("PC", "DC"):
                            key = OpKey(
                                op_type=op_name,
                                M=shape.M, N=shape.N, K=shape.K,
                                n_heads=shape.n_heads,
                                dtype="fp16",
                                chiplet_type=chiplet,
                                phase=phase,
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
    p.add_argument("--phases", nargs="+", default=None)
    p.add_argument("--max-tiles", type=int, default=5)
    p.add_argument("--max-splits", type=int, default=6)
    p.add_argument("--temp-C", type=float, default=65.0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.quick:
        build(models=["qwq-32b"], batch_sizes=[1, 4], seq_lens=[1024],
              max_tiles=4, max_splits=4, temp_C=args.temp_C,
              verbose=args.verbose)
    else:
        build(models=args.models, batch_sizes=args.batch_sizes,
              seq_lens=args.seq_lens, phases=args.phases,
              max_tiles=args.max_tiles, max_splits=args.max_splits,
              temp_C=args.temp_C, verbose=args.verbose)
