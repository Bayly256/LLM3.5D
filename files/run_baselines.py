"""
Day 13 driver: build TETRIS / TS / ARU baseline libraries.

Each baseline restricts D³ search to a single RU policy. Outputs a pkl
containing {baseline_name: MappingLibrary}.
"""

from __future__ import annotations
import sys
import time
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cost_model"))
sys.path.insert(0, str(ROOT))

from configs import load_chiplet  # type: ignore
from operators import operator_shapes, ALL_MODELS  # type: ignore

from mapping_lib import OpKey
from baselines import build_baseline_libraries
from thermal_proxy import LinearThermalProxy
from real_evaluator import Week1CostEvaluator


def run_all_baselines(
    models=None, batch_sizes=None, seq_lens=None, phases=None,
    output_pkl=None, max_tiles=5, max_splits=6, temp_C=65.0,
):
    models = models or ["qwq-32b"]
    batch_sizes = batch_sizes or [1, 4, 16, 32]
    seq_lens = seq_lens or [1024]
    phases = phases or ["prefill", "decode"]
    output_pkl = output_pkl or str(ROOT / "cost_model" / "mapping_library_baselines.pkl")

    cfgs = {"PC": load_chiplet("PC"), "DC": load_chiplet("DC")}
    evaluator = Week1CostEvaluator(temp_C=temp_C)
    proxy = LinearThermalProxy(ambient_C=45.0)

    op_keys = []
    seen = set()
    for model_name in models:
        if model_name not in ALL_MODELS:
            continue
        model = ALL_MODELS[model_name]
        for bs in batch_sizes:
            for sl in seq_lens:
                for phase in phases:
                    for op_name, shape in operator_shapes(model, bs, sl, phase).items():
                        if shape.M < 1 or shape.N < 1 or shape.K < 1:
                            continue
                        for chiplet in ("PC", "DC"):
                            k = OpKey(
                                op_type=op_name, M=shape.M, N=shape.N, K=shape.K,
                                n_heads=shape.n_heads, dtype="fp16",
                                chiplet_type=chiplet, phase=phase, batch_size=bs,
                            )
                            if k in seen:
                                continue
                            seen.add(k)
                            op_keys.append(k)

    print(f"Running 3 baselines × {len(op_keys)} OpKeys...")
    t_start = time.time()
    libs = build_baseline_libraries(
        op_keys, cfgs, evaluator,
        max_tiles_per_dim=max_tiles,
        max_core_splits=max_splits,
    )
    print(f"\nAll baselines done in {time.time()-t_start:.1f}s")

    for name, lib in libs.items():
        lib.attach_thermal(proxy)
        print(f"  {name}: thermal attached ({lib.thermal_method})")

    Path(output_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(libs, f)
    print(f"\nSaved: {output_pkl}")
    for name, lib in libs.items():
        s = lib.stats()
        print(f"  {name}: {s['n_op_entries']} entries, "
              f"{s['total_pareto_candidates']} Pareto, "
              f"{s['total_search_time_s']:.1f}s")
    return libs


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--max-tiles", type=int, default=5)
    p.add_argument("--max-splits", type=int, default=6)
    args = p.parse_args()

    if args.quick:
        run_all_baselines(batch_sizes=[1, 4], max_tiles=args.max_tiles,
                          max_splits=args.max_splits)
    else:
        run_all_baselines(max_tiles=args.max_tiles, max_splits=args.max_splits)
