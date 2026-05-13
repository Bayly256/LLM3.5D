"""
Day 13 driver: build TETRIS / TS / ARU baseline libraries.

Each baseline restricts the D³ search to a single RU policy.
Output is a single pkl containing a dict {baseline_name: MappingLibrary}.

Run after build_mapping_lib.py; the Fig.9 script consumes both.
"""

from __future__ import annotations
import sys
import time
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cost_model"))
sys.path.insert(0, str(ROOT / "scripts"))

from mapping_lib import OpKey
from baselines import build_baseline_libraries
from thermal_proxy import LinearThermalProxy
from build_mapping_lib import op_shapes, make_chiplet_configs, get_evaluator


def run_all_baselines(
    models=None,
    batch_sizes=None,
    seq_lens=None,
    phases=None,
    output_pkl=None,
    max_tiles=5,
    max_splits=5,
):
    """Build TETRIS/TS/ARU libraries on the same OpKey set used by Fig.9."""
    models = models or ["qwq-32b"]
    batch_sizes = batch_sizes or [1, 4, 16, 32]
    seq_lens = seq_lens or [1024]
    phases = phases or ["prefill"]
    output_pkl = output_pkl or str(ROOT / "cost_model" / "mapping_library_baselines.pkl")

    cfgs = make_chiplet_configs()
    evaluator = get_evaluator()
    proxy = LinearThermalProxy()

    # Build OpKey list (same convention as build_mapping_lib.py)
    op_keys = []
    seen = set()
    for model in models:
        for bs in batch_sizes:
            for sl in seq_lens:
                for phase in phases:
                    for op_type, M, N, K in op_shapes(model, bs, sl, phase):
                        if M < 1 or N < 1 or K < 1:
                            continue
                        for chiplet in ("PC", "DC"):
                            k = OpKey(op_type, M, N, K, "fp16", chiplet, bs)
                            if k in seen:
                                continue
                            seen.add(k)
                            op_keys.append(k)

    print(f"Running 3 baselines × {len(op_keys)} OpKeys "
          f"({len(models)} models, bs={batch_sizes}, sl={seq_lens})...")
    t_start = time.time()
    libs = build_baseline_libraries(
        op_keys, cfgs, evaluator,
        max_tiles_per_dim=max_tiles,
        max_core_splits=max_splits,
    )
    print(f"\nAll baselines done in {time.time()-t_start:.1f}s")

    # Attach thermal labels to each (uses the same proxy as D³)
    for name, lib in libs.items():
        lib.attach_thermal(proxy)
        print(f"  {name}: thermal attached ({lib.thermal_method})")

    Path(output_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(libs, f)
    print(f"\nSaved: {output_pkl}")
    print(f"Contents: {list(libs.keys())}")
    for name, lib in libs.items():
        s = lib.stats()
        print(f"  {name}: {s['n_op_entries']} entries, "
              f"{s['total_pareto_candidates']} Pareto, "
              f"{s['total_search_time_s']:.1f}s")
    return libs


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="bs=1,4 only for testing")
    p.add_argument("--max-tiles", type=int, default=5)
    p.add_argument("--max-splits", type=int, default=5)
    args = p.parse_args()

    if args.quick:
        run_all_baselines(
            batch_sizes=[1, 4],
            max_tiles=args.max_tiles,
            max_splits=args.max_splits,
        )
    else:
        run_all_baselines(
            max_tiles=args.max_tiles,
            max_splits=args.max_splits,
        )
