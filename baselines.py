"""
Day 13: Baseline dataflow strategies for Fig.9 comparison.

Each baseline restricts the D³ search space to mimic a specific prior work:

- TETRIS: forces ARU policy (stage all operands in SRAM, minimize DRAM traffic).
  Approximates the SRAM-staging philosophy of TETRIS for systolic arrays.

- TokenStationary (TS, from 3D-TokSIM): stages input activations (A-tiles)
  on chip while streaming weights from DRAM. Maps to IRU in our taxonomy.

- ARU-only: classical LLMCompass SRAM-reuse philosophy — force ARU regardless
  of whether direct-DRAM-delivery (D³ insight) would be cheaper.

These build parallel libraries used for Day 11 Fig.9 EDP comparison.
"""

from __future__ import annotations
import time

from mapping_lib import OpKey, MappingLibrary
from d3_search import d3_search


def _search_with_ru_restriction(
    op_key: OpKey, chiplet_cfg: dict, evaluator,
    ru_set: tuple, **kwargs
):
    """Run D³ search but restricted to a subset of RU policies."""
    return d3_search(
        op_key, chiplet_cfg, evaluator,
        ru_policies=ru_set,
        **kwargs,
    )


def search_tetris(op_key, chiplet_cfg, evaluator, **kwargs):
    """TETRIS: stage everything in SRAM. ARU only."""
    return _search_with_ru_restriction(
        op_key, chiplet_cfg, evaluator, ru_set=("ARU",), **kwargs
    )


def search_token_stationary(op_key, chiplet_cfg, evaluator, **kwargs):
    """TS: stage activations (A), stream weights. IRU only."""
    return _search_with_ru_restriction(
        op_key, chiplet_cfg, evaluator, ru_set=("IRU",), **kwargs
    )


def search_aru_only(op_key, chiplet_cfg, evaluator, **kwargs):
    """SRAM-reuse-centric (LLMCompass-style). ARU only."""
    return _search_with_ru_restriction(
        op_key, chiplet_cfg, evaluator, ru_set=("ARU",), **kwargs
    )


BASELINE_STRATEGIES = {
    "TETRIS": search_tetris,
    "TS": search_token_stationary,
    "ARU": search_aru_only,
}


def build_baseline_libraries(
    op_keys: list, chiplet_cfgs: dict, evaluator,
    max_tiles_per_dim: int = 5, max_core_splits: int = 5,
) -> dict:
    """Build all three baseline libraries.
    Returns dict {baseline_name: MappingLibrary}."""
    results = {}
    for baseline_name, search_fn in BASELINE_STRATEGIES.items():
        print(f"  Building baseline: {baseline_name}")
        t0 = time.time()
        lib = MappingLibrary(
            version=f"baseline_{baseline_name}_v0.1",
            cost_model_version="M2",
        )
        for op_key in op_keys:
            cfg = chiplet_cfgs[op_key.chiplet_type]
            entry = search_fn(
                op_key, cfg, evaluator,
                max_tiles_per_dim=max_tiles_per_dim,
                max_core_splits=max_core_splits,
            )
            lib.entries[op_key] = entry
        lib.meta["baseline"] = baseline_name
        lib.meta["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        lib.meta["build_time_s"] = time.time() - t0
        results[baseline_name] = lib
        print(f"    {baseline_name}: {len(lib.entries)} entries in {time.time()-t0:.1f}s")
    return results
