# Week 2 Summary — Innovation #1 thermal-labeled mapping library

**Status**: M2 deliverable complete. 9/9 acceptance tests pass.

## What was built

A library mapping `(op_type, M, N, K, n_heads, dtype, chiplet, phase, batch)`
→ Pareto front of mapping candidates over `(latency, energy, sram)`, each
labeled with a steady-state ΔT estimate and tagged with mode metadata
(`perf_optimal`, `energy_optimal`, `thermal_optimal`, `balanced_knee`).

Eight modules: `mapping_lib.py`, `d3_search.py`, `pareto.py`,
`thermal_proxy.py`, `baselines.py`, `real_evaluator.py` (cost_model/);
`build_mapping_lib.py`, `run_baselines.py`, `fig9_repro.py` (scripts/);
plus 9-test acceptance suite.

## Numbers from the real build

| Metric | Value |
|---|---|
| OpKeys in library | 330 |
| Pareto candidates (total) | 711 |
| Avg Pareto front size | 2.15 |
| Thermal label coverage | 100% (711/711) |
| Full-sweep build time | 0.14s |
| Models × phases × bs × sl | 3 × 2 × 3 × 2 = 36 |

**Adapter validation against Week 1 LUT** (at Week 1's deterministic split):

| Case | Adapter | LUT | Ratio |
|---|---|---|---|
| ffn_up prefill DC bs=4 | 1113.65 μs | 1113.65 μs | 1.000× |
| ffn_up decode DC bs=4 | 8.43 μs | 8.43 μs | 1.000× |
| attn_qk prefill PC bs=4 | 54.76 μs | 54.76 μs | 1.000× |

**D³ improvements over Week 1 deterministic split**:

| Case | Week 1 LUT | D³ best | Improvement |
|---|---|---|---|
| ffn_up prefill DC bs=4 | 1113.65 μs | 1024.12 μs | 8% (split (256,1)→(1,256), full SA vs baseSA) |
| ffn_up decode DC bs=4 | 8.43 μs | 8.42 μs | 0% (Week 1's heuristic already optimal) |
| attn_qk prefill PC bs=4 | 54.76 μs | 54.76 μs | 0% |

**EDP gap vs baselines (DC, decode, bs=16, qwq-32b)**:

| Op | TS / D³ | TETRIS / D³ | ARU / D³ |
|---|---|---|---|
| ffn_up | 1.83× | infeasible | infeasible |
| ffn_down | 1.82× | infeasible | infeasible |
| o_proj | 2.85× | 1.00× | 1.00× |
| qkv_proj | 1.00× | infeasible | infeasible |

Paper reports 1.25-1.53× — our 1.8-2.85× on decode FFN/O is in the same
direction and stronger than expected. Prefill shows ~1.0× (all baselines
identical when compute-bound, as predicted).

**Thermal heterogeneity** (validates Innovation #2 motivation):

| Chiplet | Median ΔT_steady | Max ΔT_steady |
|---|---|---|
| PC (4 DRAM layers) | 65.7°C | 133.5°C |
| DC (8 DRAM layers) | 133.0°C | 187.6°C |
| **DC − PC (median)** | **+67.3°C** | |

This 67°C delta is well above the paper's reported 15-20°C heterogeneity
because the M2 proxy uses high R_th values (calibration target for M4).

## Adapter design

`Week1CostEvaluator` (cost_model/real_evaluator.py) wraps Week 1's
`compute.py` / `memory.py` / `operators.py` primitives into the D³ search
Protocol. Key decisions:

- **Per-core feasibility, not tile-level**: SRAM check uses
  `operator_sram_bytes_needed(per_core_shape, RU)` against `SRAM_per_core_KB`.
  Closes the "ARU free lunch" loophole where tile=(1,1,1) trivially fits.
- **Power decomposition**: `p_mpu_W = e_comp_pJ / latency_ns × 1e-3`,
  `p_dram_W = e_mem_pJ / latency_ns × 1e-3`; `p_noc=10W`, `p_vpu=5W` fixed.
- **Heads-first split**: when `n_heads > 1`, mirrors Week 1's
  `chiplet_2d_split` — heads distributed first, then within-head (cM, cN).
- **STREAM RU added**: always feasible; fallback when no other RU fits.

## Limitations carried forward (to error ledger)

| Limitation | Effect | Fix milestone |
|---|---|---|
| No tile-aware DRAM reuse | Pareto fronts thinner than paper's; loop_order frozen to ("MNK",) | M5 |
| p_noc, p_vpu fixed estimates | thermal proxy uses approximated rates | M4 |
| Memory/compute use max(), not pipelined | ≤30% latency error (inherited from Wk1) | M4 |
| SA peak 838 vs paper 400 TFLOPS | absolute latencies 2.1× off | M9 calibration |
| Thermal proxy R_th values uncalibrated | ΔT median 67°C high vs paper 15-20°C | M4 (HotSpot ground truth) |

## Files produced

```
cost_model/mapping_library_v0.1.pkl              (main library, 330 entries)
cost_model/mapping_library_v0.1.parquet          (analytical export)
cost_model/mapping_library_baselines.pkl         ({TETRIS, TS, ARU})
figs/fig9_repro_DC_decode.png + .csv             (Fig.9 reproduction)
figs/fig9_repro_DC_prefill.png + .csv
figs/fig9_repro_PC_decode.png + .csv
```

## What Week 3 inherits

- `MappingLibrary.compare_chiplets(op, M, N, K, dtype, phase, bs, n_heads)`
  → `{"PC": MappingCandidate, "DC": MappingCandidate}`  (scheduler migration API)
- `MappingLibrary.best_under_thermal(op_key, delta_T_budget_C)`
  → thermal-aware perf lookup (Innovation #2 hook)
- All Pareto candidates have `mode_tags` for scheduler mode-switching
- Thermal method recorded in `lib.thermal_method`; swap to HotSpot in Week 4
  by calling `lib.attach_thermal(new_proxy)` — no D³ rerun needed
