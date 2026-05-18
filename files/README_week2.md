# Week 2 — Thermal-labeled mapping library (Innovation #1)

M2 deliverable: schema + D³ search + thermal proxy + Pareto mapping library
built directly on top of Week 1's cost primitives.

## Layout

```
cost_model/
├── memory.py compute.py comm.py operators.py build_lut.py  (Week 1, unchanged)
├── lut.pkl                                                  (Week 1 output)
├── mapping_lib.py        OpKey, MappingCandidate, ThermalLabel, MappingLibrary
├── d3_search.py          Algorithm 1, RU/loop_order/tile enumeration
├── pareto.py             3D Pareto filter + dedup + mode tagging
├── thermal_proxy.py      M2 linear ΔT model (logic + DRAM stack)
├── baselines.py          TETRIS / TS / ARU baseline searches
├── real_evaluator.py     Week1CostEvaluator adapter — uses Week 1 primitives
├── mapping_library_v0.1.{pkl,parquet}    main library
└── mapping_library_baselines.pkl         {TETRIS,TS,ARU} libraries
configs/  (Week 1 unchanged: PC.yaml, DC.yaml, system.yaml, __init__.py)
scripts/
├── build_mapping_lib.py  Day 12 driver
├── run_baselines.py      Day 13 driver
└── fig9_repro.py         Day 11 plot
tests/
└── test_week2.py         9-test acceptance suite (Day 14)
figs/
└── fig9_repro_{DC,PC}_{prefill,decode}.{png,csv}
```

## Quickstart

```bash
python tests/test_week2.py                # 9 tests
python scripts/build_mapping_lib.py --quick --max-tiles 1
python scripts/build_mapping_lib.py --max-tiles 1
python scripts/run_baselines.py --max-tiles 1
python scripts/fig9_repro.py --chiplet DC --phase decode --batch-sizes 1 4 16
```

## Using the library

```python
from cost_model.mapping_lib import OpKey, MappingLibrary

lib = MappingLibrary.load("cost_model/mapping_library_v0.1.pkl")

# Perf-optimal lookup
k = OpKey("ffn_up", M=16, N=27648, K=5120, n_heads=1,
          dtype="fp16", chiplet_type="DC", phase="decode", batch_size=16)
best = lib.best_perf(k)
print(f"RU={best.RU}, split=({best.cores_M},{best.cores_N}), "
      f"lat={best.latency_us:.2f}μs, ΔT={best.thermal.delta_T_steady_C:.1f}°C")

# Migration query (Week 3 scheduler will use this)
result = lib.compare_chiplets("ffn_up", 16, 27648, 5120, "fp16",
                              "decode", 16, n_heads=1)
# → {"PC": MappingCandidate(...), "DC": MappingCandidate(...)}

# Thermal-budget query (Innovation #2)
cool = lib.best_under_thermal(k, delta_T_budget_C=30.0)

# Pareto front for analysis
for cand in lib.lookup(k).pareto_front:
    print(f"  {cand.RU:6s} lat={cand.latency_us:7.2f} eng={cand.energy_uJ:7.2f} "
          f"tags={cand.mode_tags}")
```

## Architecture: Week 1 ↔ Week 2 wiring

`Week1CostEvaluator` (real_evaluator.py) adapts Week 1's per-(op, RU)
primitives to the Week 2 D³ Protocol. Validated against `lut.pkl`:
calling the adapter at Week 1's exact deterministic split reproduces LUT
latency to 1.000×. D³ search then explores `(RU, cores_M, cores_N)` and
finds improvements (e.g., 8% on ffn_up prefill via cores=(1,256) full-SA
vs Week 1's heuristic (256,1) baseSA).

Power decomposition is derived at the adapter:
- `p_mpu_W = e_comp_pJ / latency_ns × 1e-3` (MAC energy rate)
- `p_dram_W = e_mem_pJ / latency_ns × 1e-3` (DRAM access rate)
- `p_noc_W = 10W`, `p_vpu_W = 5W` (fixed estimates; Week 1 doesn't model)
- `p_peak_W = max(p_mpu, p_dram) × 1.5 + p_noc + p_vpu` (peak/avg=1.5)

## Known limitations (Week 1 cost model gaps)

Documented in error ledger; addressed in M4/M5:

| Limitation | Effect | Fix milestone |
|---|---|---|
| No tile-aware DRAM reuse | tile size affects feasibility only | M5 |
| loop_order has no cost effect | restricted to ("MNK",) by default | M5 |
| Memory/compute use max, not pipelined | ≤30% latency error | M4 |
| SA peak 838 vs paper 400 TFLOPS (2.1×) | LUT magnitude off | M9 calibration |
| p_noc, p_vpu fixed estimates | thermal proxy uses approximated rates | M4 |
| DRAM timing mostly est:M9 | only tRFI_lowT is JEDEC truth | M9 |
| No NoC contention term | comm cost ideal | Week 3 |

## Thermal proxy (M2)

```
ΔT_steady = p_logic · R_th_logic + p_dram · (R_th_logic + R_th_dram_per_layer · N_layers)
```

With R_th_logic=0.45 K/W, R_th_dram=0.06 K/W/layer, τ=0.20s. DC's 8-layer
DRAM stack produces higher ΔT than PC's 4-layer at same power — used in
Innovation #2 for migration decisions. Replaced by HotSpot (M4) and
surrogate (M3) in Week 4.
