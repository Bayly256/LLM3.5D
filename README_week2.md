# Week 2 — Thermal-labeled Mapping Library

Innovation #1 deliverable: a queryable Pareto-front library of intra-PE
mappings with thermal ΔT labels attached out-of-band, output of the D³
exhaustive search (paper Algorithm 1) extended with chiplet-internal 2D
core split.

## File layout

```
outputs/
├── cost_model/
│   ├── mapping_lib.py        # Schema (OpKey, MappingCandidate, ThermalLabel, ...)
│   ├── d3_search.py          # Algorithm 1 + CostEvaluator protocol + MockCostEvaluator
│   ├── thermal_proxy.py      # M2 linear thermal proxy
│   ├── pareto.py             # Pareto filter + mode tagging
│   ├── baselines.py          # TETRIS / TS / ARU restricted searches
│   ├── mapping_library_v0.1.pkl       <- built by scripts/build_mapping_lib.py
│   ├── mapping_library_v0.1.parquet
│   └── mapping_library_baselines.pkl  <- built by scripts/run_baselines.py
├── scripts/
│   ├── build_mapping_lib.py  # Day 12 driver (main innovation #1 output)
│   ├── run_baselines.py      # Day 13 baselines for Fig.9 comparison
│   └── fig9_repro.py         # Day 11 EDP-vs-batch plotting
├── tests/
│   └── test_week2.py         # Day 14 acceptance suite (9 tests)
└── figs/
    ├── fig9_repro_DC.png
    └── fig9_ru_table.csv
```

## Quick start (smoke test, no Week 1 wired)

```bash
# 1. Acceptance tests (9 tests, <1 minute)
python tests/test_week2.py

# 2. Build a small library (qwq-32b, bs=1,4; ~20s with MockCostEvaluator)
python scripts/build_mapping_lib.py --quick

# 3. Build baseline libraries (~30s)
python scripts/run_baselines.py --quick

# 4. Generate Fig.9-style plot
python scripts/fig9_repro.py --batch-sizes 1 4
```

With MockCostEvaluator the numbers are NOT meaningful (compute-bound
everywhere, all dataflows look identical at 1.00× ratio). The pipeline
itself runs correctly — that's what `--quick` validates.

## Full Day 12 build (after Week 1 wired)

```bash
python scripts/build_mapping_lib.py            # full sweep ~3 models × 3 bs × 2 sl
python scripts/run_baselines.py --max-tiles 5  # baselines on bs=[1,4,16,32]
python scripts/fig9_repro.py                   # produces final Fig.9
```

Estimated time with a real Week 1 evaluator (assuming ~5 ms per `.evaluate()` call):
- D³ build: ~10–15 minutes
- baselines: ~5 minutes per baseline × 3 = 15 minutes
- Fig.9 plot: <1 minute

If too slow, lower `--max-tiles` and `--max-splits` from 5 to 4 (≈3× speedup).

## Week 1 wire-in (the one thing you must do)

### Create `cost_model/real_evaluator.py`

```python
# cost_model/real_evaluator.py
import pickle
from pathlib import Path

class Week1CostEvaluator:
    """Implements the CostEvaluator protocol from d3_search.py.

    Reads from your Week 1 lut.pkl + cost models.
    """
    def __init__(self, lut_path=None):
        lut_path = lut_path or (Path(__file__).parent / "lut.pkl")
        with open(lut_path, "rb") as f:
            self.lut = pickle.load(f)
        # any other state your cost model needs

    def evaluate(self, op_key, tile, ru, loop_order, cores_split, chiplet_cfg) -> dict:
        # Use your Week 1 memory.py / compute.py / comm.py to fill in the dict.
        # Returns:
        return {
            "latency_us":   ...,  # float
            "energy_uJ":    ...,  # float
            "sram_bytes":   ...,  # int — actual on-chip footprint
            "p_mpu_W":      ...,  # MPU compute power
            "p_vpu_W":      ...,  # vector unit power
            "p_dram_W":     ...,  # DRAM stack power (separate! thermal needs this)
            "p_noc_W":      ...,  # NoC + interconnect power
            "p_avg_W":      ...,  # = p_mpu+p_vpu+p_dram+p_noc
            "p_peak_W":     ...,  # peak instantaneous
            "bw_utilization":     ...,  # in [0, 1]
            "mpu_utilization":    ...,
            "sa_active_rows_frac": ...,  # GEMV degradation
            "is_compute_bound":   ...,  # bool
            "provenance": {
                "latency_us": ["sim:M2"],         # per-field source tags
                "energy_uJ":  ["sim:M2"],
                "p_dram_W":   ["est:M9"],         # if still estimated
                # ...
            }
        }
```

`build_mapping_lib.py` auto-detects this class via `import real_evaluator`.

### Pre-Day-8 blocker check

The thermal proxy needs **decomposed power** (`p_mpu_W`, `p_dram_W`, `p_noc_W`
separately). If your Week 1 `compute.py`/`memory.py` only return a single
`p_total`, you must split them before continuing — DC's 8 DRAM layers vs
PC's 4 layers is the entire physical basis of Innovation #2's motivation.

Quick check:
```bash
python -c "
import pickle
with open('cost_model/lut.pkl', 'rb') as f: lut = pickle.load(f)
sample = next(iter(lut.values())) if isinstance(lut, dict) else lut[0]
print('Sample LUT entry keys:', list(sample.keys()) if hasattr(sample, 'keys') else 'not a dict')
"
```

You should see `p_mpu`, `p_dram`, `p_noc` (or equivalents). If only `p_total`,
add the decomposition before Day 9.

### Optional: real chiplet configs

`make_chiplet_configs()` in `build_mapping_lib.py` auto-loads
`outputs/configs/PC.yaml` and `DC.yaml` if present. Expected keys (defaults
shown — see `make_chiplet_configs()` for full schema):

```yaml
# configs/DC.yaml
sa_rows: 64
sa_cols: 32
freq_hz: 800e6
bw_Bps: 26.2e12          # 26.2 TB/s (DC's 8-layer DRAM)
sram_budget_bytes: 262144
total_cores: 256
p_mpu_max: 200
p_dram_max: 100
p_noc_max: 30
n_dram_layers: 8
```

PC is identical structure but `bw_Bps: 10.6e12`, `n_dram_layers: 4`.

## Library query API

After build:

```python
from mapping_lib import MappingLibrary, OpKey

lib = MappingLibrary.load("cost_model/mapping_library_v0.1.pkl")

# Innovation #1: query best mapping for serving simulator
key = OpKey("ffn_up", M=4096, N=27648, K=5120, dtype="fp16",
            chiplet_type="DC", batch_size=4)
best = lib.best_perf(key)
print(f"  Use RU={best.RU}, tile=({best.T_M},{best.T_N},{best.T_K}), "
      f"split=({best.cores_M},{best.cores_N})")

# Innovation #2 mode: thermal-aware fallback
safe = lib.best_under_thermal(key, delta_T_budget_C=40.0)

# Innovation #2 migration query: which chiplet runs this op cheaper?
options = lib.compare_chiplets("ffn_up", 4096, 27648, 5120, "fp16", 4)
print(f"  PC: lat={options['PC'].latency_us:.1f}μs ΔT={options['PC'].thermal.delta_T_steady_C:.1f}°C")
print(f"  DC: lat={options['DC'].latency_us:.1f}μs ΔT={options['DC'].thermal.delta_T_steady_C:.1f}°C")
```

## Day-by-day mapping → deliverables

| Day | Deliverable | File |
|-----|-------------|------|
| 8  | D³ Algorithm 1 + Pareto over (lat, eng, sram) | `cost_model/d3_search.py`, `cost_model/pareto.py` |
| 9  | M2 linear thermal proxy (PC vs DC) | `cost_model/thermal_proxy.py` |
| 10 | attach_thermal + mode tagging pipeline | `mapping_lib.attach_thermal()`, `pareto.tag_modes()` |
| 11 | Fig.9 EDP-vs-batch repro | `scripts/fig9_repro.py`, `figs/fig9_repro_DC.png` |
| 12 | Full mapping_library_v0.1.pkl | `scripts/build_mapping_lib.py` |
| 13 | Baselines for Fig.9 comparison | `scripts/run_baselines.py`, `cost_model/baselines.py` |
| 14 | Acceptance suite + this README + summary | `tests/test_week2.py`, `README_week2.md`, `week_2_summary.md` |

## Acceptance criteria (Day 14)

- [x] `tests/test_week2.py`: 9/9 passing
- [ ] After Week 1 wired: full build runs without errors
- [ ] Library has ≥ 252 entries (3 models × 3 bs × 2 sl × 7 ops × 2 chiplets / minus dups)
- [ ] All Pareto candidates have thermal labels (`thermal_attached: True` in `lib.stats()`)
- [ ] DC ΔT > PC ΔT for same power (`test_thermal_proxy`)
- [ ] (with real evaluator) ffn_up DC best RU differs between bs=1 (BW-bound) and bs=16 (compute-bound)
- [ ] (with real evaluator) Fig.9 shows D³ EDP < baselines at bs ≥ 8 (paper: 1.25–1.53×)

## Known limitations / error budget (carry forward)

- **Thermal proxy ±15°C uncertainty.** Linear `R_th_eff(n_layers)` is rough.
  Day 22–24 (Week 4) replaces with HotSpot-trained M3 surrogate.
- **`time_to_85C_s` uses single thermal time constant.** Real 3D-DRAM has
  multi-mode transients; M3 should capture per-layer dynamics.
- **MockCostEvaluator numbers are NOT meaningful.** Smoke test only.
- **Pareto over (lat, eng, sram) only.** Thermal is a LABEL, not an axis
  (see week_2_design.md for rationale).
- **Tile/split enumeration is sampled** (max ~5 per dim). True exhaustive
  over all divisors is ~10× larger search space; sampled set has produced
  no major misses on synthetic data but should be revisited if Fig.9
  ratios miss paper claim.
- **D³ search uses max(compute, mem) for latency** via cost model's
  current convention. Week 1 M4 milestone replaces with pipelining; this
  may shift Pareto fronts and require library rebuild.

## Carries from Week 1 (not addressed in Week 2)

- SA peak 838 vs paper 400 TFLOPS (2.1× off) — M9 calibration
- DRAM timing mostly `est:M9` (only `tRFI_lowT` is JEDEC truth)
- No NoC contention term — Week 3 fixes for TP≥8
- Energy = `e_mem + e_comp` (sum not aligned; small error) — M8
