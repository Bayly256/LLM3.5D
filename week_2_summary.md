# Week 2 Summary

_(Fill in numbers in italicized sections after running with the real Week 1
evaluator. Sections with checkboxes self-fill from `tests/test_week2.py` and
`lib.stats()`.)_

## Built

Thermal-labeled mapping library — Innovation #1 deliverable.

- 5 cost_model modules: schema, D³ search, thermal proxy, Pareto, baselines
- 3 driver scripts: build / baselines / Fig.9 plot
- 9-test acceptance suite — _9/9 passing on MockCostEvaluator_

### LOC summary

```
$ wc -l cost_model/{mapping_lib,d3_search,thermal_proxy,pareto,baselines}.py \
        scripts/{build_mapping_lib,run_baselines,fig9_repro}.py \
        tests/test_week2.py
```

## Acceptance checklist

- [x] `tests/test_week2.py` 9/9 (mock)
- [ ] Schema round-trip survives pkl + parquet
- [ ] DC ΔT > PC ΔT for same power _(observed: PC=__°C, DC=__°C)_
- [ ] `time_to_85C_s` reasonable at high power, None at low power
- [ ] Full build runs without errors (real evaluator)
- [ ] Library size ≥ 252 entries _(observed: __)_
- [ ] All Pareto candidates have thermal labels _(`thermal_attached: True`)_
- [ ] ffn_up DC RU flip across batch sizes _(bs=1 → __; bs=16 → __)_
- [ ] Fig.9 D³/baseline EDP ratio at large batch in [1.25, 1.53]× _(observed: __)_

## Library stats (post-build, real evaluator)

```
n_op_entries:              ____
total_candidates_evaluated: ____
total_pareto_candidates:    ____
avg_pareto_size:            ____
total_search_time_s:        ____
thermal_method:             linear_proxy_M2
```

## Key findings — Day 11 Fig.9

### Best-RU table (DC chiplet)

| op_type   | bs=1 | bs=4 | bs=16 |
|-----------|------|------|-------|
| qkv_proj  | ____ | ____ | ____  |
| o_proj    | ____ | ____ | ____  |
| ffn_up    | ____ | ____ | ____  |
| ffn_down  | ____ | ____ | ____  |
| ffn_gate  | ____ | ____ | ____  |

**Expected regime flip:** at small bs the workload is BW-bound (best RU
typically WRU or IRU, depending on shape); at large bs compute-bound, RU
choice shifts. Specifically watch ffn_up DC — paper Fig.9 shows the flip
around bs=8–16.

### EDP comparison at bs=32 (DC, ffn_up)

| Dataflow | EDP (μJ·μs) | Ratio vs D³ |
|----------|-------------|-------------|
| D³ (ours) | ____ | 1.00× |
| TETRIS   | ____ | __× |
| TS       | ____ | __× |
| ARU      | ____ | __× |

Paper claim: D³ achieves 1.25–1.53× lower EDP than baselines at large batch.

### Chiplet comparison (Innovation #2 motivation)

For the same op at the same shape:

| op_type | PC latency (μs) | DC latency (μs) | PC ΔT (°C) | DC ΔT (°C) | Migration insight |
|---------|-----------------|-----------------|-----------|-----------|-------------------|
| ffn_up bs=4 | ____ | ____ | ____ | ____ | ____ |
| ffn_down bs=4 (decode) | ____ | ____ | ____ | ____ | ____ |

## Open items / blocking for Week 3

- [ ] **Real evaluator wired** — Week 1 `compute.py`/`memory.py` integrated
      via `cost_model/real_evaluator.py`. _(blocker for everything else)_
- [ ] **Power decomposition verified** — `p_mpu_W`, `p_dram_W`, `p_noc_W`
      separately. Single `p_total` not enough.
- [ ] **YAML chiplet configs** — replace hardcoded defaults in
      `make_chiplet_configs()` with `configs/PC.yaml` / `DC.yaml` loader.

## Risks flagged for Week 3–4

- **Search time at full sweep.** With real evaluator at ~5ms/eval and
  ~6000 candidates/OpKey, full build is ~15min. If unacceptable, lower
  `--max-tiles` from 5 to 4 (~3× speedup) before going wider.
- **D³ vs baseline EDP gap might be <1.25×** with our cost model. Paper
  uses ATSim (closed-source). If our ratio is too low, primary suspect is
  NoC contention term (Week 1 known issue); secondary is tile enumeration
  granularity.
- **DC ΔT - PC ΔT** in real numbers — need to verify Day 25 (Week 4) sees
  the 15–20°C heterogeneity needed for Innovation #2 motivation. If
  proxy doesn't show it, Day 22–24 HotSpot calibration should fix it
  (real thermal physics > linear proxy).

## Error budget delta (Week 1 → Week 2)

| Source | Magnitude | Status | Next milestone |
|--------|-----------|--------|----------------|
| SA peak 838 vs 400 TFLOPS | 2.1× | open (carry) | M9 freq_MHz calibration |
| memory/compute use max not pipelined | ≤30% | open (carry) | M4 |
| DRAM timing est:M9 | unknown | open (carry) | M9 JEDEC datasheet refresh |
| NoC contention missing | TP≥8 only | open | Week 3 Day 16 |
| **Thermal proxy uncertainty** | **±15°C** | **NEW Week 2** | **Week 4 Day 22–24 (HotSpot)** |
| **Pareto under sampled tiles/splits** | **unknown** | **NEW Week 2** | **Revisit if Fig.9 misses target** |

## Carryover to Week 3 (lock these contracts early)

- `scheduler.dispatch(op, mode, thermal_state) -> mapping` — signature
  must be stable by end of Week 3 Day 19 (stub OK, signature contract).
- Thermal state struct (per-chiplet T, JEDEC mode, runway) — Week 3
  Day 17.
- Mapping query helpers: `lib.best_perf`, `lib.best_under_thermal`,
  `lib.compare_chiplets` — Week 3 simulator's only API into the library.
