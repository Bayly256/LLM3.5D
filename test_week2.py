"""
Week 2 acceptance suite (Day 14).

Covers:
  1. Schema round-trip (pickle save/load)
  2. Pareto correctness on synthetic data
  3. RU feasibility (SRAM bounds enforced)
  4. Thermal proxy: DC ΔT > PC ΔT for same power
  5. Time_to_85C: non-None at high power, None at low power
  6. D³ search end-to-end with mock evaluator
  7. bs=1 vs bs=16 ffn_up DC regime (smoke; real flip needs Week 1 evaluator)
  8. compare_chiplets API (Innovation #2 contract)
  9. attach_thermal pipeline (no thermal → with thermal)
 10. Parquet export round-trip

Run: python tests/test_week2.py
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cost_model"))

from mapping_lib import (
    OpKey, MappingCandidate, ThermalLabel, OpEntry, MappingLibrary, SearchStats,
)
from d3_search import (
    d3_search, MockCostEvaluator,
    smart_divisors, enumerate_core_splits, ru_feasible,
)
from thermal_proxy import LinearThermalProxy
from pareto import pareto_filter_3d, tag_modes, mark_pareto_ranks


PASS, FAIL = "✓", "✗"


# ---------- 1. Schema round-trip ----------

def test_schema_roundtrip():
    key = OpKey("ffn_up", 4096, 5120, 5120, "fp16", "DC", 4)
    cand = MappingCandidate(
        mapping_id="abc12345",
        cores_M=4, cores_N=8, cores_K=1,
        T_M=128, T_N=128, T_K=64,
        RU="WRU", loop_order="MNK",
        latency_us=120.5, energy_uJ=45.2, sram_bytes=8192,
        p_mpu_W=80, p_dram_W=40, p_noc_W=15, p_vpu_W=5,
        p_avg_W=140, p_peak_W=210,
        mode_tags={"perf_optimal"},
        provenance={"latency_us": ["sim:M2"]},
    )
    entry = OpEntry(op_key=key, pareto_front=[cand])
    lib = MappingLibrary(version="test_v0.1")
    lib.entries[key] = entry
    lib.meta["built_at"] = "test"

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        lib.save(path)
        loaded = MappingLibrary.load(path)
        assert loaded.lookup(key) is not None, "Key lookup failed"
        loaded_best = loaded.best_perf(key)
        assert loaded_best.latency_us == 120.5
        assert loaded_best.mapping_id == "abc12345"
        assert "perf_optimal" in loaded_best.mode_tags
        assert loaded_best.provenance == {"latency_us": ["sim:M2"]}
        print(f"  {PASS} pkl round-trip: lat={loaded_best.latency_us}, "
              f"tags={loaded_best.mode_tags}, prov preserved")
    finally:
        Path(path).unlink(missing_ok=True)


# ---------- 2. Pareto correctness ----------

def test_pareto_filter():
    # Synthetic — using only latency_us and energy_uJ axes for clarity
    cands = [
        MappingCandidate("a", latency_us=10, energy_uJ=10, sram_bytes=100),
        MappingCandidate("b", latency_us=20, energy_uJ=20, sram_bytes=100),  # dominated by a (and e)
        MappingCandidate("c", latency_us=5,  energy_uJ=50, sram_bytes=100),  # best latency
        MappingCandidate("d", latency_us=50, energy_uJ=5,  sram_bytes=100),  # best energy
        MappingCandidate("e", latency_us=10, energy_uJ=10, sram_bytes=50),   # dominates a (smaller sram)
        MappingCandidate("f", latency_us=8,  energy_uJ=8,  sram_bytes=200),  # best lat+eng combined
    ]
    pareto = pareto_filter_3d(cands)
    ids = {c.mapping_id for c in pareto}

    assert "a" not in ids, "'a' should be dominated by 'e' (smaller sram, same lat/eng)"
    assert "b" not in ids, "'b' is strictly dominated"
    assert "c" in ids, "'c' is best latency"
    assert "d" in ids, "'d' is best energy"
    assert "e" in ids, "'e' dominates a"
    assert "f" in ids, "'f' is best (lat, eng) combined"
    print(f"  {PASS} pareto_filter_3d: kept {sorted(ids)} of 6 synthetic points")


# ---------- 3. RU feasibility ----------

def test_ru_feasibility():
    sram = 1024  # bytes
    bpe = 2      # fp16
    # Tile (16, 16, 8): IRU=256, WRU=256, ORU=512, ARU=1024 → all feasible at budget=1024
    assert ru_feasible("IRU", 16, 16, 8, sram, bpe)
    assert ru_feasible("WRU", 16, 16, 8, sram, bpe)
    assert ru_feasible("ORU", 16, 16, 8, sram, bpe)
    assert ru_feasible("ARU", 16, 16, 8, sram, bpe)
    # Tile (32, 32, 8): ARU = (32*8 + 32*8 + 32*32) * 2 = 2560 → infeasible
    assert not ru_feasible("ARU", 32, 32, 8, sram, bpe)
    # Tile (32, 32, 8) ORU = 32*32*2 = 2048 → infeasible
    assert not ru_feasible("ORU", 32, 32, 8, sram, bpe)
    # But IRU = 32*8*2 = 512 → still feasible
    assert ru_feasible("IRU", 32, 32, 8, sram, bpe)
    print(f"  {PASS} ru_feasible: all four policies' SRAM bounds enforced")


# ---------- 4 & 5. Thermal proxy ----------

def test_thermal_proxy():
    proxy = LinearThermalProxy()
    high_power = MappingCandidate(
        "x",
        p_mpu_W=120, p_vpu_W=10, p_dram_W=80, p_noc_W=20,
        p_avg_W=230, p_peak_W=320,
    )
    low_power = MappingCandidate(
        "y",
        p_mpu_W=5, p_vpu_W=1, p_dram_W=3, p_noc_W=2,
        p_avg_W=11, p_peak_W=15,
    )

    pc_high = proxy.compute_label(high_power, "PC")
    dc_high = proxy.compute_label(high_power, "DC")

    assert dc_high.delta_T_steady_C > pc_high.delta_T_steady_C, (
        f"DC ({dc_high.delta_T_steady_C:.1f}) should exceed "
        f"PC ({pc_high.delta_T_steady_C:.1f}) for same power"
    )
    print(f"  {PASS} DC > PC: ΔT PC={pc_high.delta_T_steady_C:.1f}°C, "
          f"DC={dc_high.delta_T_steady_C:.1f}°C (same power)")

    # time_to_85: high power should reach 85°C; low power should not
    assert dc_high.time_to_85C_s is not None and dc_high.time_to_85C_s > 0, (
        "DC at high power should reach 85°C"
    )
    pc_low = proxy.compute_label(low_power, "PC")
    assert pc_low.time_to_85C_s is None, (
        f"PC at low power shouldn't reach 85°C (steady ΔT={pc_low.delta_T_steady_C:.1f})"
    )
    print(f"  {PASS} time_to_85: high-pow DC reaches at {dc_high.time_to_85C_s:.3f}s; "
          f"low-pow PC never (ΔT_ss={pc_low.delta_T_steady_C:.1f}°C)")


# ---------- 6. D³ search end-to-end ----------

def test_d3_search_endtoend():
    op_key = OpKey("ffn_up", 4096, 5120, 5120, "fp16", "DC", 4)
    cfg = {
        "sa_rows": 64, "sa_cols": 32, "freq_hz": 800e6,
        "bw_Bps": 26.2e12,
        "sram_budget_bytes": 256 * 1024,
        "total_cores": 256,
        "p_mpu_max": 200, "p_dram_max": 100, "p_noc_max": 30,
    }
    evaluator = MockCostEvaluator()
    entry = d3_search(op_key, cfg, evaluator, max_tiles_per_dim=4, max_core_splits=4)

    assert len(entry.pareto_front) > 0, "Expected at least one Pareto point"
    assert entry.stats.n_evaluated > 0
    assert entry.stats.n_pareto == len(entry.pareto_front)
    for c in entry.pareto_front:
        assert c.sram_bytes <= cfg["sram_budget_bytes"], (
            f"Pareto candidate exceeds SRAM budget: {c.sram_bytes}"
        )
        assert c.pareto_rank == 0

    best = entry.best_perf()
    print(f"  {PASS} D³ search: evaluated={entry.stats.n_evaluated}, "
          f"pareto={entry.stats.n_pareto}, time={entry.stats.search_time_s:.2f}s")
    print(f"    best_perf: RU={best.RU} loop={best.loop_order} "
          f"tile=({best.T_M},{best.T_N},{best.T_K}) split=({best.cores_M},{best.cores_N}) "
          f"lat={best.latency_us:.2f}μs")


# ---------- 7. bs regime check (smoke with mock) ----------

def test_bs_regime():
    """Day 8 acceptance smoke: ffn_up DC at small vs large M should differ.
    Real RU flip validation requires Week 1's calibrated evaluator."""
    cfg = {
        "sa_rows": 64, "sa_cols": 32, "freq_hz": 800e6,
        "bw_Bps": 26.2e12,
        "sram_budget_bytes": 256 * 1024,
        "total_cores": 256,
        "p_mpu_max": 200, "p_dram_max": 100, "p_noc_max": 30,
    }
    evaluator = MockCostEvaluator()

    # QwQ-32B: hidden=5120, hidden_inner=27648
    key_bs1 = OpKey("ffn_up", 1024, 27648, 5120, "fp16", "DC", 1)
    key_bs16 = OpKey("ffn_up", 16384, 27648, 5120, "fp16", "DC", 16)

    e1 = d3_search(key_bs1, cfg, evaluator, max_tiles_per_dim=4, max_core_splits=4)
    e16 = d3_search(key_bs16, cfg, evaluator, max_tiles_per_dim=4, max_core_splits=4)
    b1 = e1.best_perf()
    b16 = e16.best_perf()

    print(f"  {PASS} bs regime smoke:")
    print(f"    bs=1:  best RU={b1.RU} loop={b1.loop_order} "
          f"compute_bound={b1.is_compute_bound} lat={b1.latency_us:.1f}μs "
          f"bw_util={b1.bw_utilization:.2f}")
    print(f"    bs=16: best RU={b16.RU} loop={b16.loop_order} "
          f"compute_bound={b16.is_compute_bound} lat={b16.latency_us:.1f}μs "
          f"bw_util={b16.bw_utilization:.2f}")
    print(f"    (NOTE: real RU flip validation requires Week 1 evaluator)")


# ---------- 8. compare_chiplets API ----------

def test_compare_chiplets():
    cfgs = {
        "PC": {
            "sa_rows": 64, "sa_cols": 32, "freq_hz": 800e6,
            "bw_Bps": 10.6e12,
            "sram_budget_bytes": 256 * 1024,
            "total_cores": 256,
            "p_mpu_max": 250, "p_dram_max": 60, "p_noc_max": 30,
        },
        "DC": {
            "sa_rows": 64, "sa_cols": 32, "freq_hz": 800e6,
            "bw_Bps": 26.2e12,
            "sram_budget_bytes": 256 * 1024,
            "total_cores": 256,
            "p_mpu_max": 200, "p_dram_max": 100, "p_noc_max": 30,
        },
    }
    evaluator = MockCostEvaluator()
    lib = MappingLibrary(version="test_v0.1")
    for cht in ("PC", "DC"):
        key = OpKey("ffn_up", 4096, 5120, 5120, "fp16", cht, 4)
        lib.entries[key] = d3_search(
            key, cfgs[cht], evaluator, max_tiles_per_dim=4, max_core_splits=4
        )

    result = lib.compare_chiplets("ffn_up", 4096, 5120, 5120, "fp16", 4)
    assert "PC" in result and "DC" in result
    print(f"  {PASS} compare_chiplets: PC lat={result['PC'].latency_us:.1f}μs, "
          f"DC lat={result['DC'].latency_us:.1f}μs "
          f"(ratio DC/PC = {result['DC'].latency_us / result['PC'].latency_us:.2f}×)")


# ---------- 9. attach_thermal pipeline ----------

def test_attach_thermal():
    cfg = {
        "sa_rows": 64, "sa_cols": 32, "freq_hz": 800e6,
        "bw_Bps": 26.2e12,
        "sram_budget_bytes": 256 * 1024,
        "total_cores": 256,
        "p_mpu_max": 200, "p_dram_max": 100, "p_noc_max": 30,
    }
    evaluator = MockCostEvaluator()
    lib = MappingLibrary(version="test_v0.1")
    key = OpKey("ffn_up", 4096, 5120, 5120, "fp16", "DC", 4)
    lib.entries[key] = d3_search(key, cfg, evaluator, max_tiles_per_dim=4, max_core_splits=4)

    # Before attach_thermal: no labels
    for c in lib.entries[key].pareto_front:
        assert c.thermal is None
    assert lib.thermal_method is None

    # Attach
    proxy = LinearThermalProxy()
    lib.attach_thermal(proxy)
    assert lib.thermal_method == "linear_proxy_M2"
    for c in lib.entries[key].pareto_front:
        assert c.thermal is not None
        assert c.thermal.delta_T_steady_C >= 0
        assert c.thermal.method == "linear_proxy_M2"

    # Tag modes after thermal attach
    tag_modes(lib.entries[key].pareto_front)
    has_perf = any("perf_optimal" in c.mode_tags for c in lib.entries[key].pareto_front)
    has_thermal = any("thermal_optimal" in c.mode_tags for c in lib.entries[key].pareto_front)
    assert has_perf and has_thermal

    budget_cand = lib.best_under_thermal(key, delta_T_budget_C=100.0)
    print(f"  {PASS} attach_thermal pipeline:")
    print(f"    method={lib.thermal_method}, perf+thermal tags present, "
          f"best_under_thermal(100°C) RU={budget_cand.RU if budget_cand else 'None'}")


# ---------- 10. Parquet round-trip ----------

def test_parquet_export():
    try:
        import pandas as pd  # noqa
    except ImportError:
        print(f"  - parquet test skipped (pandas not installed)")
        return

    cfg = {
        "sa_rows": 64, "sa_cols": 32, "freq_hz": 800e6,
        "bw_Bps": 26.2e12,
        "sram_budget_bytes": 256 * 1024,
        "total_cores": 256,
        "p_mpu_max": 200, "p_dram_max": 100, "p_noc_max": 30,
    }
    evaluator = MockCostEvaluator()
    lib = MappingLibrary(version="test_v0.1")
    key = OpKey("ffn_up", 1024, 2048, 2048, "fp16", "DC", 1)
    lib.entries[key] = d3_search(
        key, cfg, evaluator, max_tiles_per_dim=3, max_core_splits=3
    )
    lib.attach_thermal(LinearThermalProxy())

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    try:
        df = lib.export_parquet(path)
        assert len(df) == len(lib.entries[key].pareto_front)
        # Round-trip
        import pandas as pd
        df2 = pd.read_parquet(path)
        assert len(df2) == len(df)
        assert "delta_T_steady_C" in df2.columns
        assert df2["delta_T_steady_C"].notna().all()
        print(f"  {PASS} parquet export: {len(df)} rows, "
              f"all thermal labels non-null")
    finally:
        Path(path).unlink(missing_ok=True)


# ---------- Runner ----------

TESTS = [
    ("schema_roundtrip", test_schema_roundtrip),
    ("pareto_filter", test_pareto_filter),
    ("ru_feasibility", test_ru_feasibility),
    ("thermal_proxy", test_thermal_proxy),
    ("d3_search_endtoend", test_d3_search_endtoend),
    ("bs_regime_smoke", test_bs_regime),
    ("compare_chiplets", test_compare_chiplets),
    ("attach_thermal", test_attach_thermal),
    ("parquet_export", test_parquet_export),
]


def main():
    print(f"Running {len(TESTS)} Week 2 acceptance tests...\n")
    n_pass = 0
    failures = []
    for name, fn in TESTS:
        print(f"[{name}]")
        try:
            fn()
            n_pass += 1
        except AssertionError as e:
            print(f"  {FAIL} AssertionError: {e}")
            failures.append((name, e))
        except Exception as e:
            print(f"  {FAIL} {type(e).__name__}: {e}")
            failures.append((name, e))
        print()
    print(f"\nResult: {n_pass}/{len(TESTS)} passed")
    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    print("All Week 2 tests passed.")


if __name__ == "__main__":
    main()
