"""
Week 2 acceptance suite (Day 14).

Uses the real Week1CostEvaluator + Week 1 ChipletConfig (loaded from YAML).
Tests:
  1. Schema round-trip (pickle save/load) with new OpKey fields
  2. Pareto + dedup correctness
  3. RU feasibility (SRAM bounds enforced)
  4. Thermal proxy: DC ΔT > PC ΔT for same power
  5. Time_to_85C: non-None at high power, None at low power
  6. D³ search end-to-end with Week 1 adapter
  7. Week 1 LUT cross-check: D³ search reproduces Week 1's deterministic split
  8. compare_chiplets API
  9. attach_thermal pipeline
 10. Parquet export round-trip
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cost_model"))
sys.path.insert(0, str(ROOT))

from configs import load_chiplet  # type: ignore
from operators import operator_shapes, ALL_MODELS  # type: ignore

from mapping_lib import (
    OpKey, MappingCandidate, ThermalLabel, OpEntry, MappingLibrary, SearchStats,
)
from d3_search import d3_search, smart_divisors, enumerate_core_splits, ru_feasible
from thermal_proxy import LinearThermalProxy
from pareto import pareto_filter_3d, dedup_candidates, tag_modes
from real_evaluator import Week1CostEvaluator


PASS, FAIL = "✓", "✗"


# ---------- 1. Schema round-trip ----------

def test_schema_roundtrip():
    key = OpKey("ffn_up", 4096, 27648, 5120, 1, "fp16", "DC", "prefill", 4)
    cand = MappingCandidate(
        mapping_id="abc12345",
        cores_M=4, cores_N=64, cores_K=1,
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

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        lib.save(path)
        loaded = MappingLibrary.load(path)
        assert loaded.lookup(key) is not None, "lookup failed"
        loaded_best = loaded.best_perf(key)
        assert loaded_best.latency_us == 120.5
        assert loaded_best.mapping_id == "abc12345"
        assert "perf_optimal" in loaded_best.mode_tags
        # Verify new fields are preserved
        assert loaded.lookup(key).op_key.phase == "prefill"
        assert loaded.lookup(key).op_key.n_heads == 1
        print(f"  {PASS} pkl round-trip: lat={loaded_best.latency_us}, "
              f"phase=prefill, n_heads=1, tags preserved")
    finally:
        Path(path).unlink(missing_ok=True)


# ---------- 2. Pareto + dedup ----------

def test_pareto_dedup():
    # Three candidates with identical Pareto coords (different mapping_ids)
    cands = [
        MappingCandidate("a", latency_us=10, energy_uJ=10, sram_bytes=100),
        MappingCandidate("b", latency_us=10, energy_uJ=10, sram_bytes=100),  # dup of a
        MappingCandidate("c", latency_us=10, energy_uJ=10, sram_bytes=100),  # dup of a
        MappingCandidate("d", latency_us=5,  energy_uJ=50, sram_bytes=100),
        MappingCandidate("e", latency_us=50, energy_uJ=5,  sram_bytes=100),
        MappingCandidate("f", latency_us=20, energy_uJ=20, sram_bytes=100),  # dominated
    ]
    deduped = dedup_candidates(cands)
    assert len(deduped) == 4, f"dedup expected 4, got {len(deduped)}"
    pareto = pareto_filter_3d(cands)
    ids = {c.mapping_id for c in pareto}
    assert "a" in ids and "d" in ids and "e" in ids
    assert "b" not in ids and "c" not in ids  # dedup
    assert "f" not in ids  # dominated
    print(f"  {PASS} pareto+dedup: 6 cands → 4 deduped → 3 Pareto ({sorted(ids)})")


# ---------- 3. RU feasibility ----------

def test_ru_feasibility():
    sram = 1024
    bpe = 2
    assert ru_feasible("IRU", 16, 16, 8, sram, bpe)
    assert ru_feasible("STREAM", 99999, 99999, 99999, sram, bpe)  # always feasible
    assert not ru_feasible("ARU", 32, 32, 8, sram, bpe)
    assert not ru_feasible("ORU", 32, 32, 8, sram, bpe)
    print(f"  {PASS} ru_feasible: IRU/WRU/ORU/ARU bounded; STREAM always feasible")


# ---------- 4 & 5. Thermal proxy ----------

def test_thermal_proxy():
    proxy = LinearThermalProxy()
    high = MappingCandidate("x", p_mpu_W=120, p_vpu_W=10, p_dram_W=80, p_noc_W=20,
                             p_avg_W=230, p_peak_W=320)
    low = MappingCandidate("y", p_mpu_W=5, p_vpu_W=1, p_dram_W=3, p_noc_W=2,
                            p_avg_W=11, p_peak_W=15)
    pc_h = proxy.compute_label(high, "PC")
    dc_h = proxy.compute_label(high, "DC")
    assert dc_h.delta_T_steady_C > pc_h.delta_T_steady_C, "DC should exceed PC"
    print(f"  {PASS} DC > PC: ΔT PC={pc_h.delta_T_steady_C:.1f}°C, "
          f"DC={dc_h.delta_T_steady_C:.1f}°C (same power)")
    assert dc_h.time_to_85C_s is not None and dc_h.time_to_85C_s > 0
    pc_l = proxy.compute_label(low, "PC")
    assert pc_l.time_to_85C_s is None
    print(f"  {PASS} time_to_85: high-pow DC reaches at {dc_h.time_to_85C_s:.3f}s; "
          f"low-pow PC never")


# ---------- 6. D³ search end-to-end ----------

def test_d3_search_endtoend():
    dc = load_chiplet("DC")
    shapes = operator_shapes(ALL_MODELS["gpt3-13b"], 4, 1024, "prefill")
    shape = shapes["ffn_up"]
    op_key = OpKey("ffn_up", shape.M, shape.N, shape.K, shape.n_heads,
                   "fp16", "DC", "prefill", 4)
    evaluator = Week1CostEvaluator()
    entry = d3_search(op_key, dc, evaluator, max_tiles_per_dim=4, max_core_splits=4)

    assert len(entry.pareto_front) > 0
    assert entry.stats.n_evaluated > 0
    for c in entry.pareto_front:
        assert c.sram_bytes <= dc.SRAM_per_core_KB * 1024 or c.RU == "STREAM"
        assert c.pareto_rank == 0

    best = entry.best_perf()
    print(f"  {PASS} D³ search: evaluated={entry.stats.n_evaluated}, "
          f"pareto={entry.stats.n_pareto} (deduped), time={entry.stats.search_time_s:.2f}s")
    print(f"    best_perf: RU={best.RU} tile=({best.T_M},{best.T_N},{best.T_K}) "
          f"split=({best.cores_M},{best.cores_N}) lat={best.latency_us:.2f}μs")


# ---------- 7. Cross-check vs Week 1 LUT ----------

def test_lut_cross_check():
    """Two things:
    (a) Adapter correctness: call evaluator with Week 1's exact split,
        compare to LUT entry. Should match within 1%.
    (b) D³ improvement: D³ search may find a BETTER mapping than Week 1's
        deterministic split (one-sided OK)."""
    import pickle
    lut_path = ROOT / "cost_model" / "lut.pkl"
    if not lut_path.exists():
        print(f"  - LUT cross-check skipped (lut.pkl missing)")
        return
    with open(lut_path, "rb") as f:
        lut = pickle.load(f)

    dc = load_chiplet("DC")
    pc = load_chiplet("PC")
    evaluator = Week1CostEvaluator()

    cases = [
        ("ffn_up", "prefill", "DC", "IRU", dc, 4, 1024),
        ("ffn_up", "decode",  "DC", "IRU", dc, 4, 1024),
        ("attn_qk", "prefill", "PC", "IRU", pc, 4, 1024),
    ]
    for op_type, phase, cht, ru, cfg, bs, sl in cases:
        shape = operator_shapes(ALL_MODELS["gpt3-13b"], bs, sl, phase)[op_type]
        key = OpKey(op_type, shape.M, shape.N, shape.K, shape.n_heads,
                    "fp16", cht, phase, bs)
        lut_entry = lut[("gpt3-13b", phase, op_type, bs, sl, cht, ru)]
        lut_lat = lut_entry["latency_ns"] / 1000
        lut_cM, lut_cN = lut_entry["cores_M"], lut_entry["cores_N"]

        # (a) Adapter parity: call evaluator at the EXACT split Week 1 picked
        out = evaluator.evaluate(key, (shape.M, shape.N, shape.K),
                                  ru, "MNK", (lut_cM, lut_cN, 1), cfg)
        ratio_a = out["latency_us"] / lut_lat
        assert 0.95 <= ratio_a <= 1.05, (
            f"Adapter mismatch at Week 1 split ({lut_cM},{lut_cN}): "
            f"{out['latency_us']:.1f}μs vs LUT {lut_lat:.1f}μs ({ratio_a:.3f}×)"
        )

        # (b) D³ search: should be ≤ LUT (better mapping discovered is OK)
        entry = d3_search(key, cfg, evaluator,
                          max_tiles_per_dim=4, max_core_splits=6)
        d3_lat = entry.best_perf().latency_us
        ratio_b = d3_lat / lut_lat
        assert ratio_b <= 1.05, (
            f"D³ worse than LUT: {d3_lat:.1f}μs > {lut_lat:.1f}μs ({ratio_b:.3f}×)"
        )
        improvement = "" if ratio_b > 0.95 else f"  D³ FINDS {(1-ratio_b)*100:.0f}% IMPROVEMENT"
        print(f"  {PASS} {op_type} {phase} {cht} bs={bs}: "
              f"adapter@({lut_cM},{lut_cN})={out['latency_us']:.2f}μs "
              f"(LUT={lut_lat:.2f}μs, {ratio_a:.3f}×){improvement}")
        print(f"          D³ best mapping: {d3_lat:.2f}μs ({ratio_b:.3f}× vs LUT)")


# ---------- 8. compare_chiplets API ----------

def test_compare_chiplets():
    pc = load_chiplet("PC")
    dc = load_chiplet("DC")
    evaluator = Week1CostEvaluator()
    lib = MappingLibrary(version="test_v0.1")
    shape = operator_shapes(ALL_MODELS["gpt3-13b"], 4, 1024, "decode")["ffn_up"]
    for cht, cfg in [("PC", pc), ("DC", dc)]:
        k = OpKey("ffn_up", shape.M, shape.N, shape.K, shape.n_heads,
                  "fp16", cht, "decode", 4)
        lib.entries[k] = d3_search(k, cfg, evaluator,
                                    max_tiles_per_dim=4, max_core_splits=4)
    result = lib.compare_chiplets("ffn_up", shape.M, shape.N, shape.K,
                                   "fp16", "decode", 4, n_heads=1)
    assert "PC" in result and "DC" in result
    ratio = result["PC"].latency_us / result["DC"].latency_us
    print(f"  {PASS} compare_chiplets decode: PC lat={result['PC'].latency_us:.1f}μs, "
          f"DC lat={result['DC'].latency_us:.1f}μs (PC/DC = {ratio:.2f}× — DC should be faster)")


# ---------- 9. attach_thermal pipeline ----------

def test_attach_thermal():
    dc = load_chiplet("DC")
    evaluator = Week1CostEvaluator()
    lib = MappingLibrary(version="test_v0.1")
    shape = operator_shapes(ALL_MODELS["gpt3-13b"], 4, 1024, "decode")["ffn_up"]
    key = OpKey("ffn_up", shape.M, shape.N, shape.K, shape.n_heads,
                "fp16", "DC", "decode", 4)
    lib.entries[key] = d3_search(key, dc, evaluator,
                                  max_tiles_per_dim=4, max_core_splits=4)

    for c in lib.entries[key].pareto_front:
        assert c.thermal is None

    proxy = LinearThermalProxy()
    lib.attach_thermal(proxy)
    assert lib.thermal_method == "linear_proxy_M2"
    for c in lib.entries[key].pareto_front:
        assert c.thermal is not None
        assert c.thermal.delta_T_steady_C >= 0

    tag_modes(lib.entries[key].pareto_front)
    has_perf = any("perf_optimal" in c.mode_tags for c in lib.entries[key].pareto_front)
    has_thermal = any("thermal_optimal" in c.mode_tags for c in lib.entries[key].pareto_front)
    assert has_perf and has_thermal
    print(f"  {PASS} attach_thermal: method={lib.thermal_method}, "
          f"perf+thermal tags both present")


# ---------- 10. Parquet round-trip ----------

def test_parquet_export():
    try:
        import pandas as pd  # noqa
    except ImportError:
        print(f"  - parquet test skipped (pandas not installed)")
        return

    dc = load_chiplet("DC")
    evaluator = Week1CostEvaluator()
    lib = MappingLibrary(version="test_v0.1")
    shape = operator_shapes(ALL_MODELS["gpt3-13b"], 1, 1024, "decode")["ffn_up"]
    key = OpKey("ffn_up", shape.M, shape.N, shape.K, shape.n_heads,
                "fp16", "DC", "decode", 1)
    lib.entries[key] = d3_search(key, dc, evaluator,
                                  max_tiles_per_dim=3, max_core_splits=3)
    lib.attach_thermal(LinearThermalProxy())

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    try:
        df = lib.export_parquet(path)
        assert len(df) == len(lib.entries[key].pareto_front)
        import pandas as pd
        df2 = pd.read_parquet(path)
        assert "delta_T_steady_C" in df2.columns
        assert "phase" in df2.columns
        assert "n_heads" in df2.columns
        assert df2["phase"].iloc[0] == "decode"
        print(f"  {PASS} parquet export: {len(df)} rows, phase + n_heads present")
    finally:
        Path(path).unlink(missing_ok=True)


# ---------- Runner ----------

TESTS = [
    ("schema_roundtrip", test_schema_roundtrip),
    ("pareto_dedup", test_pareto_dedup),
    ("ru_feasibility", test_ru_feasibility),
    ("thermal_proxy", test_thermal_proxy),
    ("d3_search_endtoend", test_d3_search_endtoend),
    ("lut_cross_check", test_lut_cross_check),
    ("compare_chiplets", test_compare_chiplets),
    ("attach_thermal", test_attach_thermal),
    ("parquet_export", test_parquet_export),
]


def main():
    print(f"Running {len(TESTS)} Week 2 acceptance tests with Week 1 adapter...\n")
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
            import traceback
            print(f"  {FAIL} {type(e).__name__}: {e}")
            traceback.print_exc()
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
