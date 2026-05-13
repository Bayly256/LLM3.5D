"""
3D-DRAM access cost model for LaMoSys3.5D simulator.

Implements the formulas in LaMoSys3.5D §V-C:
    N_cmd = DATA / (N_bank * N_io * BL)
    t_mem = t_RCD + t_CAS + t_RP + BL * N_cmd
    t_refresh ratio = t_RFC / t_RFI, scales with temperature

DRAM refresh follows JEDEC: t_RFI is halved every 10°C above 85°C.

Author: <you>
References:
    - LaMoSys3.5D, arXiv:2512.08731, §V-C
    - JEDEC JESD79-4 (DDR4 refresh spec)
"""
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ChipletConfig:
    """3D-DRAM chiplet configuration.

    All timing values are in ns. Capacity is in GB. Bandwidth in TB/s.
    Energy in pJ/bit (DRAM, SRAM) or pJ/op (MAC).
    """
    name: str

    # --- DRAM stack ---
    n_layers: int
    capacity_GB: int
    n_banks: int            # banks per die
    io_width: int           # bits per bank IO (TSV width)
    page_size: int          # bytes
    BL: int = 8             # burst length

    # --- Timing (ns) — calibrate against silicon in M9 ---
    tRCD: float = 13.0
    tCAS: float = 13.0
    tRP: float = 13.0
    tRFC: float = 350.0
    tRFI_lowT: float = 7800.0   # at T <= 85°C
    t_tsv: float = 1.0          # TSV traversal under layer-interleaving

    # --- Bandwidth target (used as ground truth; paper §VI-A) ---
    peak_bw_TBs: float = 10.6

    # --- Compute ---
    n_PE: int = 16
    n_cores_per_PE: int = 16
    freq_MHz: int = 800
    SA_rows: int = 64
    SA_cols: int = 32
    baseSA_rows: int = 8
    SRAM_per_core_KB: int = 256
    peak_TFLOPS: float = 400.0

    # --- Energy ---
    dram_energy_pJ_bit: float = 0.7
    sram_energy_pJ_bit: float = 0.5
    mac_energy_pJ_op: float = 0.5

    # --- Power / Area ---
    peak_power_W: float = 438
    area_mm2: float = 546

    # --- Interconnect ---
    noc_bw_GBs: float = 200


# Reference configurations from LaMoSys3.5D §VI-A
PC_CONFIG = ChipletConfig(
    name="PC",
    n_layers=4, capacity_GB=32, n_banks=32, io_width=256, page_size=2048,
    peak_bw_TBs=10.6,
    n_PE=16, n_cores_per_PE=16,
    peak_TFLOPS=400, peak_power_W=438, area_mm2=546,
)

DC_CONFIG = ChipletConfig(
    name="DC",
    n_layers=8, capacity_GB=64, n_banks=32, io_width=256, page_size=2048,
    peak_bw_TBs=26.2,
    n_PE=32, n_cores_per_PE=8,
    peak_TFLOPS=400, peak_power_W=638, area_mm2=584,
)


# ---------------------------------------------------------------------------
# Refresh model
# ---------------------------------------------------------------------------

def refresh_interval_ns(cfg: ChipletConfig, temperature_C: float) -> float:
    """tRFI scales with temperature.

    JEDEC: retention halves every 10°C above 85°C, so tRFI halves too.
    Below 85°C: nominal interval.
    """
    if temperature_C <= 85.0:
        return cfg.tRFI_lowT
    return cfg.tRFI_lowT / (2.0 ** ((temperature_C - 85.0) / 10.0))


def refresh_overhead_ratio(cfg: ChipletConfig, temperature_C: float) -> float:
    """Fraction of time the DRAM spends refreshing instead of serving reads."""
    tRFI = refresh_interval_ns(cfg, temperature_C)
    return cfg.tRFC / tRFI


def effective_bandwidth_TBs(cfg: ChipletConfig, temperature_C: float = 65.0) -> float:
    """Useful bandwidth after refresh overhead is subtracted."""
    return cfg.peak_bw_TBs * (1.0 - refresh_overhead_ratio(cfg, temperature_C))


# ---------------------------------------------------------------------------
# Access latency + energy
# ---------------------------------------------------------------------------

def n_commands(data_bytes: float, cfg: ChipletConfig) -> float:
    """N_cmd = DATA / (N_bank * N_io * BL).  Returns float (no rounding)."""
    return (data_bytes * 8.0) / (cfg.n_banks * cfg.io_width * cfg.BL)


def memory_access_latency_ns(data_bytes: float, cfg: ChipletConfig,
                              temperature_C: float = 65.0) -> float:
    """Latency to read/write `data_bytes` from 3D-DRAM, in ns.

    Decomposed as:
        t_open  : fixed row activation overhead, paid once
        t_burst : data_bytes / effective_bandwidth
        t_tsv   : TSV traversal under layer-interleaving

    Note: this is equivalent to the paper formula
        t_mem = tRCD + tCAS + tRP + BL * N_cmd
    where the BL * N_cmd term represents burst time computed
    from aggregate bandwidth.
    """
    t_open = cfg.tRCD + cfg.tCAS + cfg.tRP
    bw_Bps = effective_bandwidth_TBs(cfg, temperature_C) * 1e12
    if bw_Bps <= 0:
        return float("inf")
    t_burst = data_bytes / bw_Bps * 1e9   # bytes / (B/s) → s → ns
    return t_open + t_burst + cfg.t_tsv


def memory_energy_pJ(data_bytes: float, cfg: ChipletConfig,
                      temperature_C: float = 65.0,
                      include_refresh: bool = True) -> float:
    """Energy for moving `data_bytes` through 3D-DRAM, in pJ.

    Dynamic = bits × pJ/bit.
    Refresh = data_bytes × pJ/bit × refresh_overhead_ratio
              (approximation: refresh is proportional to active time)
    """
    bits = data_bytes * 8.0
    e_dyn = bits * cfg.dram_energy_pJ_bit
    if not include_refresh:
        return e_dyn
    e_refresh = e_dyn * refresh_overhead_ratio(cfg, temperature_C)
    return e_dyn + e_refresh


# ---------------------------------------------------------------------------
# Derivation helpers (for sanity-checking parameter sweeps in M5+)
# ---------------------------------------------------------------------------

def derive_peak_bw_TBs(cfg: ChipletConfig, tCK_ns: float = 0.4) -> float:
    """Compute aggregate peak BW from low-level params.

    peak_BW = n_layers × n_banks × io_width × (1 / tCK)
    Useful when you want to see if your chiplet sweep is self-consistent.

    HB-DRAM tCK is typically 0.3–0.4 ns (≈2.5–3 Gbps per pin).
    """
    bits_per_cycle = cfg.n_layers * cfg.n_banks * cfg.io_width
    bps = bits_per_cycle / (tCK_ns * 1e-9)
    return bps / 8.0 / 1e12  # bits/s → bytes/s → TB/s


# ---------------------------------------------------------------------------
# Self-tests (run `python memory.py` to validate Day 2 outputs)
# ---------------------------------------------------------------------------

def _run_self_test():
    print("=" * 68)
    print("LaMoSys3.5D memory cost model — sanity checks")
    print("=" * 68)

    for cfg in (PC_CONFIG, DC_CONFIG):
        print(f"\n[{cfg.name}] {cfg.n_layers}L × {cfg.n_banks}B × {cfg.io_width}b")
        print(f"  peak BW (paper):       {cfg.peak_bw_TBs:6.2f} TB/s")
        print(f"  peak BW (derived@0.4ns): {derive_peak_bw_TBs(cfg):6.2f} TB/s")

        # Effective BW at various temperatures
        for T in (25, 65, 85, 95, 105, 115):
            bw = effective_bandwidth_TBs(cfg, T)
            ratio = bw / effective_bandwidth_TBs(cfg, 65)
            print(f"  eff BW @ {T:3d}°C: {bw:6.2f} TB/s  ({100*ratio:5.1f}% of 65°C)")

        # BW drop check vs paper Fig. 3(b)
        drop = 1.0 - effective_bandwidth_TBs(cfg, 105) / effective_bandwidth_TBs(cfg, 65)
        status = "✓" if 0.05 <= drop <= 0.20 else "✗ tune tRFC"
        print(f"  65→105°C BW drop: {100*drop:.1f}%  (paper ~10%)  {status}")

        # Access latency at a few request sizes
        print(f"  Access latency (ns) @ 65°C:")
        for sz in (1024, 64*1024, 1024*1024, 16*1024*1024):
            t = memory_access_latency_ns(sz, cfg, 65)
            e = memory_energy_pJ(sz, cfg, 65)
            print(f"    {sz//1024:6d} KB → {t:8.1f} ns  ({e/1e6:7.2f} μJ)")

    # Cross-check: how long does it take to drain peak BW for 1 GB at 65°C?
    print("\n--- Drain time for 1 GB at 65°C ---")
    for cfg in (PC_CONFIG, DC_CONFIG):
        t = memory_access_latency_ns(1024**3, cfg, 65)
        achieved_BW = 1024**3 / (t * 1e-9) / 1e12
        print(f"  {cfg.name}: {t:8.1f} ns  →  achieved {achieved_BW:5.2f} TB/s")


if __name__ == "__main__":
    _run_self_test()