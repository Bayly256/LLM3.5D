"""
Configuration loader for LaMoSys3.5D simulator.

Loads chiplet (PC, DC) and system (NoP, package, thermal) parameters from
YAML and converts them into the dataclasses used by the cost model.

Usage:
    from configs import load_chiplet, load_system

    pc  = load_chiplet('PC')           # returns ChipletConfig
    dc  = load_chiplet('DC')
    sys = load_system()                # returns SystemConfig

The flat ChipletConfig schema in memory.py is preserved (so existing code
continues to work). YAML sections {dram, timing_ns, bandwidth, compute,
energy, budget, noc} are flattened into the dataclass fields at load time.

Author: <you>
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yaml

# Import path setup: configs/ is a sibling of cost_model/
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(_ROOT, 'cost_model'))

from memory import ChipletConfig  # noqa: E402


CONFIG_DIR = _THIS_DIR


# ---------------------------------------------------------------------------
# Chiplet loader
# ---------------------------------------------------------------------------

def load_chiplet(name: str, config_dir: str = CONFIG_DIR) -> ChipletConfig:
    """Load a chiplet config from YAML.

    `name` is one of 'PC', 'DC' (matches the YAML filename without extension).
    Raises FileNotFoundError if the YAML is missing.
    """
    path = os.path.join(config_dir, f"{name}.yaml")
    with open(path) as f:
        raw = yaml.safe_load(f)

    return ChipletConfig(
        name=raw['name'],
        # DRAM
        n_layers=raw['dram']['n_layers'],
        capacity_GB=raw['dram']['capacity_GB'],
        n_banks=raw['dram']['n_banks'],
        io_width=raw['dram']['io_width'],
        page_size=raw['dram']['page_size'],
        BL=raw['dram']['BL'],
        # Timing
        tRCD=raw['timing_ns']['tRCD'],
        tCAS=raw['timing_ns']['tCAS'],
        tRP=raw['timing_ns']['tRP'],
        tRFC=raw['timing_ns']['tRFC'],
        tRFI_lowT=raw['timing_ns']['tRFI_lowT'],
        t_tsv=raw['timing_ns']['t_tsv'],
        # Bandwidth
        peak_bw_TBs=raw['bandwidth']['peak_TBs'],
        # Compute
        n_PE=raw['compute']['n_PE'],
        n_cores_per_PE=raw['compute']['n_cores_per_PE'],
        freq_MHz=raw['compute']['freq_MHz'],
        SA_rows=raw['compute']['SA_rows'],
        SA_cols=raw['compute']['SA_cols'],
        baseSA_rows=raw['compute']['baseSA_rows'],
        SRAM_per_core_KB=raw['compute']['SRAM_per_core_KB'],
        peak_TFLOPS=raw['compute']['peak_TFLOPS'],
        # Energy
        dram_energy_pJ_bit=raw['energy']['dram_pJ_per_bit'],
        sram_energy_pJ_bit=raw['energy']['sram_pJ_per_bit'],
        mac_energy_pJ_op=raw['energy']['mac_pJ_per_op'],
        # Budget
        peak_power_W=raw['budget']['peak_power_W'],
        area_mm2=raw['budget']['area_mm2'],
        # NoC
        noc_bw_GBs=raw['noc']['bw_GBs'],
    )


# ---------------------------------------------------------------------------
# System loader
# ---------------------------------------------------------------------------

@dataclass
class NoPConfig:
    bw_GBs: float
    per_hop_ns: float
    pJ_per_bit: float
    topology: str


@dataclass
class PackageConfig:
    n_PC_chiplets: int
    n_DC_chiplets: int
    grid_shape: Tuple[int, int]


@dataclass
class ThermalConfig:
    T_ambient_C: float
    T_nominal_C: float
    T_jedec_C: float
    T_critical_C: float
    hotspot_floorplan: Optional[str] = None
    thermal_coupling: Optional[List[List[float]]] = None


@dataclass
class SystemConfig:
    package: PackageConfig
    nop: NoPConfig
    thermal: ThermalConfig


def load_system(path: Optional[str] = None) -> SystemConfig:
    if path is None:
        path = os.path.join(CONFIG_DIR, 'system.yaml')
    with open(path) as f:
        raw = yaml.safe_load(f)

    return SystemConfig(
        package=PackageConfig(
            n_PC_chiplets=raw['package']['n_PC_chiplets'],
            n_DC_chiplets=raw['package']['n_DC_chiplets'],
            grid_shape=tuple(raw['package']['grid_shape']),
        ),
        nop=NoPConfig(
            bw_GBs=raw['nop']['bw_GBs'],
            per_hop_ns=raw['nop']['per_hop_ns'],
            pJ_per_bit=raw['nop']['pJ_per_bit'],
            topology=raw['nop']['topology'],
        ),
        thermal=ThermalConfig(
            T_ambient_C=raw['thermal']['T_ambient_C'],
            T_nominal_C=raw['thermal']['T_nominal_C'],
            T_jedec_C=raw['thermal']['T_jedec_C'],
            T_critical_C=raw['thermal']['T_critical_C'],
            hotspot_floorplan=raw['thermal'].get('hotspot_floorplan'),
            thermal_coupling=raw['thermal'].get('thermal_coupling'),
        ),
    )


# ---------------------------------------------------------------------------
# Validation against memory.py hardcoded baseline
# ---------------------------------------------------------------------------

def _check_parity_with_hardcoded() -> None:
    """Verify YAML loads match the hardcoded PC_CONFIG/DC_CONFIG in memory.py.

    This catches drift: if someone edits the YAML, this fails until they also
    update memory.py (or remove the hardcoded versions).
    """
    from memory import PC_CONFIG, DC_CONFIG

    pc = load_chiplet('PC')
    dc = load_chiplet('DC')

    fields_to_check = [
        'name', 'n_layers', 'capacity_GB', 'n_banks', 'io_width',
        'tRCD', 'tCAS', 'tRP', 'tRFC', 'tRFI_lowT',
        'peak_bw_TBs', 'n_PE', 'n_cores_per_PE',
        'SA_rows', 'SA_cols', 'baseSA_rows', 'SRAM_per_core_KB',
        'peak_TFLOPS', 'peak_power_W', 'area_mm2',
        'dram_energy_pJ_bit', 'sram_energy_pJ_bit', 'mac_energy_pJ_op',
        'noc_bw_GBs',
    ]

    print("Parity check: YAML vs memory.py hardcoded")
    print("-" * 60)
    all_ok = True
    for hardcoded, loaded, label in [(PC_CONFIG, pc, 'PC'), (DC_CONFIG, dc, 'DC')]:
        mismatches = []
        for f in fields_to_check:
            hv = getattr(hardcoded, f)
            lv = getattr(loaded, f)
            if hv != lv:
                mismatches.append((f, hv, lv))
        if mismatches:
            all_ok = False
            print(f"  {label}: ✗ {len(mismatches)} mismatch(es)")
            for f, hv, lv in mismatches:
                print(f"    {f:20s} hardcoded={hv!r:>12} yaml={lv!r:>12}")
        else:
            print(f"  {label}: ✓ all {len(fields_to_check)} fields match")
    print()
    if not all_ok:
        raise AssertionError("YAML and hardcoded configs disagree")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _summary_table(pc: ChipletConfig, dc: ChipletConfig) -> None:
    print("Chiplet parameter summary")
    print("=" * 68)
    rows = [
        ('DRAM layers',      f"{pc.n_layers}",                 f"{dc.n_layers}"),
        ('Capacity (GB)',    f"{pc.capacity_GB}",              f"{dc.capacity_GB}"),
        ('Peak BW (TB/s)',   f"{pc.peak_bw_TBs}",              f"{dc.peak_bw_TBs}"),
        ('PE × cores',       f"{pc.n_PE} × {pc.n_cores_per_PE}", f"{dc.n_PE} × {dc.n_cores_per_PE}"),
        ('Total cores',      f"{pc.n_PE * pc.n_cores_per_PE}", f"{dc.n_PE * dc.n_cores_per_PE}"),
        ('SA shape',         f"{pc.SA_rows}×{pc.SA_cols}",      f"{dc.SA_rows}×{dc.SA_cols}"),
        ('SRAM/core (KB)',   f"{pc.SRAM_per_core_KB}",          f"{dc.SRAM_per_core_KB}"),
        ('Peak TFLOPS',      f"{pc.peak_TFLOPS}",               f"{dc.peak_TFLOPS}"),
        ('Power (W)',        f"{pc.peak_power_W}",              f"{dc.peak_power_W}"),
        ('Area (mm²)',       f"{pc.area_mm2}",                  f"{dc.area_mm2}"),
        ('NoC BW (GB/s)',    f"{pc.noc_bw_GBs}",                f"{dc.noc_bw_GBs}"),
    ]
    print(f"  {'parameter':22s} {'PC':>16s} {'DC':>16s}   {'ratio (DC/PC)':>14s}")
    print("  " + "-" * 64)
    for label, pv, dv in rows:
        # try to compute a numeric ratio if both sides are numbers
        try:
            r = float(dv) / float(pv) if float(pv) else float('inf')
            r_str = f"{r:.2f}×"
        except ValueError:
            r_str = "—"
        print(f"  {label:22s} {pv:>16s} {dv:>16s}   {r_str:>14s}")
    print()


def _system_table(sysc: SystemConfig) -> None:
    print("System / package parameters")
    print("=" * 68)
    p = sysc.package
    print(f"  package chiplets:  {p.n_PC_chiplets} PC + {p.n_DC_chiplets} DC  "
          f"on a {p.grid_shape[0]}×{p.grid_shape[1]} grid")
    n = sysc.nop
    print(f"  NoP:               {n.bw_GBs} GB/s, {n.per_hop_ns} ns/hop, "
          f"{n.pJ_per_bit} pJ/bit, {n.topology}")
    t = sysc.thermal
    print(f"  thermal window:    ambient={t.T_ambient_C}°C  "
          f"nominal={t.T_nominal_C}°C  jedec={t.T_jedec_C}°C  "
          f"critical={t.T_critical_C}°C")
    print()


if __name__ == "__main__":
    pc = load_chiplet('PC')
    dc = load_chiplet('DC')
    sysc = load_system()

    _summary_table(pc, dc)
    _system_table(sysc)
    _check_parity_with_hardcoded()

    # Sanity check: feed loaded configs into the cost model to confirm
    # they produce the same numbers as the hardcoded versions.
    from memory import memory_access_latency_ns, effective_bandwidth_TBs
    print("Cost-model sanity check")
    print("=" * 68)
    for cfg in (pc, dc):
        bw65 = effective_bandwidth_TBs(cfg, 65)
        bw105 = effective_bandwidth_TBs(cfg, 105)
        drop = (1 - bw105 / bw65) * 100
        print(f"  {cfg.name}:  BW@65°C = {bw65:5.2f} TB/s   "
              f"BW@105°C = {bw105:5.2f} TB/s   ({drop:4.1f}% drop)")
    print("\n✓ Day 1 config loader operational.")
