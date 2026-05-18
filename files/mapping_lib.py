"""
Thermal-labeled mapping library schema.

Core deliverable of Innovation #1. Holds the Pareto front of intra-PE
mappings (over latency, energy, SRAM) per (op, chiplet) combination, with
thermal ΔT labels attached out-of-band so the thermal model can swap
M2→M3→M4 without rerunning D³ search.

Schema decisions:
- Thermal is a LABEL, not a Pareto axis.
- Power is decomposed (p_mpu, p_dram, p_noc) so the thermal model can treat
  logic and DRAM layers separately.
- chiplet_type, phase, n_heads, and batch_size are all in OpKey: scheduler
  queries need to disambiguate prefill vs decode and per-head attention.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Iterator
from pathlib import Path
import hashlib
import pickle


@dataclass(frozen=True)
class OpKey:
    """Library lookup key. frozen=True so it's dict-hashable.

    Fields match Week 1's LUT key tuple structure plus n_heads:
        (model_name, phase, op_type, bs, sl) → unique GEMM via (M, N, K, n_heads)
    """
    op_type: str       # 'qkv_proj','attn_qk','attn_av','o_proj','ffn_up','ffn_down','ffn_gate'
    M: int
    N: int
    K: int
    n_heads: int       # 1 for linear layers; h_q for attention
    dtype: str         # 'fp16'
    chiplet_type: str  # 'PC' | 'DC'
    phase: str         # 'prefill' | 'decode'
    batch_size: int

    def canonical_str(self) -> str:
        h = f"_h{self.n_heads}" if self.n_heads > 1 else ""
        return (
            f"{self.op_type}_{self.M}x{self.N}x{self.K}{h}_"
            f"{self.dtype}_{self.chiplet_type}_{self.phase}_bs{self.batch_size}"
        )


@dataclass
class ThermalLabel:
    """Lazily attached thermal annotation."""
    delta_T_steady_C: float
    delta_T_peak_C: float
    time_to_85C_s: Optional[float] = None
    method: str = "linear_proxy_M2"
    ambient_C: float = 45.0
    neighbor_assumption: str = "isolated"
    uncertainty_C: float = 15.0


@dataclass
class MappingCandidate:
    """One feasible mapping with cost, power decomposition, optional thermal."""
    mapping_id: str

    # 2D core split within chiplet
    cores_M: int = 1
    cores_N: int = 1
    cores_K: int = 1

    # Per-PE D³ params (Algorithm 1 of paper)
    T_M: int = 0
    T_N: int = 0
    T_K: int = 0
    RU: str = "WRU"
    loop_order: str = "MNK"

    # Pareto axes
    latency_us: float = 0.0
    energy_uJ: float = 0.0
    sram_bytes: int = 0

    # Power decomposition
    p_mpu_W: float = 0.0
    p_vpu_W: float = 0.0
    p_dram_W: float = 0.0
    p_noc_W: float = 0.0
    p_avg_W: float = 0.0
    p_peak_W: float = 0.0

    # Thermal label (None until attach_thermal is called)
    thermal: Optional[ThermalLabel] = None

    # Regime / debug fields
    is_compute_bound: bool = False
    bw_utilization: float = 0.0
    mpu_utilization: float = 0.0
    sa_active_rows_frac: float = 1.0

    # Pareto / mode metadata
    pareto_rank: int = -1
    mode_tags: set = field(default_factory=set)

    # Provenance: field name -> list of source tags
    provenance: dict = field(default_factory=dict)
    cost_model_version: str = "M2"

    @staticmethod
    def make_id(cores_M, cores_N, cores_K, T_M, T_N, T_K, RU, loop_order) -> str:
        payload = f"{cores_M}-{cores_N}-{cores_K}-{T_M}-{T_N}-{T_K}-{RU}-{loop_order}"
        return hashlib.md5(payload.encode()).hexdigest()[:12]


@dataclass
class SearchStats:
    n_evaluated: int = 0
    n_feasible: int = 0
    n_pruned_sram: int = 0
    n_pareto: int = 0
    search_time_s: float = 0.0


@dataclass
class OpEntry:
    """Pareto front + search statistics for one OpKey."""
    op_key: OpKey
    pareto_front: list  # list[MappingCandidate]
    stats: SearchStats = field(default_factory=SearchStats)

    def best_perf(self) -> Optional[MappingCandidate]:
        if not self.pareto_front:
            return None
        return min(self.pareto_front, key=lambda c: c.latency_us)

    def best_energy(self) -> Optional[MappingCandidate]:
        if not self.pareto_front:
            return None
        return min(self.pareto_front, key=lambda c: c.energy_uJ)

    def best_under_thermal(self, delta_T_budget_C: float) -> Optional[MappingCandidate]:
        feasible = [c for c in self.pareto_front
                    if c.thermal is not None and c.thermal.delta_T_steady_C <= delta_T_budget_C]
        if not feasible:
            return None
        return min(feasible, key=lambda c: c.latency_us)

    def best_thermal(self) -> Optional[MappingCandidate]:
        labeled = [c for c in self.pareto_front if c.thermal is not None]
        if not labeled:
            return None
        return min(labeled, key=lambda c: c.thermal.delta_T_steady_C)


@dataclass
class MappingLibrary:
    """Top-level container: dict of OpKey -> OpEntry."""
    version: str
    cost_model_version: str = "M2"
    thermal_method: Optional[str] = None
    entries: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def lookup(self, op_key: OpKey) -> Optional[OpEntry]:
        return self.entries.get(op_key)

    def best_perf(self, op_key: OpKey) -> Optional[MappingCandidate]:
        entry = self.lookup(op_key)
        return entry.best_perf() if entry else None

    def best_under_thermal(self, op_key: OpKey, delta_T_budget_C: float) -> Optional[MappingCandidate]:
        entry = self.lookup(op_key)
        return entry.best_under_thermal(delta_T_budget_C) if entry else None

    def compare_chiplets(self, op_type: str, M: int, N: int, K: int,
                         dtype: str, phase: str, batch_size: int,
                         n_heads: int = 1) -> dict:
        """Migration-decision query: returns {'PC': best_mapping, 'DC': best_mapping}."""
        result = {}
        for cht in ("PC", "DC"):
            k = OpKey(op_type=op_type, M=M, N=N, K=K, n_heads=n_heads,
                      dtype=dtype, chiplet_type=cht, phase=phase, batch_size=batch_size)
            cand = self.best_perf(k)
            if cand is not None:
                result[cht] = cand
        return result

    def attach_thermal(self, thermal_model) -> None:
        for entry in self.entries.values():
            for cand in entry.pareto_front:
                cand.thermal = thermal_model.compute_label(cand, entry.op_key.chiplet_type)
        self.thermal_method = thermal_model.method_tag()

    def query_fig9(self, op_type: str, chiplet_type: str,
                   phase: str = "prefill", dtype: str = "fp16"):
        """DataFrame of (batch_size, best_RU, EDP) for one op."""
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas required for query_fig9")
        rows = []
        for k, entry in self.entries.items():
            if (k.op_type != op_type or k.chiplet_type != chiplet_type
                    or k.phase != phase or k.dtype != dtype):
                continue
            best = entry.best_perf()
            if best is None:
                continue
            rows.append({
                "op_type": k.op_type, "chiplet": k.chiplet_type, "phase": k.phase,
                "batch_size": k.batch_size, "M": k.M, "N": k.N, "K": k.K,
                "best_RU": best.RU, "best_loop_order": best.loop_order,
                "latency_us": best.latency_us, "energy_uJ": best.energy_uJ,
                "EDP": best.latency_us * best.energy_uJ,
                "is_compute_bound": best.is_compute_bound,
                "bw_util": best.bw_utilization, "mpu_util": best.mpu_utilization,
                "delta_T_C": best.thermal.delta_T_steady_C if best.thermal else None,
            })
        return pd.DataFrame(rows).sort_values(["batch_size"])

    def stats(self) -> dict:
        total_eval = sum(e.stats.n_evaluated for e in self.entries.values())
        total_pareto = sum(len(e.pareto_front) for e in self.entries.values())
        total_t = sum(e.stats.search_time_s for e in self.entries.values())
        return {
            "n_op_entries": len(self.entries),
            "total_candidates_evaluated": total_eval,
            "total_pareto_candidates": total_pareto,
            "avg_pareto_size": total_pareto / max(len(self.entries), 1),
            "total_search_time_s": total_t,
            "thermal_attached": self.thermal_method is not None,
            "thermal_method": self.thermal_method,
            "schema_version": self.version,
            "cost_model_version": self.cost_model_version,
        }

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "MappingLibrary":
        with open(path, "rb") as f:
            lib = pickle.load(f)
        assert isinstance(lib, MappingLibrary), f"Loaded object is not a MappingLibrary: {type(lib)}"
        return lib

    def export_parquet(self, path) -> None:
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas required for export_parquet")
        rows = []
        for op_key, entry in self.entries.items():
            for cand in entry.pareto_front:
                rows.append({
                    "op_type": op_key.op_type,
                    "M": op_key.M, "N": op_key.N, "K": op_key.K,
                    "n_heads": op_key.n_heads,
                    "dtype": op_key.dtype, "chiplet_type": op_key.chiplet_type,
                    "phase": op_key.phase, "batch_size": op_key.batch_size,
                    "mapping_id": cand.mapping_id,
                    "cores_M": cand.cores_M, "cores_N": cand.cores_N, "cores_K": cand.cores_K,
                    "T_M": cand.T_M, "T_N": cand.T_N, "T_K": cand.T_K,
                    "RU": cand.RU, "loop_order": cand.loop_order,
                    "latency_us": cand.latency_us, "energy_uJ": cand.energy_uJ,
                    "sram_bytes": cand.sram_bytes,
                    "p_mpu_W": cand.p_mpu_W, "p_vpu_W": cand.p_vpu_W,
                    "p_dram_W": cand.p_dram_W, "p_noc_W": cand.p_noc_W,
                    "p_avg_W": cand.p_avg_W, "p_peak_W": cand.p_peak_W,
                    "bw_utilization": cand.bw_utilization,
                    "mpu_utilization": cand.mpu_utilization,
                    "sa_active_rows_frac": cand.sa_active_rows_frac,
                    "is_compute_bound": cand.is_compute_bound,
                    "delta_T_steady_C": cand.thermal.delta_T_steady_C if cand.thermal else None,
                    "delta_T_peak_C": cand.thermal.delta_T_peak_C if cand.thermal else None,
                    "time_to_85C_s": cand.thermal.time_to_85C_s if cand.thermal else None,
                    "thermal_method": cand.thermal.method if cand.thermal else None,
                    "mode_tags": ",".join(sorted(cand.mode_tags)) if cand.mode_tags else "",
                    "pareto_rank": cand.pareto_rank,
                })
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        return df

    def __iter__(self) -> Iterator:
        return iter(self.entries.items())

    def __len__(self) -> int:
        return len(self.entries)
