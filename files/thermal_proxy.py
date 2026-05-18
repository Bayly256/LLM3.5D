"""
M2 placeholder thermal proxy.

    ΔT_steady = P_logic · R_th_logic + P_dram · (R_th_logic + R_th_dram_per_layer · N_layers)

PC (4 DRAM layers) sees lower R_th_dram_eff than DC (8 layers), so for the
same total power, DC's ΔT is higher. Matches paper Fig.11 qualitatively.

Acknowledged uncertainty: ±15°C. Day 22-24 (Week 4) replaces with HotSpot.
"""

from __future__ import annotations
import math
from typing import Optional

from mapping_lib import MappingCandidate, ThermalLabel


class LinearThermalProxy:
    def __init__(
        self,
        ambient_C: float = 45.0,
        R_th_logic_K_per_W: float = 0.45,
        R_th_dram_per_layer_K_per_W: float = 0.06,
        thermal_time_const_s: float = 0.20,
        uncertainty_C: float = 15.0,
    ):
        self.ambient_C = ambient_C
        self.R_th_logic = R_th_logic_K_per_W
        self.R_th_dram_per_layer = R_th_dram_per_layer_K_per_W
        self.tau_s = thermal_time_const_s
        self.uncertainty_C = uncertainty_C

    def method_tag(self) -> str:
        return "linear_proxy_M2"

    @staticmethod
    def chiplet_layers(chiplet_type: str) -> int:
        return 4 if chiplet_type == "PC" else 8

    def compute_label(
        self, cand: MappingCandidate, chiplet_type: str
    ) -> ThermalLabel:
        n_layers = self.chiplet_layers(chiplet_type)
        p_logic = cand.p_mpu_W + cand.p_vpu_W + cand.p_noc_W
        p_dram = cand.p_dram_W

        R_th_dram_eff = self.R_th_dram_per_layer * n_layers
        delta_T_steady = (
            p_logic * self.R_th_logic
            + p_dram * (self.R_th_logic + R_th_dram_eff)
        )

        if cand.p_avg_W > 0:
            peak_scale = cand.p_peak_W / cand.p_avg_W
        else:
            peak_scale = 1.0
        delta_T_peak = delta_T_steady * peak_scale

        threshold = 85.0
        if self.ambient_C + delta_T_steady <= threshold:
            time_to_85 = None
        else:
            need_rise = threshold - self.ambient_C
            ratio = need_rise / delta_T_steady
            if 0 < ratio < 1:
                time_to_85 = -self.tau_s * math.log(1 - ratio)
            else:
                time_to_85 = None

        return ThermalLabel(
            delta_T_steady_C=delta_T_steady,
            delta_T_peak_C=delta_T_peak,
            time_to_85C_s=time_to_85,
            method=self.method_tag(),
            ambient_C=self.ambient_C,
            neighbor_assumption="isolated",
            uncertainty_C=self.uncertainty_C,
        )
