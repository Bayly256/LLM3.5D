"""
Day 11: Reproduce paper Fig.9 — dataflow EDP comparison vs batch_size.

Caveat: under Week 1 cost model (no tile-aware reuse), the EDP gap between D³
and baselines collapses to ~1×. Paper's 1.25-1.53× requires tile-aware reuse
which is a Week 2.5 / M5 cost-model upgrade. This script still produces the
plot and reports the actual ratios honestly.
"""

from __future__ import annotations
import sys
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cost_model"))

from mapping_lib import MappingLibrary


def collect_edp_rows(lib, dataflow_name, op_types, chiplet, phase, batch_sizes):
    rows = []
    for bs in batch_sizes:
        for op_type in op_types:
            best_lat = None; best_eng = None; best_ru = None; best_cb = None
            for op_key, entry in lib.entries.items():
                if (op_key.op_type == op_type
                        and op_key.chiplet_type == chiplet
                        and op_key.phase == phase
                        and op_key.batch_size == bs):
                    best = entry.best_perf()
                    if best is None:
                        continue
                    if best_lat is None or best.latency_us < best_lat:
                        best_lat = best.latency_us
                        best_eng = best.energy_uJ
                        best_ru = best.RU
                        best_cb = best.is_compute_bound
            if best_lat is None:
                continue
            rows.append({
                "dataflow": dataflow_name, "op_type": op_type, "chiplet": chiplet,
                "phase": phase, "batch_size": bs,
                "latency_us": best_lat, "energy_uJ": best_eng,
                "EDP": best_lat * best_eng, "best_RU": best_ru,
                "is_compute_bound": best_cb,
            })
    return rows


def plot_fig9(
    lib_d3_path=None, lib_baselines_path=None, output_dir=None,
    op_types=("ffn_up", "ffn_down", "qkv_proj", "o_proj"),
    chiplet="DC", phase="prefill", batch_sizes=(1, 4, 16, 32),
):
    lib_d3_path = lib_d3_path or str(ROOT / "cost_model" / "mapping_library_v0.1.pkl")
    lib_baselines_path = lib_baselines_path or str(ROOT / "cost_model" / "mapping_library_baselines.pkl")
    output_dir = Path(output_dir or (ROOT / "figs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pandas required.")
        return

    try:
        lib_d3 = MappingLibrary.load(lib_d3_path)
        print(f"Loaded D³ library: {lib_d3.stats()['n_op_entries']} entries")
    except FileNotFoundError:
        print(f"[ERROR] {lib_d3_path} not found. Run build_mapping_lib.py first.")
        return

    try:
        with open(lib_baselines_path, "rb") as f:
            lib_baselines = pickle.load(f)
        print(f"Loaded baselines: {list(lib_baselines.keys())}")
    except FileNotFoundError:
        print(f"[WARN] {lib_baselines_path} not found.")
        lib_baselines = {}

    all_rows = collect_edp_rows(lib_d3, "D³ (ours)", op_types, chiplet, phase, batch_sizes)
    for bl_name, bl_lib in lib_baselines.items():
        all_rows.extend(collect_edp_rows(bl_lib, bl_name, op_types, chiplet, phase, batch_sizes))

    df = pd.DataFrame(all_rows)
    if df.empty:
        print(f"[ERROR] No data collected. Check that {phase} entries exist in lib.")
        return

    csv_path = output_dir / f"fig9_ru_table_{chiplet}_{phase}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved table: {csv_path}")
    print(f"\nBest-RU summary (chiplet={chiplet}, phase={phase}):")
    pivot = df[df["dataflow"] == "D³ (ours)"].pivot_table(
        index="op_type", columns="batch_size", values="best_RU", aggfunc="first"
    )
    print(pivot.to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping plot")
        return

    n_ops = len(op_types)
    fig, axes = plt.subplots(1, n_ops, figsize=(4 * n_ops, 4), sharey=False)
    if n_ops == 1:
        axes = [axes]

    dataflow_order = ["D³ (ours)"] + [k for k in lib_baselines.keys()]
    colors = {"D³ (ours)": "C3", "TETRIS": "C0", "TS": "C2", "ARU": "C1"}
    markers = {"D³ (ours)": "o", "TETRIS": "s", "TS": "^", "ARU": "D"}

    for ax, op_type in zip(axes, op_types):
        sub = df[(df["op_type"] == op_type) & (df["chiplet"] == chiplet) & (df["phase"] == phase)]
        if sub.empty:
            ax.set_title(f"{op_type}: no data"); continue

        d3 = sub[sub["dataflow"] == "D³ (ours)"].set_index("batch_size")["EDP"]
        for dataflow in dataflow_order:
            if dataflow not in sub["dataflow"].values:
                continue
            data = sub[sub["dataflow"] == dataflow].set_index("batch_size")["EDP"]
            norm = data / d3.reindex(data.index)
            ax.plot(norm.index, norm.values,
                    marker=markers.get(dataflow, "o"),
                    color=colors.get(dataflow, "gray"),
                    label=dataflow, linewidth=2, markersize=8)
        ax.set_xscale("log", base=2)
        ax.set_title(f"{op_type} ({chiplet}, {phase})")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("EDP / EDP(D³)")
        ax.axhline(1.0, linestyle="--", color="gray", alpha=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"Fig.9 reproduction: dataflow EDP vs batch_size "
                 f"({chiplet}, {phase}; under Week 1 cost model)", fontsize=11)
    fig.tight_layout()
    png_path = output_dir / f"fig9_repro_{chiplet}_{phase}.png"
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {png_path}")

    print(f"\n=== Acceptance check ===")
    print(f"Paper claim: D³ 1.25-1.53× lower EDP vs baselines at large batch")
    print(f"Under Week 1 cost model: tile-aware reuse is missing, so ratios should be ~1.0×.")
    for op_type in op_types:
        d3_max_bs = df[(df["dataflow"] == "D³ (ours)") & (df["op_type"] == op_type)]
        if d3_max_bs.empty:
            continue
        max_bs = d3_max_bs["batch_size"].max()
        d3_edp = d3_max_bs[d3_max_bs["batch_size"] == max_bs]["EDP"].values
        if len(d3_edp) == 0:
            continue
        d3_edp = d3_edp[0]
        for bl in dataflow_order[1:]:
            bl_row = df[(df["dataflow"] == bl) & (df["op_type"] == op_type)
                        & (df["batch_size"] == max_bs)]
            if not bl_row.empty:
                bl_edp = bl_row["EDP"].values[0]
                ratio = bl_edp / d3_edp
                marker = (" ✓ matches paper-like" if 1.20 <= ratio <= 1.60
                          else " ~ expected under Wk1 cost model" if 0.95 <= ratio <= 1.05
                          else " ? unexpected")
                print(f"  {op_type} bs={max_bs}: {bl}/D³ EDP = {ratio:.3f}×{marker}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--chiplet", default="DC", choices=["PC", "DC"])
    p.add_argument("--phase", default="prefill", choices=["prefill", "decode"])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 16, 32])
    args = p.parse_args()
    plot_fig9(chiplet=args.chiplet, phase=args.phase,
              batch_sizes=tuple(args.batch_sizes))
