"""
Pareto front extraction + mode tagging.

Used by:
- d3_search.py: pareto_filter_3d called after candidate enumeration
- scripts/build_mapping_lib.py: tag_modes called after Pareto extraction

The dedup step is important under Week 1 cost model: many (cores_M, cores_N)
splits collapse to identical (latency, energy, sram) coordinates because the
adapter's per_core_shape caps to within-head budget. Without dedup the Pareto
front gets bloated 5-10×.
"""

from __future__ import annotations


def _vals(c, axes):
    return tuple(getattr(c, ax) for ax in axes)


def dedup_candidates(
    cands: list,
    axes: tuple = ("latency_us", "energy_uJ", "sram_bytes"),
) -> list:
    """Keep only the first candidate for each unique (axes,) tuple.

    Preserves order of first appearance — typically the "simpler" mapping
    (smaller cores_M, etc.) ranks first since enumerate_core_splits emits in
    ascending cM order.
    """
    seen = {}
    for c in cands:
        key = _vals(c, axes)
        if key not in seen:
            seen[key] = c
    return list(seen.values())


def pareto_filter_3d(
    cands: list,
    axes: tuple = ("latency_us", "energy_uJ", "sram_bytes"),
    dedup_first: bool = True,
) -> list:
    """Return Pareto-optimal candidates on given axes (all minimized).

    Sort-then-sweep implementation. Strictly dominated points are dropped;
    duplicates on all axes are kept as one via dedup_first.
    """
    if not cands:
        return []
    if dedup_first:
        cands = dedup_candidates(cands, axes)

    n_ax = len(axes)
    decorated = [(_vals(c, axes), i) for i, c in enumerate(cands)]
    decorated.sort()

    kept_vals = []
    kept_idx = []
    for vals, idx in decorated:
        dominated = False
        for prev in kept_vals:
            ge = all(prev[k] <= vals[k] for k in range(n_ax))
            gt = any(prev[k] < vals[k] for k in range(n_ax))
            if ge and gt:
                dominated = True
                break
        if not dominated:
            # Remove kept points dominated by the new one
            new_kept_vals, new_kept_idx = [], []
            for v, i_ in zip(kept_vals, kept_idx):
                dom_by_new = (all(vals[k] <= v[k] for k in range(n_ax))
                              and any(vals[k] < v[k] for k in range(n_ax)))
                if not dom_by_new:
                    new_kept_vals.append(v)
                    new_kept_idx.append(i_)
            kept_vals = new_kept_vals + [vals]
            kept_idx = new_kept_idx + [idx]

    return [cands[i] for i in kept_idx]


def mark_pareto_ranks(
    cands: list,
    axes: tuple = ("latency_us", "energy_uJ", "sram_bytes"),
) -> None:
    """In-place: assign pareto_rank = 0 for front, 1 for next layer, etc."""
    remaining = list(cands)
    rank = 0
    while remaining:
        front = pareto_filter_3d(remaining, axes, dedup_first=False)
        front_ids = {id(c) for c in front}
        for c in front:
            c.pareto_rank = rank
        remaining = [c for c in remaining if id(c) not in front_ids]
        rank += 1


def tag_modes(pareto: list) -> None:
    """Annotate mode_tags on each Pareto candidate.
    Tags used by Innovation #2 scheduler:
        perf_optimal     — lowest latency
        energy_optimal   — lowest energy
        thermal_optimal  — lowest ΔT_steady (requires thermal labels)
        balanced_knee    — closest to origin in (lat_norm, eng_norm)
    """
    if not pareto:
        return

    perf_best = min(pareto, key=lambda c: c.latency_us)
    perf_best.mode_tags.add("perf_optimal")

    energy_best = min(pareto, key=lambda c: c.energy_uJ)
    energy_best.mode_tags.add("energy_optimal")

    labeled = [c for c in pareto if c.thermal is not None]
    if labeled:
        thermal_best = min(labeled, key=lambda c: c.thermal.delta_T_steady_C)
        thermal_best.mode_tags.add("thermal_optimal")

    lat_max = max(c.latency_us for c in pareto)
    eng_max = max(c.energy_uJ for c in pareto)
    if lat_max > 0 and eng_max > 0:
        knee = min(
            pareto,
            key=lambda c: (c.latency_us / lat_max) ** 2 + (c.energy_uJ / eng_max) ** 2,
        )
        knee.mode_tags.add("balanced_knee")
