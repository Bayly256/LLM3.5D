"""
Pareto front extraction + mode tagging.

Used by:
- d3_search.py: pareto_filter_3d called after candidate enumeration
- scripts/build_mapping_lib.py: tag_modes called after Pareto extraction
- Innovation #2 scheduler (Week 3+): queries mode_tags directly
"""

from __future__ import annotations


def pareto_filter_3d(
    cands: list,
    axes: tuple = ("latency_us", "energy_uJ", "sram_bytes"),
) -> list:
    """Return Pareto-optimal candidates on given axes (all minimized).

    Implementation: sort by first axis ascending, then sweep; maintain a list
    of running minimums for the other axes. A point is Pareto-optimal iff no
    earlier point dominates it on ALL other axes. O(N²) worst-case but with
    a much smaller constant than naive double loop, and typically O(N·k) in
    practice (k = Pareto front size).
    """
    if not cands:
        return []

    n_ax = len(axes)
    # Decorate: (axis_values_tuple, index)
    decorated = [(tuple(getattr(c, ax) for ax in axes), i) for i, c in enumerate(cands)]
    decorated.sort()  # lexicographic by axes — primary key is axes[0]

    pareto_indices = []
    kept = []  # list of axis-tuples for Pareto candidates so far

    for vals, idx in decorated:
        dominated = False
        for prev in kept:
            # prev is on the Pareto front *up to here*; check if it dominates vals
            ge = all(prev[k] <= vals[k] for k in range(n_ax))
            gt = any(prev[k] < vals[k] for k in range(n_ax))
            if ge and gt:
                dominated = True
                break
        if not dominated:
            # vals could also dominate previously-kept points if equal on axis[0]
            # Remove kept points dominated by vals
            kept = [
                p for p in kept
                if not (all(vals[k] <= p[k] for k in range(n_ax))
                        and any(vals[k] < p[k] for k in range(n_ax)))
            ]
            kept.append(vals)
            pareto_indices.append(idx)

    # Filter the kept indices to those actually still on the front
    # (kept and pareto_indices may have diverged due to dominated-removal)
    kept_set = set(kept)
    return [cands[idx] for vals, idx in
            [(tuple(getattr(c, ax) for ax in axes), i) for i, c in enumerate(cands)]
            if vals in kept_set]


def mark_pareto_ranks(
    cands: list,
    axes: tuple = ("latency_us", "energy_uJ", "sram_bytes"),
) -> None:
    """In-place: assign pareto_rank = 0 for front, 1 for next layer, etc.
    Useful when keeping dominated points for Week 4 motivation experiments."""
    remaining = list(cands)
    rank = 0
    while remaining:
        front = pareto_filter_3d(remaining, axes)
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
