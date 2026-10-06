"""Statistical estimates only after equilibrium and event-count requirements."""

import numpy as np


def equilibrium_event_statistics(
    ionic_traces,
    dt_fs,
    kT,
    *,
    equilibrated=False,
    kinetic_clock_validated=False,
    min_transitions=20,
    seed=719,
):
    """Bootstrap whole independent trajectories, preserving time correlation.

    The basin definition belongs to the observer. This function never changes
    microscopic trajectories or replaces missing events with pseudocounts.
    """
    traces = [np.asarray(x, dtype=bool) for x in ionic_traces]
    if not traces or any(len(x) < 2 for x in traces):
        return {
            "status": "insufficient_trajectories",
            "free_energy_hartree": None,
            "rates_per_fs": None,
        }
    forward = sum(int((~x[:-1] & x[1:]).sum()) for x in traces)
    reverse = sum(int((x[:-1] & ~x[1:]).sum()) for x in traces)
    if not equilibrated or len(traces) < 3 or min(forward, reverse) < min_transitions:
        return {
            "status": "insufficient_equilibrium_or_transitions",
            "formations": forward,
            "recombinations": reverse,
            "free_energy_hartree": None,
            "rates_per_fs": None,
        }
    fractions = np.array([x.mean() for x in traces])
    fraction = float(fractions.mean())
    if not 0 < fraction < 1:
        return {"status": "unvisited_basin", "free_energy_hartree": None, "rates_per_fs": None}
    free = -kT * np.log(fraction / (1 - fraction))
    rng = np.random.default_rng(seed)
    sampled = fractions[rng.integers(0, len(traces), (2000, len(traces)))].mean(1)
    finite = sampled[(sampled > 0) & (sampled < 1)]
    interval = np.quantile(-kT * np.log(finite / (1 - finite)), [0.025, 0.975]).tolist()
    neutral_time = sum(int((~x[:-1]).sum()) for x in traces) * dt_fs
    ionic_time = sum(int(x[:-1].sum()) for x in traces) * dt_fs
    rates = (
        {"formation": forward / neutral_time, "recombination": reverse / ionic_time}
        if kinetic_clock_validated
        else None
    )
    return {
        "status": "equilibrium_basin_estimate",
        "free_energy_hartree": float(free),
        "free_energy_95_percent_interval_hartree": interval,
        "rates_per_fs": rates,
        "formations": forward,
        "recombinations": reverse,
        "scope": "specified finite system and observer basins; no bulk standard-state conversion",
    }
