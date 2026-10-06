"""Post-hoc geometric observations with explicit, non-physical thresholds."""

import numpy as np

from ..datasets import BOHR_ANGSTROM


def observe_protons(numbers, positions, *, previous=None):
    z = np.asarray(numbers)
    r = np.asarray(positions)
    oxygen = np.where(z == 8)[0]
    hydrogen = np.where(z == 1)[0]
    delta = r[hydrogen, None] - r[oxygen][None]
    d = np.linalg.norm(delta, axis=-1) * BOHR_ANGSTROM
    owners = oxygen[d.argmin(1)]
    nearest = d.min(1)
    counts = np.array([(owners == i).sum() for i in oxygen])
    ambiguous = np.sort(d, axis=1)[:, 1] - nearest < 0.15
    transfer = []
    if previous is not None:
        for h, old, new, uncertain in zip(hydrogen, previous["proton_owners"], owners, ambiguous):
            if old != new and not uncertain:
                transfer.append(
                    {"hydrogen_id": int(h), "from_oxygen": int(old), "to_oxygen": int(new)}
                )
    return {
        "oxygen_ids": oxygen.tolist(),
        "proton_owners": owners.tolist(),
        "oxygen_coordination": counts.tolist(),
        "ionic_pair_candidate": bool((counts == 1).any() and (counts == 3).any()),
        "ambiguous_protons": int(ambiguous.sum()),
        "max_nearest_oh_angstrom": float(nearest.max()),
        "proton_owner_changes": transfer,
        "interpretation": "geometric candidate only; requires electronic charge/density and persistence validation",
    }


def summarize_trajectories(numbers, trajectories):
    result = []
    for trajectory in trajectories:
        old = None
        observations = []
        changes = 0
        formed = 0
        recombined = 0
        for r in trajectory:
            current = observe_protons(numbers, r, previous=old)
            changes += len(current["proton_owner_changes"])
            if old:
                formed += int(current["ionic_pair_candidate"] and not old["ionic_pair_candidate"])
                recombined += int(
                    old["ionic_pair_candidate"] and not current["ionic_pair_candidate"]
                )
            observations.append(current)
            old = current
        result.append(
            {
                "candidate_formations": formed,
                "candidate_recombinations": recombined,
                "proton_owner_changes": changes,
                "frames": len(observations),
                "observations": observations,
            }
        )
    return result
