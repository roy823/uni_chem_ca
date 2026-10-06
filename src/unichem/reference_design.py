"""Unlabeled, symmetric geometries for the next reference acquisition.

No energies, forces or guessed density labels are generated here. The
manifest cannot be consumed by the training data loader without actual SCF.
"""

import json
from pathlib import Path

import numpy as np


def write_reference_design(output="outputs/reference_design"):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    centre = np.full(3, 14.4 * 0.529177210903)
    jobs = []

    def add(family, numbers, positions, charge, split, field=(0.0, 0.0, 0.0), **coordinates):
        jobs.append(
            {
                "index": len(jobs),
                "family": family,
                "atomic_numbers": numbers,
                "positions_angstrom": (np.array(positions) + centre).tolist(),
                "net_charge": charge,
                "spin_excess": 0,
                "external_field_au": list(field),
                "split": split,
                "coordinates": coordinates,
                "labels_available": False,
                "reference_method": "PBE0/aug-cc-pVDZ",
                "boundary": "isolated",
            }
        )

    for r1 in [0.92, 0.96, 1.00]:
        for r2 in [0.92, 0.96, 1.00]:
            for angle in [100.0, 104.5, 109.0]:
                radians = np.deg2rad(angle)
                add(
                    "water_local_surface",
                    [8, 1, 1],
                    [[0, 0, 0], [r1, 0, 0], [r2 * np.cos(radians), r2 * np.sin(radians), 0]],
                    0,
                    "validation" if angle == 104.5 else "training",
                    r1_angstrom=r1,
                    r2_angstrom=r2,
                    angle_degrees=angle,
                )
    for axis in range(3):
        for strength in [-0.004, -0.002, -0.001, 0.0, 0.001, 0.002, 0.004]:
            field = [0.0, 0.0, 0.0]
            field[axis] = strength
            angle = np.deg2rad(104.5)
            add(
                "water_field_response",
                [8, 1, 1],
                [[0, 0, 0], [0.96, 0, 0], [0.96 * np.cos(angle), 0.96 * np.sin(angle), 0]],
                0,
                "test" if abs(strength) == 0.004 else "training",
                field,
                axis=axis,
                strength_au=strength,
            )
    for distance in np.round(np.arange(2.3, 3.11, 0.1), 1):
        split = (
            "validation"
            if distance in [2.4, 2.8]
            else ("test" if distance in [2.6, 3.0, 3.1] else "training")
        )
        for delta in np.round(np.arange(-0.6, 0.61, 0.1), 1):
            positions = [
                [-distance / 2, 0, 0],
                [distance / 2, 0, 0],
                [delta / 2, 0, 0],
                [-distance / 2 - 0.586, 0.757, 0],
                [-distance / 2 - 0.586, -0.757, 0],
                [distance / 2 + 0.586, 0.757, 0],
                [distance / 2 + 0.586, -0.757, 0],
            ]
            add(
                "symmetric_zundel_surface",
                [8, 8, 1, 1, 1, 1, 1],
                positions,
                1,
                split,
                oo_angstrom=float(distance),
                proton_distance_difference_angstrom=float(delta),
            )
    # Identical zero-field geometries are one physical reference, even when
    # their scan axes differ. Preserve all experimental roles as metadata.
    unique = {}
    for job in jobs:
        key = (
            tuple(job["atomic_numbers"]),
            tuple(np.round(np.array(job["positions_angstrom"]).ravel(), 8)),
            job["net_charge"],
            tuple(job["external_field_au"]),
        )
        if key not in unique:
            job["roles"] = [{"family": job["family"], "coordinates": job["coordinates"]}]
            unique[key] = job
        else:
            unique[key]["roles"].append(
                {"family": job["family"], "coordinates": job["coordinates"]}
            )
    jobs = list(unique.values())
    for index, job in enumerate(jobs):
        job["index"] = index
    manifest = {
        "schema_version": 1,
        "status": "unlabeled_geometry_design",
        "jobs": jobs,
        "training_requirement": "Train a fresh model on this partition; old checkpoint has seen some O-O distances",
        "zundel_holdout": "Entire O-O distance surfaces held out; reflected geometries stay on the same surface split",
        "reference_requirements": [
            "SCF convergence",
            "analytic force check",
            "valence/core convention",
            "field vector and origin",
            "matched density grid",
        ],
        "not_a_training_dataset": True,
    }
    (root / "jobs.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    return {
        "jobs": len(jobs),
        "manifest": str(root / "jobs.json"),
        "labels_available": False,
        "families": sorted({role["family"] for job in jobs for role in job["roles"]}),
    }
