"""Additional actual DFT force checks along each applied field direction.

Existing labels remain immutable; the audit is a separate evaluation artifact.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from generate_ho_dft import BOHR, calculate
from pyscf import lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/designed_pbe0")
    parser.add_argument("--output", default="outputs/designed_reference_audit")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    lib.num_threads(args.threads)
    root = Path(args.root)
    jobs = json.loads((root / "jobs.json").read_text())
    checks = []
    eps = 2e-4
    for index, job in enumerate(jobs):
        coordinates = job["design_coordinates"]
        selections = []
        if np.any(job["external_field_au"]):
            selections = [(1, int(np.argmax(abs(np.array(job["external_field_au"])))))]
        elif index == 13:
            selections = [(1, axis) for axis in range(3)]
        elif (
            job["group"] == "symmetric_zundel_surface"
            and coordinates["oo_angstrom"] == 2.6
            and abs(coordinates["proton_distance_difference_angstrom"]) in (0, 0.4)
        ):
            selections = [(2, 0)]
        if not selections:
            continue
        with np.load(root / f"{index:04d}.npz", allow_pickle=False) as data:
            force = data["forces"].copy()
        for atom, axis in selections:
            plus, minus = json.loads(json.dumps(job)), json.loads(json.dumps(job))
            plus["positions_angstrom"][atom][axis] += eps * BOHR
            minus["positions_angstrom"][atom][axis] -= eps * BOHR
            fd = -(calculate(plus)[2] - calculate(minus)[2]) / (2 * eps)
            checks.append(
                {
                    "index": index,
                    "family": job["group"],
                    "atom": atom,
                    "axis": axis,
                    "field_au": job["external_field_au"],
                    "epsilon_bohr": eps,
                    "analytic_force_au": float(force[atom, axis]),
                    "fd_force_au": fd,
                    "absolute_error_au": float(abs(fd - force[atom, axis])),
                }
            )
            print(json.dumps(checks[-1]), flush=True)
    report = {
        "method": "PBE0/aug-cc-pVDZ",
        "checks": checks,
        "maximum_error_hartree_per_bohr": max(r["absolute_error_au"] for r in checks),
        "threshold_hartree_per_bohr": 2e-5,
        "passed": all(r["absolute_error_au"] <= 2e-5 for r in checks),
        "scope": "Finite-difference samples of actual DFT forces; not model accuracy",
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    if not report["passed"]:
        raise RuntimeError("DFT force audit failed")


if __name__ == "__main__":
    main()
