"""Independent water geometry and harmonic reference, executed with PySCF.

This evaluation-only reference is never imported into the designed train split.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from generate_ho_dft import BOHR, calculate
from pyscf import lib
from pyscf.hessian.thermo import harmonic_analysis
from scipy.optimize import minimize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/water_dft_reference")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    lib.num_threads(args.threads)
    centre = np.full(3, 14.4)

    def geometry(q):
        radius, angle = q
        directions = np.array(
            [[np.cos(angle / 2), np.sin(angle / 2), 0], [np.cos(angle / 2), -np.sin(angle / 2), 0]]
        )
        positions = np.vstack((centre, centre + radius * directions))
        dr = np.vstack((np.zeros(3), directions))
        dt = (
            radius
            / 2
            * np.array(
                [
                    [0, 0, 0],
                    [-np.sin(angle / 2), np.cos(angle / 2), 0],
                    [-np.sin(angle / 2), -np.cos(angle / 2), 0],
                ]
            )
        )
        job = {
            "numbers": [8, 1, 1],
            "positions_angstrom": (positions * BOHR).tolist(),
            "net_charge": 0,
            "spin_excess": 0,
            "external_field_au": [0, 0, 0],
        }
        return job, dr, dt

    calls = []

    def objective(q):
        job, dr, dt = geometry(q)
        _, _, energy, force, _, _ = calculate(job)
        gradient = -np.array([(force * dr).sum(), (force * dt).sum()])
        calls.append({"energy_hartree": energy, "gradient_norm": float(np.linalg.norm(gradient))})
        return energy, gradient

    result = minimize(
        objective,
        [0.96 / BOHR, np.deg2rad(104.5)],
        jac=True,
        method="BFGS",
        options={"gtol": 2e-7, "maxiter": 40},
    )
    job, _, _ = geometry(result.x)
    mol, mf, energy, force, _, _ = calculate(job)
    if not result.success or abs(force).max() > 2e-5:
        raise RuntimeError(f"Water reference geometry unconverged: {result.message}")
    hessian = mf.Hessian().kernel()
    modes = harmonic_analysis(mol, hessian, mass=np.array([15.999, 1.008, 1.008]))
    frequencies = modes["freq_wavenumber"]
    if np.iscomplexobj(frequencies) or len(frequencies) != 3 or (frequencies <= 0).any():
        raise RuntimeError("Water reference has imaginary or missing vibrational modes")
    rigid_audit = []
    positions = mol.atom_coords()
    for axis in range(3):
        shifted = positions.copy()
        shifted[:, axis] += 0.3
        shifted_job = {**job, "positions_angstrom": (shifted * BOHR).tolist()}
        shifted_energy = calculate(shifted_job)[2]
        rigid_audit.append(
            {
                "operation": "translation",
                "axis": axis,
                "shift_bohr": 0.3,
                "energy_change_ev": (shifted_energy - energy) * 27.211386245988,
            }
        )
    for angle in (45, 90):
        theta = np.deg2rad(angle)
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        )
        rotated = (positions - positions[0]) @ rotation.T + positions[0]
        rotated_energy = calculate({**job, "positions_angstrom": (rotated * BOHR).tolist()})[2]
        rigid_audit.append(
            {
                "operation": "rotation_z",
                "angle_degrees": angle,
                "energy_change_ev": (rotated_energy - energy) * 27.211386245988,
            }
        )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "reference.npz",
        numbers=np.array([8, 1, 1]),
        positions=mol.atom_coords(),
        energy=energy,
        forces=force,
        hessian=hessian,
        frequencies_cm1=frequencies,
        normal_modes=modes["norm_mode"],
    )
    report = {
        "method": "PBE0/aug-cc-pVDZ",
        "integration_grid_level": 2,
        "optimized_oh_angstrom": float(result.x[0] * BOHR),
        "optimized_hoh_degrees": float(np.rad2deg(result.x[1])),
        "harmonic_frequencies_cm1": frequencies.tolist(),
        "maximum_force_hartree_per_bohr": float(abs(force).max()),
        "optimization_evaluations": len(calls),
        "rigid_motion_audit": rigid_audit,
        "optimizer_converged": True,
        "mass_u": [15.999, 1.008, 1.008],
        "scope": "Independent electronic-structure reference; harmonic frequencies, not experimental fundamentals or a CA trajectory",
        "implementation_source": "https://pyscf.org/pyscf_api_docs/pyscf.hessian.html",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
