"""Independent HF geometry/frequency checks and native quantum-cell NVE.

The production energy is never trained. Full Cartesian optimization is used;
the Hamiltonian does not receive bonds, target geometry or normal modes.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from pyscf import scf
from pyscf.hessian.thermo import harmonic_analysis
from scipy.optimize import minimize

from unichem import Evaluator, SynchronousEngine, make_world
from unichem.checkpoint import save_world
from unichem.landscapes import vibrational_curvatures


def geometry(r):
    vectors = r[1:] - r[0]
    lengths = np.linalg.norm(vectors, axis=1)
    angle = np.degrees(np.arccos(np.clip(np.dot(*vectors) / np.prod(lengths), -1, 1)))
    return lengths * 0.529177210903, float(angle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-fs", type=float, default=10)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--output", default="outputs/v04_validation/molecular")
    args = parser.parse_args()
    torch.set_num_threads(2)
    world = make_world(
        ["O", "H", "H"], [[0, 0, 0], [1.8, 0, 0], [-0.45, 1.74, 0]], basis=args.basis
    )
    evaluator = Evaluator()

    def objective(flat):
        r = torch.tensor(flat.reshape(3, 3), dtype=torch.float64)
        value = evaluator(world.with_positions(r))
        return value.energy, -value.forces.numpy().ravel()

    optimized = minimize(
        objective,
        world.atoms.positions.numpy().ravel(),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 60},
    )
    r = optimized.x.reshape(3, 3)
    current = world.with_positions(torch.tensor(r))
    evaluated = evaluator(current)
    if float(abs(evaluated.forces).max()) > 1e-6:
        raise RuntimeError("Production geometry not stationary")
    reference = scf.RHF(evaluated.integrals.mol)
    reference.conv_tol = 1e-12
    reference.conv_tol_grad = 1e-10
    reference_energy = reference.kernel()
    h_reference = reference.Hessian().kernel()
    masses = world.atoms.masses.numpy() / 1822.8884862
    frequencies = harmonic_analysis(reference.mol, h_reference, mass=masses)["freq_wavenumber"]
    eps = 0.0005
    h = np.empty((9, 9))
    for axis in range(9):
        plus, minus = r.copy().ravel(), r.copy().ravel()
        plus[axis] += eps
        minus[axis] -= eps
        fp = (
            evaluator(world.with_positions(torch.tensor(plus.reshape(3, 3)))).forces.numpy().ravel()
        )
        fm = (
            evaluator(world.with_positions(torch.tensor(minus.reshape(3, 3))))
            .forces.numpy()
            .ravel()
        )
        h[:, axis] = -(fp - fm) / (2 * eps)
    actual_frequencies = vibrational_curvatures(r, h, masses)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    displacement = np.random.default_rng(11).normal(size=r.shape) * 0.01
    displacement -= np.sum(displacement * masses[:, None], axis=0) / masses.sum()
    initial = world.with_positions(torch.tensor(r + displacement))
    records = []
    for dt in (0.2, 0.1, 0.05):
        steps = round(args.duration_fs / dt)
        if not np.isclose(steps * dt, args.duration_fs):
            raise ValueError("Duration must fit each time step")
        state = initial.clone()
        evaluation = evaluator(state)
        state = replace(state, densities=evaluation.electronic.densities)
        energy0 = evaluation.energy
        engine = SynchronousEngine(evaluator, dt=dt / 0.024188843265857)
        positions = [state.atoms.positions.numpy()]
        errors = []
        number = []
        purity = []
        for _ in range(steps):
            value = engine.step(state)
            state = value.world
            positions.append(state.atoms.positions.numpy())
            errors.append((value.diagnostics["total_energy_hartree"] - energy0) * 27.211386245988)
            number.append(max(value.evaluation.electronic.electron_errors))
            purity.append(max(value.evaluation.electronic.purity_errors))
        save_world(state, root / f"world_dt_{dt}.npz")
        np.savez_compressed(
            root / f"nve_dt_{dt}.npz",
            positions=np.array(positions),
            time_fs=np.arange(steps + 1) * dt,
            energy_balance_ev=np.array([0, *errors]),
        )
        row = {
            "dt_fs": dt,
            "duration_fs": steps * dt,
            "steps": steps,
            "max_energy_balance_error_ev": max(map(abs, errors)),
            "max_electron_error": max(number),
            "max_purity_error": max(purity),
        }
        records.append(row)
        print(json.dumps(row), flush=True)
    lengths, angle = geometry(r)
    report = {
        "method": "all-electron HF",
        "basis": args.basis,
        "network_enabled": False,
        "optimized_oh_angstrom": lengths.tolist(),
        "optimized_hoh_degrees": angle,
        "optimizer_success": bool(optimized.success),
        "maximum_stationary_force_au": float(abs(evaluated.forces).max()),
        "reference_energy_error_hartree": abs(evaluated.energy - reference_energy),
        "production_frequencies_cm1": actual_frequencies.tolist(),
        "reference_frequencies_cm1": frequencies.tolist(),
        "maximum_frequency_relative_error": float(
            (abs(actual_frequencies - frequencies) / frequencies).max()
        ),
        "hessian_asymmetry_au": float(abs(h - h.T).max()),
        "runs": records,
        "scope": "Native physical quantum-cell reproduction at the selected HF/basis level; not experimental vibrational fundamentals, DFT accuracy or reaction kinetics.",
    }
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
