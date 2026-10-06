"""Live integral-provider validation; PySCF SCF is used only as an oracle."""

import json
from pathlib import Path

import torch
from pyscf import scf

from unichem.quantum.gaussian import GaussianIntegrals
from unichem.runtime.scf import SCFSolver


def main():
    torch.set_num_threads(2)
    rows = []
    for numbers, r, charge, spin, basis in (
        ([1, 1], [[0, 0, 0], [1.889726125, 0, 0]], 1, 1, "sto-3g"),
        ([1, 1], [[0, 0, 0], [1.3984, 0, 0]], 0, 0, "sto-3g"),
        ([8, 1, 1], [[0, 0, 0], [1.8, 0, 0], [-0.45, 1.74, 0]], 0, 0, "sto-3g"),
        ([8, 1, 1], [[0, 0, 0], [1.8, 0, 0], [-0.45, 1.74, 0]], 0, 0, "cc-pvdz"),
    ):
        integrals = GaussianIntegrals(numbers, r, charge=charge, spin=spin, basis=basis)
        result = SCFSolver().solve(integrals.operators(), integrals.mol.nelec)
        force = integrals.forces(result)
        oracle = scf.RHF(integrals.mol) if spin == 0 else scf.UHF(integrals.mol)
        oracle.conv_tol = 1e-12
        oracle.conv_tol_grad = 1e-9
        e = oracle.kernel()
        f = -oracle.nuc_grad_method().kernel()
        row = {
            "numbers": numbers,
            "basis": basis,
            "energy_error_hartree": result.energy - e,
            "force_max_error_au": float(abs(force.numpy() - f).max()),
            "force_au": force.tolist(),
            "scf_iterations": result.iterations,
            "net_force_au": force.sum(0).tolist(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    root = Path("outputs/v04_validation/integrals")
    root.mkdir(exist_ok=True, parents=True)
    (root / "live_integral_report.json").write_text(json.dumps(rows, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
