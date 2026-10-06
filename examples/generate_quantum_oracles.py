"""Small independent all-electron integral oracles, with no learned parameters.

PySCF is used only to export tiny-system reference integrals and solutions.
These files are validation oracles, never a production global quantum backend.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from pyscf import gto, lib, scf
from scipy.linalg import eigh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/quantum_oracles")
    args = parser.parse_args()
    lib.num_threads(2)
    jobs = [
        (f"one_electron_pair_{r:.2f}", [1, 1], [[0, 0, 0], [r, 0, 0]], 1, 1)
        for r in (0.6, 0.8, 1.0, 1.2, 1.4, 1.8, 2.4, 3.0)
    ]
    jobs.append(("two_electron_pair", [1, 1], [[0, 0, 0], [0.74, 0, 0]], 0, 0))
    with np.load("outputs/water_dft_reference/reference.npz", allow_pickle=False) as data:
        # Geometry provenance only; reference energies are recomputed with HF.
        positions = data["positions"] * 0.529177210903
    jobs.append(("water", [8, 1, 1], positions.tolist(), 0, 0))
    jobs.append(("atomic_h", [1], [[0, 0, 0]], 0, 1))
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (name, numbers, positions, charge, spin) in enumerate(jobs):
        cached = root / f"{index:04d}.npz"
        if cached.exists():
            with np.load(cached, allow_pickle=False) as data:
                meta = json.loads(str(data["metadata"].item()))
                if (
                    meta["name"] != name
                    or meta["charge"] != charge
                    or meta["spin"] != spin
                    or meta["basis"] != "STO-3G"
                    or not np.array_equal(data["numbers"], numbers)
                    or not np.allclose(
                        data["positions"] * 0.529177210903, positions, atol=1e-8, rtol=0
                    )
                ):
                    raise ValueError(
                        "Cached integral geometry or method mismatch; use a new directory"
                    )
                rows.append({**meta, "energy_hartree": float(data["reference_energy"])})
            continue
        mol = gto.M(
            atom=list(zip(numbers, positions)),
            unit="Angstrom",
            basis="sto-3g",
            charge=charge,
            spin=spin,
            verbose=0,
        )
        if mol.nao_nr() > 16:
            raise ValueError("Dense reference limited to at most 16 AOs")
        mf = scf.RHF(mol) if spin == 0 else scf.UHF(mol)
        h, overlap = mf.get_hcore(), mf.get_ovlp()
        nuclear = float(mol.energy_nuc())
        eri = mol.intor("int2e", aosym="s1")
        if mol.nelectron == 1:
            energies, coefficients = eigh(h, overlap)
            dm = np.array([np.outer(coefficients[:, 0], coefficients[:, 0]), np.zeros_like(h)])
            energy = float(energies[0] + nuclear)
            method = "exact one-electron generalized eigenproblem in STO-3G"
            converged = True
        else:
            mf.conv_tol = 1e-12
            mf.conv_tol_grad = 1e-9
            energy = float(mf.kernel())
            if not mf.converged:
                raise RuntimeError("Reference HF unconverged")
            dm = np.array([mf.make_rdm1() / 2] * 2)
            method = "RHF/STO-3G"
            converged = bool(mf.converged)
        metadata = {
            "name": name,
            "method": method,
            "basis": "STO-3G",
            "all_electron": True,
            "charge": charge,
            "spin": spin,
            "nelectron": mol.nelectron,
            "spin_counts": list(mol.nelec),
            "units": "atomic",
            "converged": converged,
            "learned_parameters": False,
            "boundary": "isolated",
            "distance_angstrom": float(np.linalg.norm(np.array(positions)[1] - positions[0]))
            if len(numbers) == 2
            else None,
            "scope": "Small validation oracle; no claim of complete-basis or correlated chemistry",
        }
        np.savez_compressed(
            root / f"{index:04d}.npz",
            numbers=np.array(numbers),
            positions=mol.atom_coords(),
            overlap=overlap,
            hcore=h,
            eri=eri,
            nuclear_energy=nuclear,
            reference_energy=energy,
            reference_spin_density=dm,
            ao_counts=mol.aoslice_by_atom()[:, 3] - mol.aoslice_by_atom()[:, 2],
            metadata=np.asarray(json.dumps(metadata)),
        )
        rows.append({**metadata, "energy_hartree": energy})
        print(json.dumps(rows[-1]), flush=True)
    (root / "report.json").write_text(
        json.dumps(
            {
                "jobs": rows,
                "source": "PySCF integrals and SciPy generalized eigh; independent of CA energy and NN",
            },
            indent=2,
        ),
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
