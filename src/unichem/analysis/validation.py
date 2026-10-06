"""Independent, size-limited validation of the physical sparse electronic core."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..quantum.operators import Operators
from ..runtime.scf import SCFSolver
from .oracles import block_operator, padded_indices, spin_density_diagnostics


class IntegralReference:
    """Dense reference restricted to small validation data, never production."""

    def __init__(self, data):
        self.counts = data["ao_counts"].copy()
        indices, active = padded_indices(self.counts)
        self.select = torch.tensor(indices)
        self.eri = torch.tensor(data["eri"])
        self.operators = Operators(
            block_operator(data["overlap"], self.counts),
            block_operator(data["hcore"], self.counts),
            torch.tensor(active),
            float(data["nuclear_energy"]),
            self.coulomb_exchange,
        )

    def coulomb_exchange(self, densities):
        ps = [
            p.dense_for_validation()[self.select[:, None], self.select[None, :]] for p in densities
        ]
        j = torch.einsum("pqrs,rs->pq", self.eri, ps[0] + ps[1])
        ks = [torch.einsum("prqs,rs->pq", self.eri, p) for p in ps]
        return block_operator(j, self.counts), tuple(block_operator(k, self.counts) for k in ks)


def validate_matrix_core(reference="data/quantum_oracles", output="outputs/v04_validation/matrix"):
    rows = []
    for path in sorted(Path(reference).glob("[0-9][0-9][0-9][0-9].npz")):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            target = float(data["reference_energy"])
            overlap = data["overlap"].copy()
            target_density = data["reference_spin_density"].copy()
            oracle = IntegralReference(data)
        result = SCFSolver().solve(oracle.operators, meta["spin_counts"])
        ps = np.array(
            [
                p.dense_for_validation()[oracle.select[:, None], oracle.select[None, :]].numpy()
                for p in result.densities
            ]
        )
        stats = spin_density_diagnostics(ps, overlap, meta["spin_counts"])
        error = abs(result.energy - target)
        density_error = float(np.linalg.norm(ps - target_density))
        row = {
            "name": meta["name"],
            "method": meta["method"],
            "energy_error_hartree": error,
            "density_error_frobenius": density_error,
            "iterations": result.iterations,
            "commutator_norm": result.commutator_norm,
            "spectral_consistency": result.spectral_consistency,
            "spin_diagnostics": stats,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "passed": error < 1e-8
            and density_error < 1e-6
            and max(result.electron_errors) < 1e-9
            and max(result.purity_errors) < 1e-9
            and all(
                d["minimum_occupation"] >= -1e-9 and d["maximum_occupation"] <= 1 + 1e-9
                for d in stats
            ),
        }
        rows.append(row)
    if not rows:
        raise ValueError("Actual independent quantum integral references required")
    report = {
        "schema_version": 4,
        "method": "physical HF and zero-temperature block-sparse SCF",
        "frames": rows,
        "all_passed": all(x["passed"] for x in rows),
        "learned_energy": False,
        "production_global_diagonalization": False,
        "scope": "Same basis/operator solver validation, not complete-basis correlated chemistry or reaction statistics",
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report
