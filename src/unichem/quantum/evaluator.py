"""One physical Hamiltonian, one converged electronic state, one force law."""

from dataclasses import dataclass

import torch

from ..runtime.scf import SCFSolver
from .gaussian import GaussianIntegrals


@dataclass(frozen=True)
class Evaluation:
    energy: float
    forces: torch.Tensor
    electronic: object
    integrals: object


class Evaluator:
    def __init__(self, solver=None):
        self.solver = SCFSolver() if solver is None else solver
        self._key = None
        self._integrals = None

    def __call__(self, world):
        world.validate()
        a = world.atoms
        key = (
            a.numbers.numpy().tobytes(),
            a.positions.numpy().tobytes(),
            world.basis,
            world.charge,
            world.spin,
            world.external_field,
        )
        if key != self._key:
            self._integrals = GaussianIntegrals(
                a.numbers.numpy(),
                a.positions.numpy(),
                basis=world.basis,
                charge=world.charge,
                spin=world.spin,
                external_field=world.external_field,
            )
            self._key = key
        result = self.solver.solve(self._integrals.operators(), world.spin_counts, world.densities)
        return Evaluation(result.energy, self._integrals.forces(result), result, self._integrals)
