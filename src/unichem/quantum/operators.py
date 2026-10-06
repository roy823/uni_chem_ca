"""Hamiltonian closure independent of molecules, reactions and SCF solver."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .blocks import BlockMatrix


@dataclass
class Operators:
    overlap: BlockMatrix
    core: BlockMatrix
    active: torch.Tensor
    nuclear_energy: float
    coulomb_exchange: Callable

    def energy_focks(self, densities):
        total = densities[0].add(densities[1])
        coulomb, exchanges = self.coulomb_exchange(densities)
        energy = total.trace_product(self.core) + 0.5 * total.trace_product(coulomb)
        energy = energy - 0.5 * sum(p.trace_product(k) for p, k in zip(densities, exchanges))
        focks = tuple(self.core.add(coulomb).add(k, -1).symmetric() for k in exchanges)
        return float(energy) + self.nuclear_energy, focks
