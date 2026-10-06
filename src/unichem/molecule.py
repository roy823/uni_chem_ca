"""Canonical extended-atom state; physical spin density kernels are persisted."""

from dataclasses import dataclass, replace

import numpy as np
import torch
from ase.data import atomic_masses, atomic_numbers

from .errors import StateError
from .units import ELECTRON_MASS_PER_U


@dataclass(frozen=True)
class AtomState:
    ids: torch.Tensor
    numbers: torch.Tensor
    positions: torch.Tensor
    momenta: torch.Tensor
    masses: torch.Tensor


@dataclass(frozen=True)
class World:
    atoms: AtomState
    basis: str
    charge: int
    spin_counts: tuple[int, int]
    densities: tuple | None = None
    external_field: tuple[float, float, float] = (0.0, 0.0, 0.0)
    time: float = 0.0
    step: int = 0

    @property
    def n_atoms(self):
        return len(self.atoms.numbers)

    @property
    def electron_count(self):
        return sum(self.spin_counts)

    @property
    def spin(self):
        return self.spin_counts[0] - self.spin_counts[1]

    def with_positions(self, positions):
        return replace(self, atoms=replace(self.atoms, positions=positions))

    def clone(self):
        a = self.atoms
        copy = lambda t: t.detach().clone()
        densities = (
            None
            if self.densities is None
            else tuple(
                replace(p, row=copy(p.row), col=copy(p.col), values=copy(p.values))
                for p in self.densities
            )
        )
        return replace(
            self,
            atoms=AtomState(
                copy(a.ids), copy(a.numbers), copy(a.positions), copy(a.momenta), copy(a.masses)
            ),
            densities=densities,
        )

    def validate(self):
        n = self.n_atoms
        a = self.atoms
        if (
            n < 1
            or a.positions.shape != (n, 3)
            or a.momenta.shape != (n, 3)
            or a.masses.shape != (n,)
        ):
            raise StateError("Inconsistent atomic cells")
        if a.ids.shape != (n,) or a.ids.dtype != torch.int64 or a.ids.unique().numel() != n:
            raise StateError("Atom IDs must be persistent unique int64 values")
        if a.numbers.dtype != torch.int64 or bool(((a.numbers < 1) | (a.numbers > 118)).any()):
            raise StateError("Invalid nuclear charges")
        if any(
            t.dtype != torch.float64
            or t.device.type != "cpu"
            or t.requires_grad
            or not bool(torch.isfinite(t).all())
            for t in (a.positions, a.momenta, a.masses)
        ):
            raise StateError("Physical molecular state requires finite float64 arrays")
        if (
            bool((a.masses <= 0).any())
            or len(self.spin_counts) != 2
            or any(type(x) is not int or x < 0 for x in self.spin_counts)
        ):
            raise StateError("Invalid masses or spin counts")
        if self.electron_count != int(a.numbers.sum()) - self.charge or self.electron_count < 1:
            raise StateError("Nuclear charges, net charge and electron numbers disagree")
        if (
            not isinstance(self.basis, str)
            or not self.basis
            or len(self.external_field) != 3
            or not np.isfinite(self.external_field).all()
        ):
            raise StateError("Invalid basis or external field")
        if self.densities is not None:
            if len(self.densities) != 2:
                raise StateError("Two spin density kernels required")
            for p in self.densities:
                if p.n != n or not torch.isfinite(p.values).all():
                    raise StateError("Invalid atom-block density kernel")
                if (
                    p.values.ndim != 3
                    or p.values.shape != (len(p.row), p.width, p.width)
                    or p.col.shape != p.row.shape
                ):
                    raise StateError("Invalid density block shapes")
                if (
                    p.row.dtype != torch.int64
                    or p.col.dtype != torch.int64
                    or bool((p.row < 0).any())
                    or bool((p.col < 0).any())
                    or bool((p.row >= n).any())
                    or bool((p.col >= n).any())
                ):
                    raise StateError("Invalid density graph indices")
                keys = p.row * n + p.col
                if len(keys) > 1 and not bool((keys[1:] > keys[:-1]).all()):
                    raise StateError("Density blocks must be unique and sorted")
                if float(p.add(p.transpose(), -1).norm()) > 1e-8:
                    raise StateError("Non-Hermitian density kernel")


def make_world(symbols, positions, *, basis="sto-3g", charge=0, spin=None):
    try:
        numbers = [atomic_numbers[s] if isinstance(s, str) else int(s) for s in symbols]
        masses = atomic_masses[numbers] * ELECTRON_MASS_PER_U
    except (KeyError, IndexError, ValueError) as exc:
        raise StateError("Unknown atomic species") from exc
    ne = sum(numbers) - charge
    spin = ne % 2 if spin is None else spin
    if type(charge) is not int or type(spin) is not int or abs(spin) > ne or (ne + spin) % 2:
        raise StateError("Charge and spin must define integer spin populations")
    r = torch.as_tensor(np.asarray(positions), dtype=torch.float64).clone()
    atoms = AtomState(
        torch.arange(len(numbers)),
        torch.tensor(numbers),
        r,
        torch.zeros_like(r),
        torch.tensor(masses),
    )
    world = World(atoms, basis, charge, ((ne + spin) // 2, (ne - spin) // 2))
    world.validate()
    return world
