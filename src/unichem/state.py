from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .errors import StateError
from .grid import Grid
from .quantum.blocks import BlockMatrix
from .species import Species


def _copy(t: Tensor, detach: bool) -> Tensor:
    return (t.detach() if detach else t).clone()


@dataclass(frozen=True)
class AtomState:
    """Structure of arrays for EXTENDED atomic cells, never one atom per voxel."""

    ids: Tensor  # (N,) immutable identity, not a voxel label
    positions: Tensor  # (N,3), continuous bohr
    momenta: Tensor  # (N,3), atomic units
    log_width_scale: Tensor  # (N,), environment-dependent cloud extent


@dataclass(frozen=True)
class ElectronState:
    """Finite-range atom-block kernel coordinates and two spin number scalars.

    Local matrix functions derive a globally consistent positive 1-RDM.
    There is no persistent global occupied-subspace coefficient matrix.
    """

    kernel: BlockMatrix
    spin_counts: Tensor
    cutoff: float = 5.0


@dataclass(frozen=True)
class LatentState:
    channels: Tensor  # (nx,ny,nz,C), dimensionless
    memory: Tensor  # filter memory OR auxiliary momentum, explicitly tagged
    grid: Grid
    kind: str = "filter"  # "filter" or "auxiliary_momentum"


@dataclass(frozen=True)
class World:
    grid: Grid
    species: tuple[Species, ...]  # one definition per atom, no molecule/bond table
    atoms: AtomState
    electrons: ElectronState
    latent: LatentState
    electron_count: float
    time: float = 0.0
    step: int = 0
    external_field: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def latent_grid(self) -> Grid:
        return self.latent.grid

    def with_latent(self, channels: Tensor) -> World:
        return replace(self, latent=replace(self.latent, channels=channels))

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    @property
    def n_basis(self) -> int:
        return sum(len(s.orbitals) for s in self.species)

    def with_positions(self, positions: Tensor) -> World:
        return replace(self, atoms=replace(self.atoms, positions=positions))

    def with_kernel_values(self, values: Tensor) -> World:
        return replace(
            self,
            electrons=replace(self.electrons, kernel=replace(self.electrons.kernel, values=values)),
        )

    def clone(self, *, detach: bool = True) -> World:
        c = lambda x: _copy(x, detach)
        return replace(
            self,
            atoms=AtomState(
                c(self.atoms.ids),
                c(self.atoms.positions),
                c(self.atoms.momenta),
                c(self.atoms.log_width_scale),
            ),
            electrons=ElectronState(
                replace(
                    self.electrons.kernel,
                    row=c(self.electrons.kernel.row),
                    col=c(self.electrons.kernel.col),
                    values=c(self.electrons.kernel.values),
                ),
                c(self.electrons.spin_counts),
                self.electrons.cutoff,
            ),
            latent=replace(
                self.latent, channels=c(self.latent.channels), memory=c(self.latent.memory)
            ),
        )

    def to(self, *, device=None, dtype=None) -> World:
        f = lambda x: x.to(device=device, dtype=dtype if x.is_floating_point() else None)
        return replace(
            self,
            atoms=AtomState(
                f(self.atoms.ids),
                f(self.atoms.positions),
                f(self.atoms.momenta),
                f(self.atoms.log_width_scale),
            ),
            electrons=ElectronState(
                replace(
                    self.electrons.kernel,
                    row=f(self.electrons.kernel.row),
                    col=f(self.electrons.kernel.col),
                    values=f(self.electrons.kernel.values),
                ),
                f(self.electrons.spin_counts),
                self.electrons.cutoff,
            ),
            latent=replace(
                self.latent, channels=f(self.latent.channels), memory=f(self.latent.memory)
            ),
        )

    def validate(self) -> None:
        a, e, z = self.atoms, self.electrons, self.latent
        n, b = self.n_atoms, self.n_basis
        if len(self.external_field) != 3 or not all(math.isfinite(x) for x in self.external_field):
            raise StateError("External electric field must be a finite three-vector")
        if not math.isfinite(self.electron_count) or self.electron_count <= 0:
            raise StateError("Electron count must be finite and positive")
        if self.latent.kind not in ("filter", "auxiliary_momentum"):
            raise StateError("Unknown latent memory semantics")
        from .fields.multiscale import GridTransfer

        GridTransfer(self.grid, self.latent_grid)
        if n == 0:
            raise StateError("An empty world is not supported by the reference backend")
        if a.ids.shape != (n,) or a.ids.dtype != torch.int64 or a.ids.unique().numel() != n:
            raise StateError("Atom ids must be unique int64 values")
        if (
            a.positions.shape != (n, 3)
            or a.momenta.shape != (n, 3)
            or a.log_width_scale.shape != (n,)
        ):
            raise StateError("Invalid atomic tensor shapes")
        if a.positions.dtype not in (torch.float32, torch.float64):
            raise StateError("Use float32 or float64; float64 is recommended for validation")
        tensors = [
            a.positions,
            a.momenta,
            a.log_width_scale,
            e.kernel.values,
            e.spin_counts,
            z.channels,
            z.memory,
        ]
        for t in tensors:
            if t.dtype != a.positions.dtype or t.device != a.positions.device:
                raise StateError("All floating tensors must have the same dtype/device")
            if not torch.isfinite(t).all():
                raise StateError("World tensors must be finite")
        if a.ids.device != a.positions.device:
            raise StateError("Atom ids must be on the same device as positions")
        width = max(len(s.orbitals) for s in self.species)
        k = e.kernel
        if not math.isfinite(e.cutoff) or e.cutoff <= 0 or float(k.values.detach().abs().max()) > 8:
            raise StateError("Kernel coordinates/cutoff outside stated numerical domain")
        if k.n != n or k.values.shape != (len(k.row), width, width) or k.col.shape != k.row.shape:
            raise StateError("Invalid local kernel blocks")
        if k.row.dtype != torch.int64 or k.col.dtype != torch.int64:
            raise StateError("Kernel graph indices must be int64")
        if k.row.device != a.positions.device or k.col.device != a.positions.device:
            raise StateError("Kernel graph indices must share the world device")
        if (
            len(k.row) == 0
            or (k.row < 0).any()
            or (k.col < 0).any()
            or (k.row >= n).any()
            or (k.col >= n).any()
        ):
            raise StateError("Invalid patch graph indices")
        keys = k.row * n + k.col
        if len(keys) > 1 and not bool((keys[1:] > keys[:-1]).all()):
            raise StateError("Patch keys must be unique and sorted")
        if e.spin_counts.shape != (2,) or (e.spin_counts < 0).any() or (e.spin_counts > b).any():
            raise StateError("Invalid spin electron counts")
        tol = 1e-5 if a.positions.dtype == torch.float32 else 1e-9
        if abs(float(e.spin_counts.sum().detach()) - self.electron_count) > tol:
            raise StateError("Electron count does not match spin sector")
        if z.channels.ndim != 4 or z.channels.shape[:3] != self.latent_grid.shape:
            raise StateError("Latent channels must have shape (*latent_grid.shape,C)")
        if z.channels.shape[-1] < 1 or z.memory.shape != z.channels.shape:
            raise StateError("Invalid latent memory shape")
        if (
            abs(self.time) == float("inf")
            or self.time != self.time
            or type(self.step) is not int
            or self.step < 0
        ):
            raise StateError("Invalid simulation clock")
        if (a.log_width_scale.abs() > 2).any():
            raise StateError("Cloud scale outside reference range exp([-2,2])")
