"""Bounded local execution for externally supplied electronic and force fields.

Each cell owns a reduced spin density of fixed size. Pair channels act on two
cells, and symmetric convex weights preserve electron number and local Pauli
bounds. This is an explicit mean-field closure: intercell entanglement is not
retained. It is not the all-electron HF reference or a trained chemistry model.
"""

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from ..errors import CapacityError, StateError
from ..molecule import AtomState
from ..spatial.neighbors import PairList


@dataclass(frozen=True)
class CellSnapshot:
    atoms: AtomState
    density: Tensor  # N, spin, local channel, local channel
    channels: Tensor
    time: float = 0.0
    step: int = 0

    @property
    def n_cells(self):
        return len(self.atoms.ids)

    def populations(self):
        return self.density.diagonal(dim1=-2, dim2=-1).real.sum(-1)

    def validate(self):
        n = self.n_cells
        a = self.atoms
        if (
            n < 1
            or a.positions.shape != (n, 3)
            or a.momenta.shape != (n, 3)
            or a.masses.shape != (n,)
        ):
            raise StateError("Inconsistent local atomic cells")
        if not math.isfinite(self.time) or type(self.step) is not int or self.step < 0:
            raise StateError("Invalid local CA clock")
        if a.ids.unique().numel() != n or a.ids.dtype != torch.int64:
            raise StateError("Persistent cell IDs must be unique int64 values")
        if not bool(
            torch.isfinite(a.positions).all()
            and torch.isfinite(a.momenta).all()
            and torch.isfinite(a.masses).all()
            and (a.masses > 0).all()
        ):
            raise StateError("Finite nuclear states and positive masses required")
        if self.density.ndim != 4 or self.density.shape[:2] != (n, 2):
            raise StateError("Cell densities must be (N,2,d,d)")
        width = self.density.shape[-1]
        if self.density.shape[-2] != width or self.channels.shape != (n,):
            raise StateError("Cell channel shape mismatch")
        if not bool(((self.channels >= 1) & (self.channels <= width)).all()):
            raise StateError("Local channel counts outside capacity")
        if self.channels.device != self.density.device:
            raise StateError("Cell channel metadata must share the state device")
        active = torch.arange(width, device=self.density.device)[None] < self.channels[:, None]
        mask = active[:, :, None] & active[:, None, :]
        if float((self.density * (~mask[:, None])).abs().max()) > 1e-12:
            raise StateError("Inactive local channels must have zero density")
        if not self.density.is_complex() or not bool(torch.isfinite(self.density).all()):
            raise StateError("Finite complex reduced densities required")
        if float((self.density - self.density.mH).abs().max()) > 1e-10:
            raise StateError("Non-Hermitian reduced density")
        occupation = torch.linalg.eigvalsh(self.density)
        if float(occupation.min()) < -1e-10 or float(occupation.max()) > 1 + 1e-10:
            raise StateError("Local Pauli occupancy bounds violated")


@dataclass(frozen=True)
class FieldSnapshot:
    forces: Tensor  # N,3; supplied by the independent whole-field model
    onsite: Tensor  # N, local channel, local channel; orthogonal effective basis
    hopping: Tensor  # E, local channel, local channel; oriented first -> second


class LocalCellEngine:
    """Three fixed local stages; work O(N K d^3) for bounded K and d.

    Electronic onsite half-step, parallel two-cell channels, onsite half-step.
    Nuclear kick/drift reads the same old field snapshot. No global electronic
    SCF, trace correction, density projection or system eigendecomposition.
    """

    def __init__(self, *, dt=0.1, max_neighbors=32, max_channels=8, edge_batch=4096):
        if not 0 < dt < math.inf or min(max_neighbors, max_channels, edge_batch) < 1:
            raise StateError("Positive local CA parameters required")
        self.dt = dt
        self.max_neighbors, self.max_channels, self.edge_batch = (
            max_neighbors,
            max_channels,
            edge_batch,
        )

    def step(self, state: CellSnapshot, pairs: PairList, fields: FieldSnapshot):
        n, spins, width, other = state.density.shape
        if n != state.n_cells or spins != 2 or width != other or width > self.max_channels:
            raise StateError("Invalid fixed-size cell density")
        if fields.forces.shape != (n, 3) or fields.onsite.shape != (n, width, width):
            raise StateError("Field snapshot does not match cell state")
        if fields.hopping.shape != (len(pairs.first), width, width):
            raise StateError("Field hopping must match the current contact edges")
        device = state.density.device
        if any(
            x.device != device
            for x in (
                fields.forces,
                fields.onsite,
                fields.hopping,
                state.atoms.positions,
                state.atoms.momenta,
                state.atoms.masses,
                pairs.first,
                pairs.second,
            )
        ):
            raise StateError("All local CA inputs must use the same device")
        degree = torch.bincount(torch.cat((pairs.first, pairs.second)), minlength=n)
        if int(degree.max()) > self.max_neighbors:
            raise CapacityError("Local degree exceeds the CA neighbor capacity")
        active = torch.arange(width, device=device)[None, :] < state.channels[:, None]
        onsite_mask = active[:, :, None] & active[:, None, :]
        onsite = fields.onsite.to(state.density.dtype) * onsite_mask
        if float((onsite - onsite.mH).abs().max()) > 1e-10:
            raise StateError("Effective local Hamiltonians must be Hermitian")
        u = torch.linalg.matrix_exp(-0.5j * self.dt * onsite)
        old = u[:, None] @ state.density @ u[:, None].mH
        change = torch.zeros_like(old)
        for lo in range(0, len(pairs.first), self.edge_batch):
            i, j = pairs.first[lo : lo + self.edge_batch], pairs.second[lo : lo + self.edge_batch]
            count = len(i)
            hopping = fields.hopping[lo : lo + count].to(old.dtype)
            hopping = hopping * (active[i, :, None] & active[j, None, :])
            h = old.new_zeros((count, 2 * width, 2 * width))
            h[:, :width, width:] = hopping
            h[:, width:, :width] = hopping.mH
            gate = torch.linalg.matrix_exp(-1j * self.dt * h)
            pair_state = old.new_zeros((count, 2, 2 * width, 2 * width))
            pair_state[:, :, :width, :width] = old[i]
            pair_state[:, :, width:, width:] = old[j]
            proposed = gate[:, None] @ pair_state @ gate[:, None].mH
            # The same weight is used at both endpoints. Each node receives a
            # convex mixture, so its reduced density remains 0 <= rho <= I.
            weight = 1 / torch.maximum(degree[i], degree[j]).clamp_min(1).to(old.real.dtype)
            w = weight[:, None, None, None]
            change.index_add_(0, i, w * (proposed[:, :, :width, :width] - old[i]))
            change.index_add_(0, j, w * (proposed[:, :, width:, width:] - old[j]))
        density = u[:, None] @ (old + change) @ u[:, None].mH
        momentum = state.atoms.momenta + self.dt * fields.forces
        positions = state.atoms.positions + self.dt * momentum / state.atoms.masses[:, None]
        if not bool(torch.isfinite(density).all() and torch.isfinite(positions).all()):
            raise StateError("Nonfinite local CA proposal; no state committed")
        atoms = replace(state.atoms, positions=positions, momenta=momentum)
        return replace(
            state, atoms=atoms, density=density, time=state.time + self.dt, step=state.step + 1
        )
