from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from ..errors import StateError
from ..spatial.neighbors import CellList
from .basis import BasisSamples
from .blocks import BlockMatrix, constrained_kernel
from .patches import effective_kernel
from .support import make_layout


@dataclass(frozen=True)
class QuantumView:
    overlap: BlockMatrix
    density_matrix: BlockMatrix
    spin_kernels: tuple[BlockMatrix, BlockMatrix]
    density: Tensor
    kinetic_energy: Tensor
    diagnostics: dict


class QuantumBackend(Protocol):
    def evaluate(self, world, samples: BasisSamples) -> QuantumView: ...


@dataclass(frozen=True)
class AtomSamples:
    voxels: Tensor
    values: Tensor

    def gather(self, query):
        where = torch.searchsorted(self.voxels, query)
        safe = where.clamp_max(len(self.voxels) - 1)
        valid = (where < len(self.voxels)) & (self.voxels[safe] == query)
        return self.values[safe] * valid[:, None]


def atom_samples(world, samples):
    layout = make_layout(world)
    owners = layout.atom_of_basis[samples.basis]
    order = torch.argsort(owners, stable=True)
    counts = torch.bincount(owners, minlength=world.n_atoms)
    boundaries = torch.cat((counts.new_zeros(1), counts.cumsum(0))).tolist()
    result = []
    for i in range(world.n_atoms):
        selected = order[boundaries[i] : boundaries[i + 1]]
        voxels, inverse = torch.unique(samples.voxel[selected], sorted=True, return_inverse=True)
        basis = layout.padded_basis[i]
        local = torch.searchsorted(basis[basis >= 0], samples.basis[selected])
        values = (
            samples.value.new_zeros(len(voxels) * basis.numel())
            .index_add(0, inverse * basis.numel() + local, samples.value[selected])
            .reshape(len(voxels), len(basis))
        )
        result.append(AtomSamples(voxels, values))
    return result, layout.padded_basis >= 0


def shift(grid, flat, axis, amount):
    stride = (grid.shape[1] * grid.shape[2], grid.shape[2], 1)[axis]
    coordinate = (flat // stride) % grid.shape[axis]
    return flat + ((coordinate + amount) % grid.shape[axis] - coordinate) * stride


class LocalQuantumBackend:
    """Finite-halo atom patches; only block-sparse quantum operators.

    Polynomial products keep all fill within the mathematical halo. Capacity
    excess fails explicitly. Pauli bounds hold across patches by construction,
    not by separately clipping patch spectra. Long-range Poisson is separate.
    """

    def __init__(
        self, *, metric_degree=16, fermi_degree=4, max_blocks=100_000, metric_tolerance=None
    ):
        if metric_degree < 1 or fermi_degree < 1 or max_blocks < 1:
            raise StateError("Invalid local quantum parameters")
        self.metric_degree, self.fermi_degree = metric_degree, fermi_degree
        self.max_blocks, self.metric_tolerance = max_blocks, metric_tolerance

    def evaluate(self, world, samples):
        atoms, active = atom_samples(world, samples)
        radii = world.atoms.positions.new_tensor(
            [max(o.width for o in s.orbitals) for s in world.species]
        )
        core = world.atoms.positions.new_tensor([s.core_smearing for s in world.species])
        radii = 4.5 * torch.maximum(radii * world.atoms.log_width_scale.exp(), core)
        pair = CellList(float(2 * radii.detach().max() + max(world.grid.spacing)))(
            world.grid, world.atoms.positions
        )
        keep = pair.distance < radii[pair.first] + radii[pair.second] + max(world.grid.spacing)
        ids = torch.arange(world.n_atoms, device=pair.first.device)
        row = torch.cat((ids, pair.first[keep], pair.second[keep]))
        col = torch.cat((ids, pair.second[keep], pair.first[keep]))
        overlaps, kinetics = [], []
        for i, j in zip(row.tolist(), col.tolist()):
            ai, aj = atoms[i], atoms[j]
            overlaps.append(ai.values.T @ aj.gather(ai.voxels) * world.grid.dv)
            kinetic = overlaps[-1] * 0
            for axis, h in enumerate(world.grid.spacing):
                points = torch.unique(
                    torch.cat((ai.voxels, shift(world.grid, ai.voxels, axis, -1)))
                )
                nxt = shift(world.grid, points, axis, 1)
                di = (ai.gather(nxt) - ai.gather(points)) / h
                dj = (aj.gather(nxt) - aj.gather(points)) / h
                kinetic = kinetic + 0.5 * (di.T @ dj) * world.grid.dv
            kinetics.append(kinetic)
        overlap = BlockMatrix.assemble(
            row, col, torch.stack(overlaps), world.n_atoms, self.max_blocks
        )
        kinetic = BlockMatrix.assemble(
            row, col, torch.stack(kinetics), world.n_atoms, self.max_blocks
        )
        overlap = overlap.add(overlap.transpose()).scale(0.5)
        kinetic = kinetic.add(kinetic.transpose()).scale(0.5)
        kernels, diagnostics = constrained_kernel(
            effective_kernel(world),
            overlap,
            active,
            world.electrons.spin_counts,
            metric_degree=self.metric_degree,
            fermi_degree=self.fermi_degree,
        )
        if (
            self.metric_tolerance is not None
            and diagnostics["metric_defect_bound"] > self.metric_tolerance
        ):
            raise StateError("Local metric defect exceeds required tolerance")
        density_matrix = kernels[0].add(kernels[1])
        blocks = density_matrix.query(overlap.row, overlap.col)
        density = samples.value.new_zeros(world.grid.n_voxels)
        for block, i, j in zip(blocks, overlap.row.tolist(), overlap.col.tolist()):
            ai, aj = atoms[i], atoms[j]
            contribution = ((ai.values @ block) * aj.gather(ai.voxels)).sum(-1)
            density = density.index_add(0, ai.voxels, contribution)
        if float(density.detach().min()) < -1e-9:
            raise StateError("Negative density beyond floating-point tolerance")
        diagnostics.update(
            {
                "overlap_blocks": len(overlap.row),
                "n_atoms": world.n_atoms,
                "maximum_block_width": overlap.width,
                "persistent_blocks": len(world.electrons.kernel.row),
            }
        )
        return QuantumView(
            overlap,
            density_matrix,
            tuple(kernels),
            density,
            density_matrix.trace_product(kinetic),
            diagnostics,
        )
