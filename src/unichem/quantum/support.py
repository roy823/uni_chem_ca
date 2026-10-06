from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..errors import CapacityError, ResolutionError
from ..state import World


@dataclass(frozen=True)
class BasisLayout:
    atom_of_basis: Tensor  # (B,)
    padded_basis: Tensor  # (N,Amax), -1 for padding
    padded_width: Tensor  # (N,Amax), base widths
    padded_axis: Tensor  # (N,Amax), -1=s, 0/1/2=p
    core_charge: Tensor  # (N,)
    core_smearing: Tensor  # (N,), numerical effective-core width
    masses: Tensor  # (N,)
    atomic_numbers: Tensor  # (N,)


def make_layout(world: World) -> BasisLayout:
    """Compile static element metadata into structure-of-arrays tensors.

    Python loops here compile topology; no atom is evolved sequentially.
    """
    like = world.atoms.positions
    max_ao = max(len(s.orbitals) for s in world.species)
    padded_ids, widths, axes, owners = [], [], [], []
    cursor = 0
    for i, s in enumerate(world.species):
        count = len(s.orbitals)
        padded_ids.append(list(range(cursor, cursor + count)) + [-1] * (max_ao - count))
        widths.append([o.width for o in s.orbitals] + [1.0] * (max_ao - count))
        axes.append([o.axis for o in s.orbitals] + [-1] * (max_ao - count))
        owners.extend([i] * count)
        cursor += count
    f = lambda xs: torch.tensor(xs, dtype=like.dtype, device=like.device)
    i = lambda xs: torch.tensor(xs, dtype=torch.int64, device=like.device)
    return BasisLayout(
        i(owners),
        i(padded_ids),
        f(widths),
        i(axes),
        f([s.core_charge for s in world.species]),
        f([s.core_smearing for s in world.species]),
        f([s.mass for s in world.species]),
        f([s.atomic_number for s in world.species]),
    )


@dataclass(frozen=True)
class SupportMap:
    """Many-to-many incidence relation between atomic cells and field voxels.

    One atom occurs in MANY rows; one voxel can occur for MANY atoms.
    There is no exclusive owner[voxel] array and no one-pixel atom fallback.
    The spherical cutoff bounds the support; the cloud inside may be anisotropic.
    """

    atom_index: Tensor
    voxel_index: Tensor
    displacement: Tensor
    radii: Tensor
    n_atoms: int
    n_voxels: int

    def counts_per_atom(self) -> Tensor:
        return torch.bincount(self.atom_index, minlength=self.n_atoms)

    def multiplicity(self) -> Tensor:
        return torch.bincount(self.voxel_index, minlength=self.n_voxels)

    def reduce_to_atoms(self, values: Tensor) -> Tensor:
        shape = (self.n_atoms, *values.shape[1:])
        return values.new_zeros(shape).index_add(0, self.atom_index, values)

    def reduce_to_grid(self, values: Tensor) -> Tensor:
        shape = (self.n_voxels, *values.shape[1:])
        return values.new_zeros(shape).index_add(0, self.voxel_index, values)


def compact_window(radius: Tensor, cutoff: Tensor) -> Tensor:
    """C2 cutoff: exactly zero with zero first/second derivative at support edge."""
    x = ((radius / cutoff - 0.75) / 0.25).clamp(0.0, 1.0)
    return 1 - 10 * x**3 + 15 * x**4 - 6 * x**5


def build_support(
    world: World,
    layout: BasisLayout,
    *,
    cutoff_sigma: float = 4.5,
    max_candidates: int = 250_000,
    max_edges: int = 8_000_000,
) -> SupportMap:
    """Batched periodic local stencils; all atoms read the same positions.

    Topology selection is discrete but the sampled basis has a C2 envelope
    at the cutoff. Autodiff differentiates within the current topology.
    The cutoff is a numerical tail tolerance, NOT a hard atomic radius.
    """
    g, a = world.grid, world.atoms
    scale = a.log_width_scale.exp()
    valid = layout.padded_basis >= 0
    widths = layout.padded_width * scale[:, None]
    largest = torch.where(valid, widths, torch.zeros_like(widths)).amax(dim=1)
    radii = cutoff_sigma * torch.maximum(largest, layout.core_smearing)
    if float(radii.detach().max()) >= 0.5 * min(g.lengths):
        raise ResolutionError("Cloud support must be smaller than half the shortest box length")
    smallest = torch.where(valid, widths, torch.full_like(widths, float("inf"))).amin()
    if float(torch.minimum(smallest, layout.core_smearing.min()).detach()) < 0.5 * max(g.spacing):
        raise ResolutionError("Cloud/core width is under-resolved; refine the field lattice")
    if max_candidates < 1:
        raise CapacityError("max_candidates must be positive")
    # Bucket by stencil size. Buckets/chunks partition WORK, not physical time:
    # every displacement below reads the same immutable position tensor.
    extents = torch.ceil(radii.detach()[:, None] / g.vector(g.spacing, radii)).to(torch.int64) + 1
    groups, membership = torch.unique(extents, dim=0, return_inverse=True)
    owners_out, voxels_out, delta_out = [], [], []
    total_edges = 0
    for group_id, extent in enumerate(groups):
        atom_ids = torch.nonzero(membership == group_id, as_tuple=False).flatten()
        offsets = []
        for size, half_t in zip(g.shape, extent):
            half = int(half_t)
            offsets.append(
                torch.arange(-(size // 2), size - size // 2, device=a.positions.device)
                if 2 * half + 1 >= size
                else torch.arange(-half, half + 1, device=a.positions.device)
            )
        stencil = torch.stack(torch.meshgrid(*offsets, indexing="ij"), dim=-1).reshape(-1, 3)
        # Chunk both atoms AND stencil, so even a single broad atom is bounded.
        per_atom = min(stencil.shape[0], max_candidates)
        atom_chunk = max(1, max_candidates // per_atom)
        for selected in atom_ids.split(atom_chunk):
            positions = a.positions[selected]
            base = torch.floor(
                (positions - g.vector(g.origin, positions)) / g.vector(g.spacing, positions)
            ).long()
            for chunk in stencil.split(max(1, max_candidates // len(selected))):
                indices = torch.remainder(
                    base[:, None, :] + chunk[None, :, :],
                    torch.tensor(g.shape, device=positions.device),
                )
                centres = g.vector(g.origin, positions) + (
                    indices.to(positions.dtype) + 0.5
                ) * g.vector(g.spacing, positions)
                delta = g.minimum_image(centres - positions[:, None, :])
                keep = delta.square().sum(-1) < radii[selected, None].square()
                owner = selected[:, None].expand_as(keep)
                total_edges += int(keep.sum())
                if total_edges > max_edges:
                    raise CapacityError(
                        f"Support exceeds {max_edges:,} stored edges; reduce system/resolution or raise the explicit budget"
                    )
                owners_out.append(owner[keep])
                voxels_out.append(g.flatten(indices)[keep])
                delta_out.append(delta[keep])
    return SupportMap(
        torch.cat(owners_out),
        torch.cat(voxels_out),
        torch.cat(delta_out),
        radii,
        world.n_atoms,
        g.n_voxels,
    )
