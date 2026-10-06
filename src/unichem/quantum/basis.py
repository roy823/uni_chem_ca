from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..errors import CapacityError
from ..state import World
from .angular import real_angular
from .support import BasisLayout, SupportMap, compact_window


@dataclass(frozen=True)
class BasisSamples:
    """Sparse samples of atom-centred AOs plus positive per-atom reference weights."""

    voxel: Tensor
    basis: Tensor
    value: Tensor
    reference_per_edge: Tensor
    core_density: Tensor  # flattened grid, charge/bohr^3
    n_voxels: int
    n_basis: int

    def dense(self, *, max_entries: int = 8_000_000) -> Tensor:
        entries = self.n_voxels * self.n_basis
        if entries > max_entries:
            raise CapacityError(
                f"Dense AO reference backend needs {entries:,} entries; "
                "replace it with a block-sparse/streaming backend"
            )
        idx = self.voxel * self.n_basis + self.basis
        return (
            self.value.new_zeros(entries)
            .index_add(0, idx, self.value)
            .reshape(self.n_voxels, self.n_basis)
        )


def sample_basis(world: World, layout: BasisLayout, support: SupportMap) -> BasisSamples:
    atom, delta = support.atom_index, support.displacement
    radius = torch.sqrt(delta.square().sum(-1) + 1e-24)
    window = compact_window(radius, support.radii[atom])
    ids = layout.padded_basis[atom]
    axes = layout.padded_axis[atom]
    widths = layout.padded_width[atom] * world.atoms.log_width_scale[atom, None].exp()
    gauss = torch.exp(-0.5 * delta.square().sum(-1)[:, None] / widths.square())
    angular = real_angular(delta, axes, widths)
    ao_values = gauss * angular * window[:, None]
    valid = ids >= 0
    reference = torch.where(valid, ao_values.square(), torch.zeros_like(ao_values)).sum(-1)
    # A smooth, conservative effective-core deposition. Not a physical nuclear sphere.
    # The effective core has its OWN fixed support. Changing an electronic
    # cloud width must not change the represented nuclear/core charge cloud.
    core_window = compact_window(radius, 4.5 * layout.core_smearing[atom])
    core = (
        torch.exp(-0.5 * delta.square().sum(-1) / layout.core_smearing[atom].square()) * core_window
    )
    normalizer = support.reduce_to_atoms(core) * world.grid.dv
    core = core * layout.core_charge[atom] / normalizer[atom]
    core_density = support.reduce_to_grid(core)
    grid_ids = support.voxel_index[:, None].expand_as(ids)
    return BasisSamples(
        grid_ids[valid],
        ids[valid],
        ao_values[valid],
        reference,
        core_density,
        world.grid.n_voxels,
        world.n_basis,
    )
