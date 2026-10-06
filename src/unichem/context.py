from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .fields.multiscale import GridTransfer
from .fields.poisson import FieldView
from .grid import Grid
from .quantum.basis import BasisSamples
from .quantum.density import QuantumView
from .quantum.observables import AtomicCloudMoments
from .quantum.support import BasisLayout, SupportMap
from .state import World


@dataclass(frozen=True)
class PreparedWorld:
    """Derived views of ONE coherent snapshot; never reused after its state changes."""

    world: World
    layout: BasisLayout
    support: SupportMap
    samples: BasisSamples
    quantum: QuantumView
    clouds: AtomicCloudMoments
    fields: FieldView

    def sample_at_atoms(self, grid_values: Tensor, *, grid: Grid | None = None) -> Tensor:
        """Cloud-averaged field, not a nearest-voxel lookup."""
        if grid is not None and grid != self.world.grid:
            grid_values = GridTransfer(self.world.grid, grid).prolong(grid_values)
        values = grid_values.reshape(self.world.grid.n_voxels, -1)
        weights = self.samples.reference_per_edge
        normalizer = self.support.reduce_to_atoms(weights)
        weighted = values[self.support.voxel_index] * weights[:, None]
        return self.support.reduce_to_atoms(weighted) / normalizer[:, None]
