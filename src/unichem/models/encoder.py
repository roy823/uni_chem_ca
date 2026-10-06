from __future__ import annotations

import torch
from torch import Tensor, nn

from ..context import PreparedWorld
from ..errors import StateError
from ..fields.multiscale import GridTransfer
from ..fields.particles import deposit_cubic


class CoarseFieldEncoder(nn.Module):
    """Shared local field encoder; no single fixed-size vector for the world.

    Nuclear mode depends ONLY on nuclei/species, not variational electrons or
    cloud widths. It supports a normalized auxiliary bath without changing a
    BO potential's equilibrium marginal (see ConditionalGaussianCoupling).
    Electronic mode includes density, charge and |E|, hence gauge-independent
    scalars, but DOES NOT retain that BO marginal guarantee.

    The same face-neighbour map is used along all axes. Cubic-lattice symmetries
    are respected; arbitrary continuum rotations are not exactly equivariant.
    Learned channels are NOT automatically slow reaction coordinates.
    """

    def __init__(self, channels: int = 4, hidden: int = 16, *, source: str = "nuclear"):
        super().__init__()
        if channels < 1 or hidden < 1 or source not in ("nuclear", "electronic"):
            raise StateError("Invalid coarse encoder configuration")
        self.channels, self.hidden, self.source = channels, hidden, source
        self.site = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.neighbour = nn.Linear(hidden, hidden, bias=False)
        self.decode = nn.Sequential(nn.SiLU(), nn.Linear(hidden, channels), nn.Tanh())

    @property
    def depends_on_electrons(self) -> bool:
        return self.source == "electronic"

    def forward(self, context: PreparedWorld) -> Tensor:
        w, g = context.world, context.world.latent_grid
        if self.source == "nuclear":
            payload = torch.stack(
                (
                    torch.ones_like(context.layout.atomic_numbers),
                    context.layout.atomic_numbers / 10,
                    context.layout.core_charge / 10,
                ),
                -1,
            )
            features = torch.log1p(deposit_cubic(g, w.atoms.positions, payload))
        else:
            transfer = GridTransfer(w.grid, g)
            density = context.quantum.density.reshape(*w.grid.shape)
            charge = context.fields.charge_density.reshape(*w.grid.shape)
            faces = context.fields.electric_faces
            field = torch.stack(
                [0.5 * (faces[..., a] + torch.roll(faces[..., a], 1, a)) for a in range(3)], -1
            )
            features = transfer.restrict(
                torch.stack((torch.log1p(density), charge, torch.log1p(field.square().sum(-1))), -1)
            )
        hidden = self.site(features)
        neighbours = sum(torch.roll(hidden, s, a) for a in range(3) for s in (-1, 1)) / 6
        return self.decode(hidden + self.neighbour(neighbours))
