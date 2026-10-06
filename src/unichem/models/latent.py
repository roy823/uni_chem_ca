from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from ..context import PreparedWorld
from ..errors import StateError
from ..fields.multiscale import GridTransfer
from ..state import LatentState


class ConservativeLatentCA(nn.Module):
    """Shared synchronous nearest-neighbour flux rule plus local memory.

    This is a learnable LATENT diffusion scaffold, not calibrated chemical
    kinetics, not a GENERIC model, and not a random thermostat. Each channel
    sum is conserved by telescoping face fluxes on the periodic grid.
    """

    def __init__(
        self,
        channels: int,
        hidden: int = 12,
        max_mobility: float = 0.1,
        memory_retention: float = 0.95,
    ):
        super().__init__()
        if channels < 1 or max_mobility <= 0 or not (0 <= memory_retention <= 1):
            raise StateError("Invalid latent rule parameters")
        self.channels, self.max_mobility = channels, max_mobility
        self.memory_retention = memory_retention
        self.gate = nn.Sequential(
            nn.Linear(2 * channels + 1, hidden), nn.Tanh(), nn.Linear(hidden, channels)
        )

    def forward(self, context: PreparedWorld, *, dt: float) -> LatentState:
        w, g = context.world, context.world.latent_grid
        if w.latent.kind != "filter":
            raise StateError("Diffusive filter cannot overwrite auxiliary momentum memory")
        old, memory = w.latent.channels, w.latent.memory
        if old.shape[-1] != self.channels:
            raise StateError("Latent channel count does not match the shared rule")
        if dt <= 0 or 2 * dt * self.max_mobility * sum(1 / h**2 for h in g.spacing) > 1 + 1e-12:
            raise StateError("Latent explicit flux step violates the sufficient stability bound")
        density = GridTransfer(w.grid, g).restrict(
            context.quantum.density.reshape(*w.grid.shape, 1)
        )
        features = torch.cat((old, memory, torch.log1p(density)), dim=-1)
        mobility = self.max_mobility * torch.sigmoid(self.gate(features))
        change = torch.zeros_like(old)
        # Three axis operators, NOT a sequential loop over atoms or voxels.
        # Every face reads `old`, never an already-updated neighbour.
        for axis, h in enumerate(g.spacing):
            next_old = torch.roll(old, -1, dims=axis)
            face_mobility = 0.5 * (mobility + torch.roll(mobility, -1, dims=axis))
            flux = face_mobility * (next_old - old) / h
            change = change + (flux - torch.roll(flux, 1, dims=axis)) / h
        new_memory = self.memory_retention * memory + (1 - self.memory_retention) * old
        return replace(w.latent, channels=old + dt * change, memory=new_memory)
