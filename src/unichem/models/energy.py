from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor, nn

from ..context import PreparedWorld
from ..errors import StateError
from ..spatial.neighbors import CellList


class EnergyFunctional(Protocol):
    def __call__(self, context: PreparedWorld) -> dict[str, Tensor]: ...


class DemoOrbitalEnergy(nn.Module):
    """Numerical reference ONLY: finite-difference orbital kinetic energy,
    smoothed-core Coulomb, and an optional local exchange-like term.

    Missing: a validated frozen-core pseudopotential, correlation, controlled
    basis convergence and fitted chemistry. The Coulomb energy contains smooth
    core self energies. This is not DFT and must not be used for reaction claims.
    """

    def __init__(self, exchange_strength: float = 0.20):
        super().__init__()
        self.exchange_strength = exchange_strength

    def forward(self, context: PreparedWorld) -> dict[str, Tensor]:
        g, q = context.world.grid, context.quantum
        kinetic = q.kinetic_energy
        # Smooth the density power at vacuum for finite higher derivatives.
        eps = 1e-10
        exchange = (
            -self.exchange_strength * ((q.density + eps) ** (4 / 3) - eps ** (4 / 3)).sum() * g.dv
        )
        return {
            "kinetic": kinetic,
            "coulomb": context.fields.electrostatic_energy,
            "demo_exchange": exchange,
        }


class ZeroEnergy(nn.Module):
    """An exact free-flight test component, not a physical default."""

    def forward(self, context: PreparedWorld) -> dict[str, Tensor]:
        return {
            "zero": context.world.atoms.positions.sum() * 0
            + context.world.electrons.kernel.values.sum() * 0
        }


class SharedScalarResidual(nn.Module):
    """Trainable shared atomic + radial-neighbour residual energy.

    No species-specific update programs and no bond/reaction lookup table.
    Scalar cloud populations/radii couple the residual to electronic state.
    Radial descriptors are rotational invariants in the continuum; the sampled
    grid still has finite discretisation anisotropy. This is NOT a full E(3)
    tensor network. Neighbourhoods use a periodic linked-cell list. No dense atom-pair table is stored.
    """

    def __init__(self, hidden: int = 24, cutoff: float = 5.0, output_scale: float = 0.03):
        super().__init__()
        self.cutoff, self.output_scale = cutoff, output_scale
        self.site = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.pair = nn.Sequential(
            nn.Linear(5, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, context: PreparedWorld) -> Tensor:
        w = context.world
        if self.cutoff >= 0.5 * min(w.grid.lengths):
            raise StateError(
                "Smooth short-range residual requires cutoff < half the shortest periodic box length"
            )
        atomic_z = context.layout.atomic_numbers / 10
        population = context.clouds.electron_number / (context.layout.core_charge + 1)
        radius = context.clouds.rms_radius
        scale = w.atoms.log_width_scale
        features = torch.stack((atomic_z, population, radius, scale), dim=-1)
        energy = self.site(features).sum()
        pairs = CellList(self.cutoff)(w.grid, w.atoms.positions)
        i, j, distance = pairs.first, pairs.second, pairs.distance
        x = (distance / self.cutoff).clamp(0, 1)
        envelope = 1 - 10 * x**3 + 15 * x**4 - 6 * x**5
        pair_features = torch.stack(
            (
                atomic_z[i] + atomic_z[j],
                atomic_z[i] * atomic_z[j],
                population[i] + population[j],
                population[i] * population[j],
                x,
            ),
            dim=-1,
        )
        energy = energy + (self.pair(pair_features).squeeze(-1) * envelope).sum()
        return self.output_scale * energy


class AnalyticTeacherResidual(nn.Module):
    """Smooth synthetic label generator. It contains no ab-initio knowledge."""

    def forward(self, context: PreparedWorld) -> Tensor:
        w = context.world
        i, j = torch.triu_indices(w.n_atoms, w.n_atoms, 1, device=w.atoms.positions.device)
        d = w.grid.minimum_image(w.atoms.positions[i] - w.atoms.positions[j])
        pair = 0.02 * torch.exp(-d.square().sum(-1) / 6).sum()
        return pair + 0.015 * context.clouds.rms_radius.square().sum()
