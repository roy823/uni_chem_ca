"""Shared many-body radial/angular energy descriptors with gradient forces."""

from __future__ import annotations

import torch
from torch import nn

from ..grid import Grid
from ..spatial.neighbors import CellList


class LocalEnvironment(nn.Module):
    """Geometry-only baseline for controlled comparisons with quantum residuals.

    Vector and quadrupole contractions encode angular/many-body environments.
    No bonds, molecule identities, proton-transfer coordinates or products are
    model inputs. All lengths/energies use atomic units.
    """

    def __init__(self, hidden=48, radial=8, cutoff=8.0, reference_per_atom=0.0):
        super().__init__()
        self.hidden, self.radial, self.cutoff = hidden, radial, cutoff
        self.register_buffer("centres", torch.linspace(0.7, cutoff, radial))
        self.register_buffer("reference", torch.tensor(float(reference_per_atom)))
        self.register_buffer("reference_slope", torch.tensor(0.0))

    def descriptors(self, numbers, positions, box):
        # The cell-list only uses box lengths; a small >=4 grid avoids any
        # electronic discretization in this geometry-only baseline.
        grid = Grid((4, 4, 4), tuple(float(x) / 4 for x in box))
        pairs = CellList(self.cutoff)(grid, positions)
        first = torch.cat((pairs.first, pairs.second))
        second = torch.cat((pairs.second, pairs.first))
        delta = torch.cat((pairs.displacement, -pairs.displacement))
        distance = torch.sqrt(delta.square().sum(-1) + 1e-18)
        x = (distance / self.cutoff).clamp(0, 1)
        envelope = 1 - 10 * x**3 + 15 * x**4 - 6 * x**5
        radial = (
            torch.exp(
                -(((distance[:, None] - self.centres) / (0.6 + self.cutoff / self.radial)) ** 2)
            )
            * envelope[:, None]
        )
        z = numbers.to(positions.dtype) / 8
        weights = torch.cat((radial, radial * z[second, None]), -1)
        unit = delta / distance[:, None]
        n = len(numbers)
        c = 2 * self.radial
        scalar = positions.new_zeros(n, c).index_add(0, first, weights)
        vector = positions.new_zeros(n, c, 3).index_add(
            0, first, weights[:, :, None] * unit[:, None, :]
        )
        tensor = positions.new_zeros(n, c, 3, 3).index_add(
            0, first, weights[:, :, None, None] * unit[:, None, :, None] * unit[:, None, None, :]
        )
        return torch.cat(
            (z[:, None], scalar, vector.square().sum(-1), tensor.square().sum((-1, -2))), -1
        )


class LocalEnergyModel(LocalEnvironment):
    def __init__(self, hidden=48, radial=8, cutoff=8.0, reference_per_atom=0.0):
        super().__init__(hidden, radial, cutoff, reference_per_atom)
        self.net = nn.Sequential(
            nn.Linear(1 + 6 * radial, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def energy(self, numbers, positions, box):
        return (
            self.net(self.descriptors(numbers, positions, box)).sum()
            + self.reference * len(numbers)
            + self.reference_slope * numbers.sum()
        )

    def energy_and_forces(self, numbers, positions, box, *, create_graph=False):
        p = positions.detach().clone().requires_grad_(True)
        energy = self.energy(numbers, p, box)
        forces = -torch.autograd.grad(energy, p, create_graph=create_graph)[0]
        return energy, forces


class LocalQuantumResidual(nn.Module):
    """Many-body geometry plus density-derived scalar electronic features."""

    def __init__(self, hidden=48, radial=8, cutoff=8.0, reference_per_atom=0.0):
        super().__init__()
        self.geometry = LocalEnvironment(hidden, radial, cutoff, reference_per_atom)
        self.net = nn.Sequential(
            nn.Linear(5 + 6 * radial, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.seed_site = nn.Sequential(
            nn.Linear(4 + 6 * radial, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        self.seed_pair = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.seed_width = nn.Sequential(
            nn.Linear(1 + 6 * radial, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        for net in (self.seed_site, self.seed_pair, self.seed_width):
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def initialize_kernel(self, world):
        """Shared learned patch coordinates and widths, with no bond labels.

        Edge blocks use real angular factors so their tensor axes follow the
        AO frame. Geometry/field energy then induces the unrolled response.
        The positive local backend enforces the quantum constraints afterward.
        """
        from dataclasses import replace

        from ..quantum.angular import real_angular
        from ..quantum.support import make_layout

        layout = make_layout(world)
        k = world.electrons.kernel
        box = world.atoms.positions.new_tensor(world.grid.lengths)
        descriptor = self.geometry.descriptors(layout.atomic_numbers, world.atoms.positions, box)
        width = 0.25 * torch.tanh(self.seed_width(descriptor).squeeze(-1))
        active = layout.padded_basis >= 0
        n = descriptor.new_tensor(
            [[o.n for o in s.orbitals] + [1] * (k.width - len(s.orbitals)) for s in world.species]
        )
        ell = descriptor.new_tensor(
            [[o.ell for o in s.orbitals] + [0] * (k.width - len(s.orbitals)) for s in world.species]
        )
        inputs = torch.cat(
            (
                descriptor[:, None, :].expand(-1, k.width, -1),
                torch.stack((n / 3, ell / 2, layout.padded_width), -1),
            ),
            -1,
        )
        onsite = 2 * torch.tanh(self.seed_site(inputs).squeeze(-1)) * active
        delta = world.grid.minimum_image(
            world.atoms.positions[k.col] - world.atoms.positions[k.row]
        )
        distance = torch.sqrt(delta.square().sum(-1) + 1e-18)
        direction = delta / distance[:, None]
        z = layout.atomic_numbers / 8
        features = torch.stack(
            (z[k.row] + z[k.col], z[k.row] * z[k.col], distance / world.electrons.cutoff), -1
        )
        amplitude = 2 * torch.tanh(self.seed_pair(features).squeeze(-1))
        ai = real_angular(
            direction, layout.padded_axis[k.row], torch.ones_like(layout.padded_width[k.row])
        )
        aj = real_angular(
            -direction, layout.padded_axis[k.col], torch.ones_like(layout.padded_width[k.col])
        )
        edge = amplitude[:, None, None] * ai[:, :, None] * aj[:, None, :]
        valid = active[k.row, :, None] & active[k.col, None, :]
        same = k.row == k.col
        proposal = (
            k.values
            + torch.where(same[:, None, None], torch.diag_embed(onsite[k.row]), edge) * valid
        )
        return replace(
            world.with_kernel_values(proposal.clamp(-8, 8)),
            atoms=replace(world.atoms, log_width_scale=width),
        )

    def forward(self, context):
        w = context.world
        numbers = context.layout.atomic_numbers
        box = w.atoms.positions.new_tensor(w.grid.lengths)
        geometry = self.geometry.descriptors(numbers, w.atoms.positions, box)
        charge = (context.layout.core_charge.sum() - w.electron_count) / w.n_atoms
        electronics = torch.stack(
            (
                context.clouds.electron_number / 8,
                context.clouds.rms_radius,
                w.atoms.log_width_scale,
                charge.expand(w.n_atoms),
            ),
            -1,
        )
        return (
            self.net(torch.cat((geometry, electronics), -1)).sum()
            + self.geometry.reference * w.n_atoms
            + self.geometry.reference_slope * numbers.sum()
        )
