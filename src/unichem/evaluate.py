from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .context import PreparedWorld
from .errors import StateError
from .fields.poisson import PeriodicPoisson
from .models.energy import DemoOrbitalEnergy, EnergyFunctional
from .quantum.basis import sample_basis
from .quantum.density import LocalQuantumBackend, QuantumBackend
from .quantum.observables import cloud_moments
from .quantum.support import build_support, make_layout
from .state import World


@dataclass(frozen=True)
class EnergyEvaluation:
    context: PreparedWorld
    components: dict[str, Tensor]
    energy: Tensor


@dataclass(frozen=True)
class ForceEvaluation:
    evaluation: EnergyEvaluation
    forces: Tensor


@dataclass(frozen=True)
class GeneralizedForceEvaluation:
    evaluation: EnergyEvaluation
    forces: Tensor
    latent_forces: Tensor


class Evaluator:
    """One energy definition; all nuclear forces come from differentiating it.

    Autodiff includes basis motion, metric polynomials and charge deposition.
    Local kernel coordinates are held independent in this call.
    A Born-Oppenheimer interpretation additionally needs converged electrons.
    create_graph=True supports force-supervised training (second derivatives).
    """

    def __init__(
        self,
        *,
        backend: QuantumBackend | None = None,
        poisson: PeriodicPoisson | None = None,
        functional: EnergyFunctional | None = None,
        residual: nn.Module | None = None,
        validate: bool = True,
        support_candidates: int = 250_000,
        support_edges: int = 8_000_000,
        coupling: nn.Module | None = None,
    ):
        self.backend = backend if backend is not None else LocalQuantumBackend()
        self.poisson = poisson if poisson is not None else PeriodicPoisson()
        self.functional = functional if functional is not None else DemoOrbitalEnergy()
        self.residual = residual
        self.coupling = coupling
        self.support_candidates, self.support_edges = support_candidates, support_edges
        self.validate = validate
        self._layouts = {}

    def __call__(self, world: World) -> EnergyEvaluation:
        if self.validate:
            world.validate()
        key = (world.species, world.atoms.positions.dtype, str(world.atoms.positions.device))
        if key not in self._layouts:
            self._layouts[key] = make_layout(world)
        layout = self._layouts[key]
        support = build_support(
            world, layout, max_candidates=self.support_candidates, max_edges=self.support_edges
        )
        samples = sample_basis(world, layout, support)
        quantum = self.backend.evaluate(world, samples)
        fields = self.poisson.solve(world.grid, samples.core_density - quantum.density)
        clouds = cloud_moments(world, support, samples, quantum)
        context = PreparedWorld(world, layout, support, samples, quantum, clouds, fields)
        components = self.functional(context)
        if any(world.external_field):
            # Uniform-field scalar energy is only an isolated/vacuum-box
            # research convention, not a periodic Berry-phase field solver.
            radii = support.radii
            edge = torch.minimum(
                world.atoms.positions - world.grid.vector(world.grid.origin, world.atoms.positions),
                world.grid.vector(world.grid.origin, world.atoms.positions)
                + world.grid.vector(world.grid.lengths, world.atoms.positions)
                - world.atoms.positions,
            )
            if bool((edge.amin(-1) <= radii).any()):
                raise StateError("Uniform external field requires clouds inside a vacuum box")
            dipole = (world.atoms.positions * layout.core_charge[:, None]).sum(0)
            dipole = (
                dipole
                - (world.grid.centres(world.atoms.positions) * quantum.density[:, None]).sum(0)
                * world.grid.dv
            )
            components = {
                **components,
                "external_field": -(
                    dipole * world.atoms.positions.new_tensor(world.external_field)
                ).sum(),
            }
        if self.residual is not None:
            components = {**components, "learned_residual": self.residual(context)}
        if self.coupling is not None:
            components = {**components, "latent_coupling": self.coupling(context)}
        if not components or any(v.ndim != 0 for v in components.values()):
            raise StateError("An energy functional must return named scalar tensors")
        energy = torch.stack(list(components.values())).sum()
        if not torch.isfinite(energy):
            raise StateError("Energy evaluation produced a non-finite value")
        return EnergyEvaluation(context, components, energy)

    def energy_and_forces(self, world: World, *, create_graph: bool = False) -> ForceEvaluation:
        with torch.enable_grad():
            # Preserve input immutability and keep the training graph through network parameters.
            positions = world.atoms.positions.detach().clone().requires_grad_(True)
            evaluation = self(world.with_positions(positions))
            energy = evaluation.energy + positions.sum() * 0
            force = -torch.autograd.grad(
                energy, positions, create_graph=create_graph, retain_graph=create_graph
            )[0]
        return ForceEvaluation(evaluation, force)

    def energy_and_generalized_forces(
        self, world: World, *, create_graph: bool = False
    ) -> GeneralizedForceEvaluation:
        """Reciprocal nuclear/latent derivatives of exactly ONE scalar energy.

        Electronic coordinates are held independent; envelope-theorem BO
        forces additionally require electronic stationarity. Derivatives here
        are ordinary array gradients for discrete canonical coordinates z_k,
        not continuum functional derivatives (which differ by cell volume).
        """
        with torch.enable_grad():
            positions = world.atoms.positions.detach().clone().requires_grad_(True)
            z = world.latent.channels.detach().clone().requires_grad_(True)
            evaluation = self(world.with_positions(positions).with_latent(z))
            energy = evaluation.energy + 0 * (positions.sum() + z.sum())
            dr, dz = torch.autograd.grad(
                energy, (positions, z), create_graph=create_graph, retain_graph=create_graph
            )
        return GeneralizedForceEvaluation(evaluation, -dr, -dz)
