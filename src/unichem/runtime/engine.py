from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from ..errors import StateError
from ..evaluate import Evaluator
from ..models.latent import ConservativeLatentCA
from ..quantum.observables import electronic_invariants
from ..state import World
from .electrons import LocalKernelRelaxer, RelaxationReport
from .guards import validate_transition


@dataclass(frozen=True)
class StepResult:
    world: World
    diagnostics: dict
    electronic_reports: tuple[RelaxationReport, RelaxationReport]


class SynchronousEngine:
    """Double-buffered, phase-synchronous extended-cell execution.

    Phase 1: electronic/field relaxation at R_t; shared snapshot barrier.
    Phase 2: ALL nuclei kick/drift together.
    Phase 3: electronic/field relaxation at R_{t+dt}; shared snapshot barrier.
    Phase 4: ALL nuclei finish the kick; ALL latent cells update from old latent.
    Phase 5: one validated commit to a new World, no aliases to input tensors.

    The integrator is velocity Verlet with approximate BO forces. Neither
    unfinished electronic solves nor latent updates prove correct kinetics.
    Rollouts detach each step; differentiable E/F training is a separate API.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        dt: float = 0.1,
        relaxer: LocalKernelRelaxer | None = None,
        latent_rule: ConservativeLatentCA | None = None,
        require_electronic_convergence: bool = False,
        max_displacement_fraction: float = 0.25,
    ):
        if dt <= 0 or not dt < float("inf") or max_displacement_fraction <= 0:
            raise StateError("Invalid timestep or displacement safety fraction")
        self.evaluator, self.dt = evaluator, dt
        self.relaxer = relaxer if relaxer is not None else LocalKernelRelaxer()
        self.latent_rule = latent_rule
        self.require_convergence = require_electronic_convergence
        self.max_displacement_fraction = max_displacement_fraction

    def step(self, world: World) -> StepResult:
        world.validate()
        if self.latent_rule is not None and self.evaluator.coupling is not None:
            raise StateError(
                "A passive filter must not drive an energy-coupled field; use CoupledSynchronousEngine"
            )
        before = self.relaxer.run(world, self.evaluator)
        first = self.evaluator.energy_and_forces(before.world)
        context = first.evaluation.context
        mass = context.layout.masses[:, None]
        p_half = before.world.atoms.momenta + 0.5 * self.dt * first.forces.detach()
        displacement = self.dt * p_half / mass
        if float(displacement.abs().max()) > self.max_displacement_fraction * min(
            world.grid.spacing
        ):
            raise StateError(
                "Displacement too large for this grid; reduce dt, do not clip the physics"
            )
        drifted = replace(
            before.world,
            atoms=replace(
                before.world.atoms,
                positions=world.grid.wrap(before.world.atoms.positions + displacement),
                momenta=p_half,
            ),
        )
        after = self.relaxer.run(drifted, self.evaluator)
        second = self.evaluator.energy_and_forces(after.world)
        if self.require_convergence and not (before.report.converged and after.report.converged):
            raise StateError("Electronic solve did not converge; no state committed")
        p_new = p_half + 0.5 * self.dt * second.forces.detach()
        if self.latent_rule is None:
            latent = world.latent
        else:
            with torch.no_grad():
                # Old latent/memory and one coherent old-position field snapshot.
                latent = self.latent_rule(context, dt=self.dt)
        new_world = replace(
            after.world,
            atoms=replace(after.world.atoms, momenta=p_new),
            latent=latent,
            time=world.time + self.dt,
            step=world.step + 1,
        ).clone()
        validate_transition(world, new_world)
        final = second.evaluation
        invariants = electronic_invariants(new_world, final.context.quantum)
        kinetic = float((p_new.square() / (2 * mass)).sum())
        diagnostics = {
            **invariants,
            "step": new_world.step,
            "time_au": new_world.time,
            "potential_energy_hartree": float(final.energy.detach()),
            "nuclear_kinetic_hartree": kinetic,
            "total_energy_hartree": float(final.energy.detach()) + kinetic,
            "max_force": float(second.forces.detach().abs().max()),
            "net_charge": float(final.context.fields.net_charge.detach()),
            "electrons_converged": before.report.converged and after.report.converged,
            "electronic_residual": after.report.residual_norm,
            "voxels_per_atom": final.context.support.counts_per_atom().cpu().tolist(),
            "overlapping_voxels": int((final.context.support.multiplicity() > 1).sum()),
            "cloud_rms_bohr": final.context.clouds.rms_radius.detach().cpu().tolist(),
        }
        return StepResult(new_world, diagnostics, (before.report, after.report))
