"""Uniform synchronous nuclear proposals with self-consistent quantum patches."""

from dataclasses import dataclass, replace

import torch

from ..errors import StateError
from ..quantum.evaluator import Evaluator


@dataclass(frozen=True)
class StepResult:
    world: object
    evaluation: object
    diagnostics: dict


class SynchronousEngine:
    def __init__(self, evaluator=None, *, dt=0.05 / 0.024188843265857):
        if not 0 < dt < float("inf"):
            raise StateError("Positive finite time step required")
        self.evaluator = Evaluator() if evaluator is None else evaluator
        self.dt = dt

    def step(self, world):
        before = self.evaluator(world)
        mass = world.atoms.masses[:, None]
        half = world.atoms.momenta + 0.5 * self.dt * before.forces
        positions = world.atoms.positions + self.dt * half / mass
        proposal = replace(world.with_positions(positions), densities=before.electronic.densities)
        after = self.evaluator(proposal)
        momentum = half + 0.5 * self.dt * after.forces
        if not torch.isfinite(momentum).all():
            raise StateError("Nonfinite proposal; no state committed")
        committed = replace(
            proposal,
            atoms=replace(proposal.atoms, momenta=momentum),
            densities=after.electronic.densities,
            time=world.time + self.dt,
            step=world.step + 1,
        ).clone()
        committed.validate()
        kinetic = float((momentum.square() / (2 * mass)).sum())
        diag = {
            "step": committed.step,
            "time_au": committed.time,
            "potential_energy_hartree": after.energy,
            "kinetic_energy_hartree": kinetic,
            "total_energy_hartree": after.energy + kinetic,
            "electronic_commutator_norm": after.electronic.commutator_norm,
            "electron_number_errors": after.electronic.electron_errors,
            "purity_errors": after.electronic.purity_errors,
        }
        return StepResult(committed, after, diag)
