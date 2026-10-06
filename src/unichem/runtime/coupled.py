from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from ..errors import StateError
from ..evaluate import Evaluator
from ..quantum.observables import electronic_invariants
from ..state import World
from .electrons import LocalKernelRelaxer
from .engine import StepResult
from .guards import validate_transition
from .random import CounterNoise


@dataclass(frozen=True)
class BathConfig:
    """Constant-temperature classical bath; kT in Hartree, friction in 1/au_time.

    auxiliary_mass_density*coarse_cell_volume is a discrete conjugate mass for
    dimensionless latent coordinates, NOT the physical solvent mass density.
    A single kT is used for nuclear and latent baths; unequal temperatures
    would describe a driven system and invalidate equilibrium claims.
    """

    kT: float = 0.0
    nuclear_friction: float = 0.0
    latent_friction: float = 0.1
    auxiliary_mass_density: float = 1.0

    def __post_init__(self):
        if any(
            not math.isfinite(v) or v < 0
            for v in (self.kT, self.nuclear_friction, self.latent_friction)
        ):
            raise StateError("Bath parameters must be finite/nonnegative")
        if not 0 < self.auxiliary_mass_density < math.inf:
            raise StateError("Latent conjugate mass density must be positive")


def ou_refresh(
    momentum: Tensor, mass: Tensor | float, *, friction: float, kT: float, dt: float, noise: Tensor
) -> tuple[Tensor, Tensor]:
    """Exact constant-mass Ornstein-Uhlenbeck substep and measured bath heat."""
    if not all(math.isfinite(v) and v >= 0 for v in (friction, kT, dt)):
        raise StateError("Invalid OU parameters")
    mass = torch.as_tensor(mass, dtype=momentum.dtype, device=momentum.device)
    if not torch.isfinite(mass).all() or (mass <= 0).any() or noise.shape != momentum.shape:
        raise StateError("OU mass/noise shape is invalid")
    decay = math.exp(-friction * dt)
    variance_fraction = -math.expm1(-2 * friction * dt)
    updated = decay * momentum + torch.sqrt(mass * kT * variance_fraction) * noise
    heat = ((updated.square() - momentum.square()) / (2 * mass)).sum()
    return updated, heat


class CoupledSynchronousEngine:
    """BAOAB over nuclei AND auxiliary fields, with shared snapshot barriers.

    B: all nuclear/latent momenta receive forces from one energy evaluation.
    A: all nuclear/latent coordinates drift simultaneously for half a step.
    O: independently keyed, exact OU noise exchange (no sequential RNG state).
    A: second collective half drift.
    BO: electrons+electrostatic field relax at the new whole-world geometry.
    B: final shared-force half kick, invariants checked, one commit.

    This is an isothermal extended-system research engine, NOT closed-system
    GENERIC and NOT a validated surrogate of a chemical reaction clock.
    Electronic relaxation is numerical, not real-time quantum evolution.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        dt: float = 0.1,
        relaxer=None,
        bath: BathConfig | None = None,
        seed: int = 0,
        require_electronic_convergence: bool = False,
        max_displacement_fraction: float = 0.25,
    ):
        if not 0 < dt < math.inf or not 0 < max_displacement_fraction < math.inf:
            raise StateError("Invalid timestep or displacement safety bound")
        self.evaluator, self.dt = evaluator, dt
        self.relaxer = relaxer if relaxer is not None else LocalKernelRelaxer()
        self.bath, self.noise = bath if bath is not None else BathConfig(), CounterNoise(seed)
        self.require_convergence = require_electronic_convergence
        self.max_displacement_fraction = max_displacement_fraction

    def _force(self, world):
        result = self.relaxer.run(world, self.evaluator)
        if self.require_convergence and not result.report.converged:
            raise StateError("Electronic solve did not converge; no state committed")
        return result, self.evaluator.energy_and_generalized_forces(result.world)

    def step(self, world: World) -> StepResult:
        world.validate()
        if world.latent.kind != "auxiliary_momentum":
            raise StateError(
                "Coupled dynamics requires explicitly tagged auxiliary_momentum, not filter memory"
            )
        first, f0 = self._force(world)
        snapshot = first.world
        mass = f0.evaluation.context.layout.masses[:, None]
        zmass = self.bath.auxiliary_mass_density * world.latent_grid.dv
        dt, bath = self.dt, self.bath
        # No kernel below mutates `world` or another kernel's old-state read buffer.
        p0, q0 = snapshot.atoms.momenta, snapshot.latent.memory
        p = p0 + 0.5 * dt * f0.forces.detach()
        q = q0 + 0.5 * dt * f0.latent_forces.detach()
        delta1, dz1 = 0.5 * dt * p / mass, 0.5 * dt * q / zmass
        r_mid = snapshot.atoms.positions + delta1
        z_mid = snapshot.latent.channels + dz1
        xi = self.noise.normal(
            world.atoms.ids, 3, step=world.step, phase="nuclei.OU", dtype=p.dtype
        )
        grid_ids = torch.arange(world.latent_grid.n_voxels, device=p.device, dtype=torch.int64)
        eta = self.noise.normal(
            grid_ids, q.shape[-1], step=world.step, phase="latent.OU", dtype=q.dtype
        ).reshape(q.shape)
        p, qn = ou_refresh(p, mass, friction=bath.nuclear_friction, kT=bath.kT, dt=dt, noise=xi)
        q, qz = ou_refresh(q, zmass, friction=bath.latent_friction, kT=bath.kT, dt=dt, noise=eta)
        delta2 = 0.5 * dt * p / mass
        if float(
            torch.stack(
                (delta1.abs().max(), delta2.abs().max(), (delta1 + delta2).abs().max())
            ).max()
        ) > self.max_displacement_fraction * min(world.grid.spacing):
            raise StateError("Displacement too large; reduce dt, never clip the physical move")
        advanced = replace(
            snapshot,
            atoms=replace(snapshot.atoms, positions=world.grid.wrap(r_mid + delta2), momenta=p),
            latent=replace(snapshot.latent, channels=z_mid + 0.5 * dt * q / zmass, memory=q),
        )
        second, f1 = self._force(advanced)
        p = p + 0.5 * dt * f1.forces.detach()
        q = q + 0.5 * dt * f1.latent_forces.detach()
        candidate = replace(
            second.world,
            atoms=replace(second.world.atoms, momenta=p),
            latent=replace(second.world.latent, memory=q),
            time=world.time + dt,
            step=world.step + 1,
        ).clone()
        validate_transition(world, candidate)
        nuclear_kinetic = float((p.square() / (2 * mass)).sum())
        latent_kinetic = float((q.square() / (2 * zmass)).sum())
        initial = float(
            f0.evaluation.energy.detach()
            + (p0.square() / (2 * mass)).sum()
            + (q0.square() / (2 * zmass)).sum()
        )
        final = float(f1.evaluation.energy.detach()) + nuclear_kinetic + latent_kinetic
        heat = float(qn + qz)
        diagnostics = {
            **electronic_invariants(candidate, f1.evaluation.context.quantum),
            "step": candidate.step,
            "time_au": candidate.time,
            "potential_energy_hartree": float(f1.evaluation.energy.detach()),
            "nuclear_kinetic_hartree": nuclear_kinetic,
            "latent_kinetic_hartree": latent_kinetic,
            "total_energy_hartree": final,
            "bath_heat_hartree": heat,
            "mechanical_balance_residual_hartree": final - initial - heat,
            "electrons_converged": first.report.converged and second.report.converged,
            "electronic_residual": second.report.residual_norm,
            "electronic_residual_kind": second.report.residual_kind,
            "auxiliary_bo_marginal_condition": getattr(
                self.evaluator.coupling, "preserves_bo_marginal", False
            ),
            "max_force": float(f1.forces.detach().abs().max()),
            "cloud_rms_bohr": f1.evaluation.context.clouds.rms_radius.detach().cpu().tolist(),
            "voxels_per_atom": f1.evaluation.context.support.counts_per_atom().cpu().tolist(),
            "noise_seed": self.noise.seed,
            "integrator": "joint_BAOAB",
        }
        return StepResult(candidate, diagnostics, (first.report, second.report))
