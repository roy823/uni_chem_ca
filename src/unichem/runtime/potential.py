"""Synchronous nuclear dynamics driven solely by an energy/force callback."""

import math
from dataclasses import dataclass

import torch

from ..errors import StateError
from .coupled import ou_refresh
from .random import CounterNoise


@dataclass(frozen=True)
class NuclearFrame:
    positions: torch.Tensor
    momenta: torch.Tensor
    time_au: float
    energy: float
    bath_heat: float
    balance_residual: float
    diagnostics: dict
    step: int


def integrate_potential(
    positions,
    momenta,
    masses,
    force_callback,
    *,
    atom_ids,
    steps,
    dt,
    kT=0.0,
    friction=0.0,
    seed=0,
    start_step=0,
):
    """BAOAB with atom-batched proposals, explicit heat and no reaction rules.

    The callback defines the potential; observers never enter this function.
    Positions remain unwrapped for an isolated molecular reference.
    """
    if (
        type(steps) is not int
        or steps < 0
        or not math.isfinite(dt)
        or dt <= 0
        or type(start_step) is not int
        or start_step < 0
    ):
        raise StateError("Invalid potential integration parameters")
    n = len(positions)
    if (
        positions.shape != (n, 3)
        or momenta.shape != positions.shape
        or masses.shape != (n,)
        or atom_ids.shape != (n,)
        or atom_ids.dtype != torch.int64
        or atom_ids.device != positions.device
        or atom_ids.unique().numel() != n
    ):
        raise StateError("Nuclear arrays and persistent atom IDs are inconsistent")
    if (
        not torch.isfinite(positions).all()
        or not torch.isfinite(momenta).all()
        or not torch.isfinite(masses).all()
        or (masses <= 0).any()
    ):
        raise StateError("Nuclear initial state must be finite with positive masses")
    r = positions.detach().clone()
    p = momenta.detach().clone()
    mass = masses[:, None]
    noise = CounterNoise(seed)
    energy, force, diagnostics = force_callback(r)
    initial = float(energy) + (p.square() / (2 * mass)).sum().item()
    heat = 0.0
    yield NuclearFrame(
        r.clone(), p.clone(), start_step * dt, float(energy), 0.0, 0.0, diagnostics, start_step
    )
    for step in range(steps):
        half = p + 0.5 * dt * force.detach()
        middle = r + 0.5 * dt * half / mass
        refreshed, q = ou_refresh(
            half,
            mass,
            friction=friction,
            kT=kT,
            dt=dt,
            noise=noise.normal(atom_ids, 3, step=start_step + step, phase="nuclear_potential"),
        )
        proposed = middle + 0.5 * dt * refreshed / mass
        new_energy, new_force, new_diagnostics = force_callback(proposed)
        new_p = refreshed + 0.5 * dt * new_force.detach()
        if not torch.isfinite(new_p).all() or not torch.isfinite(proposed).all():
            raise StateError("Nonfinite nuclear proposal; no state committed")
        r, p, energy, force = proposed.detach(), new_p.detach(), new_energy, new_force
        heat += float(q)
        balance = float(energy) + (p.square() / (2 * mass)).sum().item() - initial - heat
        yield NuclearFrame(
            r.clone(),
            p.clone(),
            (start_step + step + 1) * dt,
            float(energy),
            heat,
            balance,
            new_diagnostics,
            start_step + step + 1,
        )
