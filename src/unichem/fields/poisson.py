from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..errors import ChargeError
from ..grid import Grid


@dataclass(frozen=True)
class FieldView:
    charge_density: Tensor
    potential: Tensor
    electric_faces: Tensor
    electrostatic_energy: Tensor
    net_charge: Tensor
    background_charge: Tensor


class PeriodicPoisson:
    """Solve -Delta_h phi = 4*pi*rho in atomic units, with mean(phi)=0.

    Explicit electrons/cores use vacuum Coulomb, NOT an added bulk-water epsilon.
    A nonneutral periodic cell raises unless a neutralising background is
    explicitly requested. That convention is reported, never silently assumed.

    E[...,a] is on the positive a FACE, NOT at a cell centre. Forward gradient
    and backward divergence give an exact discrete electrostatic energy identity.
    """

    def __init__(self, *, neutralizing_background: bool = False, charge_tolerance: float = 1e-7):
        self.neutralizing_background = neutralizing_background
        self.charge_tolerance = charge_tolerance

    def solve(self, grid: Grid, charge_density: Tensor) -> FieldView:
        rho = charge_density.reshape(grid.shape)
        total = rho.sum() * grid.dv
        # Scale the neutral-roundoff allowance with dtype and deposited charge.
        # A float32 neutral deposition must not be mistaken for a charged cell.
        rounding = 32 * torch.finfo(rho.dtype).eps * float((rho.abs().sum() * grid.dv).detach())
        tolerance = max(self.charge_tolerance, rounding)
        if abs(float(total.detach())) > tolerance and not self.neutralizing_background:
            raise ChargeError(
                f"Periodic net charge {float(total.detach()):.6g}; "
                "set neutralizing_background=True only with an explicit charged-cell convention"
            )
        # Remove only the uniform k=0 component (roundoff for a neutral world).
        rhs = rho - rho.mean()
        eigen = rho.new_zeros(grid.shape)
        for axis, (size, h) in enumerate(zip(grid.shape, grid.spacing)):
            f = torch.fft.fftfreq(size, d=1.0, dtype=rho.dtype, device=rho.device)
            shape = [1, 1, 1]
            shape[axis] = size
            eigen = eigen + (4 * torch.sin(math.pi * f).square() / h**2).reshape(shape)
        nonzero = eigen > 0
        denominator = torch.where(nonzero, eigen, torch.ones_like(eigen))
        phi_k = 4 * math.pi * torch.fft.fftn(rhs) / denominator
        phi_k = torch.where(nonzero, phi_k, torch.zeros_like(phi_k))
        phi = torch.fft.ifftn(phi_k).real
        faces = torch.stack(
            [-(torch.roll(phi, -1, dims=a) - phi) / h for a, h in enumerate(grid.spacing)], dim=-1
        )
        energy = 0.5 * (rhs * phi).sum() * grid.dv
        return FieldView(rho, phi, faces, energy, total, total)

    @staticmethod
    def residual(grid: Grid, view: FieldView) -> Tensor:
        lap = torch.zeros_like(view.potential)
        for a, h in enumerate(grid.spacing):
            p = view.potential
            lap = lap + (torch.roll(p, -1, dims=a) - 2 * p + torch.roll(p, 1, dims=a)) / h**2
        return -lap - 4 * math.pi * (view.charge_density - view.charge_density.mean())
