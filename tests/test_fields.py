import math

import pytest
import torch

from unichem import Grid
from unichem.errors import ChargeError
from unichem.fields.poisson import PeriodicPoisson


def test_manufactured_periodic_poisson_solution():
    grid = Grid((16, 12, 10), (0.4, 0.5, 0.6))
    rho = torch.cos(2 * math.pi * (torch.arange(16, dtype=torch.float64) + 0.5) / 16)[
        :, None, None
    ].expand(grid.shape)
    solver = PeriodicPoisson()
    result = solver.solve(grid, rho)
    eigen = 4 * math.sin(math.pi / 16) ** 2 / grid.spacing[0] ** 2
    assert torch.allclose(result.potential, 4 * math.pi * rho / eigen, atol=2e-12, rtol=2e-12)
    assert float(solver.residual(grid, result).abs().max()) < 1e-11
    assert abs(float(result.potential.mean())) < 1e-12


def test_discrete_field_energy_identity():
    grid = Grid((8,) * 3, (0.5,) * 3)
    rng = torch.Generator().manual_seed(4)
    rho = torch.randn(grid.shape, generator=rng, dtype=torch.float64)
    rho -= rho.mean()
    result = PeriodicPoisson().solve(grid, rho)
    field_energy = result.electric_faces.square().sum() * grid.dv / (8 * math.pi)
    assert float(field_energy) == pytest.approx(float(result.electrostatic_energy), rel=1e-12)


def test_periodic_charge_policy_is_explicit():
    grid = Grid((8,) * 3, (0.5,) * 3)
    rho = torch.ones(grid.shape, dtype=torch.float64)
    with pytest.raises(ChargeError):
        PeriodicPoisson().solve(grid, rho)
    result = PeriodicPoisson(neutralizing_background=True).solve(grid, rho)
    assert float(result.background_charge) == 64.0
    assert result.potential.count_nonzero() == 0


def test_poisson_energy_gradient_is_potential():
    grid = Grid((8,) * 3, (0.5,) * 3)
    rng = torch.Generator().manual_seed(11)
    raw = torch.randn(grid.shape, generator=rng, dtype=torch.float64, requires_grad=True)
    rho = raw - raw.mean()
    result = PeriodicPoisson().solve(grid, rho)
    derivative = torch.autograd.grad(result.electrostatic_energy, raw)[0] / grid.dv
    assert torch.allclose(derivative, result.potential, atol=2e-12, rtol=2e-12)
