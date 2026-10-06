import numpy as np
import pytest
import torch

from unichem import Evaluator, Grid, LocalQuantumBackend, make_world
from unichem.analysis.reactions import observe_protons
from unichem.analysis.statistics import equilibrium_event_statistics
from unichem.fields.isolated import IsolatedPoisson
from unichem.models.local_energy import LocalEnergyModel, LocalQuantumResidual
from unichem.quantum.patches import refresh_patch_graph
from unichem.runtime.potential import integrate_potential
from unichem.species import HO_RESEARCH_SPECIES
from unichem.training import KernelUnrolledPredictor


def test_isolated_coulomb_energy_gradient_is_potential():
    grid = Grid((7, 8, 9), (0.4, 0.5, 0.6))
    gen = torch.Generator().manual_seed(911)
    rho = torch.randn(grid.shape, generator=gen, dtype=torch.float64, requires_grad=True)
    result = IsolatedPoisson().solve(grid, rho)
    gradient = torch.autograd.grad(result.electrostatic_energy, rho)[0] / grid.dv
    assert torch.allclose(gradient, result.potential, atol=2e-12, rtol=2e-12)
    assert result.background_charge == 0
    assert float(result.net_charge) == pytest.approx(float(rho.sum() * grid.dv))


def test_isolated_point_pair_matches_free_space_green_function():
    grid = Grid((8, 8, 8), (0.6, 0.6, 0.6))
    rho = torch.zeros(grid.shape, dtype=torch.float64)
    rho[1, 2, 3] = 1 / grid.dv
    result = IsolatedPoisson().solve(grid, rho)
    assert float(result.potential[5, 2, 3]) == pytest.approx(1 / (4 * 0.6), abs=1e-12)
    assert float(result.potential[1, 6, 3]) == pytest.approx(1 / (4 * 0.6), abs=1e-12)


def test_geometry_model_energy_invariance_and_force_covariance():
    model = LocalEnergyModel(hidden=8).double()
    z = torch.tensor([8, 1, 1])
    p = torch.tensor(
        [[12.0, 12.0, 12.0], [13.8, 12.0, 12.0], [11.6, 13.7, 12.0]], dtype=torch.float64
    )
    box = torch.tensor([30.0, 30.0, 30.0], dtype=torch.float64)
    gen = torch.Generator().manual_seed(612)
    q = torch.linalg.qr(torch.randn(3, 3, generator=gen, dtype=p.dtype)).Q
    rotated = (p - 12) @ q.T + 12
    e, f = model.energy_and_forces(z, p, box)
    er, fr = model.energy_and_forces(z, rotated, box)
    et, ft = model.energy_and_forces(z, p + 0.234, box)
    assert float(e) == pytest.approx(float(er), abs=1e-12)
    assert float(e) == pytest.approx(float(et), abs=1e-12)
    assert torch.allclose(fr, f @ q.T, atol=2e-12)
    assert torch.allclose(ft, f, atol=2e-12)
    assert f.sum(0).abs().max() < 1e-12


def test_learned_patch_initializer_is_ao_rotation_covariant():
    grid = Grid((48,) * 3, (0.6,) * 3)
    p = torch.tensor(
        [[14.4, 14.4, 14.4], [16.2, 14.4, 14.4], [14.0, 16.1, 14.4]], dtype=torch.float64
    )
    w = make_world(["O", "H", "H"], p, grid=grid, registry=HO_RESEARCH_SPECIES)
    m = LocalQuantumResidual(hidden=8).double()
    with torch.no_grad():
        m.seed_pair[-1].bias.fill_(0.17)
    gen = torch.Generator().manual_seed(424)
    q = torch.linalg.qr(torch.randn(3, 3, generator=gen, dtype=p.dtype)).Q
    old = m.initialize_kernel(w)
    new = m.initialize_kernel(w.with_positions((p - 14.4) @ q.T + 14.4))
    transform = torch.eye(5, dtype=p.dtype)
    transform[2:, 2:] = q
    assert torch.allclose(
        new.electrons.kernel.values,
        transform @ old.electrons.kernel.values @ transform.T,
        atol=2e-12,
    )
    assert torch.allclose(old.atoms.log_width_scale, new.atoms.log_width_scale, atol=2e-12)


def test_density_supervision_reaches_learned_patch_and_width_parameters():
    grid = Grid((28,) * 3, (0.6,) * 3)
    w = make_world(
        ["O", "H", "H"],
        [[8.0, 8.0, 8.0], [9.8, 8.0, 8.0], [7.6, 9.7, 8.0]],
        grid=grid,
        registry=HO_RESEARCH_SPECIES,
    )
    model = LocalQuantumResidual(hidden=6).double()
    predictor = KernelUnrolledPredictor(
        Evaluator(
            backend=LocalQuantumBackend(metric_degree=8), poisson=IsolatedPoisson(), residual=model
        ),
        kernel_initializer=model.initialize_kernel,
    )
    p = predictor(w)
    probe = torch.linspace(0, 1, w.grid.n_voxels, dtype=torch.float64)
    loss = (p.evaluation.context.quantum.density * probe).sum() * w.grid.dv
    loss.backward()
    for name in ["seed_site", "seed_pair", "seed_width"]:
        gradients = [v.grad for v in getattr(model, name).parameters() if v.grad is not None]
        assert all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 1e-10


def test_dynamic_contact_graph_does_not_change_zero_interface_energy(h2):
    raw = h2.electrons.kernel.values.clone()
    raw[h2.electrons.kernel.row != h2.electrons.kernel.col] = 0.5
    p = h2.atoms.positions.clone()
    p[1, 0] = p[0, 0] + 5.0001
    w = h2.with_kernel_values(raw).with_positions(p)
    refreshed = refresh_patch_graph(w)
    assert len(refreshed.electrons.kernel.row) == 2
    assert float(Evaluator()(w).energy) == pytest.approx(
        float(Evaluator()(refreshed).energy), abs=1e-11
    )


def test_nve_energy_error_decreases_at_fixed_duration():
    errors = []
    r = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)

    def harmonic(p):
        return 0.5 * p.square().sum(), -p, {}

    for steps in [16, 32, 64]:
        frames = list(
            integrate_potential(
                r,
                torch.zeros_like(r),
                torch.ones(2, dtype=r.dtype),
                harmonic,
                atom_ids=torch.arange(len(r)),
                steps=steps,
                dt=2 / steps,
            )
        )
        errors.append(max(abs(x.balance_residual) for x in frames))
    assert errors[1] < errors[0] / 3 and errors[2] < errors[1] / 3


def test_no_event_statistics_do_not_invent_free_energy_or_rates():
    result = equilibrium_event_statistics(
        [np.zeros(100, dtype=bool)] * 3, 0.1, 0.001, equilibrated=True
    )
    assert result["free_energy_hartree"] is None and result["rates_per_fs"] is None


def test_equilibrium_bootstrap_has_explicit_kinetic_clock_requirement():
    trace = np.tile([False, False, False, True], 100)
    result = equilibrium_event_statistics([trace, trace, trace], 0.1, 0.001, equilibrated=True)
    assert result["free_energy_hartree"] == pytest.approx(0.001 * np.log(3))
    assert result["rates_per_fs"] is None


def test_reaction_observer_is_read_only_and_reports_candidates():
    z = np.array([8, 8, 1, 1, 1, 1])
    p = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [-0.5, 1.7, 0.0],
            [5.5, 1.7, 0.0],
            [5.5, -1.7, 0.0],
        ]
    )
    saved = p.copy()
    old = observe_protons(z, p)
    assert not old["ionic_pair_candidate"]
    p[2] = [3.3, 0.0, 0.0]
    current = observe_protons(z, p, previous=old)
    assert current["ionic_pair_candidate"] and len(current["proton_owner_changes"]) == 1
    assert np.array_equal(
        saved,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [1.8, 0.0, 0.0],
                [-0.5, 1.7, 0.0],
                [5.5, 1.7, 0.0],
                [5.5, -1.7, 0.0],
            ]
        ),
    )
