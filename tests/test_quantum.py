from dataclasses import replace

import pytest
import torch

from unichem import Evaluator, Grid, LocalQuantumBackend, make_world
from unichem.errors import CapacityError, StateError
from unichem.quantum.blocks import BlockMatrix, constrained_kernel
from unichem.quantum.observables import electronic_invariants


def physical_eigenvalues(q, spin):
    s = q.overlap.dense_for_validation()
    eigen, u = torch.linalg.eigh(s)
    root = (u * eigen.clamp_min(0).sqrt()) @ u.T
    p = q.spin_kernels[spin].dense_for_validation()
    return torch.linalg.eigvalsh(root @ p @ root)


def test_global_pauli_and_number_constraints_hold_across_overlapping_patches(water):
    q = Evaluator()(water).context.quantum
    inv = electronic_invariants(water, q)
    assert inv["electron_integral"] == pytest.approx(8.0, abs=2e-11)
    assert inv["trace_PS"] == pytest.approx(8.0, abs=2e-11)
    assert inv["minimum_density"] >= -1e-12
    for spin in range(2):
        eig = physical_eigenvalues(q, spin)
        assert float(eig.min()) >= -1e-12 and float(eig.max()) <= 1 + 1e-12


def test_locally_valid_patch_spectra_do_not_certify_global_pauli():
    d = torch.tensor([[0.5, 0.45, 0.0], [0.45, 0.5, 0.45], [0.0, 0.45, 0.5]], dtype=torch.float64)
    assert torch.linalg.eigvalsh(d[:2, :2]).min() > 0
    assert torch.linalg.eigvalsh(d[1:, 1:]).max() < 1
    assert torch.linalg.eigvalsh(d).min() < 0
    s = BlockMatrix.diagonal(torch.ones(3, 1, 1, dtype=torch.float64))
    row, col = torch.nonzero(d, as_tuple=True)
    raw = BlockMatrix.assemble(row, col, d[row, col, None, None], 3)
    kernels, _ = constrained_kernel(
        raw, s, torch.ones(3, 1, dtype=torch.bool), torch.tensor([1.5, 1.5])
    )
    for k in kernels:
        eig = torch.linalg.eigvalsh(k.dense_for_validation())
        assert eig.min() >= -1e-12 and eig.max() <= 1 + 1e-12


def test_sparse_products_match_dense_oracle_and_have_second_derivatives():
    gen = torch.Generator().manual_seed(442)
    row = torch.tensor([0, 0, 1, 1, 2])
    col = torch.tensor([0, 1, 0, 1, 2])
    values = torch.randn(5, 2, 2, dtype=torch.float64, generator=gen, requires_grad=True)
    a = BlockMatrix.assemble(row, col, values, 3)
    b = a.transpose()
    product = a.matmul(b)
    assert torch.allclose(
        product.dense_for_validation(), a.dense_for_validation() @ b.dense_for_validation()
    )
    gradient = torch.autograd.grad(product.values.square().sum(), values, create_graph=True)[0]
    second = torch.autograd.grad(gradient.square().sum(), values)[0]
    assert torch.isfinite(second).all()


def test_no_dense_quantum_factorization_is_called(water, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Global dense factorisation used")

    for name in ["cholesky", "cholesky_ex", "solve", "solve_triangular", "eigh", "eigvalsh"]:
        monkeypatch.setattr(torch.linalg, name, forbidden)
    result = Evaluator().energy_and_forces(water)
    assert torch.isfinite(result.forces).all()
    assert result.evaluation.context.quantum.diagnostics["global_dense_algebra"] is False


def test_fill_budget_fails_without_silent_truncation(water):
    with pytest.raises(CapacityError):
        Evaluator(backend=LocalQuantumBackend(max_blocks=2))(water)


def test_metric_degree_converges_towards_identity(water):
    defects = []
    for degree in [8, 16, 64]:
        q = Evaluator(backend=LocalQuantumBackend(metric_degree=degree))(water).context.quantum
        defects.append(q.diagnostics["metric_defect_bound"])
    assert defects[2] < defects[1] < defects[0]
    with pytest.raises(StateError, match="metric defect"):
        Evaluator(backend=LocalQuantumBackend(metric_tolerance=1e-12))(water)


def test_kernel_coherence_changes_density(h2):
    raw = h2.electrons.kernel.values.clone()
    graph = h2.electrons.kernel
    mask = graph.row != graph.col
    raw[mask] = 0.8
    plus = Evaluator()(h2.with_kernel_values(raw)).context.quantum.density
    raw[mask] = -0.8
    minus = Evaluator()(h2.with_kernel_values(raw)).context.quantum.density
    assert float((plus - minus).abs().max()) > 1e-4


def test_invalid_spin_sector_rejected(h2):
    bad = replace(
        h2,
        electrons=replace(h2.electrons, spin_counts=torch.tensor([-1.0, 3.0], dtype=torch.float64)),
    )
    with pytest.raises(StateError):
        bad.validate()


def test_empty_spin_sector_is_exactly_zero():
    world = make_world(["H"], [[6.0, 6.0, 6.0]], grid=Grid((20,) * 3, (0.6,) * 3))
    q = Evaluator()(world).context.quantum
    assert torch.count_nonzero(q.spin_kernels[1].values) == 0
    assert q.diagnostics["spin_counts"][1] == 0


def test_cloud_scale_gradient_is_finite_and_nonzero(h2):
    scale = h2.atoms.log_width_scale.clone().requires_grad_(True)
    w = replace(h2, atoms=replace(h2.atoms, log_width_scale=scale))
    gradient = torch.autograd.grad(Evaluator()(w).energy, scale)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 1e-4
