import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.linalg import eigh

from unichem import BlockMatrix, make_world
from unichem.analysis.oracles import block_operator, padded_indices
from unichem.analysis.validation import IntegralReference, validate_matrix_core
from unichem.checkpoint import load_world, save_world
from unichem.errors import CapacityError, StateError
from unichem.quantum.purification import ground_projector, orthogonalize
from unichem.runtime.scf import SCFSolver


def test_zero_temperature_projector_matches_generalized_eigenproblem():
    counts = [2, 1]
    s = np.array([[1, 0.15, 0.07], [0.15, 1, -0.04], [0.07, -0.04, 1]])
    h = np.array([[-1, 0.2, 0.1], [0.2, 0.4, -0.11], [0.1, -0.11, 0.9]])
    indices, active = padded_indices(counts)
    overlap, hamiltonian = block_operator(s, counts), block_operator(h, counts)
    metric = orthogonalize(overlap, torch.tensor(active))
    projected = ground_projector(hamiltonian, metric, 1)
    p = projected.density.dense_for_validation().numpy()[indices[:, None], indices[None, :]]
    _, c = eigh(h, s)
    assert p == pytest.approx(np.outer(c[:, 0], c[:, 0]), abs=1e-10)
    assert projected.purity_error < 1e-12
    assert abs(np.trace(p @ s) - 1) < 1e-11


def test_ground_projector_has_no_excited_occupation_floor():
    counts = [1, 1]
    _, active = padded_indices(counts)
    s = block_operator(np.eye(2), counts)
    h = block_operator(np.diag([-1, 2]), counts)
    p = ground_projector(
        h, orthogonalize(s, torch.tensor(active)), 1
    ).density.dense_for_validation()
    assert torch.allclose(
        p, torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64)), atol=1e-11, rtol=0
    )


def test_unresolved_degenerate_partial_sector_fails_explicitly():
    counts = [1, 1]
    _, active = padded_indices(counts)
    s = block_operator(np.eye(2), counts)
    with pytest.raises(StateError, match="degenerate"):
        ground_projector(s.scale(0), orthogonalize(s, torch.tensor(active)), 1)


def test_sparse_products_never_silently_truncate_fill():
    p = block_operator(np.ones((3, 3)), [1, 1, 1])
    with pytest.raises(CapacityError):
        BlockMatrix.assemble(p.row, p.col, p.values, p.n, budget=2)


def test_compiled_sparse_product_preserves_first_and_second_derivatives():
    ids = torch.arange(3).repeat_interleave(3)
    cols = torch.arange(3).repeat(3)
    rng = torch.Generator().manual_seed(37)
    a = torch.randn(9, 2, 2, generator=rng, dtype=torch.float64, requires_grad=True)
    b = torch.randn(9, 2, 2, generator=rng, dtype=torch.float64, requires_grad=True)

    def product(x, y):
        return BlockMatrix(ids, cols, x, 3).matmul(BlockMatrix(ids, cols, y, 3)).values

    assert torch.autograd.gradcheck(product, (a, b))
    assert torch.autograd.gradgradcheck(product, (a, b))


def test_checkpoint_preserves_physical_density_and_rejects_old_schema(tmp_path):
    world = make_world(["H", "H"], [[0, 0, 0], [1.4, 0, 0]])
    density = block_operator(np.eye(2) / 2, [1, 1])
    world = replace(world, densities=(density, density))
    path = tmp_path / "state.npz"
    save_world(world, path)
    loaded = load_world(path)
    assert torch.equal(world.atoms.positions, loaded.atoms.positions)
    assert torch.equal(world.densities[0].values, loaded.densities[0].values)
    np.savez(
        tmp_path / "old.npz",
        metadata=np.asarray(json.dumps({"schema_version": 3, "units": "atomic"})),
    )
    with pytest.raises(StateError, match="Unsupported checkpoint"):
        load_world(tmp_path / "old.npz")


@pytest.mark.skipif(
    not Path("data/quantum_oracles/0009.npz").exists(), reason="Actual integral reference absent"
)
def test_real_water_matrix_core_converges_without_raw_coordinate_fit():
    with np.load("data/quantum_oracles/0009.npz", allow_pickle=False) as data:
        oracle = IntegralReference(data)
        target = float(data["reference_energy"])
    result = SCFSolver().solve(oracle.operators, [5, 5])
    assert result.energy == pytest.approx(target, abs=1e-9)
    assert result.commutator_norm < 1e-8
    assert max(result.purity_errors) < 1e-9
    assert result.spectral_consistency < 1e-8


@pytest.mark.skipif(
    not Path("data/quantum_oracles/0009.npz").exists(), reason="Actual integral reference absent"
)
def test_failed_scf_does_not_return_a_converged_state():
    with np.load("data/quantum_oracles/0009.npz", allow_pickle=False) as data:
        oracle = IntegralReference(data)
    with pytest.raises(StateError, match="SCF not converged"):
        SCFSolver(max_iterations=1).solve(oracle.operators, [5, 5])


@pytest.mark.skipif(
    not Path("data/quantum_oracles/0010.npz").exists(), reason="Actual integral references absent"
)
def test_all_cached_physical_oracles_pass_new_core(tmp_path):
    assert validate_matrix_core(output=tmp_path)["all_passed"]
