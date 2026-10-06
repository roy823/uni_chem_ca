from dataclasses import replace

import numpy as np
import pytest
import scipy.linalg
import torch

from unichem import BlockMatrix, Evaluator, SynchronousEngine, make_world
from unichem.checkpoint import load_world, save_world
from unichem.errors import StateError
from unichem.quantum.gaussian import GaussianIntegrals
from unichem.runtime.scf import SCFSolver

scf = pytest.importorskip("pyscf.scf")


@pytest.mark.parametrize(
    "symbols,positions,charge,spin",
    [
        (["H", "H"], [[0, 0, 0], [1.9, 0, 0]], 1, 1),
        (["H", "H"], [[0, 0, 0], [1.4, 0, 0]], 0, 0),
        (["O", "H", "H"], [[0, 0, 0], [1.8, 0, 0], [-0.45, 1.74, 0]], 0, 0),
        (["O", "H"], [[0, 0, 0], [1.83, 0, 0]], -1, 0),
        (
            ["O", "H", "H", "H"],
            [[0, 0, 0], [1.85, 0, 0.4], [-0.925, 1.60, 0.4], [-0.925, -1.60, 0.4]],
            1,
            0,
        ),
    ],
)
def test_same_hf_energy_and_analytic_forces_for_multiple_compositions(
    symbols, positions, charge, spin
):
    world = make_world(symbols, positions, charge=charge, spin=spin)
    actual = Evaluator()(world)
    mol = actual.integrals.mol
    reference = scf.RHF(mol) if spin == 0 else scf.UHF(mol)
    reference.conv_tol = 1e-12
    reference.conv_tol_grad = 1e-9
    energy = reference.kernel()
    force = -reference.nuc_grad_method().kernel()
    assert reference.converged
    assert actual.energy == pytest.approx(energy, abs=1e-8)
    assert actual.forces.numpy() == pytest.approx(force, abs=1e-7)
    assert max(actual.electronic.electron_errors) < 1e-9
    assert max(actual.electronic.purity_errors) < 1e-8


def test_rigid_motion_invariance_and_covariant_forces():
    world = make_world(["O", "H", "H"], [[0, 0, 0], [1.8, 0, 0], [-0.45, 1.74, 0]])
    evaluator = Evaluator()
    base = evaluator(world)
    shifted = evaluator(
        world.with_positions(world.atoms.positions + torch.tensor([0.33, 0.27, -0.11]))
    )
    angle = 0.47
    q = torch.tensor(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
        dtype=torch.float64,
    )
    rotated = evaluator(world.with_positions(world.atoms.positions @ q.T))
    assert shifted.energy == pytest.approx(base.energy, abs=1e-9)
    assert rotated.energy == pytest.approx(base.energy, abs=1e-9)
    assert torch.allclose(shifted.forces, base.forces, atol=1e-7, rtol=0)
    assert torch.allclose(rotated.forces, base.forces @ q.T, atol=1e-7, rtol=0)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_uniform_field_forces_are_derivatives_of_the_same_energy(axis):
    world = make_world(["O", "H", "H"], [[0, 0, 0], [1.8, 0, 0], [-0.45, 1.74, 0]])
    field = [0.0, 0.0, 0.0]
    field[axis] = 0.001
    world = replace(world, external_field=tuple(field))
    evaluator = Evaluator()
    actual = evaluator(world)
    eps = 2e-4
    plus, minus = world.atoms.positions.clone(), world.atoms.positions.clone()
    plus[1, axis] += eps
    minus[1, axis] -= eps
    fd = -(
        evaluator(world.with_positions(plus)).energy - evaluator(world.with_positions(minus)).energy
    ) / (2 * eps)
    assert float(actual.forces[1, axis]) == pytest.approx(fd, abs=2e-6)


def test_production_uses_no_global_quantum_diagonalization_or_dense_state(monkeypatch):
    world = make_world(["O", "H", "H"], [[0, 0, 0], [1.8, 0, 0], [-0.45, 1.74, 0]])

    def forbidden(*args, **kwargs):
        raise AssertionError("Global quantum path used")

    monkeypatch.setattr(BlockMatrix, "dense_for_validation", forbidden)
    monkeypatch.setattr(torch.linalg, "eigh", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(scipy.linalg, "eigh", forbidden)
    assert Evaluator()(world).electronic.converged


def test_native_restart_and_reordering_keep_the_same_physical_trajectory(tmp_path):
    world = make_world(["H", "H"], [[0, 0, 0], [1.45, 0, 0]])
    engine = SynchronousEngine()
    current = world
    for _ in range(4):
        current = engine.step(current).world
    prefix = world
    for _ in range(2):
        prefix = engine.step(prefix).world
    save_world(prefix, tmp_path / "restart.npz")
    restart = load_world(tmp_path / "restart.npz")
    for _ in range(2):
        restart = engine.step(restart).world
    assert torch.allclose(current.atoms.positions, restart.atoms.positions, atol=1e-10, rtol=0)
    assert torch.equal(world.atoms.momenta, torch.zeros_like(world.atoms.momenta))
    shuffled = make_world(["H", "H"], world.atoms.positions.flip(0).numpy())
    for _ in range(4):
        shuffled = engine.step(shuffled).world
    assert torch.allclose(
        current.atoms.positions.flip(0), shuffled.atoms.positions, atol=1e-10, rtol=0
    )


def test_unconverged_electronic_proposal_never_commits_nuclear_motion():
    world = make_world(["O", "H", "H"], [[0, 0, 0], [1.8, 0, 0], [-0.45, 1.74, 0]])
    old = world.clone()
    engine = SynchronousEngine(Evaluator(SCFSolver(max_iterations=1)))
    with pytest.raises(StateError, match="SCF not converged"):
        engine.step(world)
    assert torch.equal(world.atoms.positions, old.atoms.positions)
    assert world.step == 0


def test_provider_integral_budget_is_explicit():
    with pytest.raises(StateError, match="quartet budget"):
        GaussianIntegrals(
            [1] * 3, [[0, 0, 0], [1.4, 0, 0], [0, 1.4, 0]], charge=1, quartet_budget=2
        )
