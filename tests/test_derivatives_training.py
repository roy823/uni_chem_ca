import pytest
import torch

from unichem import Evaluator
from unichem.models.energy import SharedScalarResidual
from unichem.training import KernelUnrolledPredictor, train_synthetic


def test_force_equals_finite_difference_of_same_energy(h2):
    evaluator = Evaluator()
    force = evaluator.energy_and_forces(h2).forces
    epsilon = 1e-5
    for atom, axis in [(0, 0), (0, 1), (1, 0)]:
        plus = h2.atoms.positions.clone()
        minus = h2.atoms.positions.clone()
        plus[atom, axis] += epsilon
        minus[atom, axis] -= epsilon
        fd = -(
            evaluator(h2.with_positions(plus)).energy - evaluator(h2.with_positions(minus)).energy
        ) / (2 * epsilon)
        assert float(force[atom, axis]) == pytest.approx(float(fd), rel=3e-5, abs=3e-6)


def test_force_supervision_reaches_network_parameters(h2):
    model = SharedScalarResidual(hidden=8).double()
    prediction = Evaluator(residual=model).energy_and_forces(h2, create_graph=True)
    loss = prediction.forces.square().sum()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 1e-8


def test_density_supervision_reaches_energy_model_through_parallel_relaxation(h2):
    model = SharedScalarResidual(hidden=8).double()
    predictor = KernelUnrolledPredictor(Evaluator(residual=model), electronic_steps=1)
    pred = predictor(h2)
    probe = torch.arange(h2.grid.n_voxels, dtype=torch.float64) / h2.grid.n_voxels
    loss = (pred.evaluation.context.quantum.density * probe).sum() * h2.grid.dv
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 1e-10


def test_synthetic_training_reduces_heldout_loss(tmp_path):
    report = train_synthetic(epochs=3, output=tmp_path)
    assert report["history"][-1]["train_loss"] < report["history"][0]["train_loss"]
    assert report["history"][-1]["validation_loss"] < report["history"][0]["validation_loss"]
    assert (tmp_path / "synthetic_model.pt").exists()


def test_full_unrolled_force_includes_electronic_response(h2):
    predictor = KernelUnrolledPredictor(Evaluator(), electronic_steps=1)
    force = predictor(h2, create_graph=False).forces
    eps = 1e-5
    plus = h2.atoms.positions.clone()
    plus[0, 0] += eps
    minus = h2.atoms.positions.clone()
    minus[0, 0] -= eps
    ep = predictor(h2.with_positions(plus), create_graph=False).evaluation.energy
    em = predictor(h2.with_positions(minus), create_graph=False).evaluation.energy
    assert float(force[0, 0]) == pytest.approx(
        float((-(ep - em) / (2 * eps)).detach()), abs=3e-6, rel=3e-5
    )


def test_force_training_remains_finite_at_exact_grid_centres(h2):
    positions = torch.tensor([[4.5, 5.1, 5.1], [6.3, 5.1, 5.1]], dtype=torch.float64)
    w = h2.with_positions(positions)
    model = SharedScalarResidual(hidden=8).double()
    prediction = Evaluator(residual=model).energy_and_forces(w, create_graph=True)
    prediction.forces.square().sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
