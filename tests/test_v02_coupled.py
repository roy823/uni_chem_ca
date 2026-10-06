import math
from dataclasses import replace

import pytest
import torch
from test_parallel import _permuted

from unichem import Evaluator, Grid, SynchronousEngine
from unichem.checkpoint import load_world, save_world
from unichem.errors import StateError
from unichem.models.coupling import ConditionalGaussianCoupling
from unichem.models.encoder import CoarseFieldEncoder
from unichem.models.energy import ZeroEnergy
from unichem.models.latent import ConservativeLatentCA
from unichem.runtime.coupled import BathConfig, CoupledSynchronousEngine, ou_refresh
from unichem.runtime.electrons import LocalKernelRelaxer
from unichem.runtime.random import CounterNoise, philox4x32


def auxiliary(h2):
    g = Grid((6, 6, 6), (1.8, 1.8, 1.8))
    return replace(
        h2,
        latent=replace(
            h2.latent,
            channels=torch.zeros(6, 6, 6, 4, dtype=torch.float64),
            memory=torch.zeros(6, 6, 6, 4, dtype=torch.float64),
            grid=g,
            kind="auxiliary_momentum",
        ),
    )


def test_philox_known_answer():
    assert philox4x32(torch.zeros(4, dtype=torch.int64), (0, 0)).tolist() == [
        0x6627E8D5,
        0xE169C58D,
        0xBC57AC4C,
        0x9B00DBD8,
    ]


def test_counter_noise_is_chunk_order_and_replay_invariant():
    ids = torch.tensor([7, 10002, 91, 0, 2**40 + 123], dtype=torch.int64)
    noise = CounterNoise(197)
    x = noise.normal(ids, 7, step=5, phase="test")
    assert torch.equal(
        x, torch.cat([noise.normal(t, 7, step=5, phase="test") for t in ids.split(2)])
    )
    perm = torch.tensor([3, 2, 1, 4, 0])
    assert torch.equal(x[perm], noise.normal(ids[perm], 7, step=5, phase="test"))
    assert not torch.equal(x, noise.normal(ids, 7, step=6, phase="test"))
    assert not torch.equal(x, noise.normal(ids, 7, step=5, phase="other"))


def test_noise_moments_and_exact_ou_variance():
    ids = torch.arange(50_000, dtype=torch.int64)
    x = CounterNoise(22).normal(ids, 1, step=0, phase="moments").squeeze(-1)
    assert abs(float(x.mean())) < 0.02
    assert abs(float(x.var()) - 1) < 0.03
    p, heat = ou_refresh(torch.zeros_like(x), 3.0, friction=0.4, kT=0.2, dt=0.7, noise=x)
    expected = 3 * 0.2 * (1 - math.exp(-2 * 0.4 * 0.7))
    assert float(p.var()) == pytest.approx(expected, rel=0.03)
    assert float(heat) == pytest.approx(float(p.square().sum() / 6), abs=1e-12)


def test_coupling_has_reciprocal_mixed_forces_and_no_electronic_double_counting(h2):
    w = auxiliary(h2)
    enc = CoarseFieldEncoder(4, hidden=5).double()
    coupling = ConditionalGaussianCoupling(enc)
    evaluator = Evaluator(coupling=coupling)
    base = Evaluator()
    a = evaluator.energy_and_generalized_forces(w)
    eps = 1e-5
    plus, minus = w.latent.channels.clone(), w.latent.channels.clone()
    plus[2, 2, 2, 0] += eps
    minus[2, 2, 2, 0] -= eps
    fd = -(evaluator(w.with_latent(plus)).energy - evaluator(w.with_latent(minus)).energy) / (
        2 * eps
    )
    assert float(a.latent_forces[2, 2, 2, 0]) == pytest.approx(float(fd.detach()), abs=1e-9)
    # Conditional centre depends on nuclei only, including under width changes.
    altered = replace(w, atoms=replace(w.atoms, log_width_scale=w.atoms.log_width_scale + 0.1))
    assert torch.equal(enc(base(w).context), enc(base(altered).context))
    assert coupling.preserves_bo_marginal
    # Cross Hessian reciprocity: dF_R/dz = dF_z/dR from the SAME energy.
    rp, rm = w.atoms.positions.clone(), w.atoms.positions.clone()
    rp[0, 0] += eps
    rm[0, 0] -= eps
    lhs = (
        evaluator.energy_and_generalized_forces(w.with_latent(plus)).forces[0, 0]
        - evaluator.energy_and_generalized_forces(w.with_latent(minus)).forces[0, 0]
    ) / (2 * eps)
    rhs = (
        evaluator.energy_and_generalized_forces(w.with_positions(rp)).latent_forces[2, 2, 2, 0]
        - evaluator.energy_and_generalized_forces(w.with_positions(rm)).latent_forces[2, 2, 2, 0]
    ) / (2 * eps)
    assert float(lhs) == pytest.approx(float(rhs), abs=2e-8)


def test_electronic_encoder_has_explicit_no_bo_marginal_guarantee(h2):
    w = auxiliary(h2)
    coupling = ConditionalGaussianCoupling(CoarseFieldEncoder(4, source="electronic").double())
    assert not coupling.preserves_bo_marginal
    assert torch.isfinite(Evaluator(coupling=coupling)(w).energy)


def test_joint_engine_free_flight_and_no_stale_diagnostics(h2):
    w = auxiliary(h2)
    p = torch.ones_like(w.atoms.momenta)
    w = replace(
        w,
        atoms=replace(w.atoms, momenta=p),
        latent=replace(w.latent, memory=torch.ones_like(w.latent.memory)),
    )
    evaluator = Evaluator(functional=ZeroEnergy())
    engine = CoupledSynchronousEngine(
        evaluator,
        dt=0.2,
        bath=BathConfig(latent_friction=0),
        relaxer=LocalKernelRelaxer(max_iterations=0),
    )
    result = engine.step(w)
    masses = evaluator(w).context.layout.masses[:, None]
    assert torch.allclose(
        result.world.atoms.positions, w.atoms.positions + 0.2 * p / masses, atol=1e-14
    )
    assert torch.allclose(
        result.world.latent.channels, w.latent.channels + 0.2 / w.latent_grid.dv, atol=1e-14
    )
    assert abs(result.diagnostics["mechanical_balance_residual_hartree"]) < 1e-12
    assert w.step == 0 and w.latent.channels.count_nonzero() == 0


def test_stochastic_coupled_engine_permutation_and_checkpoint_replay(h2, tmp_path):
    w = auxiliary(h2)
    coupling = ConditionalGaussianCoupling(CoarseFieldEncoder(4, hidden=5).double())
    evaluator = Evaluator(coupling=coupling)
    engine = CoupledSynchronousEngine(
        evaluator,
        bath=BathConfig(kT=0.001, nuclear_friction=0.1),
        seed=28,
        relaxer=LocalKernelRelaxer(max_iterations=0),
    )
    a, b = engine.step(w), engine.step(_permuted(w, [1, 0]))
    assert torch.allclose(a.world.atoms.momenta.flip(0), b.world.atoms.momenta, atol=1e-13)
    assert torch.allclose(a.world.latent.channels, b.world.latent.channels, atol=1e-13)
    assert float(evaluator(a.world).energy.detach()) == pytest.approx(
        a.diagnostics["potential_energy_hartree"], abs=1e-11
    )
    save_world(a.world, tmp_path / "w.npz")
    restored = load_world(tmp_path / "w.npz")
    x, y = engine.step(a.world).world, engine.step(restored).world
    assert torch.equal(x.atoms.positions, y.atoms.positions)
    assert torch.equal(x.latent.memory, y.latent.memory)


def test_incompatible_latent_semantics_fail_before_commit(h2):
    with pytest.raises(StateError, match="auxiliary_momentum"):
        CoupledSynchronousEngine(Evaluator()).step(h2)
    evaluator = Evaluator(coupling=ConditionalGaussianCoupling(CoarseFieldEncoder(4).double()))
    with pytest.raises(StateError, match="passive filter"):
        SynchronousEngine(evaluator, latent_rule=ConservativeLatentCA(4).double()).step(h2)
    assert h2.step == 0


def test_auxiliary_gaussian_partition_is_translation_invariant():
    # This is a numerical check of the explicit Gaussian normalization identity,
    # not evidence of ergodicity or a calibrated chemical trajectory.
    z = torch.linspace(-12, 12, 20001, dtype=torch.float64)
    integrals = []
    for target in (-2.1, 0.0, 3.4):
        energy = 0.5 * 0.7 * (z - target).square()
        integrals.append(torch.trapezoid(torch.exp(-energy / 0.5), z))
    assert torch.allclose(torch.stack(integrals), integrals[0].expand(3), atol=1e-12)


def test_auxiliary_field_energy_is_not_a_second_electronic_functional(h2):
    w = auxiliary(h2)
    coupling = ConditionalGaussianCoupling(CoarseFieldEncoder(4, hidden=5).double())
    c = w.electrons.kernel.values.clone().requires_grad_(True)
    context = Evaluator()(w.with_kernel_values(c)).context
    e = coupling(context) + 0 * c.sum()
    assert torch.count_nonzero(torch.autograd.grad(e, c)[0]) == 0


def test_strict_coupled_solver_rejects_without_mutating_input(h2):
    w = auxiliary(h2)
    saved = w.clone()
    engine = CoupledSynchronousEngine(
        Evaluator(),
        require_electronic_convergence=True,
        relaxer=LocalKernelRelaxer(max_iterations=0),
    )
    with pytest.raises(StateError, match="did not converge"):
        engine.step(w)
    assert torch.equal(w.atoms.positions, saved.atoms.positions)
    assert torch.equal(w.latent.memory, saved.latent.memory)


def test_latent_encoder_gets_force_supervision(h2):
    w = auxiliary(h2)
    enc = CoarseFieldEncoder(4, hidden=5).double()
    evaluation = Evaluator(coupling=ConditionalGaussianCoupling(enc)).energy_and_generalized_forces(
        w, create_graph=True
    )
    (evaluation.forces.square().sum() + evaluation.latent_forces.square().sum()).backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 1e-12
