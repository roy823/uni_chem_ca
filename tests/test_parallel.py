from dataclasses import replace
from itertools import pairwise

import pytest
import torch

from unichem import Evaluator, SynchronousEngine
from unichem.errors import StateError
from unichem.models.energy import SharedScalarResidual, ZeroEnergy
from unichem.models.latent import ConservativeLatentCA
from unichem.runtime.electrons import LocalKernelRelaxer


def _permuted(world, permutation):
    from unichem.quantum.blocks import BlockMatrix

    idx = torch.tensor(permutation, device=world.atoms.ids.device)
    inverse = torch.empty_like(idx)
    inverse[idx] = torch.arange(len(idx), device=idx.device)
    a = world.atoms
    k = world.electrons.kernel
    kernel = BlockMatrix.assemble(inverse[k.row], inverse[k.col], k.values, k.n, k.budget)
    return replace(
        world,
        species=tuple(world.species[i] for i in permutation),
        atoms=replace(
            a,
            ids=a.ids[idx],
            positions=a.positions[idx],
            momenta=a.momenta[idx],
            log_width_scale=a.log_width_scale[idx],
        ),
        electrons=replace(world.electrons, kernel=kernel),
    )


def test_site_order_does_not_change_energy_or_force(water):
    model = SharedScalarResidual().double()
    evaluator = Evaluator(residual=model)
    original = evaluator.energy_and_forces(water)
    perm = [2, 0, 1]
    reordered = evaluator.energy_and_forces(_permuted(water, perm))
    assert float(original.evaluation.energy.detach()) == pytest.approx(
        float(reordered.evaluation.energy.detach()), abs=1e-10
    )
    assert torch.allclose(original.forces[perm], reordered.forces, atol=2e-10, rtol=2e-10)


def test_whole_parallel_step_is_atom_permutation_equivariant(water):
    evaluator = Evaluator()
    engine = SynchronousEngine(evaluator, relaxer=LocalKernelRelaxer(max_iterations=1))
    perm = [2, 0, 1]
    a = engine.step(water).world
    b = engine.step(_permuted(water, perm)).world
    assert torch.allclose(a.atoms.positions[perm], b.atoms.positions, atol=1e-12)
    assert torch.allclose(a.atoms.momenta[perm], b.atoms.momenta, atol=1e-10)
    assert torch.allclose(
        evaluator(a).context.quantum.density, evaluator(b).context.quantum.density, atol=1e-12
    )


def test_double_buffer_does_not_mutate_or_alias_input(h2):
    original = h2.clone()
    engine = SynchronousEngine(Evaluator(), relaxer=LocalKernelRelaxer(max_iterations=1))
    result = engine.step(h2)
    assert torch.equal(h2.atoms.positions, original.atoms.positions)
    assert torch.equal(h2.electrons.kernel.values, original.electrons.kernel.values)
    assert torch.equal(h2.latent.channels, original.latent.channels)
    assert h2.time == 0 and h2.step == 0
    assert result.world.time == 0.1 and result.world.step == 1
    result.world.atoms.positions[0, 0] += 0.01
    assert torch.equal(h2.atoms.positions, original.atoms.positions)


def test_velocity_verlet_has_exact_free_flight(h2):
    momentum = torch.tensor([[2.0, 3.0, 4.0], [-1.0, 4.0, 2.0]], dtype=torch.float64)
    w = replace(h2, atoms=replace(h2.atoms, momenta=momentum))
    evaluator = Evaluator(functional=ZeroEnergy())
    engine = SynchronousEngine(evaluator, dt=0.2, relaxer=LocalKernelRelaxer(max_iterations=0))
    masses = evaluator(w).context.layout.masses[:, None]
    result = engine.step(w).world
    assert torch.allclose(
        result.atoms.positions, w.atoms.positions + 0.2 * momentum / masses, atol=1e-14
    )
    assert torch.equal(result.atoms.momenta, momentum)


def test_electronic_global_acceptance_monotonically_reduces_energy(h2):
    result = LocalKernelRelaxer(max_iterations=5).run(h2, Evaluator())
    history = result.report.energies
    assert len(history) > 1
    assert all(b <= a + 1e-12 for a, b in pairwise(history))
    assert torch.equal(result.world.electrons.spin_counts, h2.electrons.spin_counts)


def test_unconverged_scf_is_reported_or_rejected(h2):
    loose = SynchronousEngine(Evaluator(), relaxer=LocalKernelRelaxer(max_iterations=0))
    assert not loose.step(h2).diagnostics["electrons_converged"]
    strict = SynchronousEngine(
        Evaluator(),
        relaxer=LocalKernelRelaxer(max_iterations=0),
        require_electronic_convergence=True,
    )
    with pytest.raises(StateError, match="did not converge"):
        strict.step(h2)
    assert h2.step == 0


def test_latent_rule_conserves_channel_sums_and_does_not_update_in_place(h2):
    z = h2.latent.channels.clone()
    z[9, 9, 9, 0] = 1.0
    w = replace(h2, latent=replace(h2.latent, channels=z))
    context = Evaluator()(w).context
    rule = ConservativeLatentCA(4).double()
    out = rule(context, dt=0.1)
    assert torch.allclose(out.channels.sum((0, 1, 2)), z.sum((0, 1, 2)), atol=1e-14)
    assert z.count_nonzero() == 1
    assert out.channels[..., 0].count_nonzero() == 7
    assert out.channels.min() >= 0
    assert out.channels[11, 9, 9, 0] == 0  # no within-step cascaded update
    assert out.memory.count_nonzero() == 1


def test_latent_unstable_step_is_rejected(h2):
    with pytest.raises(StateError, match="stability"):
        ConservativeLatentCA(4).double()(Evaluator()(h2).context, dt=10.0)


def test_nuclear_displacement_is_rejected_not_clipped(h2):
    huge = replace(h2, atoms=replace(h2.atoms, momenta=torch.full_like(h2.atoms.momenta, 1e8)))
    engine = SynchronousEngine(
        Evaluator(functional=ZeroEnergy()), relaxer=LocalKernelRelaxer(max_iterations=0)
    )
    with pytest.raises(StateError, match="Displacement"):
        engine.step(huge)
