import math

import pytest
import torch

from unichem.checkpoint import load_cells, save_cells
from unichem.errors import CapacityError
from unichem.molecule import make_world
from unichem.runtime.local_cells import CellSnapshot, FieldSnapshot, LocalCellEngine
from unichem.spatial.neighbors import PairList


def test_local_quantum_channel_matches_a_single_two_site_transfer():
    atoms = make_world(["H", "H"], [[0, 0, 0], [1, 0, 0]]).atoms
    rho = torch.zeros(2, 2, 1, 1, dtype=torch.complex128)
    rho[0, 0, 0, 0] = 1
    state = CellSnapshot(atoms, rho, torch.ones(2, dtype=torch.int64))
    pairs = PairList(
        torch.tensor([0]), torch.tensor([1]), torch.tensor([[1.0, 0, 0]]), torch.tensor([1.0]), 1
    )
    fields = FieldSnapshot(torch.zeros(2, 3), torch.zeros(2, 1, 1), torch.ones(1, 1, 1))
    actual = LocalCellEngine(dt=0.3).step(state, pairs, fields)
    expected = torch.tensor([math.cos(0.3) ** 2, math.sin(0.3) ** 2], dtype=torch.float64)
    assert actual.populations()[:, 0] == pytest.approx(expected, abs=1e-12)
    assert actual.populations().sum() == pytest.approx(1, abs=1e-12)


def test_overlapping_local_channels_conserve_number_and_pauli_bounds():
    atoms = make_world(["H"] * 4, [[i, 0, 0] for i in range(4)]).atoms
    rng = torch.Generator().manual_seed(51)
    rho = torch.diag_embed(torch.rand(4, 2, 2, generator=rng)).to(torch.complex128)
    state = CellSnapshot(atoms, rho, torch.full((4,), 2, dtype=torch.int64))
    i, j = torch.tensor([0, 0, 1, 1, 2]), torch.tensor([1, 2, 2, 3, 3])
    pairs = PairList(i, j, torch.zeros(5, 3), torch.ones(5), 5)
    h = torch.randn(4, 2, 2, generator=rng)
    fields = FieldSnapshot(torch.zeros(4, 3), 0.5 * (h + h.mT), torch.randn(5, 2, 2, generator=rng))
    number = state.populations().sum(0)
    engine = LocalCellEngine(dt=0.2)
    for _ in range(15):
        state = engine.step(state, pairs, fields)
    eigenvalues = torch.linalg.eigvalsh(state.density)
    assert state.populations().sum(0) == pytest.approx(number, abs=1e-11)
    assert eigenvalues.min() >= -1e-12
    assert eigenvalues.max() <= 1 + 1e-12
    assert (state.density - state.density.mH).abs().max() < 1e-12


def test_local_degree_capacity_failure_keeps_old_state():
    atoms = make_world(["H"] * 3, [[i, 0, 0] for i in range(3)], spin=1).atoms
    rho = torch.full((3, 2, 1, 1), 0.5, dtype=torch.complex128)
    state = CellSnapshot(atoms, rho, torch.ones(3, dtype=torch.int64))
    pairs = PairList(
        torch.tensor([0, 0]), torch.tensor([1, 2]), torch.zeros(2, 3), torch.ones(2), 2
    )
    fields = FieldSnapshot(torch.zeros(3, 3), torch.zeros(3, 1, 1), torch.ones(2, 1, 1))
    with pytest.raises(CapacityError):
        LocalCellEngine(max_neighbors=1).step(state, pairs, fields)
    assert state.step == 0
    assert torch.equal(state.density, rho)


def test_local_field_hopping_has_the_expected_training_gradient():
    atoms = make_world(["H", "H"], [[0, 0, 0], [1, 0, 0]]).atoms
    rho = torch.zeros(2, 2, 1, 1, dtype=torch.complex128)
    rho[0, 0, 0, 0] = 1
    state = CellSnapshot(atoms, rho, torch.ones(2, dtype=torch.int64))
    pairs = PairList(torch.tensor([0]), torch.tensor([1]), torch.zeros(1, 3), torch.ones(1), 1)
    amplitude = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    fields = FieldSnapshot(torch.zeros(2, 3), torch.zeros(2, 1, 1), amplitude.reshape(1, 1, 1))
    after = LocalCellEngine(dt=0.3).step(state, pairs, fields)
    derivative = torch.autograd.grad(after.populations()[0, 0], amplitude)[0]
    assert float(derivative) == pytest.approx(-0.3 * math.sin(0.6), abs=1e-11)


def test_one_step_propagates_only_one_contact_hop():
    atoms = make_world(["H"] * 8, [[i, 0, 0] for i in range(8)]).atoms
    rho = torch.zeros(8, 2, 1, 1, dtype=torch.complex128)
    rho[0, 0, 0, 0] = 1
    state = CellSnapshot(atoms, rho, torch.ones(8, dtype=torch.int64))
    i, j = torch.arange(7), torch.arange(1, 8)
    pairs = PairList(i, j, torch.zeros(7, 3), torch.ones(7), 7)
    fields = FieldSnapshot(torch.zeros(8, 3), torch.zeros(8, 1, 1), torch.ones(7, 1, 1))
    one = LocalCellEngine(dt=0.3).step(state, pairs, fields)
    assert one.populations()[1, 0] > 0
    assert torch.equal(one.populations()[2:], torch.zeros(6, 2, dtype=torch.float64))
    two = LocalCellEngine(dt=0.3).step(one, pairs, fields)
    assert two.populations()[2, 0] > 0
    assert torch.equal(two.populations()[3:], torch.zeros(5, 2, dtype=torch.float64))


def test_parallel_batch_partition_does_not_change_the_ca_rule():
    atoms = make_world(["H"] * 4, [[i, 0, 0] for i in range(4)]).atoms
    rho = torch.diag_embed(torch.linspace(0.1, 0.8, 8).reshape(4, 2, 1)).to(torch.complex128)
    state = CellSnapshot(atoms, rho, torch.ones(4, dtype=torch.int64))
    pairs = PairList(
        torch.tensor([0, 0, 1, 2]), torch.tensor([1, 2, 2, 3]), torch.zeros(4, 3), torch.ones(4), 4
    )
    fields = FieldSnapshot(torch.zeros(4, 3), torch.zeros(4, 1, 1), torch.ones(4, 1, 1))
    one = LocalCellEngine(dt=0.3, edge_batch=1).step(state, pairs, fields)
    all_edges = LocalCellEngine(dt=0.3, edge_batch=4096).step(state, pairs, fields)
    assert one.density == pytest.approx(all_edges.density, abs=1e-12)
    assert state.step == 0


def test_free_nuclear_drift_and_internal_force_momentum_balance():
    atoms = make_world(["H", "H"], [[0, 0, 0], [2, 0, 0]]).atoms
    state = CellSnapshot(
        atoms, torch.zeros(2, 2, 1, 1, dtype=torch.complex128), torch.ones(2, dtype=torch.int64)
    )
    pairs = PairList(torch.tensor([0]), torch.tensor([1]), torch.zeros(1, 3), torch.ones(1), 1)
    fields = FieldSnapshot(
        torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]]), torch.zeros(2, 1, 1), torch.zeros(1, 1, 1)
    )
    after = LocalCellEngine(dt=0.3).step(state, pairs, fields)
    assert after.atoms.momenta.sum(0) == pytest.approx(torch.zeros(3), abs=1e-12)
    expected = atoms.positions + 0.3**2 * fields.forces / atoms.masses[:, None]
    assert after.atoms.positions == pytest.approx(expected, abs=1e-12)


def test_local_ca_restart_retains_the_same_state(tmp_path):
    atoms = make_world(["H", "H"], [[0, 0, 0], [1, 0, 0]]).atoms
    rho = torch.zeros(2, 2, 1, 1, dtype=torch.complex128)
    rho[0, 0, 0, 0] = 1
    initial = CellSnapshot(atoms, rho, torch.ones(2, dtype=torch.int64))
    pairs = PairList(torch.tensor([0]), torch.tensor([1]), torch.zeros(1, 3), torch.ones(1), 1)
    fields = FieldSnapshot(torch.zeros(2, 3), torch.zeros(2, 1, 1), torch.ones(1, 1, 1))
    engine = LocalCellEngine(dt=0.3)
    first = engine.step(initial, pairs, fields)
    save_cells(first, tmp_path / "cells.npz")
    restarted = engine.step(load_cells(tmp_path / "cells.npz"), pairs, fields)
    continuous = engine.step(first, pairs, fields)
    assert restarted.step == continuous.step == 2
    assert torch.equal(restarted.density, continuous.density)
    assert torch.equal(restarted.atoms.positions, continuous.atoms.positions)


def test_permuting_cell_storage_preserves_the_local_update():
    from dataclasses import fields as atom_fields
    from dataclasses import replace

    atoms = make_world(["H"] * 4, [[i, 0, 0] for i in range(4)]).atoms
    rho = torch.diag_embed(torch.linspace(0.1, 0.8, 8).reshape(4, 2, 1)).to(torch.complex128)
    state = CellSnapshot(atoms, rho, torch.ones(4, dtype=torch.int64))
    i, j = torch.tensor([0, 0, 1, 2]), torch.tensor([1, 2, 2, 3])
    pairs = PairList(i, j, torch.zeros(4, 3), torch.ones(4), 4)
    field = FieldSnapshot(torch.zeros(4, 3), torch.zeros(4, 1, 1), torch.ones(4, 1, 1))
    order = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(order)
    permuted_atoms = replace(
        atoms, **{f.name: getattr(atoms, f.name)[order] for f in atom_fields(atoms)}
    )
    permuted = CellSnapshot(permuted_atoms, rho[order], state.channels[order])
    permuted_pairs = PairList(inverse[i], inverse[j], pairs.displacement, pairs.distance, 4)
    engine = LocalCellEngine(dt=0.3)
    expected = engine.step(state, pairs, field)
    actual = engine.step(permuted, permuted_pairs, field)
    assert actual.density == pytest.approx(expected.density[order], abs=1e-12)
