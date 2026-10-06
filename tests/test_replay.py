import pytest
import torch

from unichem.errors import StateError
from unichem.runtime.potential import integrate_potential


def harmonic(r):
    return 0.5 * r.square().sum(), -r, {}


def run(r, p, ids, *, steps=12, start_step=0):
    return list(
        integrate_potential(
            r,
            p,
            torch.ones(len(r), dtype=r.dtype),
            harmonic,
            atom_ids=ids,
            steps=steps,
            dt=0.05,
            kT=0.01,
            friction=0.2,
            seed=91,
            start_step=start_step,
        )
    )


def test_stochastic_restart_uses_persistent_ids_and_global_step():
    r = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)
    ids = torch.tensor([77, 15])
    full = run(r, torch.zeros_like(r), ids)
    prefix = run(r, torch.zeros_like(r), ids, steps=5)
    last = prefix[-1]
    suffix = run(last.positions, last.momenta, ids, steps=7, start_step=last.step)
    assert torch.equal(full[-1].positions, suffix[-1].positions)
    assert torch.equal(full[-1].momenta, suffix[-1].momenta)
    assert full[-1].step == suffix[-1].step == 12
    assert full[-1].time_au == suffix[-1].time_au


def test_reordering_atoms_preserves_their_stochastic_trajectory():
    r = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)
    p = torch.zeros_like(r)
    ids = torch.tensor([77, 15])
    perm = torch.tensor([1, 0])
    a = run(r, p, ids)
    b = run(r[perm], p[perm], ids[perm])
    assert torch.equal(a[-1].positions[perm], b[-1].positions)
    assert torch.equal(a[-1].momenta[perm], b[-1].momenta)


def test_duplicate_atom_ids_rejected():
    r = torch.zeros((2, 3), dtype=torch.float64)
    with pytest.raises(StateError, match="persistent atom IDs"):
        run(r, r, torch.tensor([1, 1]))
