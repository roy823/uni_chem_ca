import pytest
import torch

from unichem import Evaluator, Grid
from unichem.errors import CapacityError, StateError
from unichem.fields.multiscale import GridTransfer
from unichem.fields.operators import face_divergence, face_gradient
from unichem.fields.particles import deposit_cubic
from unichem.interactions import InteractionComplex
from unichem.quantum.support import build_support, make_layout
from unichem.spatial.neighbors import CellList


def test_cell_list_matches_all_pairs_and_periodic_boundaries():
    gen = torch.Generator().manual_seed(417)
    grid = Grid((20, 18, 16), (0.6, 0.7, 0.8))
    positions = torch.rand((77, 3), generator=gen, dtype=torch.float64) * torch.tensor(grid.lengths)
    for cutoff in (1.2, 4.0, 15.0):
        result = CellList(cutoff, batch_atoms=7)(grid, positions)
        i, j = torch.triu_indices(len(positions), len(positions), 1)
        d = grid.minimum_image(positions[j] - positions[i])
        keep = d.square().sum(-1) < cutoff**2
        assert torch.equal(result.first, i[keep])
        assert torch.equal(result.second, j[keep])
        assert torch.allclose(result.displacement, d[keep])


def test_cell_list_budget_is_explicit():
    grid = Grid((10, 10, 10), (1.0, 1.0, 1.0))
    with pytest.raises(CapacityError):
        CellList(4.0, max_candidates=10)(grid, torch.zeros((10, 3), dtype=torch.float64))


def test_support_chunks_do_not_change_physics(water):
    layout = make_layout(water)
    a = build_support(water, layout, max_candidates=17)
    b = build_support(water, layout, max_candidates=100_000)
    ka = a.atom_index * water.grid.n_voxels + a.voxel_index
    kb = b.atom_index * water.grid.n_voxels + b.voxel_index
    ia, ib = ka.argsort(), kb.argsort()
    assert torch.equal(ka[ia], kb[ib])
    assert torch.equal(a.displacement[ia], b.displacement[ib])


def test_interaction_complex_contacts_are_symmetric_and_exchange_is_conservative(water):
    graph = InteractionComplex.build(Evaluator()(water).context)
    row, col = graph.patch_edges()
    keys = set(zip(row.tolist(), col.tolist()))
    assert all((j, i) in keys for i, j in keys)
    assert all((i, i) in keys for i in range(water.n_atoms))
    exchange = torch.randn(len(graph.contacts.first), 3, dtype=torch.float64)
    change = graph.exchange(exchange)
    assert torch.allclose(change.sum(0), torch.zeros(3, dtype=torch.float64), atol=1e-14)


def test_nested_grid_transfer_conserves_integrals_and_is_adjoint():
    fine = Grid((12, 16, 20), (0.2, 0.3, 0.4))
    coarse = Grid((4, 8, 5), (0.6, 0.6, 1.6))
    transfer = GridTransfer(fine, coarse)
    f = torch.randn(*fine.shape, 3, dtype=torch.float64)
    c = torch.randn(*coarse.shape, 3, dtype=torch.float64)
    assert torch.allclose(
        transfer.restrict(f).sum((0, 1, 2)) * coarse.dv, f.sum((0, 1, 2)) * fine.dv, atol=1e-13
    )
    assert (transfer.restrict(f) * c).sum() * coarse.dv == pytest.approx(
        float((f * transfer.prolong(c)).sum() * fine.dv), abs=1e-12
    )


def test_transfer_rejects_different_box():
    with pytest.raises(StateError, match="same physical box"):
        GridTransfer(Grid((8, 8, 8), (1.0, 1.0, 1.0)), Grid((4, 4, 4), (1.0, 1.0, 1.0)))


def test_face_operator_negative_adjoint_and_zero_integral():
    grid = Grid((7, 8, 9), (0.5, 0.8, 1.1))
    z = torch.randn(*grid.shape, 2, dtype=torch.float64)
    j = torch.randn(3, *grid.shape, 2, dtype=torch.float64)
    assert float((face_gradient(grid, z) * j).sum()) == pytest.approx(
        float(-(z * face_divergence(grid, j)).sum()), abs=1e-11
    )
    assert face_divergence(grid, j).sum().abs() < 1e-12


def test_cubic_deposition_partition_boundary_gradient_and_batch_invariance():
    grid = Grid((8, 8, 8), (0.4, 0.5, 0.6))
    r = torch.tensor(
        [[0.0, 0.0, 0.0], [1.32, 2.71, 3.73], [3.199, 0.01, 4.77]],
        dtype=torch.float64,
        requires_grad=True,
    )
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
    field = deposit_cubic(grid, r, values, batch_atoms=1)
    assert torch.allclose(field.sum((0, 1, 2)) * grid.dv, values.sum(0), atol=2e-14)
    assert torch.allclose(field, deposit_cubic(grid, r, values), atol=1e-14)
    grad = torch.autograd.grad(field.square().sum(), r)[0]
    eps = 1e-6
    rp, rm = r.detach().clone(), r.detach().clone()
    rp[1, 0] += eps
    rm[1, 0] -= eps
    fd = (
        deposit_cubic(grid, rp, values).square().sum()
        - deposit_cubic(grid, rm, values).square().sum()
    ) / (2 * eps)
    assert float(grad[1, 0]) == pytest.approx(float(fd), abs=1e-6, rel=1e-6)
