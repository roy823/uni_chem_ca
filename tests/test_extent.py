import math
from dataclasses import replace

import pytest
import torch

from unichem import Evaluator, Grid, make_world
from unichem.errors import ResolutionError
from unichem.quantum.basis import sample_basis
from unichem.quantum.support import build_support, make_layout


def test_atom_is_many_voxels_and_voxel_can_have_multiple_atoms(water):
    context = Evaluator()(water).context
    assert (context.support.counts_per_atom() > 500).all()
    assert (context.support.multiplicity() > 1).sum() > 100
    assert int(context.support.multiplicity().max()) == 3


def test_continuous_nucleus_is_not_snapped_to_a_grid_cell(h2):
    shifted = h2.with_positions(h2.atoms.positions + 0.137)
    assert torch.allclose(
        shifted.atoms.positions - h2.atoms.positions, torch.full_like(h2.atoms.positions, 0.137)
    )
    old = Evaluator()(h2).context.quantum.density
    new = Evaluator()(shifted).context.quantum.density
    assert not torch.allclose(old, new)


def test_physical_size_survives_grid_refinement():
    values = []
    for size, spacing in [(20, 0.6), (40, 0.3)]:
        world = make_world(["H"], [[6.137, 6.234, 6.351]], grid=Grid((size,) * 3, (spacing,) * 3))
        c = Evaluator()(world).context
        values.append((float(c.clouds.rms_radius[0]), int(c.support.counts_per_atom()[0])))
        assert torch.allclose(
            c.clouds.centroid_offset, torch.zeros_like(c.clouds.centroid_offset), atol=2e-4
        )
    assert abs(values[0][0] - values[1][0]) < 2e-3
    assert 7.7 < values[1][1] / values[0][1] < 8.3


def test_larger_cloud_changes_extent_not_atom_identity():
    world = make_world(["H"], [[6.137, 6.234, 6.351]], grid=Grid((24,) * 3, (0.6,) * 3))
    expanded = replace(
        world,
        atoms=replace(
            world.atoms, log_width_scale=torch.full_like(world.atoms.log_width_scale, math.log(1.3))
        ),
    )
    a, b = Evaluator()(world).context, Evaluator()(expanded).context
    assert torch.equal(world.atoms.ids, expanded.atoms.ids)
    assert b.support.counts_per_atom()[0] > a.support.counts_per_atom()[0]
    assert float(b.clouds.rms_radius[0] / a.clouds.rms_radius[0]) == pytest.approx(1.3, rel=2e-3)
    assert float(b.quantum.density.sum() * world.grid.dv) == pytest.approx(1.0, abs=1e-12)


def test_periodic_cloud_crosses_boundary_without_duplicate_incidence():
    w = make_world(["H"], [[0.13, 5.27, 5.11]], grid=Grid((18,) * 3, (0.6,) * 3))
    s = build_support(w, make_layout(w))
    keys = s.atom_index * w.grid.n_voxels + s.voxel_index
    assert keys.unique().numel() == keys.numel()
    ix = s.voxel_index // (w.grid.shape[1] * w.grid.shape[2])
    assert int(ix.min()) == 0 and int(ix.max()) == 17
    assert float(Evaluator()(w).context.quantum.density.sum() * w.grid.dv) == pytest.approx(1.0)


def test_too_coarse_grid_raises_instead_of_one_pixel_atom():
    w = make_world(["H"], [[5.0, 5.0, 5.0]], grid=Grid((8,) * 3, (2.0,) * 3))
    with pytest.raises(ResolutionError, match="under-resolved"):
        Evaluator()(w)


def test_cloud_too_large_for_box_rejected():
    w = make_world(["O"], [[2.0, 2.0, 2.0]], grid=Grid((8,) * 3, (0.6,) * 3))
    with pytest.raises(ResolutionError, match="half"):
        Evaluator()(w)


def test_core_deposition_is_conservative(water):
    c = Evaluator()(water).context
    assert float(c.samples.core_density.sum() * water.grid.dv) == pytest.approx(8.0, abs=1e-12)
    assert float(c.clouds.electron_number.sum()) == pytest.approx(8.0, abs=1e-12)


def test_s_and_p_shapes_are_not_interchangeable():
    w = make_world(["O"], [[6.3, 6.3, 6.3]], grid=Grid((20,) * 3, (0.6,) * 3))
    layout = make_layout(w)
    support = build_support(w, layout)
    ao = sample_basis(w, layout, support).dense().reshape(*w.grid.shape, 4)
    assert abs(float(ao[10, 10, 10, 1])) < 1e-12
    assert abs(float(ao[11, 10, 10, 1])) > 0.1
    assert abs(float(ao[10, 11, 10, 1])) < 1e-12
    assert float(ao[10, 10, 10, 0]) > 0.99
