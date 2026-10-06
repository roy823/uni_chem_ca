from dataclasses import replace

import pytest
import torch

from unichem import Evaluator
from unichem.errors import StateError
from unichem.models.slow import vamp2_score
from unichem.quantum.angular import real_angular
from unichem.quantum.chemistry import chemical_observables
from unichem.quantum.hamiltonian import kernel_stationarity
from unichem.species import OrbitalSpec


def test_s_p_d_angular_structure_and_nodes():
    delta = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [0.0, 2.0, 3.0]], dtype=torch.float64
    )
    code = torch.arange(-1, 8)[None, :].expand(3, -1)
    vals = real_angular(delta, code, torch.ones_like(code, dtype=torch.float64))
    assert torch.equal(vals[0, 1:4], -vals[1, 1:4])
    assert torch.equal(vals[0, 4:], vals[1, 4:])
    assert vals[2, 4] == 0  # xy node at x=0
    assert OrbitalSpec(3, 2, "dz2", 0.9).axis == 7
    with pytest.raises(StateError):
        OrbitalSpec(2, 2, "dxy", 0.9)


def test_kernel_stationarity_matches_directional_finite_difference(h2):
    evaluator = Evaluator()
    gen = torch.Generator().manual_seed(837)
    # Evaluate at a generic point, away from absolute-value/row-maximum ties
    # in the conservative spectral bound used by the matrix-function map.
    h2 = h2.with_kernel_values(
        h2.electrons.kernel.values
        + 0.03 * torch.randn(h2.electrons.kernel.values.shape, generator=gen, dtype=torch.float64)
    )
    h = kernel_stationarity(h2, evaluator)
    direction = torch.randn(h2.electrons.kernel.values.shape, generator=gen, dtype=torch.float64)
    eps = 1e-5
    raw = h2.electrons.kernel.values
    fd = (
        evaluator(h2.with_kernel_values(raw + eps * direction)).energy
        - evaluator(h2.with_kernel_values(raw - eps * direction)).energy
    ) / (2 * eps)
    assert float((h.gradient.values * direction).sum()) == pytest.approx(
        float(fd), rel=1e-5, abs=1e-7
    )


def test_chemical_observer_reads_but_cannot_define_reaction_rules(h2):
    view = chemical_observables(Evaluator()(h2).context)
    assert view.pair_first.tolist() == [0]
    assert view.pair_second.tolist() == [1]
    assert float(view.assigned_valence_charge.sum()) == pytest.approx(0.0, abs=1e-12)
    view.atomic_ids[0] = 99
    assert h2.atoms.ids[0] == 0


def test_vamp_prefers_time_correlation_rejects_collapse_and_backpropagates():
    g = torch.Generator().manual_seed(88)
    x = torch.randn(1000, 2, generator=g, dtype=torch.float64, requires_grad=True)
    correlated = 0.98 * x + 0.05 * torch.randn(1000, 2, generator=g, dtype=torch.float64)
    random = torch.randn(1000, 2, generator=g, dtype=torch.float64)
    score = vamp2_score(x, correlated)
    assert float(score.detach()) > float(vamp2_score(x, random).detach()) + 1.5
    score.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    constant = torch.ones(20, 2, dtype=torch.float64)
    assert vamp2_score(constant, constant) == 0


def test_core_charge_cloud_is_independent_of_electronic_width(h2):
    evaluator = Evaluator()
    original = evaluator(h2).context.samples.core_density
    wide = replace(h2, atoms=replace(h2.atoms, log_width_scale=h2.atoms.log_width_scale + 0.1))
    altered = evaluator(wide).context.samples.core_density
    assert torch.allclose(original, altered, atol=2e-15, rtol=1e-14)
