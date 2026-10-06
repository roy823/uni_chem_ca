from pathlib import Path

import numpy as np
import pytest

from unichem.datasets import HARTREE_EV
from unichem.landscapes import field_statistics, surface_statistics, vibrational_curvatures


def test_surface_distinguishes_centred_and_split_wells_without_energy_zero():
    x = np.linspace(-0.6, 0.6, 13)
    single = surface_statistics(x, -153 + x**2 / HARTREE_EV)
    double = surface_statistics(x, -153 + (x**4 - 0.5 * x**2) / HARTREE_EV)
    assert single["centre_shape"] == "single_well"
    assert single["centre_curvature_ev_per_angstrom2"] == pytest.approx(2)
    assert double["centre_shape"] == "double_well"
    assert double["sampled_centre_barrier_ev"] == pytest.approx(0.0625)
    assert not double["minimum_on_scan_boundary"]


def test_field_response_detects_unphysical_density_energy_disagreement():
    f = np.array([-0.002, -0.001, 0, 0.001, 0.002])
    e = -76 - 0.7 * f - 9 * f**2 / 2
    rows = field_statistics(f, e, 0.7 + 9 * f)
    assert rows[0]["polarizability_from_energy_bohr3"] == pytest.approx(9, abs=1e-7)
    assert rows[0]["dipole_consistency_error_au"] < 1e-9
    bad = field_statistics(f, e, np.full_like(f, 0.1))
    assert bad[0]["dipole_consistency_error_au"] == pytest.approx(0.6)
    assert bad[0]["polarizability_consistency_error_bohr3"] == pytest.approx(9, abs=1e-7)


def test_missing_field_sign_and_asymmetric_scan_cannot_claim_response():
    with pytest.raises(ValueError, match="Both field signs"):
        field_statistics([0, 0.001], [-76, -76.001], [0.1, 0.2])
    with pytest.raises(ValueError, match="symmetric scan"):
        surface_statistics([0, 0.1, 0.2, 0.3, 0.4], np.zeros(5))


@pytest.mark.skipif(
    not Path("outputs/water_dft_reference/reference.npz").exists(),
    reason="Local independent DFT reference absent",
)
def test_rigid_motion_projection_matches_independent_pyscf_normal_modes():
    with np.load("outputs/water_dft_reference/reference.npz", allow_pickle=False) as data:
        hessian = data["hessian"].transpose(0, 2, 1, 3).reshape(9, 9)
        frequencies = vibrational_curvatures(data["positions"], hessian, [15.999, 1.008, 1.008])
        assert frequencies == pytest.approx(data["frequencies_cm1"], abs=0.01)
