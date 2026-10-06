import json
from pathlib import Path

import numpy as np
import pytest

from unichem.datasets import BOHR_ANGSTROM, HARTREE_EV, ReferenceFrame, geometry_hash, load_deepmd


def test_deepmd_units_type_order_and_frame_alignment(tmp_path):
    (tmp_path / "type.raw").write_text("0 1 1")
    directory = tmp_path / "set.000"
    directory.mkdir()
    np.save(directory / "coord.npy", np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]))
    np.save(directory / "force.npy", np.ones((1, 9)))
    np.save(directory / "energy.npy", np.array([-27.211386245988]))
    np.save(directory / "box.npy", np.array([[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]]))
    f = next(
        load_deepmd(tmp_path, source="unit fixture", group="g", split="training", type_map=(8, 1))
    )
    assert f.atomic_numbers.tolist() == [8, 1, 1]
    assert f.energy == pytest.approx(-1.0)
    assert f.positions[1, 0] == pytest.approx(1 / BOHR_ANGSTROM)
    assert f.forces[0, 0] == pytest.approx(BOHR_ANGSTROM / HARTREE_EV)


def test_duplicate_hash_detects_periodic_image_equivalence():
    p = np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 3.0]])
    f = ReferenceFrame(
        np.array([8, 1]),
        p,
        np.array([10.0, 10.0, 10.0]),
        0.0,
        np.zeros((2, 3)),
        "s",
        "g",
        "training",
        0,
    )
    other = ReferenceFrame(
        f.atomic_numbers, p + f.box, f.box, 0.0, f.forces, "s", "g", "validation", 0
    )
    assert geometry_hash(f) == geometry_hash(other)


def test_real_reference_suite_when_available_has_converged_number_consistent_rdms():
    root = Path("data/generated_pbe0")
    if not root.exists():
        pytest.skip("Generated reference data not downloaded/calculated in this checkout")
    for path in sorted(root.glob("[0-9][0-9][0-9][0-9].npz")):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            assert meta["scf_converged"]
            electron_count = float(data["numbers"].sum() - meta["net_charge"])
            dm = data["ao_density_matrix"]
            s = data["ao_overlap"]
            assert np.trace(dm @ s) == pytest.approx(electron_count, abs=1e-8)
            valence_count = electron_count - 2 * (data["numbers"] == 8).sum()
            assert np.trace(data["valence_ao_density_matrix"] @ s) == pytest.approx(
                valence_count, abs=1e-8
            )
            if "field_force_fd_error" in meta:
                assert meta["field_force_fd_error"] < 2e-5
