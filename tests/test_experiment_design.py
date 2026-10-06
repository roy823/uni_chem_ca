import json

import numpy as np
import pytest

from unichem.benchmarks import run_numerical_controls
from unichem.reference_design import write_reference_design


def test_native_ca_controls_reproduce_known_laws_and_replay(tmp_path):
    report = run_numerical_controls(output=tmp_path, steps=4)
    assert report["passed"]
    assert report["chemistry_claim"] is False
    assert report["checkpoint_replay_bitwise"]
    assert report["stochastic_replay_bitwise"]


def test_reference_design_preserves_symmetric_pairs_and_surface_holdout(tmp_path):
    result = write_reference_design(tmp_path)
    manifest = json.loads((tmp_path / "jobs.json").read_text())
    assert result["labels_available"] is False and manifest["not_a_training_dataset"]
    jobs = [j for j in manifest["jobs"] if j["family"] == "symmetric_zundel_surface"]
    assert len(jobs) == 117
    surfaces = {}
    energy_keys = {"energy", "forces", "density"}
    for job in manifest["jobs"]:
        assert not energy_keys.intersection(job)
    for j in jobs:
        r = j["coordinates"]["oo_angstrom"]
        delta = j["coordinates"]["proton_distance_difference_angstrom"]
        surfaces.setdefault(r, set()).add(j["split"])
        partner = next(
            p
            for p in jobs
            if p["coordinates"]["oo_angstrom"] == r
            and p["coordinates"]["proton_distance_difference_angstrom"] == -delta
        )
        a = np.array(j["positions_angstrom"])
        b = np.array(partner["positions_angstrom"])
        c = a[:2].mean(0)
        assert a[2, 0] - c[0] == pytest.approx(-(b[2, 0] - c[0]))
    assert all(len(v) == 1 for v in surfaces.values())
    assert surfaces[2.6] == {"test"}


def test_manifest_does_not_duplicate_zero_field_geometries(tmp_path):
    write_reference_design(tmp_path)
    jobs = json.loads((tmp_path / "jobs.json").read_text())["jobs"]
    keys = [
        (
            tuple(np.round(np.array(j["positions_angstrom"]).ravel(), 8)),
            tuple(j["external_field_au"]),
            j["net_charge"],
        )
        for j in jobs
    ]
    assert len(keys) == len(set(keys))
