import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from unichem import Evaluator, Grid
from unichem.checkpoint import load_world, save_world
from unichem.errors import StateError


def test_checkpoint_roundtrip_preserves_canonical_world(water, tmp_path):
    path = tmp_path / "world.npz"
    save_world(water, path)
    loaded = load_world(path)
    assert loaded.grid == water.grid and loaded.species == water.species
    assert torch.equal(loaded.atoms.positions, water.atoms.positions)
    assert torch.equal(loaded.electrons.kernel.values, water.electrons.kernel.values)
    assert torch.equal(loaded.latent.memory, water.latent.memory)
    assert float(Evaluator()(loaded).energy) == pytest.approx(
        float(Evaluator()(water).energy), abs=1e-12
    )


def test_checkpoint_rejects_unknown_schema(h2, tmp_path):
    path = tmp_path / "world.npz"
    save_world(h2, path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    meta = json.loads(str(arrays["metadata"].item()))
    meta["schema_version"] = 100
    arrays["metadata"] = np.asarray(json.dumps(meta))
    np.savez(path, **arrays)
    with pytest.raises(StateError, match="schema"):
        load_world(path)


def test_checkpoint_never_uses_pickled_objects(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, metadata=np.asarray({"hello": "world"}, dtype=object))
    with pytest.raises(StateError):
        load_world(path)


def test_bad_state_is_rejected(h2):
    bad = replace(h2, atoms=replace(h2.atoms, ids=torch.tensor([1, 1])))
    with pytest.raises(StateError, match="unique"):
        bad.validate()
    with pytest.raises(StateError):
        Grid((8, 8, 8), (1.0, 0.0, 1.0))


def test_grid_minimum_image_and_wrap(h2):
    p = torch.tensor([[-0.1, 11.0, 21.7]], dtype=torch.float64)
    wrapped = h2.grid.wrap(p)
    assert torch.allclose(
        wrapped, torch.tensor([[10.7, 0.2, 0.1]], dtype=torch.float64), atol=1e-12
    )
    d = h2.grid.minimum_image(torch.tensor([[10.6, 0.0, 0.0]], dtype=torch.float64))
    assert torch.allclose(d, torch.tensor([[-0.2, 0.0, 0.0]], dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is not available")
def test_cuda_density_and_energy_match_cpu(h2):
    cpu = Evaluator()(h2)
    gpu = Evaluator()(h2.to(device="cuda"))
    assert torch.allclose(
        cpu.context.quantum.density, gpu.context.quantum.density.cpu(), atol=1e-10, rtol=1e-9
    )
    assert float(cpu.energy) == pytest.approx(float(gpu.energy.cpu()), abs=1e-9)
