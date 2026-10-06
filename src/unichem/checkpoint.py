"""Canonical v0.4 molecular checkpoints; no old auxiliary-kernel migration."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from .errors import StateError
from .molecule import AtomState, World
from .quantum.blocks import BlockMatrix
from .runtime.local_cells import CellSnapshot


def save_world(world, path):
    world.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = world.atoms
    arrays = {
        name: getattr(a, name).numpy()
        for name in ("ids", "numbers", "positions", "momenta", "masses")
    }
    metadata = {
        "schema_version": 4,
        "units": "atomic",
        "basis": world.basis,
        "charge": world.charge,
        "spin_counts": world.spin_counts,
        "external_field": world.external_field,
        "time": world.time,
        "step": world.step,
        "has_density": world.densities is not None,
    }
    if world.densities is not None:
        for spin, p in enumerate(world.densities):
            for name in ("row", "col", "values"):
                arrays[f"{spin}_{name}"] = getattr(p, name).numpy()
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_world(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"].item()))
        if meta["schema_version"] != 4 or meta["units"] != "atomic":
            raise StateError("Unsupported checkpoint; v0.4 persists physical density kernels")
        get = lambda key: torch.from_numpy(data[key].copy())
        atoms = AtomState(
            *(get(name) for name in ("ids", "numbers", "positions", "momenta", "masses"))
        )
        densities = None
        if meta["has_density"]:
            densities = tuple(
                BlockMatrix(
                    get(f"{s}_row"), get(f"{s}_col"), get(f"{s}_values"), len(atoms.numbers)
                )
                for s in range(2)
            )
        world = World(
            atoms,
            meta["basis"],
            meta["charge"],
            tuple(meta["spin_counts"]),
            densities,
            tuple(meta["external_field"]),
            float(meta["time"]),
            int(meta["step"]),
        )
    world.validate()
    return world


def save_cells(state, path):
    """Persist the bounded CA state; independent of the HF reference schema."""
    state.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "state_kind": "local_ca",
        "schema_version": 1,
        "units": "atomic",
        "time": state.time,
        "step": state.step,
    }
    arrays = {
        name: getattr(state.atoms, name).detach().cpu().numpy()
        for name in ("ids", "numbers", "positions", "momenta", "masses")
    }
    arrays.update(
        density=state.density.detach().cpu().numpy(),
        channels=state.channels.cpu().numpy(),
        metadata=np.asarray(json.dumps(metadata)),
    )
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_cells(path, *, device="cpu"):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"].item()))
        if meta != {
            "state_kind": "local_ca",
            "schema_version": 1,
            "units": "atomic",
            "time": meta.get("time"),
            "step": meta.get("step"),
        }:
            raise StateError("Unsupported local CA checkpoint")
        get = lambda name: torch.from_numpy(data[name].copy()).to(device)
        atoms = AtomState(
            *(get(name) for name in ("ids", "numbers", "positions", "momenta", "masses"))
        )
        state = CellSnapshot(
            atoms, get("density"), get("channels"), float(meta["time"]), int(meta["step"])
        )
    state.validate()
    return state
