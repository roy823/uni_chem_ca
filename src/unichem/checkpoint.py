from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .errors import StateError
from .grid import Grid
from .quantum.blocks import BlockMatrix
from .species import OrbitalSpec, Species
from .state import AtomState, ElectronState, LatentState, World

SCHEMA_VERSION = 3


def save_world(world: World, path: str | Path) -> None:
    """Atomic replacement of a versioned, non-pickle NPZ checkpoint.

    Only canonical state is persisted, never stale field/support work buffers.
    Model weights are separate artifacts. Coordinate units are explicit.
    """
    world.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "units": "atomic",
        "grid": asdict(world.grid),
        "species": [asdict(s) for s in world.species],
        "electron_count": world.electron_count,
        "time": world.time,
        "step": world.step,
        "kernel_cutoff": world.electrons.cutoff,
        "external_field": world.external_field,
        "latent_grid": asdict(world.latent_grid),
        "latent_kind": world.latent.kind,
    }
    array = lambda t: t.detach().cpu().numpy()
    arrays = {
        "metadata": np.asarray(json.dumps(metadata, ensure_ascii=False)),
        "atom_ids": array(world.atoms.ids),
        "positions": array(world.atoms.positions),
        "momenta": array(world.atoms.momenta),
        "log_width_scale": array(world.atoms.log_width_scale),
        "kernel_row": array(world.electrons.kernel.row),
        "kernel_col": array(world.electrons.kernel.col),
        "kernel_values": array(world.electrons.kernel.values),
        "spin_counts": array(world.electrons.spin_counts),
        "latent": array(world.latent.channels),
        "memory": array(world.latent.memory),
    }
    handle, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as file:
            np.savez_compressed(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_world(path: str | Path, *, device="cpu") -> World:
    """No arbitrary pickle deserialization. Files still need normal resource limits."""
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            if meta["schema_version"] != SCHEMA_VERSION or meta["units"] != "atomic":
                raise StateError("Unsupported checkpoint schema or unit system")
            grid = Grid(
                tuple(meta["grid"]["shape"]),
                tuple(meta["grid"]["spacing"]),
                tuple(meta["grid"]["origin"]),
            )
            latent_meta = meta["latent_grid"]
            latent_grid = Grid(
                tuple(latent_meta["shape"]),
                tuple(latent_meta["spacing"]),
                tuple(latent_meta["origin"]),
            )
            species = []
            for item in meta["species"]:
                spec = dict(item)
                spec["orbitals"] = tuple(OrbitalSpec(**o) for o in spec["orbitals"])
                species.append(Species(**spec))
            t = lambda key: torch.from_numpy(data[key].copy()).to(device)
            world = World(
                grid,
                tuple(species),
                AtomState(t("atom_ids"), t("positions"), t("momenta"), t("log_width_scale")),
                ElectronState(
                    BlockMatrix(t("kernel_row"), t("kernel_col"), t("kernel_values"), len(species)),
                    t("spin_counts"),
                    meta["kernel_cutoff"],
                ),
                LatentState(t("latent"), t("memory"), latent_grid, meta["latent_kind"]),
                float(meta["electron_count"]),
                float(meta["time"]),
                int(meta["step"]),
                tuple(meta["external_field"]),
            )
        world.validate()
        return world
    except StateError:
        raise
    except (KeyError, ValueError, TypeError, OSError) as exc:
        raise StateError(f"Invalid checkpoint: {exc}") from exc
