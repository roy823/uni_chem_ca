from __future__ import annotations

import itertools

import torch
from torch import Tensor

from ..errors import CapacityError, StateError
from ..grid import Grid


def deposit_cubic(
    grid: Grid, positions: Tensor, values: Tensor, *, batch_atoms: int = 2048
) -> Tensor:
    """Conservative C2 cardinal-cubic deposition to a cell-centred periodic grid.

    Exactly 4^3 weights per particle, independent of atomic cloud size. This is
    an auxiliary NUCLEAR DESCRIPTOR, never the electron-cloud representation.
    Sum(grid*dv) == sum(values). Geometry gradients flow through spline weights;
    no integer position snapping. All batches read the same source snapshot.
    """
    if positions.ndim != 2 or positions.shape[1] != 3 or values.shape[0] != len(positions):
        raise StateError("Deposition expects (N,3) positions and N payloads")
    if batch_atoms < 1:
        raise CapacityError("A positive deposition batch size is required")
    scalar = values.ndim == 1
    if scalar:
        values = values[:, None]
    if values.ndim != 2 or values.device != positions.device or values.dtype != positions.dtype:
        raise StateError("Nuclear descriptor values must be (N,C) with matching dtype/device")
    offsets = torch.tensor(list(itertools.product(range(4), repeat=3)), device=positions.device)
    output = positions.new_zeros((grid.n_voxels, values.shape[1]))
    for lo in range(0, len(positions), batch_atoms):
        r = positions[lo : lo + batch_atoms]
        x = (grid.wrap(r) - r.new_tensor(grid.origin)) / r.new_tensor(grid.spacing) - 0.5
        base = torch.floor(x).long()
        t = x - base
        weights = torch.stack(
            (
                (1 - t) ** 3 / 6,
                (3 * t**3 - 6 * t**2 + 4) / 6,
                (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6,
                t**3 / 6,
            ),
            dim=-1,
        )
        indices = torch.remainder(
            base[:, None, :] - 1 + offsets, torch.tensor(grid.shape, device=r.device)
        )
        weight = (
            weights[:, 0, offsets[:, 0]]
            * weights[:, 1, offsets[:, 1]]
            * weights[:, 2, offsets[:, 2]]
        )
        payload = weight[..., None] * values[lo : lo + batch_atoms, None, :] / grid.dv
        output = output.index_add(
            0, grid.flatten(indices).flatten(), payload.reshape(-1, values.shape[1])
        )
    shaped = output.reshape(*grid.shape, values.shape[1])
    return shaped[..., 0] if scalar else shaped
