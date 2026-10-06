from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch
from torch import Tensor

from .errors import StateError


@dataclass(frozen=True)
class Grid:
    """Periodic, orthorhombic, cell-centred finite-volume lattice in bohr.

    Positions stay continuous. Voxel i represents the volume centred at
    origin + (i + 1/2) * spacing; it is NOT an atom's position.
    """

    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(type(n) is not int or n < 4 for n in self.shape):
            raise StateError("Grid.shape requires three integer dimensions >= 4")
        if len(self.spacing) != 3 or any(not (0.0 < h < float("inf")) for h in self.spacing):
            raise StateError("Grid.spacing must be finite and positive")
        if len(self.origin) != 3 or not all(abs(x) < float("inf") for x in self.origin):
            raise StateError("Grid.origin must have three finite entries")

    @property
    def n_voxels(self) -> int:
        return prod(self.shape)

    @property
    def dv(self) -> float:
        return prod(self.spacing)

    @property
    def lengths(self) -> tuple[float, float, float]:
        return tuple(n * h for n, h in zip(self.shape, self.spacing))

    def vector(self, values, like: Tensor) -> Tensor:
        return torch.as_tensor(values, dtype=like.dtype, device=like.device)

    def wrap(self, positions: Tensor) -> Tensor:
        origin = self.vector(self.origin, positions)
        return torch.remainder(positions - origin, self.vector(self.lengths, positions)) + origin

    def minimum_image(self, displacement: Tensor) -> Tensor:
        box = self.vector(self.lengths, displacement)
        return displacement - box * torch.round(displacement / box)

    def flatten(self, indices: Tensor) -> Tensor:
        return (indices[..., 0] * self.shape[1] + indices[..., 1]) * self.shape[2] + indices[..., 2]

    def centres(self, like: Tensor) -> Tensor:
        axes = [torch.arange(n, device=like.device, dtype=like.dtype) for n in self.shape]
        ijk = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        return self.vector(self.origin, like) + (ijk + 0.5) * self.vector(self.spacing, like)
