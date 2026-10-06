from __future__ import annotations

import torch
from torch import Tensor

from ..errors import StateError
from ..grid import Grid


def face_gradient(grid: Grid, cells: Tensor) -> Tensor:
    """Forward differences on oriented positive faces, shape (3,*cells.shape)."""
    if cells.shape[:3] != grid.shape:
        raise StateError("Field shape and lattice disagree")
    return torch.stack(
        [(torch.roll(cells, -1, axis) - cells) / h for axis, h in enumerate(grid.spacing)]
    )


def face_divergence(grid: Grid, faces: Tensor) -> Tensor:
    """Negative adjoint of face_gradient in the volume-weighted inner product."""
    if faces.shape[0] != 3 or faces.shape[1:4] != grid.shape:
        raise StateError("Expected three oriented face fields")
    return sum((faces[a] - torch.roll(faces[a], 1, a)) / h for a, h in enumerate(grid.spacing))
