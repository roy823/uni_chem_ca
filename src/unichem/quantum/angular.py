from __future__ import annotations

import torch
from torch import Tensor

# These are real solid harmonics, not definite magnetic quantum number states.
COMPONENTS = {
    "s": (-1, 0),
    "px": (0, 1),
    "py": (1, 1),
    "pz": (2, 1),
    "dxy": (3, 2),
    "dyz": (4, 2),
    "dxz": (5, 2),
    "dx2-y2": (6, 2),
    "dz2": (7, 2),
}


def real_angular(delta: Tensor, code: Tensor, widths: Tensor) -> Tensor:
    """Real s/p/d solid-harmonic polynomials, batched over support edges/AOs.

    Overall normalization is set by the actual S metric. d components use the
    same relative harmonic normalization within the five-dimensional shell.
    There are no radial nodes here: n remains a shell LABEL, not an exact
    hydrogenic principal quantum number or a complete radial basis.
    """
    x, y, z = (delta[:, k, None] / widths for k in range(3))
    output = torch.ones_like(widths)
    components = (
        x,
        y,
        z,
        (3**0.5) * x * y,
        (3**0.5) * y * z,
        (3**0.5) * x * z,
        (3**0.5) / 2 * (x * x - y * y),
        0.5 * (2 * z * z - x * x - y * y),
    )
    for index, polynomial in enumerate(components):
        output = torch.where(code == index, polynomial, output)
    return output
