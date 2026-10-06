from __future__ import annotations

import math
from dataclasses import dataclass

from torch import Tensor

from ..errors import StateError
from ..grid import Grid


@dataclass(frozen=True)
class GridTransfer:
    """Nested finite-volume restriction/prolongation on the SAME physical box.

    restrict = volume mean, prolong = piecewise constant injection. The two
    are adjoint under their volume-weighted inner products, so density
    integrals and the variational chain rule agree across resolutions.
    This is not adaptive mesh refinement or interpolation of new QM data.
    """

    fine: Grid
    coarse: Grid

    def __post_init__(self):
        if self.fine.origin != self.coarse.origin:
            raise StateError("Nested grids must share an origin")
        if any(f % c != 0 for f, c in zip(self.fine.shape, self.coarse.shape)):
            raise StateError("Coarse shape must divide fine shape")
        if any(
            not math.isclose(a, b, rel_tol=1e-12)
            for a, b in zip(self.fine.lengths, self.coarse.lengths)
        ):
            raise StateError("Nested grids must represent the same physical box")

    @property
    def ratios(self) -> tuple[int, int, int]:
        return tuple(f // c for f, c in zip(self.fine.shape, self.coarse.shape))

    def restrict(self, field: Tensor) -> Tensor:
        if field.shape[:3] != self.fine.shape:
            raise StateError("Restriction expects a fine-grid field")
        dims = []
        for c, r in zip(self.coarse.shape, self.ratios):
            dims.extend((c, r))
        return field.reshape(*dims, *field.shape[3:]).mean(dim=(1, 3, 5))

    def prolong(self, field: Tensor) -> Tensor:
        if field.shape[:3] != self.coarse.shape:
            raise StateError("Prolongation expects a coarse-grid field")
        for axis, ratio in enumerate(self.ratios):
            field = field.repeat_interleave(ratio, dim=axis)
        return field
