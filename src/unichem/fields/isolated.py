"""Free-space Coulomb convolution on a finite vacuum grid."""

import math

import torch

from .poisson import FieldView


class IsolatedPoisson:
    """Zero-padded FFT evaluation of the isolated 1/r Green function.

    No charged-cell background or periodic replicas. The singular cell uses
    the mean 1/r in a same-volume sphere, an explicit quadrature approximation
    whose error must be reduced with grid refinement. Energy differentiation
    is consistent with the same symmetric convolution. The periodic discrete
    Laplacian/face-energy identity does not apply to this different operator.
    """

    def __init__(self):
        self._kernels = {}

    def solve(self, grid, charge_density):
        rho = charge_density.reshape(grid.shape)
        shape = tuple(2 * n for n in grid.shape)
        key = (grid.shape, grid.spacing, rho.dtype, str(rho.device))
        if key not in self._kernels:
            axes = []
            for n, h in zip(grid.shape, grid.spacing):
                x = torch.arange(2 * n, dtype=rho.dtype, device=rho.device)
                axes.append(torch.minimum(x, 2 * n - x) * h)
            coordinates = torch.meshgrid(*axes, indexing="ij")
            r2 = sum(c.square() for c in coordinates)
            radius = (3 * grid.dv / (4 * math.pi)) ** (1 / 3)
            kernel = torch.where(r2 > 0, r2.clamp_min(1e-30).rsqrt(), rho.new_tensor(1.5 / radius))
            self._kernels[key] = torch.fft.rfftn(kernel)
        padded = torch.nn.functional.pad(
            rho, (0, grid.shape[2], 0, grid.shape[1], 0, grid.shape[0])
        )
        extended = torch.fft.irfftn(torch.fft.rfftn(padded) * self._kernels[key], s=shape) * grid.dv
        slices = tuple(slice(0, n) for n in grid.shape)
        phi = extended[slices]
        faces = []
        for a, h in enumerate(grid.spacing):
            nxt = list(slices)
            nxt[a] = slice(1, grid.shape[a] + 1)
            faces.append(-(extended[tuple(nxt)] - phi) / h)
        return FieldView(
            rho,
            phi,
            torch.stack(faces, -1),
            0.5 * (rho * phi).sum() * grid.dv,
            rho.sum() * grid.dv,
            rho.new_zeros(()),
        )
