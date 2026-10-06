"""Tolerance-controlled sparse orthogonalization and zero-temperature TC2.

No orbital coefficient matrix, diagonalization, sigmoid occupancy floor or
bounded auxiliary generator. All quantum algebra uses atom blocks. Products
retain their fill; exhausting the explicit memory budget is a hard failure.
"""

from dataclasses import dataclass

import torch

from ..errors import StateError
from .blocks import BlockMatrix


@dataclass(frozen=True)
class Metric:
    transform: BlockMatrix
    identity: BlockMatrix
    active: torch.Tensor
    defect: float
    iterations: int
    overlap_norm: float
    compression_bound_sum: float


def orthogonalize(overlap, active, *, tolerance=1e-12, max_iterations=50):
    """Newton-Schulz inverse square root, stopped by actual metric residual."""
    diagonal = overlap.diagonal_values().diagonal(dim1=-2, dim2=-1)
    if bool((diagonal[active] <= 0).any()):
        raise StateError("Nonpositive AO overlap diagonal")
    scale = torch.zeros_like(diagonal)
    scale[active] = diagonal[active].rsqrt()
    d = BlockMatrix.diagonal(torch.diag_embed(scale), overlap.budget)
    identity = BlockMatrix.diagonal(torch.diag_embed(active.to(diagonal.dtype)), overlap.budget)
    normalized = d.matmul(overlap).matmul(d).symmetric()
    alpha = normalized.row_bound().clamp_min(1)
    a = normalized.scale(1 / alpha)
    z = identity
    compression_sum = 0.0
    rank = max(1, int(active.sum()))
    for iteration in range(max_iterations + 1):
        transform = d.matmul(z).scale(alpha.rsqrt())
        represented = transform.transpose().matmul(overlap).matmul(transform)
        defect = float(identity.add(represented, -1).row_bound())
        if defect <= tolerance:
            return Metric(
                transform,
                identity,
                active,
                defect,
                iteration,
                float(overlap.row_bound()),
                compression_sum,
            )
        if not torch.isfinite(z.values).all():
            raise StateError("AO metric iteration became nonfinite")
        correction = identity.scale(3).add(a.matmul(z.matmul(z)), -1)
        z = z.matmul(correction).scale(0.5).symmetric()
        z, discarded = z.compress(tolerance / (4 * rank**0.5))
        compression_sum += discarded
    raise StateError(f"AO metric did not converge: defect={defect:.3e}; check basis conditioning")


@dataclass(frozen=True)
class Projector:
    density: BlockMatrix
    orthogonal_density: BlockMatrix
    iterations: int
    number_error: float
    purity_error: float
    compression_bound_sum: float = 0.0


def ground_projector(fock, metric, count, *, tolerance=1e-12, max_iterations=120):
    """Second-order trace-correcting purification for a gapped spin sector.

    An unresolved Fermi-level degeneracy fails explicitly. Finite-temperature
    occupations require a separately defined thermodynamic model, not a fallback.
    """
    count = float(count)
    rank = int(metric.active.sum())
    if not count.is_integer() or not 0 <= count <= rank:
        raise StateError("Zero-temperature spin counts must be integers within AO capacity")
    identity, transform = metric.identity, metric.transform
    if count == 0 or count == rank:
        x = identity.scale(count / rank)
        p = transform.matmul(x).matmul(transform.transpose()).symmetric()
        return Projector(p, x, 0, 0.0, 0.0)
    h = transform.transpose().matmul(fock).matmul(transform).symmetric()
    lower, upper = h.spectral_bounds(metric.active)
    span = float(upper - lower)
    if span < 1e-14:
        raise StateError("Unresolved zero-temperature partial occupation of a degenerate sector")
    margin = 1e-12 * (1 + max(abs(float(lower)), abs(float(upper))))
    lower, upper = lower - margin, upper + margin
    x = identity.scale(upper).add(h, -1).scale(1 / (upper - lower))
    compression_sum = 0.0
    for iteration in range(max_iterations + 1):
        squared = x.matmul(x).symmetric()
        number = float(x.trace())
        number_error = abs(number - count)
        purity_error = float(x.add(squared, -1).norm())
        if number_error <= tolerance and purity_error <= tolerance:
            p = transform.matmul(x).matmul(transform.transpose()).symmetric()
            p, discarded = p.compress(
                tolerance / (max_iterations * rank * max(1, metric.overlap_norm))
            )
            return Projector(
                p, x, iteration, number_error, purity_error, compression_sum + discarded
            )
        x = squared if number > count else x.scale(2).add(squared, -1)
        x, discarded = x.compress(tolerance / (4 * rank**0.5))
        compression_sum += discarded
        if not torch.isfinite(x.values).all():
            raise StateError("Density purification became nonfinite")
    raise StateError(
        f"Zero-temperature projector did not converge: number={number_error:.3e}, purity={purity_error:.3e}; check the occupied/virtual gap"
    )
