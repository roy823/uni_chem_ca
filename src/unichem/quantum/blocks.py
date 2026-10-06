"""Atom-block sparse algebra with differentiable values and bounded fill.

No operation allocates a system-sized dense matrix. Integer joins are outside
autograd; ordinary batched products/index_add retain higher derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..errors import CapacityError, StateError


@dataclass(frozen=True)
class BlockMatrix:
    row: Tensor
    col: Tensor
    values: Tensor
    n: int
    budget: int = 100_000

    @property
    def width(self):
        return self.values.shape[-1]

    @classmethod
    def assemble(cls, row, col, values, n, budget=100_000):
        keys, inverse = torch.unique(row * n + col, sorted=True, return_inverse=True)
        if keys.numel() > budget:
            raise CapacityError(
                f"Sparse patch fill {keys.numel()} exceeds {budget}; no truncation applied"
            )
        out = values.new_zeros((len(keys), *values.shape[1:])).index_add(0, inverse, values)
        return cls(keys // n, keys % n, out, n, budget)

    @classmethod
    def diagonal(cls, values, budget=100_000):
        ids = torch.arange(len(values), device=values.device)
        return cls(ids, ids, values, len(values), budget)

    def transpose(self):
        return self.assemble(self.col, self.row, self.values.transpose(-1, -2), self.n, self.budget)

    def scale(self, scalar):
        return BlockMatrix(self.row, self.col, self.values * scalar, self.n, self.budget)

    def add(self, other, scale=1.0):
        return self.assemble(
            torch.cat((self.row, other.row)),
            torch.cat((self.col, other.col)),
            torch.cat((self.values, scale * other.values)),
            self.n,
            self.budget,
        )

    def matmul(self, other):
        if self.n != other.n or self.width != other.width:
            raise StateError("Incompatible atom-block operators")
        order = torch.argsort(other.row, stable=True)
        sorted_rows = other.row[order]
        lo = torch.searchsorted(sorted_rows, self.col)
        hi = torch.searchsorted(sorted_rows, self.col, right=True)
        counts = hi - lo
        total = int(counts.sum())
        if total > self.budget * max(8, self.width):
            raise CapacityError("Sparse multiplication candidate budget exceeded")
        left = torch.repeat_interleave(torch.arange(len(counts), device=counts.device), counts)
        offset = torch.arange(total, device=counts.device) - torch.repeat_interleave(
            counts.cumsum(0) - counts, counts
        )
        right = order[lo[left] + offset]
        return self.assemble(
            self.row[left],
            other.col[right],
            self.values[left] @ other.values[right],
            self.n,
            self.budget,
        )

    def diagonal_values(self):
        keep = self.row == self.col
        return self.values.new_zeros((self.n, self.width, self.width)).index_add(
            0, self.row[keep], self.values[keep]
        )

    def trace_product(self, other):
        if self.n != other.n or self.width != other.width:
            raise StateError("Incompatible trace-product operators")
        if not len(self.row) or not len(other.row):
            return (self.values.sum() + other.values.sum()) * 0
        keys = other.row * self.n + other.col
        query = self.col * self.n + self.row
        where = torch.searchsorted(keys, query)
        valid = where < len(keys)
        safe = where.clamp_max(max(0, len(keys) - 1))
        valid = valid & (keys[safe] == query)
        return (self.values[valid] * other.values[safe[valid]].transpose(-1, -2)).sum()

    def row_bound(self):
        sums = self.values.abs().sum(-1)
        return sums.new_zeros((self.n, self.width)).index_add(0, self.row, sums).max()

    def smooth_row_bound(self):
        """Differentiable upper bound, including implicit zero matrix entries.

        sqrt(x^2+eps^2) majorizes |x|. Account for absent zeros analytically,
        so adding a zero graph edge does not change the represented bound.
        Log-sum-exp majorizes max without ambiguous tied-row subgradients.
        These conservative smooth bounds vanish to the exact bound as their
        stated smoothing constants tend to zero; no dense array is allocated.
        """
        eps = 1e-5
        temperature = 1e-3
        sums = (torch.sqrt(self.values.square() + eps**2) - eps).sum(-1)
        rows = sums.new_zeros((self.n, self.width)).index_add(0, self.row, sums)
        rows = rows + self.n * self.width * eps
        return temperature * torch.logsumexp(rows.flatten() / temperature, 0)

    def query(self, row, col):
        if not len(self.row):
            return self.values.new_zeros((len(row), self.width, self.width)) + self.values.sum() * 0
        keys = self.row * self.n + self.col
        where = torch.searchsorted(keys, row * self.n + col)
        safe = where.clamp_max(len(keys) - 1)
        valid = (where < len(keys)) & (keys[safe] == row * self.n + col)
        return self.values[safe] * valid[:, None, None]

    def dense_for_validation(self, max_atoms=32):
        if self.n > max_atoms:
            raise CapacityError("Dense validation restricted to small systems")
        a = self.width
        out = self.values.new_zeros((self.n * self.n, a, a)).index_add(
            0, self.row * self.n + self.col, self.values
        )
        return out.reshape(self.n, self.n, a, a).permute(0, 2, 1, 3).reshape(self.n * a, self.n * a)


def metric_filter(overlap: BlockMatrix, active: Tensor, degree: int):
    """A finite local polynomial under-approximation of the inverse square root.

    For S>=0, alpha>=lambda_max(S), X=I-S/alpha, the positive binomial
    partial sum T_m satisfies 0<=sqrt(S)T_m<=I. Hence T_m D T_m has physical
    Pauli bounds for any 0<=D<=I. No fill is dropped. The metric defect is
    reported, not repaired with jitter or hidden global factorisation.
    """
    diagonal = overlap.diagonal_values().diagonal(dim1=-2, dim2=-1)
    normalizer = torch.where(active, diagonal.clamp_min(1e-30).rsqrt(), torch.zeros_like(diagonal))
    d = BlockMatrix.diagonal(torch.diag_embed(normalizer), overlap.budget)
    normalized = d.matmul(overlap).matmul(d)
    ident = BlockMatrix.diagonal(torch.diag_embed(torch.ones_like(normalizer)), overlap.budget)
    alpha = normalized.smooth_row_bound().clamp_min(1.0)
    x = ident.add(normalized, -1 / alpha)
    term, total = ident, ident
    coefficient = 1.0
    for k in range(1, degree + 1):
        term = term.matmul(x)
        coefficient *= (2 * k - 1) / (2 * k)
        total = total.add(term, coefficient)
    t = d.matmul(total.scale(alpha.rsqrt()))
    represented = t.transpose().matmul(overlap).matmul(t)
    target = BlockMatrix.diagonal(torch.diag_embed(active.to(normalizer.dtype)), overlap.budget)
    defect = target.add(represented, -1.0).row_bound()
    return t, defect


def constrained_kernel(
    raw: BlockMatrix,
    overlap: BlockMatrix,
    active: Tensor,
    spin_counts: Tensor,
    *,
    metric_degree=16,
    fermi_degree=4,
):
    """Local positive Bernstein matrix function plus a scalar number solve.

    M=(I+L/beta)/2 has spectrum in [0,1]. Bernstein basis operators are PSD
    and sum to I. Sigmoid coefficients lie in [0,1], so both spin kernels
    satisfy Pauli bounds after the contractive metric filter. Chemical
    potentials use scalar reductions only. Polynomial halo/fill is explicit.
    """
    t, defect = metric_filter(overlap, active, metric_degree)
    ident = BlockMatrix.diagonal(
        torch.eye(raw.width, dtype=raw.values.dtype, device=raw.values.device).expand(
            raw.n, -1, -1
        ),
        raw.budget,
    )
    symmetric = raw.add(raw.transpose()).scale(0.5)
    # A fixed positive floor makes the map differentiable at zero generator.
    beta = 1 + symmetric.smooth_row_bound()
    m = ident.add(symmetric, 1 / beta).scale(0.5)
    complement = ident.add(m, -1.0)
    import math

    left, right = [ident], [ident]
    for _ in range(fermi_degree):
        left.append(left[-1].matmul(m))
        right.append(right[-1].matmul(complement))
    basis = [
        t.matmul(left[k].matmul(right[fermi_degree - k]))
        .matmul(t.transpose())
        .scale(math.comb(fermi_degree, k))
        for k in range(fermi_degree + 1)
    ]
    weights = torch.stack([b.trace_product(overlap) for b in basis])
    capacity = weights.sum()
    if bool((spin_counts > capacity.detach() + 1e-9).any()):
        raise StateError(
            "Local metric representation has insufficient electron capacity; increase metric degree/basis"
        )
    knots = torch.linspace(-6, 6, fermi_degree + 1, device=weights.device, dtype=weights.dtype)
    potentials = []
    for target in spin_counts:
        if float(target) == 0:
            potentials.append(weights.new_tensor(-100.0))
            continue
        if abs(float(target - capacity.detach())) < 1e-10:
            potentials.append(weights.new_tensor(100.0))
            continue
        with torch.no_grad():
            lo, hi = weights.new_tensor(-80.0), weights.new_tensor(80.0)
            for _ in range(60):
                mid = (lo + hi) / 2
                if float((torch.sigmoid(knots + mid) * weights).sum()) > float(target):
                    hi = mid
                else:
                    lo = mid
            root = (lo + hi) / 2
        # Newton refinements retain implicit first/second/higher derivatives.
        for _ in range(3):
            occ = torch.sigmoid(knots + root)
            root = root - ((occ * weights).sum() - target) / (
                occ * (1 - occ) * weights
            ).sum().clamp_min(1e-20)
        potentials.append(root)
    kernels = []
    for spin, mu in enumerate(potentials):
        if float(spin_counts[spin]) == 0:
            coefficients = torch.zeros_like(knots)
        elif abs(float(spin_counts[spin] - capacity.detach())) < 1e-10:
            coefficients = torch.ones_like(knots)
        else:
            coefficients = torch.sigmoid(knots + mu)
        result = basis[0].scale(coefficients[0])
        for k in range(1, len(basis)):
            result = result.add(basis[k], coefficients[k])
        kernels.append(result)
    return kernels, {
        "metric_defect_bound": float(defect.detach()),
        "metric_degree": metric_degree,
        "fermi_degree": fermi_degree,
        "spin_counts": [float(k.trace_product(overlap).detach()) for k in kernels],
        "density_blocks": len(kernels[0].row),
        "global_dense_algebra": False,
        "pauli_construction": "positive Bernstein function and contractive metric polynomial",
    }
