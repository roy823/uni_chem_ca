"""Atom-block sparse algebra with differentiable values and bounded fill.

No operation allocates a system-sized dense matrix. CPU products use SciPy's
compiled CSR kernel, retaining a symbolic block graph and exact adjoint products.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import bsr_matrix, csr_matrix
from torch import Tensor

from ..errors import CapacityError, StateError


class _SparseProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, av, bv, ar, ac, br, bc, n, budget):
        if av.device.type != "cpu" or bv.device.type != "cpu":
            raise StateError("The physical block product requires CPU tensors")
        width = av.shape[-1]

        def layout(row):
            return np.r_[0, np.cumsum(np.bincount(row.numpy(), minlength=n))]

        ai, bi = layout(ar), layout(br)
        ga = csr_matrix((np.ones(len(ar), dtype=bool), ac.numpy(), ai), shape=(n, n))
        gb = csr_matrix((np.ones(len(br), dtype=bool), bc.numpy(), bi), shape=(n, n))
        graph = ga @ gb
        graph.sort_indices()
        size = len(graph.indices)
        if size > budget:
            raise CapacityError(f"Sparse patch fill {size} exceeds {budget}; no truncation applied")
        if size * width**2 * av.element_size() > 512 * 1024**2:
            raise CapacityError("Sparse multiplication output memory budget exceeded")
        aa = bsr_matrix((av.detach().numpy(), ac.numpy(), ai), shape=(n * width, n * width))
        bb = bsr_matrix((bv.detach().numpy(), bc.numpy(), bi), shape=(n * width, n * width))
        aa, bb = aa.tocsr(), bb.tocsr()
        aa.eliminate_zeros()
        bb.eliminate_zeros()
        product = (aa @ bb).tocoo()
        row = np.repeat(np.arange(n), np.diff(graph.indptr))
        col = graph.indices.astype(np.int64)
        keys = row * n + col
        query = (product.row // width) * n + product.col // width
        where = np.searchsorted(keys, query)
        values = np.zeros((size, width, width), dtype=av.detach().numpy().dtype)
        if len(product.data):
            values[where, product.row % width, product.col % width] = product.data
        r, c, v = torch.from_numpy(row), torch.from_numpy(col), torch.from_numpy(values)
        ctx.save_for_backward(av, bv, ar, ac, br, bc, r, c)
        ctx.n, ctx.budget = n, budget
        ctx.mark_non_differentiable(r, c)
        return r, c, v

    @staticmethod
    def backward(ctx, _gr, _gc, gv):
        av, bv, ar, ac, br, bc, row, col = ctx.saved_tensors
        a = BlockMatrix(ar, ac, av, ctx.n, ctx.budget)
        b = BlockMatrix(br, bc, bv, ctx.n, ctx.budget)
        g = BlockMatrix(row, col, gv, ctx.n, ctx.budget)
        da = g.matmul(b.transpose()).query(ar, ac)
        db = a.transpose().matmul(g).query(br, bc)
        return da, db, None, None, None, None, None, None


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
        row, col, values = _SparseProduct.apply(
            self.values,
            other.values,
            self.row,
            self.col,
            other.row,
            other.col,
            self.n,
            min(self.budget, other.budget),
        )
        return BlockMatrix(row, col, values, self.n, min(self.budget, other.budget))

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

    def symmetric(self):
        return self.add(self.transpose()).scale(0.5)

    def trace(self):
        return self.diagonal_values().diagonal(dim1=-2, dim2=-1).sum()

    def norm(self):
        return self.values.norm()

    def inner(self, other):
        return (self.values * other.query(self.row, self.col)).sum()

    def spectral_bounds(self, active):
        diagonal = self.diagonal_values().diagonal(dim1=-2, dim2=-1)
        sums = self.values.abs().sum(-1)
        rows = sums.new_zeros((self.n, self.width)).index_add(0, self.row, sums)
        radius = (rows - diagonal.abs()).clamp_min(0)
        return (diagonal - radius)[active].min(), (diagonal + radius)[active].max()

    def compress(self, tolerance):
        """Symmetric block screening with a measured global row-norm bound.

        Diagonal blocks are retained. This is an explicit numerical operation,
        not an implicit truncation in matrix multiplication or budget handling.
        """
        if tolerance < 0:
            raise StateError("Nonnegative compression tolerance required")
        degree = max(1, int(torch.bincount(self.row, minlength=self.n).max()))
        row_size = self.values.abs().sum(-1).amax(-1)
        col_size = self.values.abs().sum(-2).amax(-1)
        keep = (torch.maximum(row_size, col_size) > tolerance / degree) | (self.row == self.col)
        discarded = self.values[~keep].abs().sum(-1)
        bound = (
            discarded.new_zeros((self.n, self.width)).index_add(0, self.row[~keep], discarded).max()
        )
        result = BlockMatrix(self.row[keep], self.col[keep], self.values[keep], self.n, self.budget)
        return result, float(bound)

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
