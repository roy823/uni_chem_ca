from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..errors import CapacityError, StateError
from ..grid import Grid


@dataclass(frozen=True)
class PairList:
    first: Tensor
    second: Tensor
    displacement: Tensor
    distance: Tensor
    candidate_count: int


class CellList:
    """Periodic linked-cell search with bounded candidate batches.

    Cell width >= cutoff, so a 3x3x3 neighbourhood is sufficient. Degenerate
    periodic bins are deduplicated. Pairs are UNIQUE unordered atom pairs.
    Geometry is differentiable; integer membership is not. No stale cache is
    reused when the world moves. Expected O(N + E) at bounded density, not a
    worst-case linear guarantee (a crowded bin still contains O(N^2) pairs).
    """

    def __init__(self, cutoff: float, *, batch_atoms: int = 1024, max_candidates: int = 8_000_000):
        if not (0 < cutoff < math.inf) or batch_atoms < 1 or max_candidates < 1:
            raise StateError("Invalid linked-cell parameters")
        self.cutoff, self.batch_atoms, self.max_candidates = cutoff, batch_atoms, max_candidates

    def __call__(self, grid: Grid, positions: Tensor) -> PairList:
        if positions.ndim != 2 or positions.shape[1] != 3 or not torch.isfinite(positions).all():
            raise StateError("CellList requires finite (N,3) positions")
        n = positions.shape[0]
        device = positions.device
        counts = tuple(max(1, int(length / self.cutoff)) for length in grid.lengths)
        nc = torch.tensor(counts, dtype=torch.int64, device=device)
        widths = positions.new_tensor(grid.lengths) / nc
        bins = torch.floor(
            (grid.wrap(positions) - positions.new_tensor(grid.origin)) / widths
        ).long()
        bins = torch.minimum(bins, nc - 1)
        flatten = lambda b: (b[..., 0] * counts[1] + b[..., 1]) * counts[2] + b[..., 2]
        key = flatten(bins)
        order = torch.argsort(key, stable=True)
        sorted_key = key[order]
        # Modulo-deduplicated offsets handle one- and two-bin periodic axes.
        offset_axes = [sorted({x % c for x in (-1, 0, 1)}) for c in counts]
        offsets = torch.tensor(list(itertools.product(*offset_axes)), device=device)
        first, second = [], []
        candidate_count = 0
        for lo in range(0, n, self.batch_atoms):
            atom = torch.arange(lo, min(lo + self.batch_atoms, n), device=device)
            target_bins = torch.remainder(bins[atom, None, :] + offsets[None, :, :], nc)
            targets = flatten(target_bins).flatten()
            lower = torch.searchsorted(sorted_key, targets)
            upper = torch.searchsorted(sorted_key, targets, right=True)
            sizes = upper - lower
            count = int(sizes.sum())
            candidate_count += count
            if candidate_count > self.max_candidates:
                raise CapacityError(
                    "Linked-cell candidate budget exceeded; high local density or cutoff"
                )
            if count == 0:
                continue
            row = torch.repeat_interleave(torch.arange(targets.numel(), device=device), sizes)
            local = torch.arange(count, device=device) - torch.repeat_interleave(
                sizes.cumsum(0) - sizes, sizes
            )
            j = order[lower[row] + local]
            i = atom[torch.div(row, len(offsets), rounding_mode="floor")]
            unique = i < j
            i, j = i[unique], j[unique]
            d = grid.minimum_image(positions[j] - positions[i])
            keep = d.square().sum(-1) < self.cutoff**2
            first.append(i[keep])
            second.append(j[keep])
        empty = torch.empty(0, dtype=torch.int64, device=device)
        i = torch.cat(first) if first else empty
        j = torch.cat(second) if second else empty
        # Canonical order makes diagnostic outputs stable across batch sizes.
        permutation = torch.argsort(i * max(n, 1) + j)
        i, j = i[permutation], j[permutation]
        d = grid.minimum_image(positions[j] - positions[i])
        return PairList(i, j, d, (d.square().sum(-1) + 1e-18).sqrt(), candidate_count)
