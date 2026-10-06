"""Small-system kernel-coordinate stationarity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..quantum.blocks import BlockMatrix


@dataclass(frozen=True)
class KernelDiagnostic:
    gradient: BlockMatrix
    norm: Tensor
    energy: Tensor


def kernel_stationarity(world, evaluator):
    raw = world.electrons.kernel.values.detach().clone().requires_grad_(True)
    energy = evaluator(world.with_kernel_values(raw)).energy + raw.sum() * 0
    gradient = torch.autograd.grad(energy, raw)[0]
    graph = world.electrons.kernel
    return KernelDiagnostic(
        BlockMatrix(graph.row, graph.col, gradient.detach(), graph.n, graph.budget),
        gradient.detach().norm(),
        energy.detach(),
    )
