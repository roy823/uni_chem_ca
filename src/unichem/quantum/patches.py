"""Dynamic patch adjacency and smooth coupling on a shared old snapshot."""

from dataclasses import replace

import torch

from ..spatial.neighbors import CellList
from .blocks import BlockMatrix


def refresh_patch_graph(world):
    old = world.electrons.kernel
    pair = CellList(world.electrons.cutoff)(world.grid, world.atoms.positions)
    ids = torch.arange(world.n_atoms, device=pair.first.device)
    row = torch.cat((ids, pair.first, pair.second))
    col = torch.cat((ids, pair.second, pair.first))
    values = old.query(row, col)
    kernel = BlockMatrix.assemble(row, col, values, world.n_atoms, old.budget)
    return replace(world, electrons=replace(world.electrons, kernel=kernel))


def effective_kernel(world):
    k = world.electrons.kernel
    delta = world.grid.minimum_image(world.atoms.positions[k.row] - world.atoms.positions[k.col])
    distance = torch.sqrt(delta.square().sum(-1) + 1e-18)
    x = (distance / world.electrons.cutoff).clamp(0, 1)
    envelope = 1 - 10 * x**3 + 15 * x**4 - 6 * x**5
    envelope = torch.where(k.row == k.col, torch.ones_like(envelope), envelope)
    return replace(k, values=k.values * envelope[:, None, None])
