from __future__ import annotations

import torch

from .errors import StateError
from .grid import Grid
from .quantum.blocks import BlockMatrix
from .spatial.neighbors import CellList
from .species import DEMO_SPECIES, Species
from .state import AtomState, ElectronState, LatentState, World


def make_world(
    symbols: list[str],
    positions,
    *,
    grid: Grid,
    net_charge: int = 0,
    spin_excess: int | None = None,
    latent_channels: int = 4,
    dtype=torch.float64,
    device="cpu",
    external_field=(0.0, 0.0, 0.0),
    registry: dict[str, Species] | None = None,
    kernel_cutoff: float = 5.0,
    latent_stride: int = 1,
    latent_kind: str = "filter",
) -> World:
    """Build a neutral/charged test world without assigning any chemical bonds.

    The local kernel seed is a numerical initializer, not a ground state.
    Spin excess N_up-N_down is fixed by this factory.
    """
    if type(net_charge) is not int or (spin_excess is not None and type(spin_excess) is not int):
        raise StateError("Factory charge/spin counts must be integers")
    if type(latent_channels) is not int or latent_channels < 1:
        raise StateError("At least one latent channel is required")
    registry = DEMO_SPECIES if registry is None else registry
    try:
        species = tuple(registry[s] for s in symbols)
    except KeyError as exc:
        raise StateError(f"Unregistered species: {exc.args[0]}") from exc
    ne_float = sum(s.core_charge for s in species) - net_charge
    ne = round(ne_float)
    if abs(ne - ne_float) > 1e-10 or ne < 1:
        raise StateError("Factory requires a positive integer valence electron count")
    spin_excess = ne % 2 if spin_excess is None else spin_excess
    if abs(spin_excess) > ne or (ne + spin_excess) % 2:
        raise StateError("Electron count and spin excess must have compatible parity")
    up, down = (ne + spin_excess) // 2, (ne - spin_excess) // 2
    b = sum(len(s.orbitals) for s in species)
    if max(up, down) > b:
        raise StateError("Not enough spatial orbitals for this electron/spin count")
    p = torch.as_tensor(positions, dtype=dtype, device=device)
    n = len(species)
    if p.shape != (n, 3):
        raise StateError("Positions must have shape (number_of_atoms, 3)")
    if (
        type(latent_stride) is not int
        or latent_stride < 1
        or any(n % latent_stride for n in grid.shape)
    ):
        raise StateError("latent_stride must be a positive divisor of each grid dimension")
    latent_grid = Grid(
        tuple(n // latent_stride for n in grid.shape),
        tuple(h * latent_stride for h in grid.spacing),
        grid.origin,
    )
    latent_shape = (*latent_grid.shape, latent_channels)
    pairs = CellList(kernel_cutoff)(grid, p)
    ids = torch.arange(n, device=device)
    row = torch.cat((ids, pairs.first, pairs.second))
    col = torch.cat((ids, pairs.second, pairs.first))
    width = max(len(s.orbitals) for s in species)
    values = torch.zeros((len(row), width, width), dtype=dtype, device=device)
    for i, s in enumerate(species):
        for a, orbital in enumerate(s.orbitals):
            values[i, a, a] = -1.0 / orbital.width
    kernel = BlockMatrix.assemble(row, col, values, n)
    world = World(
        grid,
        species,
        AtomState(
            torch.arange(n, device=device),
            grid.wrap(p),
            torch.zeros_like(p),
            torch.zeros(n, dtype=dtype, device=device),
        ),
        ElectronState(kernel, torch.tensor([up, down], dtype=dtype, device=device), kernel_cutoff),
        LatentState(
            torch.zeros(latent_shape, dtype=dtype, device=device),
            torch.zeros(latent_shape, dtype=dtype, device=device),
            latent_grid,
            latent_kind,
        ),
        float(ne),
        external_field=tuple(float(x) for x in external_field),
    )
    world.validate()
    return world
