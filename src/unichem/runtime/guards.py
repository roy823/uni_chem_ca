from __future__ import annotations

import torch

from ..errors import StateError
from ..state import World


def validate_transition(before: World, after: World) -> None:
    """Fail-before-commit invariants; no silent repairs of an invalid proposal."""
    after.validate()
    if before.species != after.species or not torch.equal(before.atoms.ids, after.atoms.ids):
        raise StateError("An evolution step cannot create/delete/relabel atomic identities")
    if before.grid != after.grid or before.latent_grid != after.latent_grid:
        raise StateError("Regridding is an explicit operation, not an implicit time update")
    if before.electron_count != after.electron_count:
        raise StateError("Closed-world electron count changed")
    tol = 2e-5 if before.atoms.positions.dtype == torch.float32 else 1e-9
    if not torch.allclose(
        before.electrons.spin_counts, after.electrons.spin_counts, atol=tol, rtol=0
    ):
        raise StateError("The collinear-spin sector changed without a spin interaction model")
    if (
        before.latent.kind != after.latent.kind
        or before.latent.channels.shape != after.latent.channels.shape
    ):
        raise StateError("Latent-state semantics changed during an evolution step")
    if after.step != before.step + 1 or not after.time > before.time:
        raise StateError("World step/time must advance exactly once per commit")
