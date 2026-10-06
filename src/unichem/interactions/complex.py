from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..context import PreparedWorld
from ..errors import StateError
from ..spatial.neighbors import CellList, PairList


@dataclass(frozen=True)
class InteractionComplex:
    """Two incidence structures, one snapshot: atom<->volume and atom<->atom.

    Contacts mean intersecting computational cloud supports, NOT bonds. Overlap
    is many-to-many; no voxel has an exclusive atom owner. This object holds
    references to a prepared snapshot and must not survive a geometry change.
    Only sparse patch edges are exposed; no dense world-sized AO mask.
    """

    context: PreparedWorld
    contacts: PairList

    @classmethod
    def build(cls, context: PreparedWorld) -> InteractionComplex:
        w, radii = context.world, context.support.radii
        broad = CellList(float(2 * radii.detach().max()))(w.grid, w.atoms.positions)
        keep = broad.distance < radii[broad.first] + radii[broad.second]
        contacts = PairList(
            broad.first[keep],
            broad.second[keep],
            broad.displacement[keep],
            broad.distance[keep],
            broad.candidate_count,
        )
        return cls(context, contacts)

    def patch_edges(self) -> tuple[Tensor, Tensor]:
        """Symmetric atom-block incidence, with no world-sized AO mask."""
        ids = torch.arange(self.context.world.n_atoms, device=self.contacts.first.device)
        return (
            torch.cat((ids, self.contacts.first, self.contacts.second)),
            torch.cat((ids, self.contacts.second, self.contacts.first)),
        )

    def exchange(self, face_values: Tensor) -> Tensor:
        """Scatter one signed exchange per contact; every edge is evaluated once.

        Positive means first -> second. The site sum is zero to roundoff for
        scalars/vectors/tensors. This is an algebraic primitive, not a claimed
        electron-transfer rate or an extra force added to the quantum model.
        """
        if face_values.shape[0] != self.contacts.first.numel():
            raise StateError("One signed exchange is required per contact")
        result = face_values.new_zeros((self.context.world.n_atoms, *face_values.shape[1:]))
        return result.index_add(0, self.contacts.first, -face_values).index_add(
            0, self.contacts.second, face_values
        )
