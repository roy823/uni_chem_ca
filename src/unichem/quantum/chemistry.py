from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ..context import PreparedWorld


@dataclass(frozen=True)
class ChemicalView:
    atomic_ids: Tensor
    assigned_valence_charge: Tensor
    pair_first: Tensor
    pair_second: Tensor
    mayer_like: Tensor
    spin_counts: Tensor


def chemical_observables(context: PreparedWorld) -> ChemicalView:
    """Read-only electronic diagnostics. NEVER feeds bond labels to the engine.

    Atomic charge uses the documented positive density partition, not a unique
    observable. The spin-summed Mayer-like AO indicator sum_AB (PS)_mu,nu
    (PS)_nu,mu is basis dependent and is not a calibrated reaction detector.
    In open shells it is not the full unrestricted Mayer prescription. A large
    value need not mean energetic bonding (e.g. antibonding occupation).
    """
    w, q = context.world, context.quantum
    ps = q.density_matrix.matmul(q.overlap)
    keep = ps.row < ps.col
    i, j = ps.row[keep], ps.col[keep]
    reverse = ps.query(j, i).transpose(-1, -2)
    indicators = (ps.values[keep] * reverse).sum((-1, -2))
    return ChemicalView(
        w.atoms.ids.detach().clone(),
        (context.layout.core_charge - context.clouds.electron_number).detach(),
        w.atoms.ids[i].detach().clone(),
        w.atoms.ids[j].detach().clone(),
        indicators.detach(),
        w.electrons.spin_counts.detach().clone(),
    )
