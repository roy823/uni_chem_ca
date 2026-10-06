from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..state import World
from .basis import BasisSamples
from .density import QuantumView
from .support import SupportMap


@dataclass(frozen=True)
class AtomicCloudMoments:
    electron_number: Tensor
    centroid_offset: Tensor
    rms_radius: Tensor
    covariance: Tensor
    partition_weights: Tensor


def cloud_moments(
    world: World, support: SupportMap, samples: BasisSamples, quantum: QuantumView
) -> AtomicCloudMoments:
    """A documented computational density partition, NOT a unique atomic observable.

    Positive reference-AO weights assign overlapping density fractions to atoms.
    Interatomic interference remains in total rho before partitioning. The sum
    of assigned populations equals the total number of electrons.
    """
    tiny = torch.finfo(samples.value.dtype).tiny
    denom = support.reduce_to_grid(samples.reference_per_edge)
    weights = samples.reference_per_edge / denom[support.voxel_index].clamp_min(tiny)
    mass = weights * quantum.density[support.voxel_index] * world.grid.dv
    population = support.reduce_to_atoms(mass)
    safe = population.clamp_min(tiny)
    first = support.reduce_to_atoms(mass[:, None] * support.displacement) / safe[:, None]
    delta = support.displacement - first[support.atom_index]
    covariance = (
        support.reduce_to_atoms(mass[:, None, None] * delta[:, :, None] * delta[:, None, :])
        / safe[:, None, None]
    )
    r2 = support.reduce_to_atoms(mass * support.displacement.square().sum(-1)) / safe
    return AtomicCloudMoments(population, first, r2.clamp_min(0).sqrt(), covariance, weights)


def electronic_invariants(world: World, q: QuantumView) -> dict[str, float]:
    return {
        "electron_integral": float((q.density.sum() * world.grid.dv).detach()),
        "trace_PS": float(q.density_matrix.trace_product(q.overlap).detach()),
        "metric_defect_bound": q.diagnostics["metric_defect_bound"],
        "minimum_density": float(q.density.min().detach()),
    }
