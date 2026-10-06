from __future__ import annotations

import math

import torch
from torch import Tensor

from ..errors import StateError


def vamp2_score(past: Tensor, future: Tensor, *, regularization: float = 1e-6) -> Tensor:
    """Centered, regularized VAMP-2 score for explicit time-lagged samples.

    Inputs (sample, feature) must represent paired trajectory frames/patches,
    not unpaired configurations. The constant mode is centered out, so a
    constant encoder has zero score. Maximize to retain slow predictive modes.
    Time/trajectory-separated validation is necessary to prevent leakage.
    This objective alone does not guarantee a calibrated reaction clock or
    prove that latent dynamics is Markovian. References are in docs/REFERENCES.
    """
    if (
        past.ndim != 2
        or future.shape != past.shape
        or len(past) < 3
        or not 0 < regularization < math.inf
    ):
        raise StateError("VAMP needs >=3 paired samples of the same feature dimension")
    if not torch.isfinite(past).all() or not torch.isfinite(future).all():
        raise StateError("Nonfinite time-lagged samples")
    x, y = past - past.mean(0), future - future.mean(0)
    norm = len(x) - 1
    ridge = regularization * torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    c00, c11, c01 = x.T @ x / norm + ridge, y.T @ y / norm + ridge, x.T @ y / norm
    l0, l1 = torch.linalg.cholesky(c00), torch.linalg.cholesky(c11)
    whitened = torch.linalg.solve_triangular(l0, c01, upper=False)
    whitened = torch.linalg.solve_triangular(l1, whitened.T, upper=False).T
    return whitened.square().sum()
