from __future__ import annotations

import math

from torch import Tensor, nn

from ..context import PreparedWorld
from ..errors import StateError
from ..fields.operators import face_gradient
from .encoder import CoarseFieldEncoder


class ConditionalGaussianCoupling(nn.Module):
    r"""One bidirectional scalar coupling, not arbitrary latent driving forces.

    Let d=z-encoder(R). V_aux=dV/2*(a||d||^2+b||grad d||^2), with constants a>0,b>=0.
    The Gaussian integral over ALL unconstrained z is independent of R: a
    translation z -> d has unit Jacobian and constant stiffness determinant.
    Thus, for a nuclear-only encoder and both baths at the same temperature,
    adding this auxiliary system preserves the ideal canonical nuclear/BO
    equilibrium marginal. Projected reaction kinetics are NOT preserved by this
    identity and require calibration. Finite timesteps/unfinished BO solves
    have additional bias. This is not a general GENERIC or quantum bath.

    Electronic-source encoders are allowed as an explicit research extension,
    but minimizing electrons before integrating z does not commute in general;
    the BO marginal-preservation claim then does NOT apply.
    """

    def __init__(
        self,
        encoder: CoarseFieldEncoder,
        *,
        stiffness: float = 0.1,
        gradient_stiffness: float = 0.02,
    ):
        super().__init__()
        if not (0 < stiffness < math.inf) or not (0 <= gradient_stiffness < math.inf):
            raise StateError("Constant Gaussian stiffnesses must be positive/nonnegative")
        self.encoder = encoder
        self.stiffness, self.gradient_stiffness = stiffness, gradient_stiffness

    @property
    def preserves_bo_marginal(self) -> bool:
        return not self.encoder.depends_on_electrons

    def forward(self, context: PreparedWorld) -> Tensor:
        w = context.world
        target = self.encoder(context)
        if target.shape != w.latent.channels.shape:
            raise StateError("Encoder output must match the complete latent field")
        deviation = w.latent.channels - target
        gradient = face_gradient(w.latent_grid, deviation)
        return (
            0.5
            * w.latent_grid.dv
            * (
                self.stiffness * deviation.square().sum()
                + self.gradient_stiffness * gradient.square().sum()
            )
        )
