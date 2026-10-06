from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize

from ..errors import StateError
from ..quantum.patches import refresh_patch_graph
from ..state import World


@dataclass(frozen=True)
class RelaxationReport:
    energies: tuple[float, ...]
    iterations: int
    residual_norm: float
    converged: bool
    stalled: bool
    algorithm: str = "local_kernel"
    residual_kind: str = "projected_kernel_coordinate_gradient"


@dataclass(frozen=True)
class RelaxationResult:
    world: World
    report: RelaxationReport


class LocalKernelRelaxer:
    """Synchronous variational kernel proposals with scalar global acceptance.

    Quantum constraints are enforced by the local backend map on every
    proposal. No global coefficient normalization/Cholesky/Cayley is used.
    Convergence means stationarity in the stated finite-halo representation.
    """

    def __init__(self, *, max_iterations=60, tolerance=1e-6, max_backtracks=20):
        if max_iterations < 0 or tolerance <= 0 or max_backtracks < 1:
            raise StateError("Invalid local relaxation parameters")
        self.max_iterations = max_iterations
        self.tolerance, self.max_backtracks = tolerance, max_backtracks

    def run(self, world, evaluator):
        current = refresh_patch_graph(world.clone())
        like = current.electrons.kernel.values
        shape = like.shape

        def objective(x):
            values = (
                torch.as_tensor(x.reshape(shape), device=like.device, dtype=like.dtype)
                .clone()
                .requires_grad_(True)
            )
            energy = evaluator(current.with_kernel_values(values)).energy + 0 * values.sum()
            gradient = torch.autograd.grad(energy, values)[0]
            return float(energy.detach()), gradient.detach().cpu().numpy().astype(
                np.float64
            ).reshape(-1)

        x0 = like.detach().cpu().numpy().astype(np.float64).reshape(-1)
        initial, _ = objective(x0)
        history = [initial]
        iterations = 0
        stalled = False
        if self.max_iterations:

            def accepted(x):
                history.append(objective(x)[0])

            result = minimize(
                objective,
                x0,
                jac=True,
                method="L-BFGS-B",
                bounds=[(-8.0, 8.0)] * len(x0),
                callback=accepted,
                options={
                    "maxiter": self.max_iterations,
                    "maxls": self.max_backtracks,
                    "maxcor": 10,
                    "gtol": self.tolerance / max(1.0, len(x0) ** 0.5),
                    "ftol": 1e-15,
                },
            )
            x = result.x
            iterations = int(result.nit)
            stalled = iterations < self.max_iterations and not result.success
        else:
            x = x0
        final, gradient = objective(x)
        residual = float(np.linalg.norm(x - np.clip(x - gradient, -8.0, 8.0)))
        values = torch.as_tensor(x.reshape(shape), device=like.device, dtype=like.dtype)
        output = current.with_kernel_values(values).clone()
        if len(history) == 1 or abs(history[-1] - final) > 1e-14:
            history.append(final)
        return RelaxationResult(
            output,
            RelaxationReport(
                tuple(history), iterations, residual, residual <= self.tolerance, stalled
            ),
        )
