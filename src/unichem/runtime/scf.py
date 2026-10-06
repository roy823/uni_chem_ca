"""Self-consistent physical Fock updates with sparse zero-temperature states."""

from dataclasses import dataclass

import torch

from ..errors import StateError
from ..quantum.purification import ground_projector, orthogonalize


class DIIS:
    """Pulay extrapolation: only a bounded history-sized scalar Gram solve."""

    def __init__(self, size=6):
        self.size = size
        self.focks = []
        self.errors = []

    def update(self, focks, errors):
        self.focks.append(focks)
        self.errors.append(errors)
        self.focks = self.focks[-self.size :]
        self.errors = self.errors[-self.size :]
        n = len(self.errors)
        if n < 2:
            return focks
        gram = torch.stack(
            [
                torch.stack([sum(a.inner(b) for a, b in zip(x, y)) for y in self.errors])
                for x in self.errors
            ]
        )
        amplitude = float(gram.abs().max())
        if amplitude < 1e-30:
            return focks
        gram = gram / amplitude
        system = gram.new_zeros((n + 1, n + 1))
        system[:n, :n] = gram
        system[n, :n] = system[:n, n] = -1
        rhs = gram.new_zeros(n + 1)
        rhs[n] = -1
        # Redundant error vectors are normal near convergence. The small
        # least-squares solve implements DIIS without arbitrary regularization.
        weights = torch.linalg.lstsq(system, rhs).solution[:n]
        result = []
        for spin in range(2):
            f = self.focks[0][spin].scale(weights[0])
            for index in range(1, n):
                f = f.add(self.focks[index][spin], weights[index])
            result.append(f.symmetric())
        return tuple(result)


@dataclass(frozen=True)
class SCFResult:
    densities: tuple
    focks: tuple
    energy: float
    iterations: int
    commutator_norm: float
    density_change: float
    energy_change: float
    electron_errors: tuple
    purity_errors: tuple
    metric_defect: float
    spectral_consistency: float
    compression_bound_sum: float
    converged: bool


class SCFSolver:
    def __init__(self, *, max_iterations=80, tolerance=1e-8):
        if max_iterations < 1 or tolerance <= 0:
            raise StateError("Invalid SCF convergence settings")
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    @torch.no_grad()
    def solve(self, operators, counts, initial=None):
        metric = orthogonalize(operators.overlap, operators.active)
        t = metric.transform
        s = operators.overlap
        compression_sum = metric.compression_bound_sum

        def project(fs):
            nonlocal compression_sum
            result = tuple(ground_projector(f, metric, n) for f, n in zip(fs, counts))
            compression_sum += sum(p.compression_bound_sum for p in result)
            return result

        if initial is None:
            densities = tuple(p.density for p in project((operators.core, operators.core)))
        else:
            densities = initial
        diis = DIIS()
        previous_energy = None
        previous_densities = None
        for iteration in range(1, self.max_iterations + 1):
            energy, focks = operators.energy_focks(densities)
            errors = tuple(
                t.transpose().matmul(f.matmul(p).matmul(s).add(s.matmul(p).matmul(f), -1)).matmul(t)
                for f, p in zip(focks, densities)
            )
            residual = float(torch.sqrt(sum(e.inner(e) for e in errors)))
            de = abs(energy - previous_energy) if previous_energy is not None else float("inf")
            change = float("inf")
            if previous_densities is not None:
                differences = tuple(
                    t.transpose().matmul(s).matmul(p.add(q, -1)).matmul(s).matmul(t)
                    for p, q in zip(densities, previous_densities)
                )
                change = float(torch.sqrt(sum(d.inner(d) for d in differences)))
            electron_errors = tuple(
                abs(float(p.trace_product(s)) - float(n)) for p, n in zip(densities, counts)
            )
            orthogonal_densities = tuple(
                t.transpose().matmul(s).matmul(p).matmul(s).matmul(t) for p in densities
            )
            purity_errors = tuple(
                float(d.matmul(d).add(d, -1).norm()) for d in orthogonal_densities
            )
            converged = (
                residual < self.tolerance
                and de < 1e-10
                and change < self.tolerance
                and max(electron_errors) < 1e-8
                and max(purity_errors) < 1e-10
            )
            consistency = float("inf")
            if (
                residual < self.tolerance
                and max(electron_errors) < 1e-8
                and max(purity_errors) < 1e-10
            ):
                ground = tuple(p.density for p in project(focks))
                differences = tuple(
                    t.transpose().matmul(s).matmul(p.add(q, -1)).matmul(s).matmul(t)
                    for p, q in zip(densities, ground)
                )
                consistency = float(torch.sqrt(sum(d.inner(d) for d in differences)))
                converged = consistency < self.tolerance and (converged or iteration == 1)
                if converged and iteration == 1:
                    de, change = 0.0, consistency
            result = SCFResult(
                densities,
                focks,
                energy,
                iteration,
                residual,
                change,
                de,
                electron_errors,
                purity_errors,
                metric.defect,
                consistency,
                compression_sum,
                converged,
            )
            if converged:
                return result
            mixed = diis.update(focks, errors)
            previous_energy, previous_densities = energy, densities
            densities = tuple(p.density for p in project(mixed))
        raise StateError(
            f"SCF not converged after {iteration} iterations: commutator={residual:.3e}, dP={change:.3e}, dE={de:.3e}"
        )
