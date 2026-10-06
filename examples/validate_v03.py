"""Measured local-kernel, quadrature and dynamics validation; no chemical claims."""

import json
import time
from pathlib import Path

import torch

from unichem import Evaluator, Grid, LocalKernelRelaxer, LocalQuantumBackend, make_world
from unichem.fields.isolated import IsolatedPoisson
from unichem.quantum.blocks import BlockMatrix, constrained_kernel
from unichem.quantum.observables import electronic_invariants
from unichem.runtime.potential import integrate_potential


def main():
    torch.set_num_threads(1)
    report = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "scope": "numerical properties of the stated approximation, not chemical validation",
    }
    water = make_world(
        ["O", "H", "H"],
        [[8.03, 8.07, 8.11], [9.63, 8.07, 8.11], [7.63, 9.57, 8.11]],
        grid=Grid((28,) * 3, (0.6,) * 3),
    )
    evaluation = Evaluator().energy_and_forces(water)
    report["water_invariants"] = electronic_invariants(water, evaluation.evaluation.context.quantum)
    report["water_backend"] = evaluation.evaluation.context.quantum.diagnostics
    defects = []
    for degree in [8, 16, 32, 64, 128]:
        q = Evaluator(backend=LocalQuantumBackend(metric_degree=degree))(water).context.quantum
        defects.append(
            {
                "degree": degree,
                "metric_defect_bound": q.diagnostics["metric_defect_bound"],
                "electron_number": float(q.density.sum() * water.grid.dv),
            }
        )
    report["metric_degree_convergence"] = defects
    refinement = []
    for n, h in [(20, 0.6), (30, 0.4), (40, 0.3)]:
        w = make_world(["H"], [[6.137, 6.234, 6.351]], grid=Grid((n,) * 3, (h,) * 3))
        result = Evaluator(poisson=IsolatedPoisson()).energy_and_forces(w)
        refinement.append(
            {
                "spacing_bohr": h,
                "energy_hartree": float(result.evaluation.energy.detach()),
                "rms_radius_bohr": float(result.evaluation.context.clouds.rms_radius[0].detach()),
                "spurious_force": float(result.forces.abs().max()),
                "electron_number": float(
                    result.evaluation.context.quantum.density.sum().detach() * w.grid.dv
                ),
            }
        )
    report["single_atom_refinement"] = refinement
    eps = 1e-5
    rp = water.atoms.positions.clone()
    rm = rp.clone()
    rp[0, 0] += eps
    rm[0, 0] -= eps
    evaluator = Evaluator()
    fd = -(
        evaluator(water.with_positions(rp)).energy - evaluator(water.with_positions(rm)).energy
    ) / (2 * eps)
    report["force_fd_error"] = abs(float(fd.detach()) - float(evaluation.forces[0, 0]))
    chain = []
    for n in [32, 128, 512, 2048]:
        row = torch.cat((torch.arange(n), torch.arange(n - 1), torch.arange(1, n)))
        col = torch.cat((torch.arange(n), torch.arange(1, n), torch.arange(n - 1)))
        values = (
            torch.cat((torch.ones(n), torch.full((2 * (n - 1),), 0.08))).double().reshape(-1, 1, 1)
        )
        s = BlockMatrix.assemble(row, col, values, n)
        raw = BlockMatrix.assemble(row, col, torch.zeros_like(values), n)
        started = time.perf_counter()
        kernels, diag = constrained_kernel(
            raw,
            s,
            torch.ones(n, 1, dtype=torch.bool),
            torch.tensor([0.3 * n, 0.3 * n], dtype=torch.float64),
            metric_degree=4,
            fermi_degree=2,
        )
        stored = sum(
            x.numel() * x.element_size()
            for x in [kernels[0].row, kernels[0].col, kernels[0].values]
        )
        chain.append(
            {
                "atoms": n,
                "density_blocks": len(kernels[0].row),
                "stored_kernel_bytes": stored,
                "seconds": time.perf_counter() - started,
                "metric_defect_bound": diag["metric_defect_bound"],
            }
        )
    report["controlled_banded_operator_scaling"] = chain
    report["scaling_limits"] = (
        "Fixed degrees and a one-dimensional bounded-degree graph; not a liquid-water or distributed benchmark"
    )
    nve = []
    p = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)

    def harmonic(r):
        return 0.5 * r.square().sum(), -r, {}

    for steps in [16, 32, 64]:
        frames = list(
            integrate_potential(
                p,
                torch.zeros_like(p),
                torch.ones(2, dtype=p.dtype),
                harmonic,
                atom_ids=torch.arange(len(p)),
                steps=steps,
                dt=2 / steps,
            )
        )
        nve.append(
            {
                "dt_au": 2 / steps,
                "fixed_duration_au": 2.0,
                "maximum_energy_error": max(abs(f.balance_residual) for f in frames),
            }
        )
    report["integrator_fixed_duration_convergence"] = nve
    w = make_world(
        ["H", "H"], [[4.63, 5.27, 5.11], [6.23, 5.27, 5.11]], grid=Grid((18,) * 3, (0.6,) * 3)
    )
    result = LocalKernelRelaxer(max_iterations=200).run(w, Evaluator())
    report["variational_relaxation"] = {
        "iterations": result.report.iterations,
        "converged": result.report.converged,
        "residual": result.report.residual_norm,
        "initial_energy": result.report.energies[0],
        "final_energy": result.report.energies[-1],
        "scope": "stationarity in the finite local kernel parameterization",
    }
    out = Path("outputs/v03_validation")
    out.mkdir(parents=True, exist_ok=True)
    (out / "numerics.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
