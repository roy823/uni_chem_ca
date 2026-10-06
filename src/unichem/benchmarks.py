"""Pre-registered numerical controls and reference screens for emergence work."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_world, save_world
from .datasets import BOHR_ANGSTROM, HARTREE_EV, load_reference_groups
from .evaluate import Evaluator
from .experiment import load_quantum_potential
from .factory import make_world
from .grid import Grid
from .models.energy import ZeroEnergy
from .models.latent import ConservativeLatentCA
from .quantum.blocks import BlockMatrix
from .reference_training import frame_world
from .runtime.electrons import LocalKernelRelaxer
from .runtime.engine import SynchronousEngine
from .runtime.potential import integrate_potential


def _reorder(world, order):
    indices = torch.tensor(order, device=world.atoms.ids.device)
    inverse = torch.empty_like(indices)
    inverse[indices] = torch.arange(len(indices), device=indices.device)
    a, k = world.atoms, world.electrons.kernel
    kernel = BlockMatrix.assemble(inverse[k.row], inverse[k.col], k.values, k.n, k.budget)
    return replace(
        world,
        species=tuple(world.species[i] for i in order),
        atoms=replace(
            a,
            ids=a.ids[indices],
            positions=a.positions[indices],
            momenta=a.momenta[indices],
            log_width_scale=a.log_width_scale[indices],
        ),
        electrons=replace(world.electrons, kernel=kernel),
    )


def _advance(engine, world, steps):
    for _ in range(steps):
        world = engine.step(world).world
    return world


def run_numerical_controls(*, output="outputs/emergence_controls", steps=20):
    """Use the native synchronous World engine, not the learned predictor MD.

    ZeroEnergy and a constant mobility define explicit manufactured controls.
    Agreement reproduces mathematical rules; it is not evidence of chemistry.
    """
    if steps < 2:
        raise ValueError("At least two steps required for replay controls")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    world = make_world(
        ["H", "H"], [[4.63, 5.27, 5.11], [6.23, 5.27, 5.11]], grid=Grid((18,) * 3, (0.6,) * 3)
    )
    momentum = torch.tensor([[2.0, 3.0, 4.0], [-1.0, 2.0, 3.0]], dtype=torch.float64)
    world = replace(world, atoms=replace(world.atoms, momenta=momentum))
    evaluator = Evaluator(functional=ZeroEnergy())
    engine = SynchronousEngine(
        evaluator,
        dt=0.1,
        relaxer=LocalKernelRelaxer(max_iterations=0),
        require_electronic_convergence=True,
    )
    full = _advance(engine, world, steps)
    prefix = _advance(engine, world, steps // 2)
    save_world(prefix, out / "restart.npz")
    replay = _advance(engine, load_world(out / "restart.npz"), steps - steps // 2)
    order = [1, 0]
    reordered = _advance(engine, _reorder(world, order), steps)
    mass = evaluator(world).context.layout.masses[:, None]
    expected = world.grid.wrap(world.atoms.positions + steps * 0.1 * momentum / mass)
    exact_replay = torch.equal(full.atoms.positions, replay.atoms.positions) and torch.equal(
        full.electrons.kernel.values, replay.electrons.kernel.values
    )
    free_error = float((full.atoms.positions - expected).abs().max())
    order_error = float((full.atoms.positions[order] - reordered.atoms.positions).abs().max())

    # A Fourier mode has a known exact discrete amplification factor.
    grid = Grid((20,) * 3, (0.6,) * 3)
    one = make_world(["H"], [[6.137, 6.234, 6.351]], grid=grid)
    mode = torch.cos(2 * torch.pi * (torch.arange(20, dtype=torch.float64) + 0.5) / 20)
    mode = mode[:, None, None].expand(grid.shape)
    initial = one.latent.channels.clone()
    initial[..., 0] = 1 + 0.1 * mode
    one = one.with_latent(initial)
    rule = ConservativeLatentCA(4).double()
    with torch.no_grad():
        for parameter in rule.parameters():
            parameter.zero_()
    dt = 0.1
    diffusivity = rule.max_mobility * 0.5
    diffusion_engine = SynchronousEngine(
        Evaluator(functional=ZeroEnergy()),
        dt=dt,
        relaxer=LocalKernelRelaxer(max_iterations=0),
        latent_rule=rule,
        require_electronic_convergence=True,
    )
    final = _advance(diffusion_engine, one, steps)
    amplification = 1 - 4 * diffusivity * dt * np.sin(np.pi / 20) ** 2 / 0.6**2
    reference = 1 + 0.1 * amplification**steps * mode
    diffusion_error = float((final.latent.channels[..., 0] - reference).abs().max())
    conserved = float((final.latent.channels.sum((0, 1, 2)) - initial.sum((0, 1, 2))).abs().max())
    checkpoint = out / "diffusion_final.npz"
    save_world(final, checkpoint)
    np.savez_compressed(
        out / "diffusion.npz",
        initial=initial.numpy(),
        final=final.latent.channels.numpy(),
        reference=reference.numpy(),
    )

    # Stochastic nuclear replay must retain both identity and the global step.
    positions = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)
    ids = torch.tensor([77, 15])
    p = torch.zeros_like(positions)

    def harmonic(r):
        return 0.5 * r.square().sum(), -r, {}

    def trajectory(r, m, identity, n, start=0):
        return list(
            integrate_potential(
                r,
                m,
                torch.ones(len(r), dtype=r.dtype),
                harmonic,
                atom_ids=identity,
                steps=n,
                start_step=start,
                dt=0.05,
                kT=0.01,
                friction=0.2,
                seed=91,
            )
        )

    direct = trajectory(positions, p, ids, 12)
    start = trajectory(positions, p, ids, 5)[-1]
    restart = trajectory(start.positions, start.momenta, ids, 7, start.step)
    shuffled = trajectory(positions[order], p[order], ids[order], 12)
    stochastic_replay = torch.equal(direct[-1].positions, restart[-1].positions) and torch.equal(
        direct[-1].momenta, restart[-1].momenta
    )
    stochastic_order = torch.equal(direct[-1].positions[order], shuffled[-1].positions)
    report = {
        "engine": "native SynchronousEngine plus persistent-ID nuclear replay control",
        "steps": steps,
        "free_flight_max_error_bohr": free_error,
        "checkpoint_replay_bitwise": exact_replay,
        "atom_reordering_error_bohr": order_error,
        "diffusion_fourier_max_error": diffusion_error,
        "diffusion_channel_sum_error": conserved,
        "diffusion_discrete_amplification": amplification,
        "stochastic_replay_bitwise": stochastic_replay,
        "stochastic_atom_reordering_bitwise": stochastic_order,
        "acceptance": {
            "coordinate_abs_tolerance_bohr": 1e-11,
            "fourier_abs_tolerance": 1e-11,
            "channel_sum_abs_tolerance": 1e-9,
        },
        "chemistry_claim": False,
        "scope": "manufactured numerical laws, not learned molecular physics",
    }
    report["passed"] = bool(
        exact_replay
        and stochastic_replay
        and stochastic_order
        and free_error < 1e-11
        and order_error < 1e-11
        and diffusion_error < 1e-11
        and conserved < 1e-9
    )
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report


def run_reference_screen(
    *,
    checkpoint="outputs/ho_final_quantum/model.pt",
    dataset="data/ho_pbe0",
    output="outputs/emergence_screen",
):
    """Screen each chemical family and audit deployed force definitions.

    A one-point relative-energy error would be trivially zero, so it cannot
    pass the curve gate. These small validation sets are preliminary screens.
    """
    _, predictor = load_quantum_potential(checkpoint)
    families = {}
    for frame in load_reference_groups(dataset, split="validation"):
        family = (
            frame.group.rsplit("_", 1)[0]
            if frame.group.startswith("external_field")
            else frame.group
        )
        families.setdefault(family, []).append(frame)
    if not families:
        raise ValueError("Independent validation references required")
    rows = []
    for family, frames in sorted(families.items()):
        energy = []
        forces = []
        force_gaps = []
        density = []
        number_error = []
        for frame in frames:
            world = frame_world(frame)
            prediction = predictor(world, create_graph=False)
            context = prediction.evaluation.context
            energy.append(float(prediction.evaluation.energy.detach()) - frame.energy)
            forces.append(
                float((prediction.forces.detach() - torch.tensor(frame.forces)).square().mean())
            )
            fixed = predictor.evaluator.energy_and_forces(context.world.clone()).forces
            force_gaps.append(float((prediction.forces.detach() - fixed).square().mean()))
            number_error.append(
                abs(
                    float(context.quantum.density.detach().sum() * world.grid.dv)
                    - world.electron_count
                )
            )
            if frame.density is not None:
                density.append(
                    float(
                        (context.quantum.density.detach() - torch.tensor(frame.density))
                        .square()
                        .sum()
                    )
                    * world.grid.dv
                    / world.electron_count
                )
        # Constant energy zero is removed using the whole validation family.
        relative = (
            float(np.sqrt(np.mean((np.array(energy) - np.mean(energy)) ** 2)) * HARTREE_EV)
            if len(frames) >= 3
            else None
        )
        force = float(np.sqrt(np.mean(forces)) * HARTREE_EV / BOHR_ANGSTROM)
        gap = float(np.sqrt(np.mean(force_gaps)) * HARTREE_EV / BOHR_ANGSTROM)
        enough = len(frames) >= 5
        accuracy = relative is not None and relative <= 0.043 and force <= 0.1
        rows.append(
            {
                "family": family,
                "validation_frames": len(frames),
                "relative_energy_rmse_ev": relative,
                "force_rmse_ev_per_angstrom": force,
                "full_vs_fixed_electronic_force_rmse_ev_per_angstrom": gap,
                "density_l2_per_electron": float(np.mean(density)) if density else None,
                "maximum_electron_number_error": max(number_error),
                "force_accuracy_passed": force <= 0.1,
                "curve_accuracy_passed": relative is not None and relative <= 0.043,
                "coverage_passed": enough,
                "status": "eligible_for_targeted_dynamics_screen"
                if accuracy and enough
                else ("insufficient_curve_coverage" if not enough else "reference_accuracy_failed"),
                "native_variational_ca_equivalence_verified": False,
            }
        )
    report = {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "families": rows,
        "thresholds": {
            "relative_energy_rmse_ev": 0.043,
            "force_rmse_ev_per_angstrom": 0.1,
            "minimum_validation_geometries": 5,
        },
        "force_definition_audit": "Full finite-predictor derivative versus partial derivative at the same electronic state; not two trajectories with identical physics",
        "scope": "Static preregistered screen; not equilibrium, reaction-rate, or CA-emergence validation",
        "all_families_eligible": all(
            row["status"] == "eligible_for_targeted_dynamics_screen" for row in rows
        ),
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report
