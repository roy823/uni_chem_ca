"""Reference-checked water curvature and short, explicitly exploratory NVE.

Uses the total unrolled-map force throughout; not native variational CA BO.
No bond, reaction, or frequency restraint is supplied to the integrator.
"""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from unichem.datasets import BOHR_ANGSTROM, HARTREE_EV, ReferenceFrame
from unichem.experiment import AU_TIME_FS, load_quantum_potential
from unichem.landscapes import vibrational_curvatures
from unichem.models.local_energy import LocalEnergyModel
from unichem.reference_training import frame_world
from unichem.runtime.potential import integrate_potential


def water_geometry(positions):
    vectors = np.asarray(positions)[1:] - positions[0]
    lengths = np.linalg.norm(vectors, axis=1)
    angle = np.degrees(np.arccos(np.clip(np.dot(*vectors) / np.prod(lengths), -1, 1)))
    return lengths * BOHR_ANGSTROM, float(angle)


def classify_molecular_control(report):
    optimization = report["geometry_optimization"]
    runs = report["runs"]
    drifts = [r["maximum_energy_balance_error_ev"] for r in runs]
    convergence = all(
        a is not None and b is not None and b > 0 and 3.5 < a / b < 4.5 for a, b in pairwise(drifts)
    )
    passed = (
        report["kind"] == "geometry"
        and report["force_accuracy_passed"]
        and report["stationary_geometry_verified"]
        and report["frequency_accuracy_verified"]
        and optimization["maximum_oh_error_angstrom"] <= 0.02
        and optimization["hoh_error_degrees"] <= 2
        and all(r["status"] == "completed" and r["completed_duration_fs"] >= 999.999 for r in runs)
        and convergence
    )
    report["nve_second_order_convergence_verified"] = convergence
    report["molecular_geometry_control_passed"] = bool(passed)
    report["dynamics_status"] = (
        "molecular_geometry_control_passed" if passed else "exploratory_unvalidated"
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/designed_quantum/model.pt")
    parser.add_argument("--kind", choices=("quantum", "geometry"), default="quantum")
    parser.add_argument("--reference", default="outputs/water_dft_reference/reference.npz")
    parser.add_argument("--output", default="outputs/designed_water")
    parser.add_argument("--duration-fs", type=float, default=10)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.duration_fs <= 0:
        raise ValueError("Positive duration required")
    torch.set_num_threads(args.threads)
    predictor, geometry_model = None, None
    if args.kind == "quantum":
        _, predictor = load_quantum_potential(args.checkpoint)
    else:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if payload["kind"] != "geometry" or payload["feature_version"] != "joint_patch_sp_v3":
            raise ValueError("Current geometry-only control checkpoint required")
        geometry_model = LocalEnergyModel(
            hidden=payload["hidden"], reference_per_atom=payload["reference_per_atom"]
        ).double()
        geometry_model.load_state_dict(payload["state_dict"])
    with np.load(args.reference, allow_pickle=False) as data:
        r = data["positions"].copy()
        fref = data["forces"].copy()
        frequencies = data["frequencies_cm1"].copy()
        href = data["hessian"].transpose(0, 2, 1, 3).reshape(9, 9)
        reference = ReferenceFrame(
            data["numbers"].copy(),
            r,
            np.array([28.8] * 3),
            float(data["energy"]),
            fref,
            "independent optimized PySCF",
            "water",
            "evaluation",
            0,
            density_grid_shape=(48, 48, 48),
        )
    world = frame_world(reference)

    def callback(positions):
        if geometry_model is not None:
            energy, force = geometry_model.energy_and_forces(
                torch.tensor(reference.atomic_numbers),
                positions,
                positions.new_tensor(reference.box),
            )
            return energy.detach(), force.detach(), {"electron_number_error": None}
        prediction = predictor(world.with_positions(positions), create_graph=False)
        energy = prediction.evaluation.energy.detach()
        density = prediction.evaluation.context.quantum.density.detach()
        diagnostics = {
            "electron_number_error": abs(
                float(density.sum() * world.grid.dv) - world.electron_count
            )
        }
        return energy, prediction.forces.detach(), diagnostics

    origin = torch.tensor(r, dtype=torch.float64)
    force = callback(origin)[1].numpy()
    force_error = float(np.sqrt(np.mean((force - fref) ** 2)) * HARTREE_EV / BOHR_ANGSTROM)
    optimized = origin.clone()
    optimization_report = None
    if geometry_model is not None:

        def objective(q):
            candidate = origin.clone()
            candidate[1:] = torch.tensor(q.reshape(2, 3), dtype=torch.float64)
            energy, gradient, _ = callback(candidate)
            return float(energy), -gradient[1:].numpy().ravel()

        result = minimize(
            objective,
            r[1:].ravel(),
            jac=True,
            method="BFGS",
            options={"gtol": 1e-8, "maxiter": 100},
        )
        optimized[1:] = torch.tensor(result.x.reshape(2, 3), dtype=torch.float64)
        length, angle = water_geometry(optimized.numpy())
        ref_length, ref_angle = water_geometry(r)
        max_force = float(abs(callback(optimized)[1]).max())
        optimization_report = {
            "optimizer_success": bool(result.success),
            "message": str(result.message),
            "maximum_force_au": max_force,
            "oh_angstrom": length.tolist(),
            "hoh_degrees": angle,
            "maximum_oh_error_angstrom": float(abs(length - ref_length).max()),
            "hoh_error_degrees": abs(angle - ref_angle),
        }
    origin_energy = float(callback(origin)[0])
    rigid_audit = []
    for axis in range(3):
        shifted = origin.clone()
        shifted[:, axis] += 0.3
        energy, shifted_force, _ = callback(shifted)
        rigid_audit.append(
            {
                "operation": "translation",
                "axis": axis,
                "shift_bohr": 0.3,
                "energy_change_ev": (float(energy) - origin_energy) * HARTREE_EV,
                "force_change_rmse_ev_per_angstrom": float(
                    (shifted_force - torch.tensor(force)).square().mean().sqrt()
                )
                * HARTREE_EV
                / BOHR_ANGSTROM,
            }
        )
    for angle in (45, 90):
        theta = np.deg2rad(angle)
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        )
        rotated = (r - r[0]) @ rotation.T + r[0]
        energy, rotated_force, _ = callback(torch.tensor(rotated, dtype=torch.float64))
        rigid_audit.append(
            {
                "operation": "rotation_z",
                "angle_degrees": angle,
                "energy_change_ev": (float(energy) - origin_energy) * HARTREE_EV,
                "force_covariance_rmse_ev_per_angstrom": float(
                    np.sqrt(np.mean((rotated_force.numpy() - force @ rotation.T) ** 2))
                )
                * HARTREE_EV
                / BOHR_ANGSTROM,
            }
        )
    curvatures = []
    for eps in (0.001, 0.0005):
        h = np.empty((9, 9))
        for axis in range(9):
            plus, minus = optimized.clone(), optimized.clone()
            plus.reshape(-1)[axis] += eps
            minus.reshape(-1)[axis] -= eps
            h[:, axis] = -(callback(plus)[1] - callback(minus)[1]).numpy().ravel() / (2 * eps)
        modes = vibrational_curvatures(optimized.numpy(), h, [15.999, 1.008, 1.008])
        curvatures.append(
            {
                "epsilon_bohr": eps,
                "signed_curvature_cm1": modes.tolist(),
                "hessian_asymmetry_max_au": float(abs(h - h.T).max()),
                "reference_relative_errors": (abs(modes - frequencies) / frequencies).tolist(),
            }
        )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    initial = optimized.clone()
    initial[1:] = optimized[0] + 1.02 * (optimized[1:] - optimized[0])
    masses = torch.tensor([s.mass for s in world.species], dtype=torch.float64)
    for dt_fs in (0.2, 0.1, 0.05):
        steps = round(args.duration_fs / dt_fs)
        if not np.isclose(steps * dt_fs, args.duration_fs):
            raise ValueError("Duration must be an integer number of both time steps")
        positions, times, balances, energies, geometries, number_errors = [], [], [], [], [], []
        status = "completed"
        try:
            for frame in integrate_potential(
                initial,
                torch.zeros_like(initial),
                masses,
                callback,
                atom_ids=torch.arange(3),
                steps=steps,
                dt=dt_fs / AU_TIME_FS,
            ):
                position = frame.positions.numpy()
                lengths, angle = water_geometry(position)
                positions.append(position)
                times.append(frame.time_au * AU_TIME_FS)
                balances.append(frame.balance_residual * HARTREE_EV)
                energies.append(frame.energy)
                geometries.append([*lengths, angle])
                if frame.diagnostics["electron_number_error"] is not None:
                    number_errors.append(frame.diagnostics["electron_number_error"])
                if min(lengths) < 0.75 or max(lengths) > 1.3 or not 70 < angle < 145:
                    status = "stopped_on_departure_from_water_geometry_domain"
                    break
        except (ValueError, RuntimeError) as exc:
            status = "stopped: " + str(exc)
        np.savez_compressed(
            out / f"nve_dt_{dt_fs}.npz",
            positions=np.array(positions),
            time_fs=np.array(times),
            energy_hartree=np.array(energies),
            balance_residual_ev=np.array(balances),
            geometry=np.array(geometries),
        )
        geo = np.array(geometries)
        runs.append(
            {
                "dt_fs": dt_fs,
                "requested_duration_fs": args.duration_fs,
                "completed_duration_fs": times[-1] if times else 0,
                "status": status,
                "maximum_energy_balance_error_ev": max(map(abs, balances), default=None),
                "maximum_electron_number_error": max(number_errors, default=None),
                "oh_range_angstrom": [float(geo[:, :2].min()), float(geo[:, :2].max())]
                if len(geo)
                else None,
                "hoh_range_degrees": [float(geo[:, 2].min()), float(geo[:, 2].max())]
                if len(geo)
                else None,
            }
        )
        print(json.dumps(runs[-1]), flush=True)
    report = {
        "checkpoint": args.checkpoint,
        "kind": args.kind,
        "reference": args.reference,
        "reference_force_rmse_ev_per_angstrom": force_error,
        "net_force_at_reference_ev_per_angstrom": (
            force.sum(0) * HARTREE_EV / BOHR_ANGSTROM
        ).tolist(),
        "rigid_motion_audit": rigid_audit,
        "reference_frequencies_cm1": frequencies.tolist(),
        "reference_projection_error_cm1": float(
            abs(vibrational_curvatures(r, href, [15.999, 1.008, 1.008]) - frequencies).max()
        ),
        "curvature_measurements": curvatures,
        "curvature_geometry": "optimized_geometry_control"
        if geometry_model is not None
        else "DFT_reference_geometry",
        "geometry_optimization": optimization_report,
        "stationary_geometry_verified": optimization_report is not None
        and optimization_report["maximum_force_au"] < 1e-6,
        "frequency_accuracy_verified": optimization_report is not None
        and optimization_report["maximum_force_au"] < 1e-6
        and max(curvatures[-1]["reference_relative_errors"]) <= 0.05,
        "force_accuracy_passed": force_error <= 0.1,
        "initial_condition": "Evaluation geometry with both O-H distances scaled by 1.02, zero momentum, NVE",
        "runs": runs,
        "dynamics_status": "exploratory_unvalidated",
        "force_definition": "Geometry-only control energy derivative"
        if geometry_model is not None
        else "Total derivative of finite unrolled electronic map; identical to training",
        "native_variational_ca_equivalence_verified": False,
        "scope": "NVE stability and reference curvature comparisons; geometry-only control has no electronic response. Reactive chemistry, free energy and rates remain unvalidated",
    }
    report = classify_molecular_control(report)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
