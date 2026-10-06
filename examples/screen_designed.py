"""Measure actual designed DFT surfaces and a fresh model against held-out data.

Usage: python examples/screen_designed.py --checkpoint outputs/designed_quantum/model.pt
Omit --checkpoint for a reference-only report. No trajectory is driven here.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from unichem.datasets import BOHR_ANGSTROM, HARTREE_EV, ReferenceFrame
from unichem.experiment import load_quantum_potential
from unichem.landscapes import field_statistics, surface_statistics
from unichem.reference_training import frame_world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/designed_pbe0")
    parser.add_argument("--dataset", default="data/ho_designed_pbe0")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="outputs/designed_screen")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(args.root)
    jobs = json.loads((root / "jobs.json").read_text())
    predictor = None
    if args.checkpoint:
        payload, predictor = load_quantum_potential(args.checkpoint)
        if payload["dataset"] != args.dataset:
            raise ValueError("A fresh model trained on the frozen designed split is required")
    rows = []
    fd_checks = []
    for index, job in enumerate(jobs):
        with np.load(root / f"{index:04d}.npz", allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            if not meta["scf_converged"] or meta["design_split"] != job["design_split"]:
                raise ValueError("Converged labels matching the frozen split required")
            row = {
                "index": index,
                "family": meta["group"],
                "split": meta["design_split"],
                "coordinates": meta["design_coordinates"],
                "roles": meta["design_roles"],
                "reference_energy_hartree": float(data["energy"]),
                "reference_dipole_au": data["dipole_au"].tolist(),
                "field_force_fd_error_hartree_per_bohr": meta.get("field_force_fd_error"),
                "ao_electron_number_error": float(
                    abs(
                        np.einsum("ij,ji->", data["ao_density_matrix"], data["ao_overlap"])
                        - (data["numbers"].sum() - meta["net_charge"])
                    )
                ),
            }
            with np.load(root / f"{index:04d}.density.npz", allow_pickle=False) as rho:
                density = rho["density"].copy()
            numbers = data["numbers"].copy()
            positions = data["positions"].copy()
            force = data["forces"].copy()
        ne = float(numbers.sum() - 2 * np.count_nonzero(numbers == 8) - meta["net_charge"])
        row["reference_density_integral_relative_error"] = abs(density.sum() * 0.6**3 - ne) / ne
        if predictor is not None:
            frame = ReferenceFrame(
                numbers,
                positions,
                np.array([28.8] * 3),
                row["reference_energy_hartree"],
                force,
                "designed PySCF",
                row["family"],
                row["split"],
                index,
                meta["net_charge"],
                tuple(meta["external_field_au"]),
                density,
                (48, 48, 48),
            )
            world = frame_world(frame)
            prediction = predictor(world, create_graph=False)
            context = prediction.evaluation.context
            q = context.quantum
            dipole = (positions * context.layout.core_charge.detach().numpy()[:, None]).sum(0)
            dipole -= (world.grid.centres(world.atoms.positions) * q.density[:, None]).detach().sum(
                0
            ).numpy() * world.grid.dv
            row.update(
                {
                    "model_energy_hartree": float(prediction.evaluation.energy.detach()),
                    "model_density_dipole_au": dipole.tolist(),
                    "force_mse_au": float(
                        (prediction.forces.detach() - torch.tensor(force)).square().mean()
                    ),
                    "density_l2_per_electron": float(
                        (q.density.detach() - torch.tensor(density)).square().sum()
                    )
                    * world.grid.dv
                    / ne,
                    "electron_number_error": abs(
                        float(q.density.detach().sum() * world.grid.dv) - ne
                    ),
                }
            )
            c = row["coordinates"]
            audit = index == 13 or (
                row["family"] == "symmetric_zundel_surface"
                and c["proton_distance_difference_angstrom"] == 0
            )
            if audit:
                fixed = predictor.evaluator.energy_and_forces(context.world.clone()).forces
                row["full_vs_fixed_force_rmse_ev_per_angstrom"] = (
                    float((prediction.forces.detach() - fixed).square().mean().sqrt())
                    * HARTREE_EV
                    / BOHR_ANGSTROM
                )
            check = index == 13 or (
                row["family"] == "symmetric_zundel_surface"
                and c["oo_angstrom"] == 2.6
                and c["proton_distance_difference_angstrom"] == 0.2
            )
            if check:
                atom = 1 if index == 13 else 2
                eps = 5e-5
                plus, minus = world.atoms.positions.clone(), world.atoms.positions.clone()
                plus[atom, 0] += eps
                minus[atom, 0] -= eps
                ep = float(
                    predictor(
                        world.with_positions(plus), create_graph=False
                    ).evaluation.energy.detach()
                )
                em = float(
                    predictor(
                        world.with_positions(minus), create_graph=False
                    ).evaluation.energy.detach()
                )
                error = abs(-(ep - em) / (2 * eps) - float(prediction.forces[atom, 0].detach()))
                fd_checks.append(
                    {
                        "index": index,
                        "atom": atom,
                        "component": 0,
                        "epsilon_bohr": eps,
                        "force_fd_error_ev_per_angstrom": error * HARTREE_EV / BOHR_ANGSTROM,
                    }
                )
        rows.append(row)
        if (index + 1) % 20 == 0:
            print(json.dumps({"screened": index + 1, "total": len(jobs)}), flush=True)

    surfaces = []
    for distance in sorted(
        {r["coordinates"]["oo_angstrom"] for r in rows if r["family"] == "symmetric_zundel_surface"}
    ):
        scan = sorted(
            [
                r
                for r in rows
                if r["family"] == "symmetric_zundel_surface"
                and r["coordinates"]["oo_angstrom"] == distance
            ],
            key=lambda r: r["coordinates"]["proton_distance_difference_angstrom"],
        )
        delta = [r["coordinates"]["proton_distance_difference_angstrom"] for r in scan]
        ref = surface_statistics(delta, [r["reference_energy_hartree"] for r in scan])
        item = {"oo_angstrom": distance, "split": scan[0]["split"], "reference": ref}
        if predictor is not None:
            pred = surface_statistics(delta, [r["model_energy_hartree"] for r in scan])
            error = np.array(pred["centred_energy_ev"]) - ref["centred_energy_ev"]
            item.update(
                {
                    "model": pred,
                    "centre_anchored_energy_rmse_ev": float(np.sqrt(np.mean(error**2))),
                    "centred_error_rmse_ev": float(np.std(error)),
                    "force_rmse_ev_per_angstrom": float(
                        np.sqrt(np.mean([r["force_mse_au"] for r in scan]))
                    )
                    * HARTREE_EV
                    / BOHR_ANGSTROM,
                    "well_shape_matches": pred["centre_shape"] == ref["centre_shape"],
                }
            )
            item["reference_gate_passed"] = (
                item["centre_anchored_energy_rmse_ev"] <= 0.043
                and item["force_rmse_ev_per_angstrom"] <= 0.1
            )
        surfaces.append(item)

    fields = []
    for axis in range(3):
        response = []
        for row in rows:
            for role in row["roles"]:
                if role["family"] == "water_field_response" and role["coordinates"]["axis"] == axis:
                    response.append((role["coordinates"]["strength_au"], row))
        response.sort(key=lambda pair: pair[0])
        f, members = zip(*response)
        reference = field_statistics(
            f,
            [r["reference_energy_hartree"] for r in members],
            [r["reference_dipole_au"][axis] for r in members],
        )
        item = {"axis": axis, "reference": reference}
        if predictor is not None:
            model = field_statistics(
                f,
                [r["model_energy_hartree"] for r in members],
                [r["model_density_dipole_au"][axis] for r in members],
            )
            errors = [
                abs(m["polarizability_from_energy_bohr3"] - r["polarizability_from_energy_bohr3"])
                / abs(r["polarizability_from_energy_bohr3"])
                for m, r in zip(model, reference)
            ]
            item.update(
                {
                    "model": model,
                    "polarizability_relative_errors": errors,
                    "response_gate_passed": max(errors) <= 0.1,
                    "test_field_magnitude_au": 0.004,
                }
            )
        fields.append(item)

    metrics = []
    if predictor is not None:
        for split in ("training", "validation", "test"):
            for family in sorted({r["family"] for r in rows if r["split"] == split}):
                members = [r for r in rows if r["family"] == family and r["split"] == split]
                errors = np.array(
                    [r["model_energy_hartree"] - r["reference_energy_hartree"] for r in members]
                )
                metrics.append(
                    {
                        "split": split,
                        "family": family,
                        "frames": len(members),
                        "shift_removed_energy_rmse_ev": float(np.std(errors) * HARTREE_EV),
                        "force_rmse_ev_per_angstrom": float(
                            np.sqrt(np.mean([r["force_mse_au"] for r in members]))
                        )
                        * HARTREE_EV
                        / BOHR_ANGSTROM,
                        "density_l2_per_electron": float(
                            np.mean([r["density_l2_per_electron"] for r in members])
                        ),
                    }
                )
    report = {
        "schema_version": 1,
        "source": args.root,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        if args.checkpoint
        else None,
        "design_plan_sha256": hashlib.sha256((root / "jobs.json").read_bytes()).hexdigest(),
        "frames": len(rows),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "maximum_ao_electron_number_error": max(r["ao_electron_number_error"] for r in rows),
        "maximum_reference_density_integral_relative_error": max(
            r["reference_density_integral_relative_error"] for r in rows
        ),
        "maximum_reference_field_force_fd_error_au": max(
            r["field_force_fd_error_hartree_per_bohr"] or 0 for r in rows
        ),
        "metrics": metrics,
        "zundel_surfaces": surfaces,
        "water_field_response": fields,
        "model_force_finite_difference_checks": fd_checks,
        "thresholds": {
            "curve_energy_rmse_ev": 0.043,
            "force_rmse_ev_per_angstrom": 0.1,
            "polarizability_relative_error": 0.1,
        },
        "force_definition": "Total derivative of the identical finite unrolled electronic map used in training",
        "native_variational_ca_equivalence_verified": False,
        "chemistry_emergence_validated": False,
        "scope": "Frozen-environment electronic potential surfaces and field response; not free energy, liquid-water autoionization or rates",
        "rows": rows,
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("rows", "zundel_surfaces", "water_field_response")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
