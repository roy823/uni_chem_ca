"""Two-water experiment and independent reference checks, outside runtime."""

import json
from pathlib import Path

import numpy as np
import torch

from .analysis.reactions import summarize_trajectories
from .analysis.statistics import equilibrium_event_statistics
from .datasets import BOHR_ANGSTROM, HARTREE_EV, load_reference_groups
from .evaluate import Evaluator
from .factory import make_world
from .fields.isolated import IsolatedPoisson
from .grid import Grid
from .models.local_energy import LocalQuantumResidual
from .quantum.density import LocalQuantumBackend
from .runtime.potential import integrate_potential
from .species import HO_RESEARCH_SPECIES
from .training import KernelUnrolledPredictor

AU_TIME_FS = 0.024188843265857


def load_quantum_potential(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != 3
        or payload.get("feature_version") != "joint_patch_sp_v3"
        or payload["kind"] != "quantum"
    ):
        raise ValueError("Current quantum checkpoint required; no model migration/fallback")
    model = LocalQuantumResidual(
        hidden=payload["hidden"], reference_per_atom=payload["reference_per_atom"]
    ).double()
    model.load_state_dict(payload["state_dict"])
    predictor = KernelUnrolledPredictor(
        Evaluator(
            backend=LocalQuantumBackend(metric_degree=payload["metric_degree"]),
            poisson=IsolatedPoisson(),
            residual=model,
        ),
        electronic_steps=payload["electronic_steps"],
        kernel_initializer=model.initialize_kernel,
    )
    return payload, predictor


def run_two_water_experiment(
    *,
    checkpoint="outputs/ho_final_quantum/model.pt",
    dataset="data/ho_pbe0",
    output="outputs/two_water",
    steps=100,
    seeds=(11, 29, 47),
    dt_fs=0.1,
    temperature=300.0,
    exploratory=False,
):
    _payload, predictor = load_quantum_potential(checkpoint)
    validation = load_reference_groups(dataset, split="validation")
    pairs = [f for f in validation if f.group == "neutral_pair_scan"]
    if not pairs:
        raise ValueError("Independent neutral water-pair DFT references required")
    grid = Grid((48, 48, 48), (0.6, 0.6, 0.6))
    symbols = ["H" if z == 1 else "O" for z in pairs[0].atomic_numbers]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    comparison = []
    for f in pairs:
        world = make_world(symbols, f.positions, grid=grid, registry=HO_RESEARCH_SPECIES)
        p = predictor(world, create_graph=False)
        comparison.append(
            {
                "reference_frame": f.index,
                "energy_hartree": float(p.evaluation.energy.detach()),
                "reference_energy_hartree": f.energy,
                "force_rmse_ev_per_angstrom": float(
                    (p.forces.detach() - torch.tensor(f.forces)).square().mean().sqrt()
                )
                * HARTREE_EV
                / BOHR_ANGSTROM,
            }
        )
    pred = np.array([x["energy_hartree"] for x in comparison])
    ref = np.array([x["reference_energy_hartree"] for x in comparison])
    # Relative curves test chemistry independently of atom-wise energy zeros.
    relative_error = float(np.sqrt(np.mean(((pred - pred[0]) - (ref - ref[0])) ** 2)) * HARTREE_EV)
    force_error = float(np.mean([x["force_rmse_ev_per_angstrom"] for x in comparison]))
    passed = relative_error <= 0.043 and force_error <= 0.1
    report = {
        "model": str(checkpoint),
        "reference_dataset": str(dataset),
        "reference_comparison": comparison,
        "relative_pes_rmse_ev": relative_error,
        "mean_force_rmse_ev_per_angstrom": force_error,
        "acceptance_thresholds": {"relative_pes_rmse_ev": 0.043, "force_rmse_ev_per_angstrom": 0.1},
        "reference_accuracy_passed": passed,
        "chemistry_validated": False,
        "runtime_reaction_rules": False,
        "sampling_bias": None,
        "seeds": list(seeds),
        "n_water": 2,
        "dt_fs": dt_fs,
        "temperature_kelvin": temperature,
        "steps": steps,
        "requested_duration_fs_per_seed": steps * dt_fs,
        "nuclear_friction_per_au": 0.001,
        "equilibrated": False,
        "kinetic_clock_validated": False,
        "electronic_force_definition": "total derivative of fixed-count local electronic predictor",
        "free_energy_and_rates_validated": False,
        "scope": "isolated two-water research experiment; not bulk water pKw or physical rate validation",
    }
    if not passed and not exploratory:
        report["status"] = "rejected_by_independent_reference_accuracy"
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
        return report
    numbers = pairs[0].atomic_numbers
    masses = torch.tensor([HO_RESEARCH_SPECIES[s].mass for s in symbols], dtype=torch.float64)
    # Start with the lowest reference energy geometry, not an imposed product.
    initial = pairs[int(np.argmin(ref))].positions
    trajectories = []
    records = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        kT = 3.166811563e-6 * temperature
        momentum = torch.randn(
            len(numbers), 3, generator=generator, dtype=torch.float64
        ) * torch.sqrt(masses[:, None] * kT)
        momentum = momentum - masses[:, None] * (momentum.sum(0) / masses.sum())

        def callback(r):
            radii = r.new_tensor(
                [
                    4.5 * 1.285 * max(o.width for o in HO_RESEARCH_SPECIES[s].orbitals)
                    for s in symbols
                ]
            )
            if bool(((r.amin(-1) <= radii) | ((28.8 - r).amin(-1) <= radii)).any()):
                raise ValueError("Isolated electronic clouds crossed the vacuum boundary")
            world = make_world(symbols, r, grid=grid, registry=HO_RESEARCH_SPECIES)
            p = predictor(world, create_graph=False)
            diagnostics = p.evaluation.context.quantum.diagnostics.copy()
            diagnostics["density_integral"] = float(
                p.evaluation.context.quantum.density.sum().detach() * grid.dv
            )
            context = p.evaluation.context
            diagnostics["assigned_valence_charge"] = (
                (context.layout.core_charge - context.clouds.electron_number).detach().tolist()
            )
            diagnostics["cloud_rms_bohr"] = context.clouds.rms_radius.detach().tolist()
            kernel = context.quantum.density_matrix
            diagnostics["offsite_coherence_l2"] = float(
                kernel.values[kernel.row != kernel.col].detach().norm()
            )
            return p.evaluation.energy.detach(), p.forces.detach(), diagnostics

        saved = []
        diag = []
        try:
            for frame in integrate_potential(
                torch.tensor(initial, dtype=torch.float64),
                momentum,
                masses,
                callback,
                atom_ids=torch.arange(len(numbers)),
                steps=steps,
                dt=dt_fs / AU_TIME_FS,
                kT=kT,
                friction=0.001,
                seed=seed,
            ):
                saved.append(frame.positions.numpy())
                diag.append(
                    {
                        "time_au": frame.time_au,
                        "energy_hartree": frame.energy,
                        "balance_residual_hartree": frame.balance_residual,
                        **frame.diagnostics,
                    }
                )
            status = "completed"
        except (ValueError, RuntimeError) as exc:
            status = "stopped_on_invalid_proposal: " + str(exc)
        trajectories.append(saved)
        records.append({"seed": seed, "status": status, "diagnostics": diag})
        np.savez_compressed(
            out / f"trajectory_{seed}.npz", numbers=numbers, positions=np.array(saved), dt_fs=dt_fs
        )
    observations = summarize_trajectories(numbers, trajectories)
    completed = all(x["status"] == "completed" for x in records)
    status = ("exploratory_trajectory_" if not passed else "reference_checked_trajectory_") + (
        "completed" if completed else "stopped"
    )
    report.update(
        {
            "status": status,
            "trajectories": records,
            "read_only_observations": observations,
            "event_counts": {
                "formations": sum(x["candidate_formations"] for x in observations),
                "recombinations": sum(x["candidate_recombinations"] for x in observations),
                "proton_owner_changes": sum(x["proton_owner_changes"] for x in observations),
            },
            "statistics_status": "insufficient for equilibrium free energy or rate constants; no estimates fabricated",
        }
    )
    report["statistical_analysis"] = equilibrium_event_statistics(
        [[o["ionic_pair_candidate"] for o in x["observations"]] for x in observations],
        dt_fs,
        3.166811563e-6 * temperature,
        equilibrated=False,
        kinetic_clock_validated=False,
    )
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report


def annotate_electronic_trajectories(checkpoint, output="outputs/two_water"):
    """Evaluate electronic observations on saved positions; never alter dynamics."""
    root = Path(output)
    report = json.loads((root / "report.json").read_text(encoding="utf8"))
    _payload, predictor = load_quantum_potential(checkpoint)
    grid = Grid((48, 48, 48), (0.6, 0.6, 0.6))
    for trajectory in report["trajectories"]:
        with np.load(root / f"trajectory_{trajectory['seed']}.npz", allow_pickle=False) as data:
            symbols = ["H" if z == 1 else "O" for z in data["numbers"]]
            for index, r in enumerate(data["positions"]):
                world = make_world(symbols, r, grid=grid, registry=HO_RESEARCH_SPECIES)
                p = predictor(world, create_graph=False)
                context = p.evaluation.context
                charges = (
                    (context.layout.core_charge - context.clouds.electron_number).detach().numpy()
                )
                trajectory["diagnostics"][index]["assigned_valence_charge"] = charges.tolist()
                kernel = context.quantum.density_matrix
                trajectory["diagnostics"][index]["offsite_coherence_l2"] = float(
                    kernel.values[kernel.row != kernel.col].detach().norm()
                )
                observation = report["read_only_observations"][
                    report["seeds"].index(trajectory["seed"])
                ]["observations"][index]
                oxygen = observation["oxygen_ids"]
                hydrogen = np.where(data["numbers"] == 1)[0]
                owners = observation["proton_owners"]
                grouped = [
                    float(
                        charges[o]
                        + sum(charges[h] for h, owner in zip(hydrogen, owners) if owner == o)
                    )
                    for o in oxygen
                ]
                observation["assigned_fragment_charge_e"] = grouped
                observation["electronic_charge_convention"] = (
                    "positive AO density partition with geometric fragment grouping; not a unique observable"
                )
    report["electronic_observations_recomputed_on_saved_positions"] = True
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report
