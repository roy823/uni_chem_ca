from __future__ import annotations

import argparse
import json
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .checkpoint import save_world
from .evaluate import Evaluator
from .factory import make_world
from .grid import Grid
from .models.latent import ConservativeLatentCA
from .quantum.observables import electronic_invariants
from .runtime.electrons import LocalKernelRelaxer
from .runtime.engine import SynchronousEngine
from .training import train_synthetic


def run_demo(*, steps: int = 3, output: str | Path = "outputs/demo", device: str = "cpu") -> dict:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grid = Grid((28, 28, 28), (0.6, 0.6, 0.6))
    world = make_world(
        ["O", "H", "H"],
        [[8.03, 8.07, 8.11], [9.63, 8.07, 8.11], [7.63, 9.57, 8.11]],
        grid=grid,
        device=device,
    )
    # A single latent perturbation makes the synchronous stencil observable.
    channels = world.latent.channels.clone()
    channels[14, 14, 14, 0] = 1.0
    world = replace(world, latent=replace(world.latent, channels=channels))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        latent = ConservativeLatentCA(4).to(device=device, dtype=torch.float64)
    evaluator = Evaluator()
    engine = SynchronousEngine(
        evaluator, dt=0.1, relaxer=LocalKernelRelaxer(max_iterations=3), latent_rule=latent
    )
    save_world(world, output / "world_initial.npz")
    initial = evaluator(world)
    baseline = {
        **electronic_invariants(world, initial.context.quantum),
        "voxels_per_atom": initial.context.support.counts_per_atom().cpu().tolist(),
        "overlapping_voxels": int((initial.context.support.multiplicity() > 1).sum()),
        "cloud_rms_bohr": initial.context.clouds.rms_radius.detach().cpu().tolist(),
    }
    records = []
    for _ in range(steps):
        result = engine.step(world)
        world = result.world
        records.append(result.diagnostics)
    save_world(world, output / "world_final.npz")
    final = evaluator(world)
    cpu = lambda t: t.detach().cpu().numpy()
    np.savez_compressed(
        output / "fields_final.npz",
        electron_density=cpu(final.context.quantum.density.reshape(grid.shape)),
        charge_density=cpu(final.context.fields.charge_density),
        potential=cpu(final.context.fields.potential),
        electric_faces=cpu(final.context.fields.electric_faces),
        support_atom=cpu(final.context.support.atom_index),
        support_voxel=cpu(final.context.support.voxel_index),
        positions=cpu(world.atoms.positions),
        spacing=np.asarray(grid.spacing),
    )
    report = {
        "status": "uncalibrated_metamodel_demo",
        "warning": "NOT DFT, NOT a validated water potential, NOT a reaction-rate prediction",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "grid_shape": list(grid.shape),
        "spacing_bohr": list(grid.spacing),
        "n_atoms": world.n_atoms,
        "n_basis": world.n_basis,
        "initial": baseline,
        "steps": records,
        "latent_sum": float(world.latent.channels.sum()),
        "basis_widths": "illustrative, not measured atomic radii",
    }
    (output / "demo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="uni-chem extended-atom synchronous metamodel")
    parser.add_argument(
        "--threads", type=int, default=1, help="CPU intra-op threads for small demos"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run the multi-voxel atomic-cell demonstration")
    demo.add_argument("--steps", type=int, default=3)
    demo.add_argument("--output", default="outputs/demo")
    demo.add_argument("--device", default="cpu")
    coupled = commands.add_parser(
        "coupled-demo", help="Joint quantum/field/latent demonstration; not calibrated chemistry"
    )
    coupled.add_argument("--steps", type=int, default=3)
    coupled.add_argument("--output", default="outputs/coupled")
    coupled.add_argument("--device", default="cpu")
    train = commands.add_parser(
        "train-demo", help="Synthetic energy/force/density training, not real chemistry"
    )
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--output", default="outputs/training")
    prepare = commands.add_parser(
        "prepare-reference", help="Import verified public and generated DFT datasets"
    )
    prepare.add_argument("--generated", action="store_true")
    reference = commands.add_parser(
        "train-reference", help="Train on real quantum reference labels"
    )
    reference.add_argument("--dataset", default="data/ho_pbe0")
    reference.add_argument("--kind", choices=["quantum", "geometry"], default="quantum")
    reference.add_argument("--epochs", type=int, default=20)
    reference.add_argument("--frames-per-group", type=int, default=16)
    reference.add_argument("--output", default="outputs/ho_final_quantum")
    reference.add_argument("--resume")
    experiment = commands.add_parser(
        "two-water", help="Reference-gated experiment, with no reaction rules"
    )
    experiment.add_argument("--checkpoint", default="outputs/ho_final_quantum/model.pt")
    experiment.add_argument("--dataset", default="data/ho_pbe0")
    experiment.add_argument("--steps", type=int, default=100)
    experiment.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    experiment.add_argument("--dt-fs", type=float, default=0.1)
    experiment.add_argument("--output", default="outputs/two_water")
    experiment.add_argument(
        "--exploratory",
        action="store_true",
        help="Record unvalidated trajectories when reference accuracy fails",
    )
    controls = commands.add_parser(
        "ca-controls", help="Native CA replay, propagation and conservation controls"
    )
    controls.add_argument("--steps", type=int, default=20)
    controls.add_argument("--output", default="outputs/emergence_controls")
    screen = commands.add_parser(
        "reference-screen", help="Per-family reference and force-definition audit"
    )
    screen.add_argument("--checkpoint", default="outputs/ho_final_quantum/model.pt")
    screen.add_argument("--dataset", default="data/ho_pbe0")
    screen.add_argument("--output", default="outputs/emergence_screen")
    design = commands.add_parser(
        "design-reference", help="Write unlabeled water and Zundel reference jobs"
    )
    design.add_argument("--output", default="outputs/reference_design")
    imported = commands.add_parser(
        "import-designed-reference", help="Import labeled designed references with frozen splits"
    )
    imported.add_argument("--root", default="data/designed_pbe0")
    imported.add_argument("--output", default="data/ho_designed_pbe0")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be >=1")
    torch.set_num_threads(args.threads)
    try:
        if args.command == "demo":
            report = run_demo(steps=args.steps, output=args.output, device=args.device)
        elif args.command == "coupled-demo":
            from .demos import run_coupled_demo

            report = run_coupled_demo(steps=args.steps, output=args.output, device=args.device)
        elif args.command == "train-demo":
            report = train_synthetic(epochs=args.epochs, output=args.output)
        elif args.command == "prepare-reference":
            from .datasets import (
                build_monomer_dataset,
                build_public_dataset,
                import_generated_references,
            )

            report = {
                "public_bulk": build_public_dataset(),
                "public_monomers": build_monomer_dataset(),
            }
            if args.generated:
                report["generated"] = import_generated_references()
        elif args.command == "train-reference":
            from .reference_training import train_reference

            report = train_reference(
                dataset=args.dataset,
                kind=args.kind,
                epochs=args.epochs,
                max_frames=args.frames_per_group,
                output=args.output,
                resume=args.resume,
            )
        elif args.command == "two-water":
            from .experiment import run_two_water_experiment

            report = run_two_water_experiment(
                checkpoint=args.checkpoint,
                dataset=args.dataset,
                steps=args.steps,
                seeds=tuple(args.seeds),
                dt_fs=args.dt_fs,
                output=args.output,
                exploratory=args.exploratory,
            )
        elif args.command == "ca-controls":
            from .benchmarks import run_numerical_controls

            report = run_numerical_controls(output=args.output, steps=args.steps)
        elif args.command == "reference-screen":
            from .benchmarks import run_reference_screen

            report = run_reference_screen(
                checkpoint=args.checkpoint, dataset=args.dataset, output=args.output
            )
        elif args.command == "design-reference":
            from .reference_design import write_reference_design

            report = write_reference_design(output=args.output)
        elif args.command == "import-designed-reference":
            from .datasets import import_generated_references

            report = import_generated_references(args.root, args.output, split_policy="manifest")
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"unichem: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
