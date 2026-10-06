"""Physical molecular evaluation, evolution and independent matrix validation."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from .checkpoint import save_world
from .molecule import make_world
from .quantum.evaluator import Evaluator
from .runtime.molecular import SynchronousEngine


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sparse zero-temperature quantum-cell chemistry")
    parser.add_argument("--threads", type=int, default=2)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "simulate"):
        command = commands.add_parser(name)
        command.add_argument("--xyz", required=True)
        command.add_argument("--basis", default="sto-3g")
        command.add_argument("--charge", type=int, default=0)
        command.add_argument("--spin", type=int)
        command.add_argument("--field", type=float, nargs=3, default=(0.0, 0.0, 0.0))
        command.add_argument("--output", default="outputs/physical_run")
        if name == "simulate":
            command.add_argument("--steps", type=int, default=20)
            command.add_argument("--dt-fs", type=float, default=0.05)
    validate = commands.add_parser("validate-core")
    validate.add_argument("--reference", default="data/quantum_oracles")
    validate.add_argument("--output", default="outputs/v04_validation/matrix")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("Positive thread count required")
    torch.set_num_threads(args.threads)
    if args.command == "validate-core":
        from .analysis.validation import validate_matrix_core

        report = validate_matrix_core(args.reference, args.output)
        print(json.dumps(report, indent=2))
        return
    atoms = read(args.xyz)
    world = make_world(
        atoms.get_chemical_symbols(),
        atoms.positions / 0.529177210903,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
    )
    world = replace(world, external_field=tuple(args.field))
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator()
    result = evaluator(world)
    record = {k: v for k, v in asdict(result.electronic).items() if k not in ("densities", "focks")}
    report = {
        "method": "all-electron HF, zero-temperature sparse SCF",
        "basis": world.basis,
        "energy_hartree": result.energy,
        "forces_au": result.forces.tolist(),
        "electronic": record,
        "learned_energy": False,
        "chemical_reaction_statistics_validated": False,
    }
    world = replace(world, densities=result.electronic.densities)
    records = []
    positions = [world.atoms.positions.numpy()]
    if args.command == "simulate":
        if args.steps < 0:
            parser.error("Nonnegative step count required")
        engine = SynchronousEngine(evaluator, dt=args.dt_fs / 0.024188843265857)
        initial = result.energy
        for _ in range(args.steps):
            step = engine.step(world)
            world = step.world
            records.append(
                {
                    **step.diagnostics,
                    "energy_balance_error_hartree": step.diagnostics["total_energy_hartree"]
                    - initial,
                }
            )
            positions.append(world.atoms.positions.numpy())
        np.savez_compressed(
            root / "trajectory.npz",
            positions=np.array(positions),
            time_fs=np.arange(len(positions)) * args.dt_fs,
        )
        report["steps"] = records
    save_world(world, root / "world.npz")
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))
